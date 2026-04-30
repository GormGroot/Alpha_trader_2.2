"""
SentimentAnalyzer – NLP-baseret sentimentanalyse med RK3588 NPU acceleration.

Prioritering:
  1. NPU (RK3588) med RKNN FinBERT model  → hurtigst (~5-15ms per tekst)
  2. CPU med HuggingFace FinBERT           → langsomt (~200-500ms per tekst)
  3. Keyword-baseret fallback              → øjeblikkeligt, lavere kvalitet
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime

from loguru import logger

from src.sentiment.news_fetcher import NewsArticle, get_source_credibility


# ── Dataklasser ──────────────────────────────────────────────

@dataclass
class SentimentScore:
    text:       str
    label:      str
    score:      float
    confidence: float
    method:     str = "finbert"
    device:     str = "cpu"

    @property
    def is_positive(self) -> bool:
        return self.score > 0.1

    @property
    def is_negative(self) -> bool:
        return self.score < -0.1


@dataclass
class WeightedSentiment:
    article_id:       str
    title:            str
    source:           str
    raw_score:        float
    source_weight:    float
    age_weight:       float
    relevance_weight: float
    weighted_score:   float

    @property
    def total_weight(self) -> float:
        return self.source_weight * self.age_weight * self.relevance_weight


@dataclass
class AggregatedSentiment:
    symbol:         str
    score:          float
    label:          str
    article_count:  int
    positive_count: int
    negative_count: int
    neutral_count:  int
    top_positive:   str = ""
    top_negative:   str = ""
    confidence:     float = 0.0
    details:        list[WeightedSentiment] = field(default_factory=list)
    npu_accelerated: bool = False

    @property
    def sentiment_ratio(self) -> float:
        if self.negative_count == 0:
            return float("inf") if self.positive_count > 0 else 1.0
        return self.positive_count / self.negative_count


# ── FinBERT CPU Model ────────────────────────────────────────

_finbert_pipeline = None
_finbert_available: bool | None = None


def _load_finbert():
    global _finbert_pipeline, _finbert_available
    if _finbert_available is not None:
        return _finbert_pipeline
    try:
        from transformers import pipeline
        logger.info("[sentiment] Indlæser FinBERT model (CPU)...")
        _finbert_pipeline = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            top_k=None,
            truncation=True,
            max_length=512,
        )
        _finbert_available = True
        logger.info("[sentiment] FinBERT (CPU) indlæst")
        return _finbert_pipeline
    except Exception as exc:
        logger.warning(f"[sentiment] FinBERT ikke tilgængelig: {exc}")
        _finbert_available = False
        return None


def is_finbert_available() -> bool:
    global _finbert_available
    if _finbert_available is not None:
        return _finbert_available
    try:
        import transformers  # noqa
        import torch          # noqa
        return True
    except ImportError:
        _finbert_available = False
        return False


# ── Keyword Fallback ─────────────────────────────────────────

_POSITIVE_WORDS = {
    "upgrade", "beat", "beats", "exceeded", "outperform", "bullish", "surge",
    "surges", "rally", "soar", "soars", "gain", "gains", "profit", "profits",
    "growth", "revenue", "raised", "raises", "buy", "strong", "positive",
    "upside", "breakout", "record", "high", "boost", "boosts", "recovery",
    "approved", "approval", "dividend", "expansion", "partnership", "deal",
    "acquisition", "innovative", "breakthrough", "optimistic", "confident",
    "opjustering", "stigning", "vækst", "overskud", "anbefaler", "køb",
    "positiv", "rekord", "gennembrud", "godkendt", "udbytte",
}

_NEGATIVE_WORDS = {
    "downgrade", "miss", "misses", "missed", "underperform", "bearish",
    "crash", "plunge", "plunges", "decline", "declines", "loss", "losses",
    "layoff", "layoffs", "cut", "cuts", "sell", "weak", "negative",
    "downside", "risk", "warning", "default", "bankruptcy", "fraud",
    "investigation", "lawsuit", "recall", "scandal", "sanctions", "tariff",
    "inflation", "recession", "downturn", "failure", "delay", "delayed",
    "fine", "fined", "penalty", "violation", "shutdown", "closure",
    "nedjustering", "fald", "tab", "underskud", "sælg", "negativ",
    "risiko", "advarsel", "konkurs", "fyringer", "straf",
}


def keyword_sentiment(text: str) -> SentimentScore:
    words = text.lower().split()
    pos   = sum(1 for w in words if w.strip(".,!?;:()\"'") in _POSITIVE_WORDS)
    neg   = sum(1 for w in words if w.strip(".,!?;:()\"'") in _NEGATIVE_WORDS)
    total = pos + neg

    if total == 0:
        return SentimentScore(text[:100], "neutral", 0.0, 0.3, "keyword", "cpu")

    score      = (pos - neg) / total
    confidence = min(1.0, total / 5)
    label      = "positive" if score > 0.1 else "negative" if score < -0.1 else "neutral"
    return SentimentScore(text[:100], label, score, confidence, "keyword", "cpu")


# ── SentimentAnalyzer ────────────────────────────────────────

class SentimentAnalyzer:
    """
    Analysér sentiment med automatisk NPU/CPU/keyword valg.

    Prioritering:
      1. NPU (hvis RKNN model er tilgængelig) → real-time
      2. FinBERT CPU (hvis transformers installeret)
      3. Keyword fallback
    """

    # Class-level cache — deles mellem alle instanser (undgår 328x init-log)
    _shared_npu = None
    _shared_npu_checked = False
    _shared_cpu_model = None
    _shared_cpu_checked = False

    def __init__(
        self,
        use_finbert: bool = True,
        age_halflife_hours: float = 24.0,
        min_confidence: float = 0.3,
    ) -> None:
        self._use_finbert     = use_finbert
        self._age_halflife    = age_halflife_hours
        self._min_confidence  = min_confidence
        self._cpu_model       = None
        self._npu_analyzer    = None
        self._use_npu         = False

        # Try NPU first (shared singleton)
        if use_finbert:
            self._try_init_npu()

        # Fall back to CPU FinBERT (shared singleton)
        if not self._use_npu and use_finbert:
            if not SentimentAnalyzer._shared_cpu_checked:
                SentimentAnalyzer._shared_cpu_model = _load_finbert()
                SentimentAnalyzer._shared_cpu_checked = True
            self._cpu_model = SentimentAnalyzer._shared_cpu_model

    def _try_init_npu(self) -> None:
        """Attempt to initialize NPU sentiment analyzer."""
        if SentimentAnalyzer._shared_npu_checked:
            # Genbrug resultat fra første init
            if SentimentAnalyzer._shared_npu is not None:
                self._npu_analyzer = SentimentAnalyzer._shared_npu
                self._use_npu = True
            return

        SentimentAnalyzer._shared_npu_checked = True
        try:
            from src.ops.npu_accelerator import NPUSentimentAnalyzer
            npu = NPUSentimentAnalyzer()
            if npu.initialize():
                test = npu.analyze("test")
                if test.get("device") in ("NPU", "CPU"):
                    self._npu_analyzer = npu
                    self._use_npu = test.get("device") == "NPU"
                    SentimentAnalyzer._shared_npu = npu
                    logger.info(
                        f"[sentiment] Using {'NPU 🚀' if self._use_npu else 'CPU'} "
                        f"via NPU accelerator"
                    )
        except Exception as e:
            logger.debug(f"[sentiment] NPU init skipped: {e}")

    @property
    def method(self) -> str:
        if self._use_npu:
            return "finbert_npu"
        elif self._cpu_model is not None:
            return "finbert_cpu"
        return "keyword"

    @property
    def device(self) -> str:
        if self._use_npu:
            return "NPU (RK3588)"
        return "CPU"

    # ── Text analysis ─────────────────────────────────────────

    def analyze_text(self, text: str) -> SentimentScore:
        if not text or not text.strip():
            return SentimentScore("", "neutral", 0.0, 0.0, "none", "cpu")

        text = text[:1000]

        # 1. NPU path
        if self._npu_analyzer is not None:
            result = self._npu_analyzer.analyze(text)
            return SentimentScore(
                text=text[:100],
                label=result["label"],
                score=result["score"],
                confidence=result["confidence"],
                method="finbert_npu" if self._use_npu else "finbert_cpu",
                device=result.get("device", "cpu"),
            )

        # 2. CPU FinBERT
        if self._cpu_model is not None:
            return self._analyze_finbert_cpu(text)

        # 3. Keyword fallback
        return keyword_sentiment(text)

    def _analyze_finbert_cpu(self, text: str) -> SentimentScore:
        try:
            results = self._cpu_model(text)
            if isinstance(results, list) and len(results) > 0:
                if isinstance(results[0], list):
                    results = results[0]

                label_scores = {r["label"]: r["score"] for r in results}
                pos = label_scores.get("positive", 0)
                neg = label_scores.get("negative", 0)
                neu = label_scores.get("neutral",  0)

                score      = pos - neg
                confidence = max(pos, neg, neu)
                label      = (
                    "positive" if pos > neg and pos > neu else
                    "negative" if neg > pos and neg > neu else
                    "neutral"
                )
                return SentimentScore(text[:100], label, score, confidence, "finbert", "cpu")
        except Exception as exc:
            logger.warning(f"[sentiment] FinBERT CPU fejl: {exc}")
        return keyword_sentiment(text)

    # ── Batch analysis ────────────────────────────────────────

    def analyze_articles(self, articles: list[NewsArticle]) -> list[SentimentScore]:
        """Analyze list of articles — uses batch NPU if available."""
        if self._npu_analyzer is not None:
            texts   = [f"{a.title}. {a.summary}" for a in articles]
            results = self._npu_analyzer.analyze_batch(texts)
            return [
                SentimentScore(
                    text=texts[i][:100],
                    label=r["label"],
                    score=r["score"],
                    confidence=r["confidence"],
                    method="finbert_npu" if self._use_npu else "finbert_cpu",
                    device=r.get("device", "cpu"),
                )
                for i, r in enumerate(results)
            ]

        # Sequential CPU
        scores = []
        for article in articles:
            text  = f"{article.title}. {article.summary}"
            score = self.analyze_text(text)
            scores.append(score)
        return scores

    # ── Weighted sentiment ────────────────────────────────────

    def compute_weighted_sentiment(
        self, article: NewsArticle, sentiment: SentimentScore
    ) -> WeightedSentiment:
        source_w = get_source_credibility(article.source)
        age_h    = article.age_hours
        age_w    = math.exp(-0.693 * age_h / self._age_halflife)
        rel_w    = article.relevance
        total_w  = source_w * age_w * rel_w
        return WeightedSentiment(
            article_id=article.id,
            title=article.title[:80],
            source=article.source,
            raw_score=sentiment.score,
            source_weight=source_w,
            age_weight=age_w,
            relevance_weight=rel_w,
            weighted_score=sentiment.score * total_w,
        )

    # ── Aggregation ───────────────────────────────────────────

    def aggregate_sentiment(
        self, symbol: str, articles: list[NewsArticle]
    ) -> AggregatedSentiment:
        if not articles:
            return AggregatedSentiment(
                symbol=symbol, score=0.0, label="neutral",
                article_count=0, positive_count=0,
                negative_count=0, neutral_count=0,
            )

        sentiments = self.analyze_articles(articles)

        weighted_list: list[WeightedSentiment] = []
        total_weight = weighted_sum = 0.0
        pos_count = neg_count = neu_count = 0
        top_pos_score = -2.0
        top_neg_score =  2.0
        top_pos_title = top_neg_title = ""

        for article, sentiment in zip(articles, sentiments):
            if sentiment.confidence < self._min_confidence:
                continue

            ws = self.compute_weighted_sentiment(article, sentiment)
            weighted_list.append(ws)

            w = ws.total_weight
            if w > 0:
                total_weight += w
                weighted_sum += ws.weighted_score

            if sentiment.score > 0.1:
                pos_count += 1
                if sentiment.score > top_pos_score:
                    top_pos_score = sentiment.score
                    top_pos_title = article.title
            elif sentiment.score < -0.1:
                neg_count += 1
                if sentiment.score < top_neg_score:
                    top_neg_score = sentiment.score
                    top_neg_title = article.title
            else:
                neu_count += 1

        final_score = (weighted_sum / total_weight) if total_weight > 0 else 0.0
        final_score = max(-1.0, min(1.0, final_score))

        label = (
            "bullish" if final_score > 0.15 else
            "bearish" if final_score < -0.15 else
            "neutral"
        )

        n = len(weighted_list)
        confidence = min(1.0, n / 10) * 0.5 + abs(final_score) * 0.5

        return AggregatedSentiment(
            symbol=symbol,
            score=final_score,
            label=label,
            article_count=n,
            positive_count=pos_count,
            negative_count=neg_count,
            neutral_count=neu_count,
            top_positive=top_pos_title,
            top_negative=top_neg_title,
            confidence=confidence,
            details=weighted_list,
            npu_accelerated=self._use_npu,
        )

    # ── Convenience ───────────────────────────────────────────

    def quick_score(self, text: str) -> float:
        return self.analyze_text(text).score

    def batch_scores(self, texts: list[str]) -> list[float]:
        return [self.quick_score(t) for t in texts]
