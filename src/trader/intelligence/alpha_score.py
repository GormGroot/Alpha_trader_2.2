"""
Alpha Score Engine — samlet score (0-100) per aktie.

Kombinerer ALLE datakilder til ét enkelt tal der indikerer styrken
af et instrument. Bygger på eksisterende moduler i codebasen.

Dimensioner:
  1. Tekniske indikatorer (RSI, MACD, SMA, BB, ADX, volume) → 25%
  2. Sentiment (FinBERT + nyheder + social)                  → 20%
  3. ML-ensemble prediction (XGBoost, RF, LogReg)            → 20%
  4. Makro-regime tilpasning                                  → 10%
  5. Alternativ data (Google Trends, insider, options flow)   → 15%
  6. Sæsonmønstre + earnings proximity                       → 10%

Signaler:
  - STRONG_BUY:  > 80
  - BUY:         65-80
  - HOLD:        35-65
  - SELL:        20-35
  - STRONG_SELL: < 20
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

# Eksisterende moduler
from src.data.indicators import (
    add_all_indicators,
    add_advanced_indicators,
    add_adx,
    add_atr,
)
from src.data.market_data import MarketDataFetcher
from src.sentiment.news_fetcher import NewsFetcher
from src.sentiment.sentiment_analyzer import SentimentAnalyzer, AggregatedSentiment
from src.sentiment.event_detector import EventDetector, EventSentiment, EventImpact


# ── Dataklasser ──────────────────────────────────────────────

@dataclass
class SubScore:
    """Score for én dimension."""
    name: str
    score: float               # 0-100
    weight: float              # 0-1 (procentuel andel)
    confidence: float          # 0-1 (hvor sikker er vi)
    details: dict[str, Any] = field(default_factory=dict)
    explanation: str = ""


@dataclass
class AlphaScore:
    """Samlet Alpha Score for ét symbol."""
    symbol: str
    total: float               # 0-100 vægtet samlet score
    signal: str                # STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL
    confidence: float          # 0-1 samlet confidence
    breakdown: dict[str, SubScore] = field(default_factory=dict)
    explanation: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    previous_score: float | None = None  # Forrige score til trend-tracking

    @property
    def trend(self) -> str:
        """Score-trend vs. forrige beregning."""
        if self.previous_score is None:
            return "new"
        diff = self.total - self.previous_score
        if diff > 5:
            return "improving"
        elif diff < -5:
            return "deteriorating"
        return "stable"

    @property
    def weighted_confidence(self) -> float:
        """Confidence vægtet efter sub-score confidence."""
        if not self.breakdown:
            return 0.0
        total_w = 0.0
        conf_sum = 0.0
        for ss in self.breakdown.values():
            total_w += ss.weight
            conf_sum += ss.confidence * ss.weight
        return conf_sum / total_w if total_w > 0 else 0.0

    def to_dict(self) -> dict:
        """Konvertér til dict for serialisering."""
        return {
            "symbol": self.symbol,
            "total": round(self.total, 1),
            "signal": self.signal,
            "confidence": round(self.confidence, 2),
            "trend": self.trend,
            "breakdown": {
                name: {
                    "score": round(ss.score, 1),
                    "weight": ss.weight,
                    "confidence": round(ss.confidence, 2),
                    "explanation": ss.explanation,
                }
                for name, ss in self.breakdown.items()
            },
            "explanation": self.explanation,
            "timestamp": self.timestamp.isoformat(),
        }


# ── Signal-tærskelværdier ───────────────────────────────────

SIGNAL_THRESHOLDS = {
    "STRONG_BUY": 80,
    "BUY": 65,
    "HOLD": 35,
    "SELL": 20,
    # Alt under 20 = STRONG_SELL
}


def score_to_signal(score: float) -> str:
    """Konvertér numerisk score til signal-label."""
    if score >= SIGNAL_THRESHOLDS["STRONG_BUY"]:
        return "STRONG_BUY"
    elif score >= SIGNAL_THRESHOLDS["BUY"]:
        return "BUY"
    elif score >= SIGNAL_THRESHOLDS["HOLD"]:
        return "HOLD"
    elif score >= SIGNAL_THRESHOLDS["SELL"]:
        return "SELL"
    return "STRONG_SELL"


# ── Vægte ────────────────────────────────────────────────────

DEFAULT_WEIGHTS = {
    "technicals": 0.25,
    "sentiment": 0.20,
    "ml_prediction": 0.20,
    "macro": 0.10,
    "alternative": 0.15,
    "seasonality": 0.10,
}


# ── Alpha Score Engine ───────────────────────────────────────

class AlphaScoreEngine:
    """
    Beregner Alpha Score for aktier ved at kombinere alle datakilder.

    Brug:
        engine = AlphaScoreEngine()
        score = engine.calculate(symbol="AAPL")
        print(f"{score.symbol}: {score.total:.0f} ({score.signal})")

        # Batch
        scores = engine.calculate_batch(["AAPL", "MSFT", "NOVO-B.CO"])
        ranked = engine.rank(scores)
    """

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        finnhub_key: str = "",
        av_key: str = "",
        fred_key: str = "",
        cache_dir: str = "data_cache",
    ) -> None:
        self._weights = weights or DEFAULT_WEIGHTS.copy()
        self._cache_dir = cache_dir

        # Initialiser eksisterende moduler
        self._data_fetcher = MarketDataFetcher()
        self._news_fetcher = NewsFetcher(
            finnhub_key=finnhub_key,
            av_key=av_key,
            cache_dir=cache_dir,
        )
        self._sentiment = SentimentAnalyzer()
        self._event_detector = EventDetector()

        # Score-historik for trend-tracking
        self._score_history: dict[str, list[tuple[datetime, float]]] = {}

    # ── Hoved-metode ─────────────────────────────────────────

    def calculate(self, symbol: str, df: pd.DataFrame | None = None) -> AlphaScore:
        """
        Beregn Alpha Score for ét symbol.

        Args:
            symbol: Aktiesymbol (f.eks. "AAPL", "NOVO-B.CO").
            df: Valgfrit — forudhentet OHLCV DataFrame. Hentes automatisk hvis None.

        Returns:
            AlphaScore med total, signal, breakdown og forklaring.
        """
        logger.info(f"[alpha_score] Beregner score for {symbol}...")

        # Hent data hvis ikke leveret
        if df is None:
            try:
                df = self._data_fetcher.fetch(symbol, period="1y")
            except Exception as exc:
                logger.error(f"[alpha_score] Kunne ikke hente data for {symbol}: {exc}")
                return self._empty_score(symbol, f"Datafejl: {exc}")

        if df is None or df.empty or len(df) < 30:
            return self._empty_score(symbol, "Utilstrækkeligt data (< 30 bars)")

        # Tilføj indikatorer
        try:
            df = add_all_indicators(df)
            df = add_advanced_indicators(df, fibonacci=False, ichimoku=False,
                                         volume_profile=False, momentum=True,
                                         volatility=True)
        except Exception as exc:
            logger.warning(f"[alpha_score] Indikator-fejl for {symbol}: {exc}")

        # Beregn sub-scores
        breakdown: dict[str, SubScore] = {}

        breakdown["technicals"] = self._technical_score(symbol, df)
        breakdown["sentiment"] = self._sentiment_score(symbol)
        breakdown["ml_prediction"] = self._ml_score(symbol, df)
        breakdown["macro"] = self._macro_score(symbol, df)
        breakdown["alternative"] = self._alternative_score(symbol)
        breakdown["seasonality"] = self._seasonality_score(symbol, df)

        # Vægtet total — confidence-vægtet så default-50 scores med lav confidence tæller mindre
        total = 0.0
        effective_weight_sum = 0.0
        for name, ss in breakdown.items():
            w = self._weights.get(name, 0.0)
            ss.weight = w
            effective_w = w * ss.confidence
            total += ss.score * effective_w
            effective_weight_sum += effective_w

        if effective_weight_sum > 0:
            total = total / effective_weight_sum
        else:
            total = 50.0
        total = max(0.0, min(100.0, total))

        # Confidence = vægtet gennemsnit af sub-confidence
        confidence = sum(
            ss.confidence * ss.weight for ss in breakdown.values()
        )

        # Hent forrige score
        previous = None
        history = self._score_history.get(symbol, [])
        if history:
            previous = history[-1][1]

        # Gem i historik
        now = datetime.now()
        if symbol not in self._score_history:
            self._score_history[symbol] = []
        self._score_history[symbol].append((now, total))
        # Hold max 100 historiske scores
        self._score_history[symbol] = self._score_history[symbol][-100:]

        # Generér forklaring
        explanation = self._generate_explanation(symbol, total, breakdown)

        alpha_score = AlphaScore(
            symbol=symbol,
            total=total,
            signal=score_to_signal(total),
            confidence=confidence,
            breakdown=breakdown,
            explanation=explanation,
            timestamp=now,
            previous_score=previous,
        )

        logger.info(
            f"[alpha_score] {symbol}: {total:.0f} ({alpha_score.signal}) "
            f"[conf={confidence:.0%}] [{alpha_score.trend}]"
        )
        return alpha_score

    def calculate_batch(self, symbols: list[str]) -> list[AlphaScore]:
        """Beregn Alpha Score for flere symboler."""
        scores = []
        for symbol in symbols:
            try:
                scores.append(self.calculate(symbol))
            except Exception as exc:
                logger.error(f"[alpha_score] Fejl for {symbol}: {exc}")
                scores.append(self._empty_score(symbol, str(exc)))
        return scores

    def rank(self, scores: list[AlphaScore]) -> list[AlphaScore]:
        """Rangér scores fra højest til lavest."""
        return sorted(scores, key=lambda s: s.total, reverse=True)

    def get_top(self, symbols: list[str], n: int = 10) -> list[AlphaScore]:
        """Beregn og returnér top-N aktier efter Alpha Score."""
        all_scores = self.calculate_batch(symbols)
        ranked = self.rank(all_scores)
        return ranked[:n]

    # ── Sub-score beregninger ────────────────────────────────

    def _technical_score(self, symbol: str, df: pd.DataFrame) -> SubScore:
        """
        Teknisk analyse score (0-100).

        Baseret på:
          - RSI position (oversold=høj, overbought=lav)
          - MACD signal (bullish cross=høj)
          - Price vs SMA20/50/200
          - Bollinger Band position
          - Volume confirmation
          - ADX trend strength
        """
        try:
            last = df.iloc[-1]
            scores: list[float] = []
            details: dict[str, Any] = {}

            # 1. RSI (0-100) → oversold er positivt (contrarian)
            rsi = last.get("RSI", 50)
            if pd.notna(rsi):
                if rsi < 30:
                    rsi_score = 80 + (30 - rsi)  # Oversold = stærkt signal
                elif rsi < 40:
                    rsi_score = 65
                elif rsi < 60:
                    rsi_score = 50  # Neutral
                elif rsi < 70:
                    rsi_score = 35
                else:
                    rsi_score = 20 - (rsi - 70) * 0.5  # Overbought
                rsi_score = max(0, min(100, rsi_score))
                scores.append(rsi_score)
                details["rsi"] = round(rsi, 1)
                details["rsi_score"] = round(rsi_score, 1)

            # 2. MACD signal
            macd = last.get("MACD", 0)
            macd_signal = last.get("MACD_Signal", 0)
            macd_hist = last.get("MACD_Hist", 0)
            if pd.notna(macd) and pd.notna(macd_signal):
                if macd > macd_signal and macd_hist > 0:
                    macd_score = 75
                    if len(df) >= 2:
                        prev_hist = df.iloc[-2].get("MACD_Hist", 0)
                        if pd.notna(prev_hist) and prev_hist <= 0 and macd_hist > 0:
                            macd_score = 90  # Bullish crossover
                elif macd < macd_signal and macd_hist < 0:
                    macd_score = 25
                    if len(df) >= 2:
                        prev_hist = df.iloc[-2].get("MACD_Hist", 0)
                        if pd.notna(prev_hist) and prev_hist >= 0 and macd_hist < 0:
                            macd_score = 10  # Bearish crossover
                else:
                    macd_score = 50
                scores.append(macd_score)
                details["macd_score"] = round(macd_score, 1)

            # 3. Price vs SMA (trend)
            close = last.get("Close", 0)
            sma_scores = []
            for sma_col in ["SMA_20", "SMA_50", "SMA_200"]:
                sma_val = last.get(sma_col)
                if pd.notna(sma_val) and sma_val > 0:
                    pct_above = (close - sma_val) / sma_val * 100
                    # Over SMA = positiv, under = negativ
                    sma_s = 50 + pct_above * 5  # Skaler: +10% over SMA → 100
                    sma_scores.append(max(0, min(100, sma_s)))

            if sma_scores:
                trend_score = statistics.mean(sma_scores)
                scores.append(trend_score)
                details["trend_score"] = round(trend_score, 1)

            # 4. Bollinger Band position
            bb_upper = last.get("BB_Upper")
            bb_lower = last.get("BB_Lower")
            if pd.notna(bb_upper) and pd.notna(bb_lower) and bb_upper > bb_lower:
                bb_pos = (close - bb_lower) / (bb_upper - bb_lower)
                # Near lower band = oversold (contrarian bullish)
                bb_score = 100 - bb_pos * 100  # 0 at top, 100 at bottom
                scores.append(max(0, min(100, bb_score)))
                details["bb_position"] = round(bb_pos, 2)

            # 5. Volume confirmation
            vol_ratio = last.get("Volume_Ratio")
            if pd.notna(vol_ratio):
                # High volume on up move = confirmed strength
                ret_1d = (close / df.iloc[-2]["Close"] - 1) if len(df) >= 2 else 0
                if ret_1d > 0 and vol_ratio > 1.2:
                    vol_score = 70 + min(30, (vol_ratio - 1) * 30)
                elif ret_1d < 0 and vol_ratio > 1.2:
                    vol_score = 30 - min(30, (vol_ratio - 1) * 15)
                else:
                    vol_score = 50
                scores.append(max(0, min(100, vol_score)))
                details["volume_ratio"] = round(vol_ratio, 2)

            # 6. ADX trend strength
            adx = last.get("ADX")
            plus_di = last.get("Plus_DI")
            minus_di = last.get("Minus_DI")
            if pd.notna(adx):
                if adx > 25:
                    # Strong trend — use direction
                    if pd.notna(plus_di) and pd.notna(minus_di):
                        adx_score = 70 if plus_di > minus_di else 30
                    else:
                        adx_score = 50
                else:
                    adx_score = 45  # Weak/no trend
                scores.append(adx_score)
                details["adx"] = round(adx, 1)

            # Saml
            final = statistics.mean(scores) if scores else 50.0
            confidence = min(1.0, len(scores) / 6)

            return SubScore(
                name="technicals",
                score=final,
                weight=self._weights.get("technicals", 0.25),
                confidence=confidence,
                details=details,
                explanation=self._tech_explanation(final, details),
            )

        except Exception as exc:
            logger.warning(f"[alpha_score] Technical score fejl for {symbol}: {exc}")
            return SubScore("technicals", 50.0, 0.25, 0.1,
                            explanation=f"Fejl: {exc}")

    def _sentiment_score(self, symbol: str) -> SubScore:
        """
        Sentiment score (0-100).

        Baseret på FinBERT/keyword analyse af seneste nyheder.
        """
        try:
            # Hent nyheder
            articles = self._news_fetcher.fetch_company_news(symbol, days_back=3)

            if not articles:
                return SubScore("sentiment", 50.0, 0.20, 0.2,
                                explanation="Ingen nyheder fundet — neutral")

            # Aggregér sentiment
            agg: AggregatedSentiment = self._sentiment.aggregate_sentiment(
                symbol, articles
            )

            # Konvertér -1/+1 score til 0-100
            # -1 → 0, 0 → 50, +1 → 100
            score = (agg.score + 1) * 50
            score = max(0, min(100, score))

            # Detekter events (from articles, returns DetectedEvent objects)
            events = self._event_detector.detect_from_articles(articles)
            event_bonus = 0.0
            event_texts: list[str] = []
            for event in events:
                if event.sentiment == EventSentiment.BULLISH:
                    bonus = 5 if event.impact == EventImpact.HIGH else 2
                    event_bonus += bonus
                    event_texts.append(f"+{bonus}: {event.event_type.value}")
                elif event.sentiment == EventSentiment.BEARISH:
                    penalty = 5 if event.impact == EventImpact.HIGH else 2
                    event_bonus -= penalty
                    event_texts.append(f"-{penalty}: {event.event_type.value}")

            score = max(0, min(100, score + event_bonus))

            details = {
                "raw_sentiment": round(agg.score, 3),
                "article_count": agg.article_count,
                "positive": agg.positive_count,
                "negative": agg.negative_count,
                "neutral": agg.neutral_count,
                "events": event_texts,
                "top_positive": agg.top_positive[:60] if agg.top_positive else "",
                "top_negative": agg.top_negative[:60] if agg.top_negative else "",
            }

            label = agg.label  # "bullish", "bearish", "neutral"
            explanation = (
                f"Sentiment: {label} ({agg.score:+.2f}) baseret på "
                f"{agg.article_count} artikler. "
                f"{agg.positive_count} pos / {agg.negative_count} neg."
            )
            if event_texts:
                explanation += f" Events: {', '.join(event_texts[:3])}"

            return SubScore(
                name="sentiment",
                score=score,
                weight=self._weights.get("sentiment", 0.20),
                confidence=agg.confidence,
                details=details,
                explanation=explanation,
            )

        except Exception as exc:
            logger.warning(f"[alpha_score] Sentiment score fejl for {symbol}: {exc}")
            return SubScore("sentiment", 50.0, 0.20, 0.1,
                            explanation=f"Fejl: {exc}")

    def _ml_score(self, symbol: str, df: pd.DataFrame) -> SubScore:
        """
        ML-ensemble prediction score (0-100).

        Phase A2: Hvis ingen trænede modeller er tilgængelige, returneres
        et SubScore med confidence=0 så det IKKE påvirker det vægtede
        gennemsnit (calculate() bruger confidence-vægtet aggregation).
        """
        try:
            from src.strategy.ml_strategy import MLStrategy
            from src.strategy.ensemble_ml_strategy import EnsembleMLStrategy
            from pathlib import Path

            scores_list: list[float] = []
            details: dict[str, Any] = {}
            trained_models = 0

            # 1. Basis ML — kun hvis trænet
            ml_path = Path("models/ml_latest.joblib")
            if ml_path.exists():
                try:
                    ml = MLStrategy.load(ml_path)
                    if getattr(ml, "is_trained", False):
                        trained_models += 1
                        ml_result = ml.analyze(df)
                        if ml_result and ml_result.signal != Signal.HOLD:
                            if ml_result.signal == Signal.BUY:
                                ml_s = 50 + ml_result.confidence * 50
                            else:
                                ml_s = 50 - ml_result.confidence * 50
                            scores_list.append(max(0, min(100, ml_s)))
                            details["ml_basic"] = round(ml_s, 1)
                except Exception as exc:
                    logger.debug(f"[alpha_score] ml_basic skip: {exc}")

            # 2. Ensemble ML — kun hvis trænet
            ens_path = Path("models/ensemble_latest.joblib")
            if ens_path.exists():
                try:
                    ensemble = EnsembleMLStrategy.load(ens_path)
                    if getattr(ensemble, "is_trained", False):
                        trained_models += 1
                        ens_result = ensemble.analyze(df)
                        if ens_result and ens_result.signal != Signal.HOLD:
                            if ens_result.signal == Signal.BUY:
                                ens_s = 50 + ens_result.confidence * 50
                            else:
                                ens_s = 50 - ens_result.confidence * 50
                            scores_list.append(max(0, min(100, ens_s)))
                            details["ml_ensemble"] = round(ens_s, 1)
                except Exception as exc:
                    logger.debug(f"[alpha_score] ml_ensemble skip: {exc}")

            if trained_models == 0:
                # Ingen trænede modeller — returnér confidence=0 så det ekskluderes
                # fra weighted average i calculate()
                return SubScore(
                    name="ml_prediction",
                    score=50.0,
                    weight=self._weights.get("ml_prediction", 0.20),
                    confidence=0.0,
                    details={"trained_models": 0},
                    explanation="ML-modeller ikke trænet — ekskluderet fra score",
                )

            score = statistics.mean(scores_list) if scores_list else 50.0
            confidence = min(1.0, trained_models / 2.0)

            explanation = (
                f"ML prediction: {score:.0f}/100 ({trained_models} trænede modeller). "
                f"Basis={details.get('ml_basic', 'N/A')}, "
                f"Ensemble={details.get('ml_ensemble', 'N/A')}"
            )

            return SubScore(
                name="ml_prediction",
                score=score,
                weight=self._weights.get("ml_prediction", 0.20),
                confidence=confidence,
                details=details,
                explanation=explanation,
            )

        except ImportError:
            logger.debug("[alpha_score] ML moduler ikke tilgængelige")
            return SubScore("ml_prediction", 50.0, 0.20, 0.0,
                            explanation="ML moduler ikke installeret")
        except Exception as exc:
            logger.warning(f"[alpha_score] ML score fejl for {symbol}: {exc}")
            return SubScore("ml_prediction", 50.0, 0.20, 0.0,
                            explanation=f"Fejl: {exc}")

    def _macro_score(self, symbol: str, df: pd.DataFrame) -> SubScore:
        """
        Makro-regime tilpasning score (0-100).

        Vurderer om aktien passer til det nuværende markedsregime.
        """
        try:
            from src.strategy.regime import RegimeDetector, MarketRegime

            detector = RegimeDetector()

            # Brug S&P 500 som markedsdata (eller symbolets eget data)
            regime_result = detector.detect(df)
            regime = regime_result.regime if hasattr(regime_result, 'regime') else MarketRegime.SIDEWAYS

            # Sektor-mapping (simpel)
            sector = self._guess_sector(symbol)

            # Scoring baseret på regime + sektor
            regime_scores = {
                MarketRegime.BULL: {"tech": 85, "cyclical": 80, "defensive": 55, "default": 70},
                MarketRegime.BEAR: {"tech": 25, "cyclical": 20, "defensive": 70, "default": 35},
                MarketRegime.SIDEWAYS: {"tech": 50, "cyclical": 45, "defensive": 55, "default": 50},
                MarketRegime.CRASH: {"tech": 15, "cyclical": 10, "defensive": 60, "default": 20},
                MarketRegime.RECOVERY: {"tech": 75, "cyclical": 80, "defensive": 45, "default": 65},
                MarketRegime.EUPHORIA: {"tech": 60, "cyclical": 55, "defensive": 40, "default": 50},
            }

            score_map = regime_scores.get(regime, {"default": 50})
            score = score_map.get(sector, score_map.get("default", 50))

            details = {
                "regime": regime.value if hasattr(regime, 'value') else str(regime),
                "sector": sector,
            }

            explanation = (
                f"Regime: {regime.value if hasattr(regime, 'value') else regime}. "
                f"Sektor '{sector}' scores {score}/100 i dette regime."
            )

            return SubScore(
                name="macro",
                score=score,
                weight=self._weights.get("macro", 0.10),
                confidence=0.5,
                details=details,
                explanation=explanation,
            )

        except Exception as exc:
            logger.warning(f"[alpha_score] Macro score fejl: {exc}")
            return SubScore("macro", 50.0, 0.10, 0.2,
                            explanation=f"Fejl: {exc}")

    def _alternative_score(self, symbol: str) -> SubScore:
        """
        Alternativ data score (0-100).

        Bruger Google Trends, insider trading, options flow.
        """
        try:
            scores_list: list[float] = []
            details: dict[str, Any] = {}

            # 1. Google Trends (via AlternativeDataTracker)
            try:
                from src.data.alternative_data import AlternativeDataTracker
                alt = AlternativeDataTracker()
                trends = alt.get_google_trends(symbol)
                if trends is not None:
                    trend_score = trends.score  # 0-100, pre-calculated
                    scores_list.append(max(20, min(80, trend_score)))
                    details["google_trends"] = round(trend_score, 1)
                    details["trend_direction"] = trends.trend_direction.value
                    details["trend_change_30d"] = round(trends.change_pct_30d, 1)
                    if trends.spike_detected:
                        details["trend_spike"] = True
            except Exception:
                pass

            # 2. Options flow (via OptionsFlowTracker)
            try:
                from src.data.options_flow import OptionsFlowTracker
                options = OptionsFlowTracker()
                pcr = options.get_put_call_ratio(symbol)
                if pcr:
                    pc_ratio = pcr.ratio
                    # Low P/C = bullish, High P/C = bearish (eller contrarian bullish)
                    if pc_ratio < 0.7:
                        opt_score = 70  # Bullish sentiment
                    elif pc_ratio < 1.0:
                        opt_score = 55
                    elif pc_ratio < 1.3:
                        opt_score = 45
                    else:
                        opt_score = 60  # Extreme fear = contrarian
                    scores_list.append(opt_score)
                    details["put_call_ratio"] = round(pc_ratio, 2)
                    details["options_score"] = opt_score
                    details["pc_interpretation"] = pcr.interpretation
            except Exception:
                pass

            # 3. Insider trading (via InsiderSentimentScore)
            try:
                from src.data.insider_tracking import InsiderTracker
                insider = InsiderTracker()
                sentiment = insider.get_insider_sentiment(symbol, lookback_days=90)
                if sentiment:
                    # Score ranges from -100 to +100, map to 10-90
                    ins_score = 50 + (sentiment.score / 100) * 40  # -100→10, 0→50, +100→90
                    # Boost for cluster buying and C-suite activity
                    if sentiment.cluster_buying:
                        ins_score = min(ins_score + 10, 90)
                    if sentiment.c_suite_buying:
                        ins_score = min(ins_score + 5, 90)
                    scores_list.append(max(10, min(90, ins_score)))
                    details["insider_score"] = round(sentiment.score, 1)
                    details["insider_sentiment"] = sentiment.sentiment.value
                    details["cluster_buying"] = sentiment.cluster_buying
                    details["c_suite_buying"] = sentiment.c_suite_buying
            except Exception:
                pass

            # Saml
            score = statistics.mean(scores_list) if scores_list else 50.0
            confidence = min(1.0, len(scores_list) / 3)

            explanation = f"Alt data: {len(scores_list)}/3 kilder tilgængelige."
            if details:
                explanation += f" Detaljer: {details}"

            return SubScore(
                name="alternative",
                score=score,
                weight=self._weights.get("alternative", 0.15),
                confidence=confidence,
                details=details,
                explanation=explanation,
            )

        except Exception as exc:
            logger.warning(f"[alpha_score] Alt data fejl: {exc}")
            return SubScore("alternative", 50.0, 0.15, 0.1,
                            explanation=f"Fejl: {exc}")

    def _seasonality_score(self, symbol: str, df: pd.DataFrame) -> SubScore:
        """
        Sæsonmønstre score (0-100).

        Baseret på historisk performance denne måned + earnings proximity.
        """
        try:
            details: dict[str, Any] = {}
            scores_list: list[float] = []

            # 1. Historisk månedsperformance
            if len(df) > 252:
                month = datetime.now().month
                df_copy = df.copy()
                df_copy["month"] = pd.to_datetime(df_copy.index).month
                df_copy["return"] = df_copy["Close"].pct_change()

                monthly = df_copy[df_copy["month"] == month]["return"]
                if len(monthly) > 10:
                    avg_return = monthly.mean()
                    win_rate = (monthly > 0).mean()

                    # Positiv historisk return denne måned = høj score
                    seasonal_score = 50 + avg_return * 5000  # Skaler
                    seasonal_score = max(20, min(80, seasonal_score))
                    scores_list.append(seasonal_score)
                    details["month_avg_return"] = round(avg_return * 100, 2)
                    details["month_win_rate"] = round(win_rate * 100, 1)

            # 2. Earnings proximity
            try:
                earnings = self._news_fetcher.fetch_earnings_calendar(symbol)
                if earnings:
                    now = datetime.now()
                    for e in earnings:
                        try:
                            event_date = datetime.strptime(e.date, "%Y-%m-%d")
                            days_to = (event_date - now).days
                            if 0 <= days_to <= 14:
                                # Tæt på earnings = høj vol → reducer certainty
                                details["days_to_earnings"] = days_to
                                details["earnings_date"] = e.date
                                # Score neutral men confidence lavere
                                scores_list.append(50)
                                break
                        except (ValueError, TypeError):
                            continue
            except Exception:
                pass

            score = statistics.mean(scores_list) if scores_list else 50.0
            confidence = 0.4 if scores_list else 0.1

            # Reducer confidence tæt på earnings
            if details.get("days_to_earnings", 999) < 7:
                confidence *= 0.5

            explanation = f"Sæson: Score {score:.0f}."
            if "month_avg_return" in details:
                explanation += (
                    f" Denne måned: avg return {details['month_avg_return']:.1f}%, "
                    f"win rate {details.get('month_win_rate', 0):.0f}%."
                )
            if "days_to_earnings" in details:
                explanation += (
                    f" OBS: Earnings om {details['days_to_earnings']} dage — "
                    f"høj usikkerhed."
                )

            return SubScore(
                name="seasonality",
                score=score,
                weight=self._weights.get("seasonality", 0.10),
                confidence=confidence,
                details=details,
                explanation=explanation,
            )

        except Exception as exc:
            logger.warning(f"[alpha_score] Seasonality fejl: {exc}")
            return SubScore("seasonality", 50.0, 0.10, 0.1,
                            explanation=f"Fejl: {exc}")

    # ── Hjælpefunktioner ─────────────────────────────────────

    def _guess_sector(self, symbol: str) -> str:
        """Simpel sektor-gætning baseret på symbol."""
        tech = {"AAPL", "MSFT", "GOOGL", "GOOG", "META", "AMZN", "NVDA", "AMD",
                "TSLA", "CRM", "ADBE", "INTC", "ASML", "ASML.AS", "SAP", "SAP.DE"}
        healthcare = {"JNJ", "PFE", "UNH", "ABBV", "MRK", "LLY", "NOVO-B.CO",
                      "NVO", "AMGN", "GILD", "AZN", "AZN.L"}
        financials = {"JPM", "BAC", "GS", "MS", "V", "MA", "BRK-B",
                      "DANSKE.CO", "JYSK.CO", "TRYG.CO"}
        energy = {"XOM", "CVX", "COP", "EQNR", "EQNR.OL", "BP", "BP.L",
                  "TTE", "TTE.PA", "ENI", "ENI.MI", "SHEL", "SHEL.L"}
        defensive = {"KO", "PG", "PEP", "WMT", "COST", "CL", "GIS",
                     "CARL-B.CO", "COLO-B.CO"}

        sym = symbol.upper()
        if sym in tech:
            return "tech"
        elif sym in healthcare:
            return "defensive"  # Healthcare er ofte defensiv
        elif sym in financials:
            return "cyclical"
        elif sym in energy:
            return "cyclical"
        elif sym in defensive:
            return "defensive"
        return "default"

    def _tech_explanation(self, score: float, details: dict) -> str:
        """Generér tekst-forklaring for teknisk score."""
        parts: list[str] = []
        rsi = details.get("rsi")
        if rsi is not None:
            if rsi < 30:
                parts.append(f"RSI oversold ({rsi:.0f})")
            elif rsi > 70:
                parts.append(f"RSI overbought ({rsi:.0f})")
            else:
                parts.append(f"RSI neutral ({rsi:.0f})")

        if details.get("adx"):
            adx_val = details["adx"]
            parts.append(f"ADX {adx_val:.0f} ({'stærk' if adx_val > 25 else 'svag'} trend)")

        if details.get("volume_ratio"):
            vr = details["volume_ratio"]
            if vr > 1.5:
                parts.append(f"Høj volumen ({vr:.1f}x)")

        return f"Teknisk score {score:.0f}/100. " + ". ".join(parts) if parts else f"Teknisk score {score:.0f}/100"

    def _generate_explanation(
        self,
        symbol: str,
        total: float,
        breakdown: dict[str, SubScore],
    ) -> str:
        """Generér samlet tekstforklaring."""
        signal = score_to_signal(total)
        lines: list[str] = [f"{symbol}: Alpha Score {total:.0f}/100 → {signal}"]

        # Top 3 drivere
        sorted_dims = sorted(
            breakdown.items(),
            key=lambda x: abs(x[1].score - 50),
            reverse=True,
        )
        for name, ss in sorted_dims[:3]:
            direction = "positiv" if ss.score > 55 else "negativ" if ss.score < 45 else "neutral"
            lines.append(f"  • {name}: {ss.score:.0f}/100 ({direction})")

        return "\n".join(lines)

    def _empty_score(self, symbol: str, reason: str) -> AlphaScore:
        """Returnér en tom/neutral score ved fejl."""
        return AlphaScore(
            symbol=symbol,
            total=50.0,
            signal="HOLD",
            confidence=0.0,
            explanation=f"{symbol}: Kan ikke beregne — {reason}",
            timestamp=datetime.now(),
        )
