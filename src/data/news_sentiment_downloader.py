"""
News Sentiment Historical Downloader — builds sentiment history from free sources.

Free sources used:
  1. yfinance ticker.news — per-symbol news (free, no key)
  2. RSS feeds — Reuters, CNBC, Yahoo Finance, MarketWatch, Investing.com
  3. Finnhub free tier — 60 calls/min if FINNHUB_API_KEY is set

Builds a daily sentiment score per symbol stored in data_cache/news_sentiment.db.
The data processor reads these scores and adds them as features #23-24
(news_sentiment, news_volume) to the ML training pipeline.

Usage:
  python -m src.data.news_sentiment_downloader --backfill
  python -m src.data.news_sentiment_downloader --daily
  python -m src.data.news_sentiment_downloader --stats
"""

from __future__ import annotations

import gc
import json
import os
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

try:
    from src.data.universe import AssetUniverse
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from src.data.universe import AssetUniverse

# ── Config ────────────────────────────────────────────────

_ROOT = Path(__file__).resolve().parent.parent.parent
_DATA_DIR = _ROOT / "data_cache"
_DB_PATH = _DATA_DIR / "news_sentiment.db"
_PROGRESS_FILE = _DATA_DIR / "news_sentiment_progress.json"

_DELAY_BETWEEN_TICKERS = 0.5   # seconds (yfinance rate limit)
_DELAY_BETWEEN_BATCHES = 2.0
_BATCH_SIZE = 20

# RSS feeds (free, no key needed)
RSS_FEEDS = {
    "Reuters Business": "https://feeds.reuters.com/reuters/businessNews",
    "CNBC Top News": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114",
    "Yahoo Finance": "https://finance.yahoo.com/news/rssindex",
    "MarketWatch": "https://feeds.marketwatch.com/marketwatch/topstories/",
    "Investing.com": "https://www.investing.com/rss/news.rss",
}


# ── Keyword Sentiment (fast, no model needed for backfill) ─────

_POSITIVE_WORDS = {
    "beat", "beats", "exceeded", "surge", "surges", "rally", "rallies",
    "bullish", "upgrade", "upgrades", "outperform", "growth", "record high",
    "profit", "profits", "dividend", "acquisition", "buyback", "partnership",
    "approval", "fda approved", "breakthrough", "innovation", "expansion",
    "revenue growth", "strong earnings", "beat estimates", "buy rating",
    "positive", "optimistic", "upbeat", "soar", "soars", "boom",
    "recovery", "rebound", "highest", "all-time high", "momentum",
}

_NEGATIVE_WORDS = {
    "miss", "misses", "missed", "crash", "crashes", "plunge", "plunges",
    "bearish", "downgrade", "downgrades", "underperform", "decline",
    "loss", "losses", "bankruptcy", "lawsuit", "investigation", "fraud",
    "recall", "fda rejected", "layoff", "layoffs", "restructuring",
    "warning", "debt", "default", "sell-off", "selloff", "weak earnings",
    "missed estimates", "sell rating", "negative", "pessimistic",
    "slump", "tumble", "tumbles", "plummet", "worst", "risk", "crisis",
    "recession", "inflation fear", "downward", "cut dividend",
}


def _keyword_sentiment(text: str) -> float:
    """Fast keyword-based sentiment scoring. Returns -1 to 1."""
    text_lower = text.lower()
    pos = sum(1 for w in _POSITIVE_WORDS if w in text_lower)
    neg = sum(1 for w in _NEGATIVE_WORDS if w in text_lower)
    total = pos + neg
    if total == 0:
        return 0.0
    return (pos - neg) / total


def _try_finbert_sentiment(texts: list[str]) -> list[float] | None:
    """Try FinBERT sentiment, return None if not available."""
    try:
        from transformers import pipeline as _hf_pipeline
        _pipe = _hf_pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            top_k=None,
            truncation=True,
            max_length=512,
        )

        scores = []
        for text in texts:
            result = _pipe(text[:512])
            if result and isinstance(result[0], list):
                result = result[0]
            score_map = {r["label"]: r["score"] for r in result}
            score = score_map.get("positive", 0) - score_map.get("negative", 0)
            scores.append(score)
        return scores
    except ImportError:
        return None
    except Exception:
        return None


# ── Database ──────────────────────────────────────────────

def _init_db(db_path: Path = _DB_PATH) -> None:
    """Create news sentiment tables."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db_path)) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS news_articles (
                id              TEXT PRIMARY KEY,
                symbol          TEXT NOT NULL,
                title           TEXT NOT NULL,
                source          TEXT,
                published       TEXT,
                sentiment_score REAL,
                sentiment_method TEXT DEFAULT 'keyword',
                fetched_at      TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS daily_sentiment (
                symbol          TEXT NOT NULL,
                date            TEXT NOT NULL,
                sentiment_avg   REAL,
                sentiment_std   REAL,
                news_count      INTEGER,
                positive_count  INTEGER,
                negative_count  INTEGER,
                neutral_count   INTEGER,
                PRIMARY KEY (symbol, date)
            );

            CREATE TABLE IF NOT EXISTS rss_articles (
                id              TEXT PRIMARY KEY,
                title           TEXT NOT NULL,
                source          TEXT,
                published       TEXT,
                sentiment_score REAL,
                symbols_matched TEXT,
                fetched_at      TEXT DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_news_symbol ON news_articles(symbol);
            CREATE INDEX IF NOT EXISTS idx_news_date ON news_articles(published);
            CREATE INDEX IF NOT EXISTS idx_daily_symbol ON daily_sentiment(symbol);
            CREATE INDEX IF NOT EXISTS idx_daily_date ON daily_sentiment(date);

            PRAGMA journal_mode = WAL;
            PRAGMA synchronous = NORMAL;
        """)
    logger.info(f"[news_sentiment] Database ready: {db_path}")


# ── Progress Tracking ─────────────────────────────────────

def _load_progress() -> dict:
    if _PROGRESS_FILE.exists():
        try:
            return json.loads(_PROGRESS_FILE.read_text())
        except Exception:
            pass
    return {"completed_symbols": [], "failed": [], "last_rss_fetch": "", "last_daily_update": ""}


def _save_progress(progress: dict) -> None:
    _PROGRESS_FILE.write_text(json.dumps(progress, indent=2))


# ── yfinance News Fetcher ─────────────────────────────────

def _fetch_yfinance_news(symbol: str) -> list[dict]:
    """Fetch news for a symbol via yfinance. Returns list of {title, published, source}."""
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        news = ticker.news
        if not news:
            return []

        articles = []
        for item in news:
            # yfinance >= 0.2.28 wraps news in {id, content} dicts
            content = item.get("content", item) if isinstance(item, dict) else item
            if isinstance(content, dict):
                title = content.get("title", "")
                summary = content.get("summary", "")

                # Parse published date
                pub_date = content.get("pubDate", "")
                if pub_date:
                    try:
                        dt = datetime.fromisoformat(pub_date.replace("Z", "+00:00"))
                        published = dt.strftime("%Y-%m-%d %H:%M:%S")
                    except (ValueError, TypeError):
                        published = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                else:
                    # Legacy format: providerPublishTime as unix timestamp
                    pub_ts = content.get("providerPublishTime", 0)
                    if pub_ts:
                        published = datetime.fromtimestamp(pub_ts).strftime("%Y-%m-%d %H:%M:%S")
                    else:
                        published = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # Source
                provider = content.get("provider", {})
                if isinstance(provider, dict):
                    source = provider.get("displayName", "Yahoo Finance")
                else:
                    source = content.get("publisher", "Yahoo Finance")

                # URL
                canon = content.get("canonicalUrl", {})
                if isinstance(canon, dict):
                    url = canon.get("url", "")
                else:
                    url = content.get("link", "")
            else:
                continue

            if not title:
                continue

            articles.append({
                "title": title,
                "summary": summary[:500] if summary else "",
                "source": source,
                "published": published,
                "url": url,
            })
        return articles
    except Exception as e:
        logger.debug(f"[news_sentiment] yfinance news error for {symbol}: {e}")
        return []


# ── RSS News Fetcher ──────────────────────────────────────

def _fetch_rss_news() -> list[dict]:
    """Fetch latest news from all RSS feeds."""
    try:
        import feedparser
    except ImportError:
        logger.warning("[news_sentiment] feedparser not installed, skipping RSS")
        return []

    articles = []
    for source, url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:30]:
                title = entry.get("title", "")
                if not title:
                    continue

                # Parse published date
                published = ""
                for date_field in ["published", "updated", "created"]:
                    if hasattr(entry, date_field) and getattr(entry, date_field):
                        published = getattr(entry, date_field)
                        break

                if not published:
                    published = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                articles.append({
                    "title": title,
                    "summary": entry.get("summary", "")[:500],
                    "source": source,
                    "published": published,
                    "url": entry.get("link", ""),
                })
        except Exception as e:
            logger.debug(f"[news_sentiment] RSS error {source}: {e}")

    return articles


# ── Symbol Matching for RSS ───────────────────────────────

# Map company names to tickers for RSS matching
_COMPANY_NAMES: dict[str, list[str]] = {
    "AAPL": ["apple", "iphone", "ipad", "mac"],
    "MSFT": ["microsoft", "azure", "windows", "xbox"],
    "GOOGL": ["google", "alphabet", "youtube", "android"],
    "AMZN": ["amazon", "aws", "prime"],
    "META": ["meta", "facebook", "instagram", "whatsapp"],
    "NVDA": ["nvidia", "geforce", "gpu"],
    "TSLA": ["tesla", "elon musk", "spacex"],
    "AMD": ["amd", "advanced micro", "ryzen", "radeon"],
    "NFLX": ["netflix"],
    "JPM": ["jpmorgan", "jp morgan", "jamie dimon"],
    "BAC": ["bank of america"],
    "GS": ["goldman sachs"],
    "V": ["visa"],
    "MA": ["mastercard"],
    "JNJ": ["johnson & johnson", "johnson and johnson"],
    "PFE": ["pfizer"],
    "UNH": ["unitedhealth"],
    "WMT": ["walmart"],
    "DIS": ["disney"],
    "INTC": ["intel"],
    "CRM": ["salesforce"],
    "ORCL": ["oracle"],
    "CSCO": ["cisco"],
    "QCOM": ["qualcomm"],
    "AVGO": ["broadcom"],
    "ASML.AS": ["asml"],
    "NVO": ["novo nordisk", "ozempic", "wegovy"],
    "TSM": ["tsmc", "taiwan semi"],
    "BTC-USD": ["bitcoin", "btc"],
    "ETH-USD": ["ethereum", "ether", "eth"],
    "SOL-USD": ["solana"],
    "XRP-USD": ["ripple", "xrp"],
    "BNB-USD": ["binance"],
    "XOM": ["exxon"],
    "CVX": ["chevron"],
    "BA": ["boeing"],
    "SHEL.L": ["shell"],
    "BP.L": ["bp "],
    "TTE.PA": ["totalenergies"],
}


def _match_symbols(text: str, symbols: list[str]) -> list[str]:
    """Match text to known symbols by company name or ticker."""
    text_lower = text.lower()
    matched = []

    # Check company name mappings
    for symbol, names in _COMPANY_NAMES.items():
        if symbol in symbols:
            for name in names:
                if name in text_lower:
                    matched.append(symbol)
                    break

    # Check ticker mentions (e.g., "$AAPL" or "AAPL")
    for symbol in symbols:
        ticker = symbol.split(".")[0].split("-")[0]
        if len(ticker) >= 2 and (f"${ticker}" in text or f" {ticker} " in text):
            if symbol not in matched:
                matched.append(symbol)

    return matched


# ── Backfill: Build Historical Sentiment ──────────────────

def _backfill_symbol(
    symbol: str,
    conn: sqlite3.Connection,
    use_finbert: bool = False,
) -> int:
    """Download and score news for a single symbol. Returns article count."""
    articles = _fetch_yfinance_news(symbol)
    if not articles:
        return 0

    stored = 0
    titles = [a["title"] for a in articles]

    # Score all titles
    if use_finbert:
        scores = _try_finbert_sentiment(titles)
    else:
        scores = None

    if scores is None:
        scores = [_keyword_sentiment(t) for t in titles]
        method = "keyword"
    else:
        method = "finbert"

    for article, score in zip(articles, scores):
        import hashlib
        art_id = hashlib.md5(
            (article["url"] or article["title"]).encode()
        ).hexdigest()[:12]

        try:
            conn.execute("""
                INSERT OR IGNORE INTO news_articles
                (id, symbol, title, source, published, sentiment_score, sentiment_method)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                art_id, symbol, article["title"], article["source"],
                article["published"], score, method,
            ))
            stored += 1
        except Exception:
            pass

    return stored


def _aggregate_daily_sentiment(conn: sqlite3.Connection) -> int:
    """Aggregate article-level scores into daily_sentiment table."""
    rows = conn.execute("""
        SELECT symbol,
               SUBSTR(published, 1, 10) as date,
               AVG(sentiment_score) as avg_score,
               COUNT(*) as cnt,
               SUM(CASE WHEN sentiment_score > 0.1 THEN 1 ELSE 0 END) as pos,
               SUM(CASE WHEN sentiment_score < -0.1 THEN 1 ELSE 0 END) as neg,
               SUM(CASE WHEN sentiment_score BETWEEN -0.1 AND 0.1 THEN 1 ELSE 0 END) as neu
        FROM news_articles
        WHERE published IS NOT NULL AND published != ''
        GROUP BY symbol, SUBSTR(published, 1, 10)
    """).fetchall()

    if not rows:
        return 0

    conn.executemany("""
        INSERT OR REPLACE INTO daily_sentiment
        (symbol, date, sentiment_avg, sentiment_std, news_count,
         positive_count, negative_count, neutral_count)
        VALUES (?, ?, ?, NULL, ?, ?, ?, ?)
    """, rows)

    conn.commit()
    return len(rows)


# ── RSS Aggregation ───────────────────────────────────────

def _process_rss_articles(
    conn: sqlite3.Connection,
    symbols: list[str],
) -> int:
    """Fetch RSS, match to symbols, score, and store."""
    articles = _fetch_rss_news()
    if not articles:
        return 0

    stored = 0
    for article in articles:
        text = article["title"] + " " + article.get("summary", "")
        matched = _match_symbols(text, symbols)
        score = _keyword_sentiment(text)

        import hashlib
        art_id = hashlib.md5(
            (article.get("url", "") or article["title"]).encode()
        ).hexdigest()[:12]

        # Store in rss_articles
        try:
            conn.execute("""
                INSERT OR IGNORE INTO rss_articles
                (id, title, source, published, sentiment_score, symbols_matched)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                art_id, article["title"], article["source"],
                article["published"], score,
                json.dumps(matched),
            ))
        except Exception:
            pass

        # Also store per-symbol if matched
        for sym in matched:
            sym_art_id = f"{art_id}_{sym}"
            try:
                conn.execute("""
                    INSERT OR IGNORE INTO news_articles
                    (id, symbol, title, source, published, sentiment_score, sentiment_method)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    sym_art_id, sym, article["title"], article["source"],
                    article["published"], score, "keyword",
                ))
                stored += 1
            except Exception:
                pass

    conn.commit()
    return stored


# ── Continuous Fetcher (background thread) ───────────────

class ContinuousNewsFetcher:
    """
    Background thread that cycles through all universe symbols in small
    batches, fetching news every ``interval_seconds`` (default 300 = 5 min).

    After one full cycle through all symbols it starts over, so sentiment
    scores stay fresh throughout the day instead of arriving in one nightly
    bulk download.

    RSS feeds are refreshed once per cycle (after all symbols are done).
    """

    def __init__(
        self,
        db_path: Path | None = None,
        batch_size: int = 10,
        interval_seconds: int = 300,
    ) -> None:
        self._db_path = db_path or _DB_PATH
        _init_db(self._db_path)
        try:
            universe = AssetUniverse()
            self._symbols = list(universe.active_symbols)
        except Exception:
            self._symbols = []
        self._batch_size = batch_size
        self._interval = interval_seconds
        self._cursor = 0          # rotating index into _symbols
        self._cycle_count = 0
        self._articles_total = 0
        self._running = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    # ── public API ───────────────────────────────────────

    def start(self) -> None:
        if self._running:
            return
        if not self._symbols:
            logger.warning("[news-continuous] No symbols — not starting")
            return
        self._running = True
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name="ContinuousNewsFetcher",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            f"[news-continuous] Started — {len(self._symbols)} symbols, "
            f"batch={self._batch_size}, interval={self._interval}s"
        )

    def stop(self) -> None:
        self._running = False
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10)
        logger.info("[news-continuous] Stopped")

    @property
    def status(self) -> dict:
        with self._lock:
            return {
                "running": self._running,
                "cursor": self._cursor,
                "total_symbols": len(self._symbols),
                "cycle": self._cycle_count,
                "articles_total": self._articles_total,
                "progress_pct": round(
                    self._cursor / max(len(self._symbols), 1) * 100, 1
                ),
            }

    # ── internal ─────────────────────────────────────────

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._tick()
            except Exception as e:
                logger.error(f"[news-continuous] Tick error: {e}")
            self._stop.wait(self._interval)

    def _tick(self) -> None:
        n_symbols = len(self._symbols)
        if n_symbols == 0:
            return

        # Pick the next batch
        start = self._cursor
        end = min(start + self._batch_size, n_symbols)
        batch = self._symbols[start:end]

        conn = sqlite3.connect(str(self._db_path), timeout=60)
        articles = 0
        for symbol in batch:
            try:
                n = _backfill_symbol(symbol, conn, use_finbert=False)
                articles += n
            except Exception as e:
                logger.debug(f"[news-continuous] {symbol}: {e}")
            time.sleep(_DELAY_BETWEEN_TICKERS)

        conn.commit()

        # Advance cursor
        with self._lock:
            self._cursor = end
            self._articles_total += articles

        if articles > 0:
            logger.debug(
                f"[news-continuous] Batch {start}-{end}: "
                f"{articles} new articles"
            )

        # End of cycle — fetch RSS, aggregate, reset cursor
        if end >= n_symbols:
            try:
                n_rss = _process_rss_articles(conn, self._symbols)
                if n_rss:
                    logger.info(f"[news-continuous] RSS: {n_rss} articles matched")
            except Exception as e:
                logger.debug(f"[news-continuous] RSS error: {e}")

            try:
                _aggregate_daily_sentiment(conn)
            except Exception:
                pass

            with self._lock:
                self._cursor = 0
                self._cycle_count += 1

            logger.info(
                f"[news-continuous] Cycle {self._cycle_count} complete — "
                f"{self._articles_total} articles total"
            )

        conn.close()


# ── Main Class ────────────────────────────────────────────

class NewsSentimentDownloader:
    """Downloads and maintains news sentiment data for all universe symbols."""

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path or _DB_PATH
        _init_db(self._db_path)
        self._universe = AssetUniverse()
        self._symbols = self._universe.active_symbols
        logger.info(f"[news_sentiment] Universe: {len(self._symbols)} symbols")

    @property
    def symbols(self) -> list[str]:
        return self._symbols

    def run_backfill(self, use_finbert: bool = False) -> dict:
        """
        Backfill news sentiment for all symbols using yfinance news.

        yfinance provides recent news (typically last 1-4 weeks).
        This gives enough data to start building sentiment features.
        Resumes from where it left off if interrupted.

        For deeper history, run daily updates over time — the DB
        accumulates sentiment scores as new articles arrive.
        """
        logger.info(
            f"[news_sentiment] === BACKFILL: {len(self._symbols)} symbols ==="
        )
        t0 = time.time()

        conn = sqlite3.connect(str(self._db_path), timeout=120)
        progress = _load_progress()

        stats = {"downloaded": 0, "skipped": 0, "failed": 0, "articles": 0}
        total = len(self._symbols)

        for i, symbol in enumerate(self._symbols):
            if symbol in progress["completed_symbols"]:
                stats["skipped"] += 1
                continue

            try:
                n_articles = _backfill_symbol(symbol, conn, use_finbert)
                stats["articles"] += n_articles
                stats["downloaded"] += 1

                progress["completed_symbols"].append(symbol)

                if (i + 1) % 10 == 0 or i == total - 1:
                    conn.commit()
                    _save_progress(progress)
                    logger.info(
                        f"[news_sentiment] {i + 1}/{total} - {symbol}: "
                        f"{n_articles} articles"
                    )

            except Exception as e:
                logger.warning(f"[news_sentiment] {symbol}: failed - {e}")
                progress["failed"].append(symbol)
                stats["failed"] += 1

            time.sleep(_DELAY_BETWEEN_TICKERS)

            if (i + 1) % _BATCH_SIZE == 0:
                time.sleep(_DELAY_BETWEEN_BATCHES)

        # Aggregate into daily scores
        logger.info("[news_sentiment] Aggregating daily sentiment scores...")
        n_daily = _aggregate_daily_sentiment(conn)
        stats["daily_rows"] = n_daily

        # Also fetch RSS and cross-match
        logger.info("[news_sentiment] Fetching RSS feeds...")
        n_rss = _process_rss_articles(conn, self._symbols)
        stats["rss_articles"] = n_rss

        # Re-aggregate after RSS
        _aggregate_daily_sentiment(conn)

        elapsed = time.time() - t0
        logger.info(
            f"[news_sentiment] === BACKFILL COMPLETE ===\n"
            f"  Symbols downloaded: {stats['downloaded']}\n"
            f"  Skipped:           {stats['skipped']}\n"
            f"  Failed:            {stats['failed']}\n"
            f"  Articles stored:   {stats['articles']}\n"
            f"  RSS articles:      {stats['rss_articles']}\n"
            f"  Daily rows:        {stats['daily_rows']}\n"
            f"  Time:              {elapsed / 60:.1f} min"
        )

        conn.close()
        return stats

    def run_daily_update(self) -> dict:
        """
        Daily update: fetch fresh news for all symbols + RSS.
        Run this daily (e.g. via scheduler) to accumulate sentiment history.
        """
        logger.info(
            f"[news_sentiment] === DAILY UPDATE: {len(self._symbols)} symbols ==="
        )
        t0 = time.time()

        conn = sqlite3.connect(str(self._db_path), timeout=120)
        stats = {"updated": 0, "articles": 0, "rss_articles": 0}
        total = len(self._symbols)

        for i, symbol in enumerate(self._symbols):
            try:
                n = _backfill_symbol(symbol, conn, use_finbert=False)
                stats["articles"] += n
                if n > 0:
                    stats["updated"] += 1
            except Exception as e:
                logger.debug(f"[news_sentiment] Daily {symbol}: {e}")

            time.sleep(_DELAY_BETWEEN_TICKERS)

            if (i + 1) % _BATCH_SIZE == 0:
                conn.commit()
                time.sleep(_DELAY_BETWEEN_BATCHES)
                if (i + 1) % 100 == 0:
                    logger.info(
                        f"[news_sentiment] Daily update: {i + 1}/{total}"
                    )

        # RSS feeds
        n_rss = _process_rss_articles(conn, self._symbols)
        stats["rss_articles"] = n_rss

        # Aggregate
        n_daily = _aggregate_daily_sentiment(conn)
        stats["daily_rows"] = n_daily

        # Update progress
        progress = _load_progress()
        progress["last_daily_update"] = datetime.now().isoformat()
        _save_progress(progress)

        elapsed = time.time() - t0
        logger.info(
            f"[news_sentiment] === DAILY UPDATE COMPLETE ===\n"
            f"  Updated:      {stats['updated']} symbols\n"
            f"  Articles:     {stats['articles']}\n"
            f"  RSS articles: {stats['rss_articles']}\n"
            f"  Daily rows:   {stats['daily_rows']}\n"
            f"  Time:         {elapsed / 60:.1f} min"
        )

        conn.close()
        return stats

    def rescore_with_finbert(self) -> dict:
        """
        Re-score all articles that were keyword-scored with FinBERT.
        Run after backfill to upgrade sentiment quality.
        """
        logger.info("[news_sentiment] Re-scoring articles with FinBERT...")

        try:
            from transformers import pipeline as _hf_pipeline
            pipe = _hf_pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                top_k=None,
                truncation=True,
                max_length=512,
            )
            logger.info("[news_sentiment] FinBERT model loaded for rescoring")
        except ImportError:
            logger.warning("[news_sentiment] transformers not installed, skipping rescore")
            return {"rescored": 0, "error": "transformers_not_installed"}
        except Exception as e:
            logger.warning(f"[news_sentiment] FinBERT load failed: {e}")
            return {"rescored": 0, "error": str(e)}

        conn = sqlite3.connect(str(self._db_path), timeout=120)
        rows = conn.execute(
            "SELECT id, title FROM news_articles WHERE sentiment_method = 'keyword'"
        ).fetchall()

        if not rows:
            conn.close()
            return {"rescored": 0}

        logger.info(f"[news_sentiment] Re-scoring {len(rows)} articles...")
        rescored = 0

        for i, (art_id, title) in enumerate(rows):
            try:
                result = pipe(title[:512])
                if result and isinstance(result[0], list):
                    result = result[0]
                score_map = {r["label"]: r["score"] for r in result}
                score = score_map.get("positive", 0) - score_map.get("negative", 0)
                conn.execute(
                    "UPDATE news_articles SET sentiment_score = ?, sentiment_method = 'finbert' "
                    "WHERE id = ?",
                    (score, art_id),
                )
                rescored += 1
            except Exception as e:
                logger.debug(f"[news_sentiment] FinBERT error: {e}")

            if (i + 1) % 50 == 0:
                conn.commit()
                logger.info(f"[news_sentiment] Rescored {rescored}/{len(rows)}")

        # Re-aggregate with better scores
        conn.commit()
        _aggregate_daily_sentiment(conn)
        conn.close()

        logger.info(f"[news_sentiment] Re-scored {rescored} articles with FinBERT")
        return {"rescored": rescored}

    def get_sentiment(self, symbol: str, days: int = 365) -> Optional[pd.DataFrame]:
        """Read daily sentiment for a symbol."""
        start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        with sqlite3.connect(str(self._db_path)) as conn:
            df = pd.read_sql_query(
                "SELECT * FROM daily_sentiment WHERE symbol = ? AND date >= ? ORDER BY date",
                conn, params=(symbol, start),
            )
        return df if not df.empty else None

    def get_all_daily_sentiment(self) -> pd.DataFrame:
        """Read all daily sentiment data (for feature engineering)."""
        with sqlite3.connect(str(self._db_path)) as conn:
            df = pd.read_sql_query(
                "SELECT symbol, date, sentiment_avg, news_count "
                "FROM daily_sentiment ORDER BY symbol, date",
                conn,
            )
        return df

    def get_stats(self) -> dict:
        """Database statistics."""
        with sqlite3.connect(str(self._db_path)) as conn:
            n_articles = conn.execute("SELECT COUNT(*) FROM news_articles").fetchone()[0]
            n_symbols = conn.execute(
                "SELECT COUNT(DISTINCT symbol) FROM news_articles"
            ).fetchone()[0]
            n_daily = conn.execute("SELECT COUNT(*) FROM daily_sentiment").fetchone()[0]
            n_rss = conn.execute("SELECT COUNT(*) FROM rss_articles").fetchone()[0]

            oldest = conn.execute("SELECT MIN(published) FROM news_articles").fetchone()[0]
            newest = conn.execute("SELECT MAX(published) FROM news_articles").fetchone()[0]

            methods = conn.execute(
                "SELECT sentiment_method, COUNT(*) FROM news_articles GROUP BY sentiment_method"
            ).fetchall()

        db_size = self._db_path.stat().st_size if self._db_path.exists() else 0

        return {
            "articles": n_articles,
            "symbols": n_symbols,
            "daily_rows": n_daily,
            "rss_articles": n_rss,
            "oldest": oldest,
            "newest": newest,
            "methods": dict(methods),
            "db_size_mb": round(db_size / (1024 * 1024), 1),
        }


# ── CLI ───────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="News Sentiment Downloader")
    parser.add_argument("--backfill", action="store_true",
                        help="Backfill news sentiment for all symbols (yfinance + RSS)")
    parser.add_argument("--daily", action="store_true",
                        help="Daily update (fetch latest news)")
    parser.add_argument("--rescore", action="store_true",
                        help="Re-score keyword articles with FinBERT/NPU")
    parser.add_argument("--stats", action="store_true",
                        help="Show database statistics")
    parser.add_argument("--finbert", action="store_true",
                        help="Use FinBERT for scoring during backfill (slower)")
    args = parser.parse_args()

    dl = NewsSentimentDownloader()

    if args.stats:
        s = dl.get_stats()
        print(f"Articles:     {s['articles']:,}")
        print(f"Symbols:      {s['symbols']}")
        print(f"Daily rows:   {s['daily_rows']:,}")
        print(f"RSS articles: {s['rss_articles']:,}")
        print(f"Date range:   {s['oldest']} -> {s['newest']}")
        print(f"Methods:      {s['methods']}")
        print(f"DB size:      {s['db_size_mb']} MB")
    elif args.backfill:
        dl.run_backfill(use_finbert=args.finbert)
        print("\nRe-scoring with FinBERT/NPU...")
        dl.rescore_with_finbert()
    elif args.daily:
        dl.run_daily_update()
    elif args.rescore:
        dl.rescore_with_finbert()
    else:
        print("Usage: python -m src.data.news_sentiment_downloader --backfill|--daily|--rescore|--stats")
        print(f"Universe: {len(dl.symbols)} symbols")
