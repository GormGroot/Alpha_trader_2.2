"""
Historical Data Downloader - Daily batch download for all universe symbols.

Two modes:
  1. Initial: Download full history (25 years daily, 60 days intraday) for all ~700 symbols
  2. Daily:   Download last 24h and append to existing database

Data is stored in data_cache/historical_master.db with pre-computed indicators,
ready for the trader to read via get_historical_from_db().

Usage:
  # From Python:
  from src.data.historical_downloader import HistoricalDownloader
  dl = HistoricalDownloader()
  dl.run_initial()       # One-time: download everything (~4-6 hours)
  dl.run_daily_update()  # Daily: last 24h (~15-20 min for 700 symbols)

  # From command line:
  python -m src.data.historical_downloader --initial
  python -m src.data.historical_downloader --daily

  # Trader reads historical data:
  dl = HistoricalDownloader()
  df = dl.get_historical("AAPL", days=365)  # Returns DataFrame with indicators
"""

from __future__ import annotations

import gc
import json
import sqlite3
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

# Project imports
try:
    from src.data.universe import AssetUniverse
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from src.data.universe import AssetUniverse

try:
    from src.data.indicators import add_all_indicators
except ImportError:
    add_all_indicators = None


# ── Config ────────────────────────────────────────────────

_ROOT = Path(__file__).resolve().parent.parent.parent
_DATA_DIR = _ROOT / "data_cache"
_DB_PATH = _DATA_DIR / "historical_master.db"
_PROGRESS_FILE = _DATA_DIR / "historical_download_progress.json"

# Rate limiting
_DELAY_BETWEEN_TICKERS = 0.4   # seconds
_DELAY_BETWEEN_BATCHES = 3.0   # seconds between batches of 50
_BATCH_SIZE = 50
_MAX_WORKERS = 8

# yfinance limits
_MAX_DAILY_YEARS = 25
_MAX_INTRADAY_DAYS = 59  # yfinance limit for 5m data


# ── Database ──────────────────────────────────────────────

def _init_db(db_path: Path = _DB_PATH) -> None:
    """Create database tables if they don't exist."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db_path)) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS daily_bars (
                symbol    TEXT NOT NULL,
                date      TEXT NOT NULL,
                open      REAL,
                high      REAL,
                low       REAL,
                close     REAL,
                volume    INTEGER,
                adj_close REAL,
                PRIMARY KEY (symbol, date)
            );

            CREATE TABLE IF NOT EXISTS intraday_bars (
                symbol    TEXT NOT NULL,
                datetime  TEXT NOT NULL,
                interval  TEXT NOT NULL DEFAULT '5m',
                open      REAL,
                high      REAL,
                low       REAL,
                close     REAL,
                volume    INTEGER,
                PRIMARY KEY (symbol, datetime, interval)
            );

            CREATE TABLE IF NOT EXISTS indicators (
                symbol        TEXT NOT NULL,
                date          TEXT NOT NULL,
                sma_20 REAL, sma_50 REAL, sma_200 REAL,
                ema_12 REAL, ema_26 REAL,
                rsi_14 REAL,
                macd REAL, macd_signal REAL, macd_hist REAL,
                bb_upper REAL, bb_middle REAL, bb_lower REAL,
                atr_14 REAL, adx_14 REAL,
                obv REAL, vwap REAL,
                stoch_k REAL, stoch_d REAL,
                cci_20 REAL, mfi_14 REAL,
                williams_r REAL,
                volatility_20d REAL,
                PRIMARY KEY (symbol, date)
            );

            CREATE TABLE IF NOT EXISTS download_log (
                symbol        TEXT NOT NULL,
                data_type     TEXT NOT NULL,
                first_date    TEXT,
                last_date     TEXT,
                bar_count     INTEGER,
                downloaded_at TEXT DEFAULT (datetime('now')),
                PRIMARY KEY (symbol, data_type)
            );

            CREATE INDEX IF NOT EXISTS idx_daily_date ON daily_bars(date);
            CREATE INDEX IF NOT EXISTS idx_daily_symbol ON daily_bars(symbol);
            CREATE INDEX IF NOT EXISTS idx_intraday_symbol ON intraday_bars(symbol);
            CREATE INDEX IF NOT EXISTS idx_intraday_dt ON intraday_bars(datetime);
            CREATE INDEX IF NOT EXISTS idx_indicators_symbol ON indicators(symbol);
            CREATE INDEX IF NOT EXISTS idx_indicators_date ON indicators(date);

            PRAGMA journal_mode = WAL;
            PRAGMA synchronous = NORMAL;
            PRAGMA cache_size = -64000;
        """)
    logger.info(f"[historical] Database ready: {db_path}")


def _get_conn(db_path: Path = _DB_PATH) -> sqlite3.Connection:
    return sqlite3.connect(str(db_path), timeout=120)


# ── Progress Tracking ─────────────────────────────────────

def _load_progress() -> dict:
    if _PROGRESS_FILE.exists():
        try:
            return json.loads(_PROGRESS_FILE.read_text())
        except Exception:
            pass
    return {"completed_daily": [], "completed_intraday": [], "failed": [], "last_daily_update": ""}


def _save_progress(progress: dict) -> None:
    _PROGRESS_FILE.write_text(json.dumps(progress, indent=2))


# ── Indicator Computation ─────────────────────────────────

def _compute_indicators(df: pd.DataFrame) -> dict:
    """Compute technical indicators from OHLCV DataFrame. Returns dict of indicator values for last row."""
    if df is None or len(df) < 20:
        return {}

    try:
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"].astype(float)

        result = {}

        # SMAs
        result["sma_20"] = close.rolling(20).mean().iloc[-1]
        result["sma_50"] = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else None
        result["sma_200"] = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else None

        # EMAs
        result["ema_12"] = close.ewm(span=12).mean().iloc[-1]
        result["ema_26"] = close.ewm(span=26).mean().iloc[-1]

        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        result["rsi_14"] = (100 - 100 / (1 + rs)).iloc[-1]

        # MACD
        macd_line = close.ewm(span=12).mean() - close.ewm(span=26).mean()
        signal_line = macd_line.ewm(span=9).mean()
        result["macd"] = macd_line.iloc[-1]
        result["macd_signal"] = signal_line.iloc[-1]
        result["macd_hist"] = (macd_line - signal_line).iloc[-1]

        # Bollinger Bands
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        result["bb_upper"] = (sma20 + 2 * std20).iloc[-1]
        result["bb_middle"] = sma20.iloc[-1]
        result["bb_lower"] = (sma20 - 2 * std20).iloc[-1]

        # ATR
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)
        result["atr_14"] = tr.rolling(14).mean().iloc[-1]

        # Volatility
        returns = close.pct_change()
        result["volatility_20d"] = returns.rolling(20).std().iloc[-1] * np.sqrt(252)

        # OBV
        obv = (volume * np.sign(close.diff())).fillna(0).cumsum()
        result["obv"] = obv.iloc[-1]

        # Stochastic
        low14 = low.rolling(14).min()
        high14 = high.rolling(14).max()
        k = 100 * (close - low14) / (high14 - low14).replace(0, np.nan)
        result["stoch_k"] = k.iloc[-1]
        result["stoch_d"] = k.rolling(3).mean().iloc[-1]

        return result

    except Exception as e:
        logger.debug(f"[historical] Indicator computation error: {e}")
        return {}


# ── Vectorised Indicator Computation ──────────────────────

def _compute_indicators_vectorised(symbol: str, df: pd.DataFrame) -> list[tuple]:
    """Compute all indicators in a single vectorised pass and return INSERT-ready tuples.

    This replaces the old O(n²) loop that created a growing DataFrame slice per row.
    All rolling/ewm operations run once over the full Series — O(n) total.
    """
    try:
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"].astype(float)

        # Rolling / EWM — computed once
        sma_20 = close.rolling(20).mean()
        sma_50 = close.rolling(50).mean()
        sma_200 = close.rolling(200).mean()
        ema_12 = close.ewm(span=12).mean()
        ema_26 = close.ewm(span=26).mean()

        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi_14 = 100 - 100 / (1 + rs)

        # MACD
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9).mean()
        macd_hist = macd_line - signal_line

        # Bollinger Bands
        std20 = close.rolling(20).std()
        bb_upper = sma_20 + 2 * std20
        bb_lower = sma_20 - 2 * std20

        # ATR
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)
        atr_14 = tr.rolling(14).mean()

        # Volatility
        returns = close.pct_change()
        vol_20d = returns.rolling(20).std() * np.sqrt(252)

        # OBV
        obv = (volume * np.sign(close.diff())).fillna(0).cumsum()

        # Stochastic
        low14 = low.rolling(14).min()
        high14 = high.rolling(14).max()
        stoch_k = 100 * (close - low14) / (high14 - low14).replace(0, np.nan)
        stoch_d = stoch_k.rolling(3).mean()

        # Build records starting from row 20 (first row with sma_20)
        records: list[tuple] = []
        _f = lambda v: float(v) if pd.notna(v) else None  # noqa: E731
        for j in range(19, len(df)):
            date_str = df.index[j].strftime("%Y-%m-%d")
            records.append((
                symbol, date_str,
                _f(sma_20.iat[j]), _f(sma_50.iat[j]), _f(sma_200.iat[j]),
                _f(ema_12.iat[j]), _f(ema_26.iat[j]),
                _f(rsi_14.iat[j]),
                _f(macd_line.iat[j]), _f(signal_line.iat[j]), _f(macd_hist.iat[j]),
                _f(bb_upper.iat[j]), _f(sma_20.iat[j]), _f(bb_lower.iat[j]),
                _f(atr_14.iat[j]), None,  # adx_14
                _f(obv.iat[j]), None,  # vwap
                _f(stoch_k.iat[j]), _f(stoch_d.iat[j]),
                None, None,  # cci_20, mfi_14
                None,  # williams_r
                _f(vol_20d.iat[j]),
            ))

        return records

    except Exception as e:
        logger.debug(f"[historical] Vectorised indicator error for {symbol}: {e}")
        return []


# ── Download Functions ────────────────────────────────────

def _download_daily_batch(
    symbols: list[str],
    years: int = _MAX_DAILY_YEARS,
    conn: sqlite3.Connection | None = None,
    progress: dict | None = None,
) -> dict:
    """Download daily OHLCV + compute indicators for a batch of symbols."""
    import yfinance as yf

    own_conn = conn is None
    if own_conn:
        conn = _get_conn()
    if progress is None:
        progress = _load_progress()

    stats = {"downloaded": 0, "skipped": 0, "failed": 0, "bars": 0}
    start_date = (datetime.now() - timedelta(days=years * 365)).strftime("%Y-%m-%d")
    total = len(symbols)

    for i, symbol in enumerate(symbols):
        if f"daily:{symbol}" in progress["completed_daily"]:
            stats["skipped"] += 1
            continue

        try:
            data = yf.download(
                symbol,
                start=start_date,
                progress=False,
                auto_adjust=True,
                timeout=30,
            )

            if data is None or data.empty:
                progress["failed"].append(f"daily:{symbol}")
                stats["failed"] += 1
                continue

            # Flatten MultiIndex
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            # Store daily bars
            records = []
            for date, row in data.iterrows():
                records.append((
                    symbol,
                    date.strftime("%Y-%m-%d"),
                    float(row.get("Open", 0) or 0),
                    float(row.get("High", 0) or 0),
                    float(row.get("Low", 0) or 0),
                    float(row.get("Close", 0) or 0),
                    int(row.get("Volume", 0) or 0),
                    float(row.get("Close", 0) or 0),
                ))

            conn.executemany("""
                INSERT OR REPLACE INTO daily_bars
                (symbol, date, open, high, low, close, volume, adj_close)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, records)

            # Compute indicators vectorised (single pass over full history)
            df_ind = data.rename(columns=str.lower)
            ind_records = []  # Init for at undgå UnboundLocalError ved del
            if len(df_ind) >= 20:
                ind_records = _compute_indicators_vectorised(symbol, df_ind)

                if ind_records:
                    conn.executemany("""
                        INSERT OR REPLACE INTO indicators
                        (symbol, date, sma_20, sma_50, sma_200, ema_12, ema_26,
                         rsi_14, macd, macd_signal, macd_hist,
                         bb_upper, bb_middle, bb_lower, atr_14, adx_14,
                         obv, vwap, stoch_k, stoch_d, cci_20, mfi_14,
                         williams_r, volatility_20d)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, ind_records)

            # Log download
            conn.execute("""
                INSERT OR REPLACE INTO download_log
                (symbol, data_type, first_date, last_date, bar_count)
                VALUES (?, ?, ?, ?, ?)
            """, (
                symbol, "daily",
                data.index[0].strftime("%Y-%m-%d"),
                data.index[-1].strftime("%Y-%m-%d"),
                len(data),
            ))

            conn.commit()

            progress["completed_daily"].append(f"daily:{symbol}")
            stats["downloaded"] += 1
            stats["bars"] += len(data)

            if (i + 1) % 10 == 0 or i == total - 1:
                _save_progress(progress)
                logger.info(
                    f"[historical] {i + 1}/{total} - {symbol}: "
                    f"{len(data)} bars ({data.index[0].strftime('%Y-%m-%d')} -> "
                    f"{data.index[-1].strftime('%Y-%m-%d')})"
                )

            # Memory cleanup — per-symbol to prevent accumulation on aarch64/16GB
            del data, df_ind, ind_records
            gc.collect()

        except Exception as e:
            logger.warning(f"[historical] {symbol}: failed - {e}")
            progress["failed"].append(f"daily:{symbol}")
            stats["failed"] += 1

        time.sleep(_DELAY_BETWEEN_TICKERS)

        if (i + 1) % _BATCH_SIZE == 0:
            time.sleep(_DELAY_BETWEEN_BATCHES)

    if own_conn:
        conn.close()

    return stats


def _download_daily_update(
    symbols: list[str],
    conn: sqlite3.Connection | None = None,
) -> dict:
    """Download last 2 days of daily data for all symbols (incremental update)."""
    import yfinance as yf

    own_conn = conn is None
    if own_conn:
        conn = _get_conn()

    stats = {"updated": 0, "failed": 0, "bars": 0}
    start_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
    total = len(symbols)

    for i, symbol in enumerate(symbols):
        try:
            data = yf.download(
                symbol,
                start=start_date,
                progress=False,
                auto_adjust=True,
                timeout=20,
            )

            if data is None or data.empty:
                stats["failed"] += 1
                continue

            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            records = []
            for date, row in data.iterrows():
                records.append((
                    symbol,
                    date.strftime("%Y-%m-%d"),
                    float(row.get("Open", 0) or 0),
                    float(row.get("High", 0) or 0),
                    float(row.get("Low", 0) or 0),
                    float(row.get("Close", 0) or 0),
                    int(row.get("Volume", 0) or 0),
                    float(row.get("Close", 0) or 0),
                ))

            conn.executemany("""
                INSERT OR REPLACE INTO daily_bars
                (symbol, date, open, high, low, close, volume, adj_close)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, records)

            # Update indicators for new dates
            # Read recent history to compute indicators properly
            rows = conn.execute(
                "SELECT date, open, high, low, close, volume FROM daily_bars "
                "WHERE symbol = ? ORDER BY date DESC LIMIT 250",
                (symbol,),
            ).fetchall()

            if len(rows) >= 20:
                df_hist = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
                df_hist = df_hist.sort_values("date").reset_index(drop=True)
                inds = _compute_indicators(df_hist)
                if inds:
                    last_date = df_hist["date"].iloc[-1]
                    conn.execute("""
                        INSERT OR REPLACE INTO indicators
                        (symbol, date, sma_20, sma_50, sma_200, ema_12, ema_26,
                         rsi_14, macd, macd_signal, macd_hist,
                         bb_upper, bb_middle, bb_lower, atr_14, adx_14,
                         obv, vwap, stoch_k, stoch_d, cci_20, mfi_14,
                         williams_r, volatility_20d)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        symbol, last_date,
                        inds.get("sma_20"), inds.get("sma_50"), inds.get("sma_200"),
                        inds.get("ema_12"), inds.get("ema_26"),
                        inds.get("rsi_14"),
                        inds.get("macd"), inds.get("macd_signal"), inds.get("macd_hist"),
                        inds.get("bb_upper"), inds.get("bb_middle"), inds.get("bb_lower"),
                        inds.get("atr_14"), None, inds.get("obv"), None,
                        inds.get("stoch_k"), inds.get("stoch_d"), None, None, None,
                        inds.get("volatility_20d"),
                    ))

            # Update download log
            conn.execute("""
                INSERT OR REPLACE INTO download_log
                (symbol, data_type, first_date, last_date, bar_count)
                VALUES (?, 'daily_update', ?, ?, ?)
            """, (
                symbol,
                data.index[0].strftime("%Y-%m-%d"),
                data.index[-1].strftime("%Y-%m-%d"),
                len(data),
            ))

            conn.commit()
            stats["updated"] += 1
            stats["bars"] += len(data)

            del data

        except Exception as e:
            logger.debug(f"[historical] Daily update {symbol}: {e}")
            stats["failed"] += 1

        time.sleep(_DELAY_BETWEEN_TICKERS)

        if (i + 1) % _BATCH_SIZE == 0:
            time.sleep(_DELAY_BETWEEN_BATCHES)
            logger.info(f"[historical] Daily update: {i + 1}/{total} symbols processed")

    if own_conn:
        conn.close()

    return stats


# ── Main Class ────────────────────────────────────────────

class HistoricalDownloader:
    """
    Downloads and maintains historical market data for all universe symbols.

    Data is stored in data_cache/historical_master.db and can be read
    by the trader via get_historical().
    """

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path or _DB_PATH
        _init_db(self._db_path)

        # Load universe (all categories enabled)
        self._universe = AssetUniverse()
        self._symbols = self._universe.active_symbols
        logger.info(f"[historical] Universe: {len(self._symbols)} symbols")

    @property
    def symbols(self) -> list[str]:
        return self._symbols

    @property
    def db_path(self) -> Path:
        return self._db_path

    def run_initial(self, years: int = _MAX_DAILY_YEARS) -> dict:
        """
        One-time full historical download for all universe symbols.
        Downloads daily data going back `years` years.
        Resumes from where it left off if interrupted.
        """
        logger.info(
            f"[historical] === INITIAL DOWNLOAD: {len(self._symbols)} symbols, "
            f"{years} years of daily data ==="
        )
        t0 = time.time()

        conn = _get_conn(self._db_path)
        progress = _load_progress()

        stats = _download_daily_batch(
            self._symbols, years=years, conn=conn, progress=progress,
        )

        elapsed = time.time() - t0
        logger.info(
            f"[historical] === INITIAL DOWNLOAD COMPLETE ===\n"
            f"  Downloaded: {stats['downloaded']}\n"
            f"  Skipped:    {stats['skipped']}\n"
            f"  Failed:     {stats['failed']}\n"
            f"  Total bars: {stats['bars']:,}\n"
            f"  Time:       {elapsed / 60:.1f} min"
        )

        conn.close()
        return stats

    def run_daily_update(self, run_processor: bool = True) -> dict:
        """
        Daily incremental update: download last 5 days of data for all symbols.
        Designed to run once per day (e.g., via DailyScheduler at 23:00 CET).
        Updates both daily bars and indicators.

        After download, launches the NPU/GPU data processor to rebuild
        the processed data block (features + predictions) so it's ready
        for the next trading session with zero startup latency.

        Args:
            run_processor: If True (default), trigger incremental data processing
                           after download completes.
        """
        logger.info(
            f"[historical] === DAILY UPDATE: {len(self._symbols)} symbols ==="
        )
        t0 = time.time()

        conn = _get_conn(self._db_path)
        stats = _download_daily_update(self._symbols, conn=conn)

        # Update progress
        progress = _load_progress()
        progress["last_daily_update"] = datetime.now().isoformat()
        _save_progress(progress)

        elapsed = time.time() - t0
        logger.info(
            f"[historical] === DAILY UPDATE COMPLETE ===\n"
            f"  Updated: {stats['updated']}\n"
            f"  Failed:  {stats['failed']}\n"
            f"  Bars:    {stats['bars']:,}\n"
            f"  Time:    {elapsed / 60:.1f} min"
        )

        conn.close()

        # Update news sentiment before processing
        if run_processor and stats.get("updated", 0) > 0:
            try:
                from src.data.news_sentiment_downloader import NewsSentimentDownloader
                nsd = NewsSentimentDownloader()
                news_stats = nsd.run_daily_update()
                stats["news_sentiment"] = news_stats
                logger.info(f"[historical] News sentiment: {news_stats.get('articles', 0)} articles")
            except Exception as e:
                logger.warning(f"[historical] News sentiment update failed: {e}")

            # Launch NPU/GPU data processor on the fresh data
            proc_result = self._run_data_processor()
            stats["processor"] = proc_result

        return stats

    def _run_data_processor(self) -> dict:
        """
        Launch the NPU/GPU data processor to rebuild the processed data block.

        Runs incremental mode (only updated symbols) so it finishes fast.
        Falls back gracefully if the processor module isn't available.
        """
        try:
            from src.ops.data_processor import DataProcessor
            logger.info("[historical] Launching NPU/GPU data processor...")
            dp = DataProcessor()
            result = dp.run_incremental()
            logger.info(
                f"[historical] Data processor done: "
                f"{result.symbols_processed} symbols, "
                f"{result.features_written} features, "
                f"{result.predictions_written} predictions "
                f"({result.device}, {result.duration_seconds:.1f}s)"
            )
            return {
                "symbols_processed": result.symbols_processed,
                "features_written": result.features_written,
                "predictions_written": result.predictions_written,
                "device": result.device,
                "duration_s": round(result.duration_seconds, 1),
            }
        except ImportError:
            logger.warning(
                "[historical] Data processor not available — "
                "processed data block will not be updated"
            )
            return {"error": "data_processor module not found"}
        except Exception as e:
            logger.error(f"[historical] Data processor failed: {e}")
            return {"error": str(e)}

    def get_historical(
        self,
        symbol: str,
        days: int = 365,
        include_indicators: bool = True,
    ) -> Optional[pd.DataFrame]:
        """
        Read historical data from the database for a symbol.
        Returns a DataFrame with OHLCV + indicators ready for the trader.

        Args:
            symbol: Ticker symbol
            days: Number of days of history to retrieve
            include_indicators: Whether to join indicators table

        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume + indicator columns
            Index is DatetimeIndex. Returns None if no data found.
        """
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        with _get_conn(self._db_path) as conn:
            if include_indicators:
                query = """
                    SELECT d.date, d.open, d.high, d.low, d.close, d.volume,
                           i.sma_20, i.sma_50, i.sma_200,
                           i.ema_12, i.ema_26, i.rsi_14,
                           i.macd, i.macd_signal, i.macd_hist,
                           i.bb_upper, i.bb_middle, i.bb_lower,
                           i.atr_14, i.volatility_20d,
                           i.obv, i.stoch_k, i.stoch_d
                    FROM daily_bars d
                    LEFT JOIN indicators i ON d.symbol = i.symbol AND d.date = i.date
                    WHERE d.symbol = ? AND d.date >= ?
                    ORDER BY d.date ASC
                """
            else:
                query = """
                    SELECT date, open, high, low, close, volume
                    FROM daily_bars
                    WHERE symbol = ? AND date >= ?
                    ORDER BY date ASC
                """

            df = pd.read_sql_query(query, conn, params=(symbol, start_date))

        if df.empty:
            return None

        # Format for trader compatibility
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        df.columns = [c.replace("open", "Open").replace("high", "High")
                       .replace("low", "Low").replace("close", "Close")
                       .replace("volume", "Volume") if c in ("open", "high", "low", "close", "volume")
                       else c for c in df.columns]

        return df

    def get_symbols_in_db(self) -> list[str]:
        """List all symbols that have data in the database."""
        with _get_conn(self._db_path) as conn:
            rows = conn.execute(
                "SELECT DISTINCT symbol FROM daily_bars ORDER BY symbol"
            ).fetchall()
        return [r[0] for r in rows]

    def get_db_stats(self) -> dict:
        """Get database statistics."""
        with _get_conn(self._db_path) as conn:
            n_symbols = conn.execute("SELECT COUNT(DISTINCT symbol) FROM daily_bars").fetchone()[0]
            n_bars = conn.execute("SELECT COUNT(*) FROM daily_bars").fetchone()[0]
            n_indicators = conn.execute("SELECT COUNT(*) FROM indicators").fetchone()[0]

            oldest = conn.execute("SELECT MIN(date) FROM daily_bars").fetchone()[0]
            newest = conn.execute("SELECT MAX(date) FROM daily_bars").fetchone()[0]

            # DB file size
            db_size = self._db_path.stat().st_size if self._db_path.exists() else 0

        return {
            "symbols": n_symbols,
            "daily_bars": n_bars,
            "indicators": n_indicators,
            "oldest_date": oldest,
            "newest_date": newest,
            "db_size_mb": round(db_size / (1024 * 1024), 1),
        }


# ── CLI Entry Point ───────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Historical Data Downloader")
    parser.add_argument("--initial", action="store_true", help="Run initial full download")
    parser.add_argument("--daily", action="store_true", help="Run daily incremental update")
    parser.add_argument("--stats", action="store_true", help="Show database statistics")
    parser.add_argument("--years", type=int, default=25, help="Years of history for initial download")
    parser.add_argument("--no-processor", action="store_true", help="Skip NPU/GPU data processor after download")
    args = parser.parse_args()

    dl = HistoricalDownloader()

    if args.stats:
        stats = dl.get_db_stats()
        print(f"Symbols:    {stats['symbols']}")
        print(f"Daily bars: {stats['daily_bars']:,}")
        print(f"Indicators: {stats['indicators']:,}")
        print(f"Date range: {stats['oldest_date']} -> {stats['newest_date']}")
        print(f"DB size:    {stats['db_size_mb']} MB")
    elif args.initial:
        dl.run_initial(years=args.years)
        if not args.no_processor:
            # Backfill news sentiment before processing
            print("\nBackfilling news sentiment...")
            try:
                from src.data.news_sentiment_downloader import NewsSentimentDownloader
                nsd = NewsSentimentDownloader()
                nsd.run_backfill()
                nsd.rescore_with_finbert()
            except Exception as e:
                print(f"News sentiment backfill error: {e}")

            print("\nRunning full data processor on downloaded data...")
            try:
                from src.ops.data_processor import DataProcessor
                dp = DataProcessor()
                dp.run(retrain=True)
            except Exception as e:
                print(f"Data processor error: {e}")
    elif args.daily:
        dl.run_daily_update(run_processor=not args.no_processor)
    else:
        print("Usage: python -m src.data.historical_downloader --initial|--daily|--stats")
        print(f"Universe: {len(dl.symbols)} symbols")
