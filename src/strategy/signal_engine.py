"""
SignalEngine – kører strategier mod markedsdata og producerer rangerede signaler.

Flow:
  1. Modtag DataFrames fra DataPipeline (dict[symbol → df])
  2. Kør alle registrerede strategier parallelt per symbol
  3. Aggregér til ét samlet signal per symbol (via CombinedStrategy-logik)
  4. Rangér symboler efter confidence
  5. Log og gem signal-historik i SQLite
"""

from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger

from config.settings import settings
from src.strategy.base_strategy import BaseStrategy, Signal, StrategyResult


# ── Dataklasser ──────────────────────────────────────────────

@dataclass
class SymbolSignal:
    """Samlet signal for ét symbol med metadata."""
    symbol: str
    signal: Signal
    confidence: float
    position_size_usd: float
    reason: str
    timestamp: str
    strategy_details: list[dict] = field(default_factory=list)

    @property
    def is_actionable(self) -> bool:
        return self.signal != Signal.HOLD and self.confidence > 0


@dataclass
class EngineResult:
    """Resultat af en fuld engine-kørsel over alle symboler."""
    timestamp: str
    signals: list[SymbolSignal]
    run_duration_ms: float

    @property
    def actionable(self) -> list[SymbolSignal]:
        """Kun BUY/SELL signaler, sorteret efter confidence (højest først)."""
        return sorted(
            [s for s in self.signals if s.is_actionable],
            key=lambda s: s.confidence,
            reverse=True,
        )

    @property
    def buys(self) -> list[SymbolSignal]:
        return [s for s in self.actionable if s.signal == Signal.BUY]

    @property
    def sells(self) -> list[SymbolSignal]:
        return [s for s in self.actionable if s.signal == Signal.SELL]


# ── Signal-historik (SQLite) ─────────────────────────────────

class SignalStore:
    """Persistent log af alle signaler for historisk analyse."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    def _init_db(self) -> None:
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS signal_history (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp   TEXT    NOT NULL,
                    symbol      TEXT    NOT NULL,
                    signal      TEXT    NOT NULL,
                    confidence  REAL    NOT NULL,
                    position_usd REAL   NOT NULL,
                    reason      TEXT,
                    strategies  TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_signal_symbol_ts
                ON signal_history (symbol, timestamp)
            """)

    def save(self, sig: SymbolSignal) -> None:
        details = "; ".join(
            f"{d['strategy']}={d['signal']}({d['confidence']:.0f})"
            for d in sig.strategy_details
        )
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO signal_history
                    (timestamp, symbol, signal, confidence, position_usd, reason, strategies)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    sig.timestamp, sig.symbol, sig.signal.value,
                    sig.confidence, sig.position_size_usd,
                    sig.reason, details,
                ),
            )

    def save_batch(self, signals: list[SymbolSignal]) -> None:
        rows = []
        for sig in signals:
            details = "; ".join(
                f"{d['strategy']}={d['signal']}({d['confidence']:.0f})"
                for d in sig.strategy_details
            )
            rows.append((
                sig.timestamp, sig.symbol, sig.signal.value,
                sig.confidence, sig.position_size_usd,
                sig.reason, details,
            ))

        with self._get_conn() as conn:
            conn.executemany(
                """
                INSERT INTO signal_history
                    (timestamp, symbol, signal, confidence, position_usd, reason, strategies)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    def get_history(
        self,
        symbol: str | None = None,
        limit: int = 100,
    ) -> pd.DataFrame:
        query = "SELECT * FROM signal_history"
        params: list = []

        if symbol:
            query += " WHERE symbol = ?"
            params.append(symbol)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with self._get_conn() as conn:
            return pd.read_sql_query(query, conn, params=params)

    def count(self, symbol: str | None = None) -> int:
        query = "SELECT COUNT(*) FROM signal_history"
        params: list = []
        if symbol:
            query += " WHERE symbol = ?"
            params.append(symbol)

        with self._get_conn() as conn:
            return conn.execute(query, params).fetchone()[0]

    def prune(self, keep_days: int = 14) -> None:
        """Remove signal history older than keep_days to prevent unbounded growth."""
        try:
            with self._get_conn() as conn:
                cutoff = (pd.Timestamp.now() - pd.Timedelta(days=keep_days)).isoformat()
                conn.execute(
                    "DELETE FROM signal_history WHERE timestamp < ?",
                    (cutoff,),
                )
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except Exception:
            pass


# ── SignalEngine ─────────────────────────────────────────────

class SignalEngine:
    """
    Kører strategier mod markedsdata og producerer rangerede signaler.

    Brug:
        engine = SignalEngine(strategies=[...])
        result = engine.process(pipeline_data)
        for sig in result.actionable:
            print(sig.symbol, sig.signal, sig.confidence)
    """

    def __init__(
        self,
        strategies: list[tuple[BaseStrategy, float]],
        min_agreement: int = 2,
        portfolio_value: float | None = None,
        max_position_pct: float | None = None,
        cache_dir: str | None = None,
        max_workers: int = 4,
    ) -> None:
        """
        Args:
            strategies: Liste af (strategi, vægt). Vægte normaliseres.
            min_agreement: Minimum enige strategier for at handle.
            portfolio_value: Porteføljeværdi i USD (til position sizing).
            max_position_pct: Maks andel per position.
            cache_dir: Mappe til signal-historik DB.
            max_workers: Antal tråde til parallel processering.
        """
        if not strategies:
            raise ValueError("Mindst én strategi kræves")

        total_weight = sum(w for _, w in strategies)
        self._strategies = [(s, w / total_weight) for s, w in strategies]
        self._min_agreement = min_agreement
        self._portfolio_value = portfolio_value or settings.backtest.initial_capital
        self._max_position_pct = max_position_pct or settings.risk.max_position_pct
        self._max_workers = max_workers

        cache_path = Path(cache_dir or settings.market_data.cache_dir)
        self.store = SignalStore(cache_path / "signals.db")

        self._run_count = 0

    # ── Public API ───────────────────────────────────────────

    def process(
        self,
        data: dict[str, pd.DataFrame],
        portfolio_value: float | None = None,
    ) -> EngineResult:
        """
        Kør alle strategier mod alle symboler.

        Args:
            data: Dict fra DataPipeline: symbol → DataFrame med indikatorer.
            portfolio_value: Overstyr porteføljeværdi for denne kørsel.

        Returns:
            EngineResult med alle signaler, rangeret efter confidence.
        """
        self._run_count += 1
        pv = portfolio_value or self._portfolio_value
        start = pd.Timestamp.now()
        ts = start.isoformat()

        logger.info(
            f"SignalEngine run #{self._run_count}: "
            f"{len(data)} symboler, {len(self._strategies)} strategier"
        )

        # Kør parallelt over symboler
        signals: list[SymbolSignal] = []

        if len(data) <= 2 or self._max_workers <= 1:
            # Sekventiel for få symboler
            for symbol, df in data.items():
                sig = self._process_symbol(symbol, df, pv, ts)
                signals.append(sig)
        else:
            # Parallel for mange symboler
            with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
                futures = {
                    executor.submit(self._process_symbol, sym, df, pv, ts): sym
                    for sym, df in data.items()
                }
                for future in as_completed(futures):
                    sym = futures[future]
                    try:
                        signals.append(future.result())
                    except Exception as exc:
                        logger.error(f"Engine fejl for {sym}: {exc}")
                        signals.append(SymbolSignal(
                            symbol=sym, signal=Signal.HOLD, confidence=0,
                            position_size_usd=0, reason=f"Fejl: {exc}",
                            timestamp=ts,
                        ))

        # Gem alle signaler i databasen
        self.store.save_batch(signals)

        # Prune old signal history periodically (not every scan)
        if not hasattr(self, "_scan_count"):
            self._scan_count = 0
        self._scan_count += 1
        if self._scan_count % 50 == 0:
            self.store.prune(keep_days=7)

        elapsed_ms = (pd.Timestamp.now() - start).total_seconds() * 1000
        result = EngineResult(timestamp=ts, signals=signals, run_duration_ms=elapsed_ms)

        # Log opsummering
        self._log_summary(result)

        return result

    def update_portfolio_value(self, value: float) -> None:
        """Opdatér porteføljeværdi (f.eks. fra broker)."""
        self._portfolio_value = value

    # ── Intern processering ──────────────────────────────────

    def _process_symbol(
        self,
        symbol: str,
        df: pd.DataFrame,
        portfolio_value: float,
        timestamp: str,
    ) -> SymbolSignal:
        """Kør alle strategier mod ét symbol og aggregér."""
        if df.empty:
            return SymbolSignal(
                symbol=symbol, signal=Signal.HOLD, confidence=0,
                position_size_usd=0, reason="Ingen data",
                timestamp=timestamp,
            )

        # Kør hver strategi — skip strategies with 0 confidence (insufficient data)
        strategy_results: list[tuple[StrategyResult, float, str]] = []
        for strategy, weight in self._strategies:
            try:
                result = strategy.analyze(df)
                if result.confidence > 0 or result.signal != Signal.HOLD:
                    strategy_results.append((result, weight, strategy.name))
                else:
                    logger.debug(f"{strategy.name} skipped for {symbol}: insufficient data")
            except Exception as exc:
                logger.warning(f"{strategy.name} fejlede for {symbol}: {exc}")

        # Aggregér signaler
        signal, confidence, reason = self._aggregate(strategy_results)

        # Position sizing
        position_usd = 0.0
        if signal != Signal.HOLD:
            fraction = confidence / 100.0
            position_usd = round(
                portfolio_value * self._max_position_pct * fraction, 2,
            )

        # Byg details
        details = [
            {
                "strategy": name,
                "signal": r.signal.value,
                "confidence": r.confidence,
                "reason": r.reason,
            }
            for r, _, name in strategy_results
        ]

        sig = SymbolSignal(
            symbol=symbol,
            signal=signal,
            confidence=confidence,
            position_size_usd=position_usd,
            reason=reason,
            timestamp=timestamp,
            strategy_details=details,
        )

        if sig.is_actionable:
            logger.info(
                f"  {symbol}: {signal.value} (conf={confidence:.0f}, "
                f"${position_usd:,.0f}) – {reason}"
            )
        else:
            logger.debug(f"  {symbol}: HOLD – {reason}")

        return sig

    def _aggregate(
        self,
        results: list[tuple[StrategyResult, float, str]],
    ) -> tuple[Signal, float, str]:
        """
        Aggregér strategi-resultater til ét signal.

        Regler:
          - Mindst `min_agreement` strategier skal pege samme vej.
          - Confidence = vægtet gennemsnit af de enige strategier.
          - Ved uenighed → HOLD.
        """
        buy_items = [(r, w) for r, w, _ in results if r.signal == Signal.BUY]
        sell_items = [(r, w) for r, w, _ in results if r.signal == Signal.SELL]

        buy_count = len(buy_items)
        sell_count = len(sell_items)
        total = len(results)

        if buy_count >= self._min_agreement and buy_count > sell_count:
            conf = self._weighted_confidence(buy_items)
            return Signal.BUY, conf, f"{buy_count}/{total} strategier siger BUY"

        if sell_count >= self._min_agreement and sell_count > buy_count:
            conf = self._weighted_confidence(sell_items)
            return Signal.SELL, conf, f"{sell_count}/{total} strategier siger SELL"

        return (
            Signal.HOLD, 0,
            f"Ingen konsensus: {buy_count} BUY, {sell_count} SELL, "
            f"{total - buy_count - sell_count} HOLD",
        )

    @staticmethod
    def _weighted_confidence(items: list[tuple[StrategyResult, float]]) -> float:
        total_w = sum(w for _, w in items)
        if total_w == 0:
            return 0.0
        return sum(r.confidence * w for r, w in items) / total_w

    def _log_summary(self, result: EngineResult) -> None:
        buys = len(result.buys)
        sells = len(result.sells)
        holds = len(result.signals) - buys - sells

        logger.info(
            f"SignalEngine run #{self._run_count} færdig i "
            f"{result.run_duration_ms:.0f}ms: "
            f"{buys} BUY, {sells} SELL, {holds} HOLD"
        )

        if result.actionable:
            top = result.actionable[0]
            logger.info(
                f"  Top signal: {top.symbol} {top.signal.value} "
                f"(conf={top.confidence:.0f}, ${top.position_size_usd:,.0f})"
            )
