"""
AutoTrader – Hjernen der kobler signaler til ordrer.

Nu med fuld 24/7 global market coverage:
  - Bruger MarketCalendar til at bestemme hvilke symboler der scannes
  - Kun aktive markeder scannes (sparer RAM og CPU)
  - Pre/post market support
  - Market handoff integration (prior session sentiment påvirker position size)

Flow:
  1. Hent aktive markeder fra MarketCalendar
  2. Hent markedsdata KUN for åbne markeder
  3. Generer signaler (SignalEngine + AlphaScore)
  4. Anvend handoff-justeringer fra tidligere sessioner
  5. Check exit-signaler (stop-loss/take-profit)
  6. Check og eksekver nye entries
  7. Log alt til SQLite + dashboard
"""

from __future__ import annotations

import gc
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from loguru import logger

from src.strategy.base_strategy import Signal, StrategyResult
from src.strategy.signal_engine import SignalEngine, EngineResult, SymbolSignal
from src.strategy.combined_strategy import CombinedStrategy
from src.strategy.rsi_strategy import RSIStrategy
from src.strategy.sma_crossover import SMACrossoverStrategy
from src.strategy.pattern_strategy import PatternStrategy
from src.broker.broker_router import BrokerRouter
from src.broker.models import Order, OrderSide, OrderStatus
from src.data.market_data import MarketDataFetcher
from src.data.indicators import add_all_indicators

CET = ZoneInfo("Europe/Copenhagen")


def _now_cet() -> datetime:
    """Get CET time from web-synced time service, fallback to local clock.

    Fallback MUST NOT recurse — if the import or call fails, return local
    wall-clock time in Europe/Copenhagen. A recursive call here caused stack
    overflow when time_service was unavailable (fixed 2026-04-17).
    """
    try:
        from src.ops.time_service import now_cet
        return now_cet()
    except Exception:
        return datetime.now(CET)


@dataclass
class TradeAction:
    symbol:           str
    side:             str
    qty:              float
    reason:           str
    signal_confidence: float
    alpha_score:      float | None = None
    risk_approved:    bool = False
    risk_message:     str = ""
    order:            Order | None = None
    executed:         bool = False
    error:            str = ""


@dataclass
class ScanResult:
    timestamp:         datetime
    symbols_scanned:   int
    signals_generated: int
    buys_proposed:     int
    sells_proposed:    int
    trades_executed:   int
    trades_rejected:   int
    actions:           list[TradeAction] = field(default_factory=list)
    exit_signals:      int = 0
    duration_sec:      float = 0.0
    open_markets:      list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Scan {self.timestamp:%H:%M:%S}: {self.symbols_scanned} symboler "
            f"({', '.join(self.open_markets)})",
            f"  Signaler: {self.signals_generated} (BUY: {self.buys_proposed}, SELL: {self.sells_proposed})",
            f"  Exits: {self.exit_signals}",
            f"  Handler: {self.trades_executed} udført, {self.trades_rejected} afvist",
            f"  Tid: {self.duration_sec:.1f}s",
        ]
        for a in self.actions:
            status = "✓" if a.executed else f"✗ {a.risk_message or a.error}"
            lines.append(f"    {a.side} {a.qty:.1f} {a.symbol} ({a.signal_confidence:.0f}%) → {status}")
        return "\n".join(lines)


class AutoTrader:
    """
    Automatisk trading engine med 24/7 global market coverage.
    Bruger MarketCalendar til dynamisk at vælge hvilke symboler der scannes.
    """

    def __init__(
        self,
        router: BrokerRouter,
        paper: bool = True,
        watchlist: list[str] | None = None,
        min_confidence: float = 40.0,
        min_alpha_score: float = 40.0,
        min_agreement: int = 1,
        max_new_positions_per_scan: int = 8,
        position_size_pct: float = 0.10,
        cooldown_minutes: int = 2,
        target_equity: float = 200_000,
        lookback_days: int = 7,
        data_interval: str = "5m",
        data_dir: str = "data_cache",
    ) -> None:
        self.router              = router
        self.paper               = paper
        self.min_confidence      = min_confidence
        self.min_alpha_score     = min_alpha_score
        self.min_agreement       = min_agreement
        self.max_new_positions   = max_new_positions_per_scan
        self.position_size_pct   = position_size_pct
        self.cooldown_minutes    = cooldown_minutes
        self.lookback_days       = lookback_days
        self.data_interval       = data_interval
        self._target_equity      = target_equity
        self._aggressive_mode    = True
        self._use_pattern_strategy = False   # Toggled via dashboard settings
        self._crypto_trading_enabled = True  # Toggled via dashboard settings
        # Max DKK per symbol. Safe default 50_000 matches CLAUDE.md spec.
        # Overridden by config/risk_sizing.json at end of __init__.
        self.max_dkk_per_symbol  = 50_000.0
        self._advanced_feedback_enabled = False  # Report recommendations → auto-apply
        self._weekend_mode = False                # Weekend crypto rotation active
        self._pre_weekend_settings: dict | None = None  # Saved settings before weekend mode

        # Load weekend rotation config
        try:
            import json as _json
            _wr_path = Path(__file__).resolve().parent.parent.parent / "config" / "weekend_rotation.json"
            if _wr_path.exists():
                _wr = _json.loads(_wr_path.read_text())
                if _wr.get("enabled", False):
                    logger.info("[auto] Weekend rotation enabled in config")
        except Exception:
            pass

        # Load advanced feedback toggle from config
        try:
            import json as _json
            _af_path = Path(__file__).resolve().parent.parent.parent / "config" / "advanced_feedback.json"
            if _af_path.exists():
                self._advanced_feedback_enabled = _json.loads(_af_path.read_text()).get("enabled", False)
                if self._advanced_feedback_enabled:
                    logger.info("[auto] Advanced feedback loop enabled — report recommendations will auto-apply")
        except Exception:
            pass

        # Market calendar — determines which symbols to scan
        self._calendar = None
        try:
            from src.ops.market_calendar import get_calendar
            self._calendar = get_calendar()
            logger.info("[auto] MarketCalendar loaded — 24/7 global coverage active")
        except Exception as e:
            logger.warning(f"[auto] MarketCalendar not available: {e}")

        # Market handoff engine
        self._handoff = None
        try:
            from src.ops.market_handoff import get_handoff_engine
            self._handoff = get_handoff_engine()
            logger.info("[auto] MarketHandoff engine loaded")
        except Exception as e:
            logger.warning(f"[auto] MarketHandoff not available: {e}")

        # Watchlist — fallback if calendar not available
        self.watchlist = watchlist or self._get_all_symbols()

        # Market data fetcher
        self._fetcher = MarketDataFetcher(cache_dir=data_dir)

        # Pattern scanner — always runs in background
        self._pattern_strategy = PatternStrategy()
        self._pattern_strategy.start_background()

        # Signal engine
        strategies = self._build_strategies()
        self._engine = SignalEngine(
            strategies=strategies,
            min_agreement=min_agreement,
            cache_dir=data_dir,
        )

        # AlphaScore
        self._alpha_engine = None
        try:
            from src.trader.intelligence.alpha_score import AlphaScoreEngine
            self._alpha_engine = AlphaScoreEngine()
            logger.info("[auto] AlphaScore engine aktiveret")
        except Exception as e:
            logger.warning(f"[auto] AlphaScore ikke tilgængelig: {e}")

        self._risk_manager = None
        self._last_trade: dict[str, datetime] = {}
        self._exchange_stop_loss: dict[str, float] = {}  # market -> stop-loss %
        self._load_exchange_stop_loss()

        # Continuous news sentiment (replaces nightly bulk download)
        self._news_fetcher = None
        try:
            from src.data.news_sentiment_downloader import ContinuousNewsFetcher
            self._news_fetcher = ContinuousNewsFetcher(
                db_path=Path(data_dir) / "news_sentiment.db",
                batch_size=10,
                interval_seconds=300,
            )
            self._news_fetcher.start()
            logger.info("[auto] ContinuousNewsFetcher started (every 5 min)")
        except Exception as e:
            logger.warning(f"[auto] ContinuousNewsFetcher not available: {e}")

        # Feedback loop: ContinuousLearner adjusts thresholds
        self._learner = None
        try:
            from src.learning.continuous_learner import ContinuousLearner
            self._learner = ContinuousLearner(
                db_path=str(Path(data_dir) / "learning.db"),
                trade_db_path=str(Path(data_dir) / "auto_trader_log.db"),
            )
            self._learner.start(interval_minutes=5)
            logger.info("[auto] Feedback loop (ContinuousLearner) aktiveret")
        except Exception as e:
            logger.warning(f"[auto] ContinuousLearner ikke tilgaengelig: {e}")

        # Load persisted position sizing overrides.
        # If the config file is missing or malformed we keep the hardcoded
        # safe defaults and log a WARNING — silent fallback hid a real
        # 5k-vs-100k mismatch for months (fixed 2026-04-17).
        try:
            import json as _json
            _rs_path = Path(__file__).resolve().parent.parent.parent / "config" / "risk_sizing.json"
            if _rs_path.exists():
                _rs = _json.loads(_rs_path.read_text())
                if "max_position_pct" in _rs:
                    _val = _rs["max_position_pct"]
                    position_size_pct = _val / 100.0 if _val > 1 else _val
                    self.position_size_pct = position_size_pct
                    logger.info(f"[auto] Loaded position_size_pct override: {position_size_pct:.1%}")
                if "max_dkk_per_symbol" in _rs:
                    self.max_dkk_per_symbol = float(_rs["max_dkk_per_symbol"])
                    logger.info(f"[auto] Loaded max_dkk_per_symbol override: {self.max_dkk_per_symbol:,.0f} DKK")
            else:
                logger.warning(
                    f"[auto] config/risk_sizing.json not found — using safe defaults "
                    f"(position_size_pct={self.position_size_pct:.1%}, "
                    f"max_dkk_per_symbol={self.max_dkk_per_symbol:,.0f}). "
                    f"Live trading should NOT start without an explicit config."
                )
        except Exception as e:
            logger.error(
                f"[auto] config/risk_sizing.json malformed ({e}) — using safe defaults. "
                f"Live trading should NOT start until this is fixed.",
                exc_info=True,
            )

        # Adaptive thresholds (adjusted by feedback loop)
        # NB: _base_position_size_pct sættes FØR config-override for korrekt feedback-loop
        self._base_min_confidence = min_confidence
        self._base_position_size_pct = self.position_size_pct

        # SQLite log
        self._db_path = Path(data_dir) / "auto_trader_log.db"
        try:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            test_file = self._db_path.parent / ".write_test"
            test_file.touch()
            test_file.unlink()
        except OSError:
            import tempfile
            self._db_path = Path(tempfile.gettempdir()) / "auto_trader_log.db"
            logger.warning(f"[auto] DB fallback til {self._db_path}")
        self._init_db()

        self._total_scans  = 0
        self._total_trades = 0

        # ── Concurrency guards (fixed 2026-04-17) ──────────
        # Serialise scan_and_trade so dashboard-triggered scans and scheduler
        # ticks cannot interleave position-sync with order execution.
        self._scan_lock = threading.RLock()
        # Per-symbol in-flight set: a symbol in here has an active order being
        # placed. Prevents duplicate entries and duplicate exits.
        self._in_flight: set[str] = set()
        self._in_flight_lock = threading.Lock()
        # Tracks last exit time per symbol to make _check_exits idempotent.
        self._last_exit: dict[str, datetime] = {}

        mode = "PAPER" if paper else "⚠️  LIVE"
        logger.info(
            f"[auto] AutoTrader initialiseret — {mode} mode, "
            f"interval={data_interval}, "
            f"max {max_new_positions_per_scan} nye positioner/scan"
        )

    def _get_all_symbols(self) -> list[str]:
        """Get all symbols from market calendar or fallback list."""
        try:
            from src.ops.market_calendar import MARKET_SYMBOLS
            all_syms = []
            for syms in MARKET_SYMBOLS.values():
                for s in syms:
                    if s not in all_syms:
                        all_syms.append(s)
            return all_syms
        except Exception:
            return [
                "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
                "SPY", "QQQ", "GLD", "BTC-USD", "ETH-USD",
                "NOVO-B.CO", "VOLV-B.ST", "EQNR.OL",
            ]

    def set_risk_manager(self, rm) -> None:
        self._risk_manager = rm
        logger.info("[auto] RiskManager tilknyttet")

    def set_crypto_trading(self, enabled: bool) -> None:
        """Enable/disable crypto symbols (BTC-USD, ETH-USD, etc.) in scan."""
        self._crypto_trading_enabled = enabled
        if not enabled:
            self.watchlist = [s for s in self.watchlist if not s.endswith("-USD")]
            logger.info(f"[auto] Crypto trading disabled — {len(self.watchlist)} symbols remain")
        else:
            # Re-add crypto symbols from full list
            all_syms = self._get_all_symbols()
            crypto = [s for s in all_syms if s.endswith("-USD")]
            for s in crypto:
                if s not in self.watchlist:
                    self.watchlist.append(s)
            logger.info(f"[auto] Crypto trading enabled — {len(self.watchlist)} symbols")

    def set_pattern_strategy(self, enabled: bool) -> None:
        """Enable/disable pattern analysis in trading pipeline."""
        if enabled != self._use_pattern_strategy:
            self._use_pattern_strategy = enabled
            strategies = self._build_strategies()
            self._engine = SignalEngine(strategies)
            logger.info(f"[auto] PatternStrategy {'enabled' if enabled else 'disabled'} — strategies rebuilt")

    def set_weekend_mode(
        self,
        enabled: bool,
        crypto_alloc_pct: int = 60,
        close_stocks: bool = True,
        close_futures: bool = True,
    ) -> None:
        """Activate or deactivate weekend crypto rotation.

        When enabled (Friday evening):
          1. Close non-crypto positions to free capital
          2. Increase crypto position sizing
          3. Tighten stop-loss / take-profit for crypto scalping

        When disabled (Monday morning):
          1. Restore original position sizing and risk parameters
          2. Resume normal multi-asset trading
        """
        import json as _json

        if enabled and not self._weekend_mode:
            # ── Save current settings (snapshot BEFORE any mutation) ──
            snapshot = {
                "position_size_pct": self.position_size_pct,
                "max_dkk_per_symbol": self.max_dkk_per_symbol,
                "cooldown_minutes": self.cooldown_minutes,
                "min_confidence": self.min_confidence,
            }
            rm = getattr(self, "_risk_manager", None)
            if rm:
                snapshot["stop_loss_pct"] = getattr(rm, "stop_loss_pct", 0.05)
                snapshot["take_profit_pct"] = getattr(rm, "take_profit_pct", 0.10)
                snapshot["trailing_stop_pct"] = getattr(rm, "trailing_stop_pct", 0.03)

            # Install snapshot BEFORE mutating state so that a rollback can
            # happen even if an exception hits mid-way.
            self._pre_weekend_settings = snapshot

            closed_count = 0
            weekend_applied = False
            try:
                # ── Close non-crypto positions ────────────────
                positions = []
                if hasattr(self, "_portfolio") and self._portfolio is not None and hasattr(self._portfolio, "positions"):
                    positions = list(self._portfolio.positions.keys())

                for sym in positions:
                    is_crypto = sym.endswith("-USD")
                    is_future = "=F" in sym
                    if is_crypto:
                        continue
                    if is_future and not close_futures:
                        continue
                    if not is_future and not close_stocks:
                        continue
                    try:
                        pos = self._portfolio.positions[sym]
                        self._portfolio.close_position(
                            symbol=sym,
                            price=pos.current_price or pos.entry_price,
                            reason="weekend_rotation",
                        )
                        closed_count += 1
                    except Exception as e:
                        logger.warning(f"[auto] Weekend rotation — could not close {sym}: {e}")

                # ── Apply weekend crypto parameters ───────────
                target_pct = crypto_alloc_pct / 100.0
                n_positions = max(1, min(4, int(target_pct / 0.05)))
                per_position_pct = target_pct / n_positions
                self.position_size_pct = per_position_pct
                self.max_dkk_per_symbol = max(self.max_dkk_per_symbol, 15000.0)
                self.cooldown_minutes = 30
                self.min_confidence = max(self.min_confidence, 55.0)

                if rm:
                    rm.stop_loss_pct = 0.015
                    rm.take_profit_pct = 0.025
                    rm.trailing_stop_pct = 0.012

                drm = getattr(self, "_dynamic_risk", None)
                if drm and hasattr(drm, "_current_params"):
                    snapshot["max_exposure_pct"] = drm._current_params.get(
                        "max_exposure_pct", 0.95
                    )
                    drm._current_params["max_exposure_pct"] = target_pct
                    logger.info(f"[auto] Weekend max_exposure capped at {target_pct:.0%}")

                weekend_applied = True
                self._weekend_mode = True
                logger.info(
                    f"[auto] WEEKEND MODE ON — closed {closed_count} non-crypto positions, "
                    f"crypto target={target_pct:.0%} ({n_positions} positions x {per_position_pct:.0%}), "
                    f"remaining {1 - target_pct:.0%} stays as cash, "
                    f"SL=1.5%, TP=2.5%, TS=1.2%"
                )
            except Exception as e:
                # Roll back any partial mutation so we don't leave the trader
                # in a half-weekend state. Fixed 2026-04-17.
                logger.error(
                    f"[auto] Weekend mode activation FAILED mid-way ({e}) — "
                    f"rolling back ({closed_count} positions already closed cannot "
                    f"be reopened automatically)",
                    exc_info=True,
                )
                if not weekend_applied:
                    self.position_size_pct = snapshot["position_size_pct"]
                    self.max_dkk_per_symbol = snapshot["max_dkk_per_symbol"]
                    self.cooldown_minutes = snapshot["cooldown_minutes"]
                    self.min_confidence = snapshot["min_confidence"]
                    if rm:
                        rm.stop_loss_pct = snapshot.get("stop_loss_pct", rm.stop_loss_pct)
                        rm.take_profit_pct = snapshot.get("take_profit_pct", rm.take_profit_pct)
                        rm.trailing_stop_pct = snapshot.get("trailing_stop_pct", rm.trailing_stop_pct)
                    self._pre_weekend_settings = None
                raise

        elif not enabled and self._weekend_mode:
            # ── Restore original settings ─────────────────
            if self._pre_weekend_settings:
                self.position_size_pct = self._pre_weekend_settings["position_size_pct"]
                self.max_dkk_per_symbol = self._pre_weekend_settings["max_dkk_per_symbol"]
                self.cooldown_minutes = self._pre_weekend_settings["cooldown_minutes"]
                self.min_confidence = self._pre_weekend_settings["min_confidence"]

                rm = getattr(self, "_risk_manager", None)
                if rm:
                    rm.stop_loss_pct = self._pre_weekend_settings.get("stop_loss_pct", 0.05)
                    rm.take_profit_pct = self._pre_weekend_settings.get("take_profit_pct", 0.10)
                    rm.trailing_stop_pct = self._pre_weekend_settings.get("trailing_stop_pct", 0.03)

                # Restore exposure cap
                drm = getattr(self, "_dynamic_risk", None)
                if drm and hasattr(drm, "_current_params") and "max_exposure_pct" in self._pre_weekend_settings:
                    drm._current_params["max_exposure_pct"] = self._pre_weekend_settings["max_exposure_pct"]

                self._pre_weekend_settings = None

            self._weekend_mode = False
            logger.info("[auto] WEEKEND MODE OFF — normal trading parameters restored")

    def _load_exchange_stop_loss(self) -> None:
        """Load per-exchange stop-loss from config/exchange_stop_loss.json."""
        try:
            import json
            sl_path = Path(__file__).resolve().parent.parent.parent / "config" / "exchange_stop_loss.json"
            if sl_path.exists():
                self._exchange_stop_loss = json.loads(sl_path.read_text())
                if self._exchange_stop_loss:
                    logger.info(f"[auto] Loaded exchange stop-loss for {len(self._exchange_stop_loss)} exchanges")
        except Exception as e:
            logger.debug(f"[auto] Could not load exchange stop-loss: {e}")

    def set_exchange_stop_loss(self, exchange: str, pct: float) -> None:
        """Set per-exchange stop-loss override (in %)."""
        self._exchange_stop_loss[exchange] = pct
        logger.info(f"[auto] Exchange stop-loss set: {exchange} = {pct:.1f}%")

    # Suffix → exchange key for individual European/Nordic exchanges
    _SUFFIX_EXCHANGE = {
        ".CO": "denmark",
        ".ST": "sweden",
        ".OL": "norway",
        ".HE": "finland",
        ".DE": "germany", ".F": "germany",
        ".PA": "france",
        ".AS": "netherlands",
        ".SW": "switzerland",
        ".MC": "spain",
        ".MI": "italy",
        ".L": "london",
        ".AX": "australia",
        ".NZ": "new_zealand",
        ".T": "japan",
        ".HK": "hong_kong",
        ".NS": "india",
    }

    def get_stop_loss_for_symbol(self, symbol: str) -> float | None:
        """Return exchange-specific stop-loss % for a symbol, or None for default."""
        if not self._exchange_stop_loss:
            return None

        # 1. Try suffix-based match (covers individual EU/Nordic exchanges)
        for suffix, exchange_key in self._SUFFIX_EXCHANGE.items():
            if symbol.endswith(suffix) and exchange_key in self._exchange_stop_loss:
                return self._exchange_stop_loss[exchange_key] / 100.0

        # 2. Crypto by suffix
        if symbol.endswith("-USD") and "crypto" in self._exchange_stop_loss:
            return self._exchange_stop_loss["crypto"] / 100.0

        # 3. Futures by suffix
        if (symbol.endswith("=F") or symbol.startswith("^")) and "chicago" in self._exchange_stop_loss:
            return self._exchange_stop_loss["chicago"] / 100.0

        # 4. Fall back to MARKET_SYMBOLS group lookup (us_stocks, etfs, etc.)
        from src.ops.market_calendar import MARKET_SYMBOLS
        for market, symbols in MARKET_SYMBOLS.items():
            if symbol in symbols and market in self._exchange_stop_loss:
                return self._exchange_stop_loss[market] / 100.0

        return None

    def _build_strategies(self) -> list[tuple]:
        strats = []
        try:
            rsi = RSIStrategy()
            strats.append((rsi, 0.30))
        except Exception as e:
            logger.warning(f"[auto] RSI strategy failed: {e}")

        try:
            sma = SMACrossoverStrategy()
            strats.append((sma, 0.30))
        except Exception as e:
            logger.warning(f"[auto] SMA strategy failed: {e}")

        try:
            if len(strats) >= 2:
                combined = CombinedStrategy(
                    strategies=[(s, w) for s, w in strats],
                    min_agreement=1,
                )
                strats.append((combined, 0.40))
        except Exception as e:
            logger.warning(f"[auto] Combined strategy failed: {e}")

        # Pattern analysis strategy (enabled via dashboard settings)
        if self._use_pattern_strategy:
            try:
                strats.append((self._pattern_strategy, 0.20))
                logger.info("[auto] PatternStrategy enabled (0.20 weight)")
            except Exception as e:
                logger.warning(f"[auto] Pattern strategy failed: {e}")

        # ML Strategy — forsøg at loade Gorm's pre-trænede model fra disk
        try:
            from src.strategy.ml_strategy import MLStrategy
            from pathlib import Path
            ml_latest = Path("models/ml_latest.joblib")
            if ml_latest.exists():
                ml = MLStrategy.load(ml_latest)
                strats.append((ml, 0.25))
                logger.info(f"[auto] MLStrategy loaded from {ml_latest} (0.25 weight)")
            else:
                # Ingen gemt model — opret ny (utrænet, bruges ikke til signaler endnu)
                ml = MLStrategy()
                strats.append((ml, 0.25))
                logger.info("[auto] MLStrategy enabled untrained — run train_remote.py on Gorm's machine (0.25 weight)")
        except Exception as e:
            logger.debug(f"[auto] ML strategy not available: {e}")

        # Ensemble ML Strategy — forsøg at loade Gorm's pre-trænede model fra disk
        try:
            from src.strategy.ensemble_ml_strategy import EnsembleMLStrategy
            from pathlib import Path
            ens_latest = Path("models/ensemble_latest.joblib")
            if ens_latest.exists():
                ensemble = EnsembleMLStrategy.load(ens_latest)
                strats.append((ensemble, 0.35))
                logger.info(f"[auto] EnsembleMLStrategy loaded from {ens_latest} (0.35 weight)")
            else:
                ensemble = EnsembleMLStrategy()
                strats.append((ensemble, 0.35))
                logger.info("[auto] EnsembleMLStrategy enabled untrained — run train_remote.py on Gorm's machine (0.35 weight)")
        except Exception as e:
            logger.debug(f"[auto] Ensemble ML strategy not available: {e}")

        return strats

    # ── Main scan loop ────────────────────────────────────

    @contextmanager
    def _claim_symbol(self, symbol: str):
        """Context manager that marks a symbol as in-flight for the duration.

        Prevents two concurrent orders for the same symbol (e.g. an exit
        signal firing while an entry for the same symbol is mid-flight).
        Yields True if the claim succeeded, False if the symbol was already
        in flight. Releases the claim on exit.
        """
        claimed = False
        with self._in_flight_lock:
            if symbol not in self._in_flight:
                self._in_flight.add(symbol)
                claimed = True
        try:
            yield claimed
        finally:
            if claimed:
                with self._in_flight_lock:
                    self._in_flight.discard(symbol)

    def scan_and_trade(self) -> ScanResult:
        """
        Run one complete scan cycle.
        Only scans symbols for currently open markets.
        Applies market handoff adjustments from prior sessions.

        Serialised via self._scan_lock so concurrent invocations (scheduler +
        dashboard trigger) cannot interleave position-sync with execution.
        """
        # Acquire scan-lock for the entire cycle. RLock so nested calls from
        # same thread (e.g. via test helpers) still work.
        with self._scan_lock:
            return self._scan_and_trade_locked()

    def _scan_and_trade_locked(self) -> ScanResult:
        t0  = time.time()
        now = _now_cet()
        self._total_scans += 1

        # Periodic cleanup of stale tracking dicts (every 100 scans)
        if self._total_scans % 100 == 0:
            self._cleanup_last_trade()
            self._prune_databases()
            gc.collect()

        # Feedback loop: adjust thresholds every 20 scans
        if self._total_scans % 20 == 0:
            self._apply_feedback_adjustments()

        # ── Determine which markets are open right now ────
        open_markets = []
        symbols_to_scan = []

        if self._calendar is not None:
            open_markets    = self._calendar.get_open_markets(now)
            symbols_to_scan = self._calendar.get_symbols_to_scan(now)
            logger.info(
                f"[auto] Open markets: {open_markets} → {len(symbols_to_scan)} symbols"
            )
        else:
            symbols_to_scan = self.watchlist
            logger.info(f"[auto] No calendar — scanning all {len(symbols_to_scan)} symbols")

        # If no markets open (should not happen with crypto), return early
        if not symbols_to_scan:
            logger.info("[auto] No markets open right now — skipping scan")
            return ScanResult(
                timestamp=now, symbols_scanned=0, signals_generated=0,
                buys_proposed=0, sells_proposed=0, trades_executed=0,
                trades_rejected=0, open_markets=open_markets,
                duration_sec=time.time() - t0,
            )

        # ── Get handoff signal for position sizing ────────
        handoff_multiplier = 1.0
        handoff_risk_off   = False
        if self._handoff is not None and open_markets:
            primary_market = open_markets[0] if open_markets[0] != "crypto" else (
                open_markets[1] if len(open_markets) > 1 else "us"
            )
            try:
                signal             = self._handoff.get_handoff_signal(primary_market)
                handoff_multiplier = signal.position_size_multiplier
                handoff_risk_off   = signal.risk_off
                if handoff_risk_off:
                    logger.warning(
                        f"[auto] RISK-OFF from prior sessions — "
                        f"reducing position sizes to {handoff_multiplier:.0%}"
                    )
            except Exception:
                pass

        # ── Target equity check ───────────────────────────
        if self._aggressive_mode:
            try:
                account = self.router.get_account()
                equity  = account.equity
                if equity >= self._target_equity:
                    logger.warning(
                        f"[auto] TARGET NÅET! ${equity:,.2f} — skifter til KONSERVATIV"
                    )
                    self.position_size_pct  = 0.02
                    self.max_new_positions  = 3
                    self.min_confidence     = 60.0
                    self.min_agreement      = 2
                    self.cooldown_minutes   = 15
                    self._aggressive_mode   = False
                    self._target_reached_at = datetime.now().isoformat()
            except Exception:
                pass

        logger.info(
            f"[auto] ═══ Scan #{self._total_scans} ({now:%H:%M:%S CET}) "
            f"{'AGGRESSIV' if self._aggressive_mode else 'KONSERVATIV'} ═══"
        )

        result = ScanResult(
            timestamp=now,
            symbols_scanned=0,
            signals_generated=0,
            buys_proposed=0,
            sells_proposed=0,
            trades_executed=0,
            trades_rejected=0,
            open_markets=open_markets,
        )

        # ── 1. Fetch market data (only open markets) ──────
        data = self._fetch_market_data(symbols_to_scan)
        result.symbols_scanned = len(data)

        # Feed data to background pattern scanner (always, even if disabled)
        if data:
            self._pattern_strategy.update_all_data(data)

        # ── 1b. Update open position prices ───────────────
        try:
            broker = self.router.get_broker("paper")
            if broker and hasattr(broker, "portfolio"):
                price_updates = {}
                for sym, df in data.items():
                    if df is not None and len(df) > 0 and "Close" in df.columns:
                        try:
                            price_updates[sym.upper()] = float(df["Close"].iloc[-1])
                        except Exception:
                            pass
                if price_updates:
                    broker.portfolio.update_prices(price_updates)
                    # Also sync to risk manager portfolio
                    if self._risk_manager and hasattr(self._risk_manager, "portfolio"):
                        try:
                            self._risk_manager.portfolio.update_prices(price_updates)
                        except Exception:
                            pass
                    logger.debug(f"[auto] Updated {len(price_updates)} position prices")
        except Exception as e:
            logger.debug(f"[auto] Price update skip: {e}")

        # ── 1b2. Sync positions to risk manager portfolio ──
        try:
            if self._risk_manager and hasattr(self._risk_manager, "portfolio"):
                positions = self.router.get_positions()
                rm_portfolio = self._risk_manager.portfolio
                # Sync any positions that exist in broker but not in risk tracker
                tracked = set(rm_portfolio.positions.keys()) if hasattr(rm_portfolio, "positions") else set()
                for pos in positions:
                    sym = getattr(pos, "symbol", "")
                    if sym and sym not in tracked:
                        try:
                            rm_portfolio.open_position(
                                symbol=sym,
                                qty=getattr(pos, "qty", 0),
                                price=getattr(pos, "entry_price", 0),
                                side=getattr(pos, "side", "long"),
                            )
                        except ValueError as ve:
                            logger.warning(f"[auto] Position sync skipped for {sym}: {ve}")
                        except Exception as e:
                            logger.debug(f"[auto] Position sync error for {sym}: {e}")
        except Exception:
            pass

        # ── 1b3. Circuit breaker check ──────────────────────
        try:
            drm = getattr(self, "_dynamic_risk_manager", None)
            if drm:
                cb_state = drm.check_circuit_breakers()
                if cb_state and cb_state.is_active:
                    logger.warning(
                        f"[risk] Circuit breaker TRIGGERED: {cb_state.level.name} — "
                        f"trading halted: {cb_state.reason}"
                    )
                    result.duration_sec = time.time() - t0
                    return result
        except Exception as e:
            logger.debug(f"[risk] Circuit breaker check skip: {e}")

        if not data:
            logger.warning("[auto] Ingen markedsdata hentet — afbryder scan")
            result.duration_sec = time.time() - t0
            return result

        # ── 1c. Regime detection ─────────────────────────
        regime_result = None
        regime_adj = None
        try:
            if not hasattr(self, "_regime_detector"):
                from src.strategy.regime import RegimeDetector, AdaptiveStrategy, REGIME_INFO
                self._regime_detector = RegimeDetector()
                self._adaptive_strategy = AdaptiveStrategy(detector=self._regime_detector)

            # Use SPY as market proxy; fall back to any available data
            market_df = data.get("SPY") or data.get("^GSPC")
            if market_df is None:
                # Use the first symbol with enough data as fallback
                for _sym, _df in data.items():
                    if _df is not None and len(_df) >= 50:
                        market_df = _df
                        break

            if market_df is not None and len(market_df) >= 20:
                regime_result = self._regime_detector.detect(market_df)
                regime_adj = self._adaptive_strategy.get_adjustment(regime_result.regime)
                self._current_regime = regime_result

                # Feed regime into DynamicRiskManager for parameter adaptation
                drm = getattr(self, "_dynamic_risk_manager", None)
                if drm and hasattr(drm, "update_regime"):
                    drm.update_regime(regime_result)

                # Apply regime max exposure to position sizing
                if regime_adj.max_exposure_pct < 1.0:
                    regime_cap = regime_adj.max_exposure_pct
                    # Beregn fra base — IKKE fra nuværende (undgår monotont fald)
                    self.position_size_pct = self._base_position_size_pct * regime_cap
                else:
                    # Regime er normalt — gendan base position size
                    self.position_size_pct = self._base_position_size_pct

                # Apply stop-loss multiplier to risk manager
                if self._risk_manager and regime_adj.stop_loss_multiplier != 1.0:
                    base_sl = getattr(self._risk_manager, "_base_stop_loss_pct",
                                      self._risk_manager.stop_loss_pct)
                    self._risk_manager.stop_loss_pct = base_sl * regime_adj.stop_loss_multiplier
                    if not hasattr(self._risk_manager, "_base_stop_loss_pct"):
                        self._risk_manager._base_stop_loss_pct = base_sl

                # Log regime (only when it changes)
                prev = getattr(self, "_prev_regime", None)
                if prev != regime_result.regime:
                    logger.info(
                        f"[regime] {regime_result.label} "
                        f"(confidence {regime_result.confidence:.0f}%, "
                        f"score {regime_result.composite_score:+.2f}) — "
                        f"max exposure {regime_adj.max_exposure_pct:.0%}, "
                        f"SL multiplier {regime_adj.stop_loss_multiplier:.1f}x, "
                        f"new buys {'YES' if regime_adj.allow_new_buys else 'BLOCKED'}"
                    )
                    self._prev_regime = regime_result.regime
        except Exception as e:
            logger.debug(f"[regime] Detection error: {e}")

        # ── 2. Generate signals ───────────────────────────
        try:
            engine_result      = self._engine.process(data)
            all_signals        = engine_result.signals
            result.signals_generated = len(all_signals)
            logger.info(f"[auto] {len(all_signals)} signaler genereret")
        except Exception as e:
            logger.error(f"[auto] SignalEngine fejl: {e}")
            result.duration_sec = time.time() - t0
            return result

        # ── 3. Check exits ────────────────────────────────
        exit_actions = self._check_exits(data)
        result.exit_signals = len(exit_actions)
        for action in exit_actions:
            self._execute_action(action)
            result.actions.append(action)
            if action.executed:
                result.trades_executed += 1
                result.sells_proposed  += 1

        # ── 4. Process entry signals with handoff adjustment
        # Block new buys if regime says so (CRASH mode)
        if regime_adj and not regime_adj.allow_new_buys:
            buy_count = sum(1 for s in all_signals if s.signal == Signal.BUY)
            all_signals = [s for s in all_signals if s.signal != Signal.BUY]
            if buy_count > 0:
                logger.warning(
                    f"[regime] {regime_result.label}: Blocked {buy_count} BUY signals — "
                    f"only exits and shorts allowed"
                )

        entry_actions = self._process_entry_signals(
            all_signals, data,
            size_multiplier=handoff_multiplier,
            risk_off=handoff_risk_off,
        )
        result.buys_proposed   = sum(1 for a in entry_actions if a.side == "BUY")
        result.sells_proposed += sum(1 for a in entry_actions if a.side == "SELL")

        # ── 5. Execute entries ────────────────────────────
        executed = 0
        for action in entry_actions:
            if executed >= self.max_new_positions:
                logger.info(f"[auto] Max {self.max_new_positions} positioner nået — stopper")
                break
            self._execute_action(action)
            result.actions.append(action)
            if action.executed:
                result.trades_executed += 1
                executed               += 1
            else:
                result.trades_rejected += 1

        # ── 6. Log ────────────────────────────────────────
        result.duration_sec = time.time() - t0
        self._log_scan(result)

        # Free large scan data
        del data
        # gc.collect() fjernet — periodisk cleanup (hvert 100. scan) er tilstrækkeligt

        logger.info(
            f"[auto] ═══ Scan #{self._total_scans} færdig: "
            f"{result.trades_executed} handler, {result.duration_sec:.1f}s ═══"
        )
        return result

    # ── Data fetching ─────────────────────────────────────

    def _fetch_market_data(self, symbols: list[str]) -> dict[str, pd.DataFrame]:
        """Fetch and calculate indicators for given symbols.

        Uses historical database as primary source for daily data (instant, no API calls).
        Falls back to yfinance for symbols not in the database or for intraday intervals.
        """
        data     = {}
        interval = self.data_interval

        max_lookback = {
            "1m": 7, "2m": 60, "5m": 60, "15m": 60,
            "30m": 60, "60m": 730, "1h": 730, "1d": 365,
        }
        lookback = min(self.lookback_days, max_lookback.get(interval, 365))
        end      = datetime.now().strftime("%Y-%m-%d")
        start    = (datetime.now() - timedelta(days=lookback)).strftime("%Y-%m-%d")
        min_rows = 55 if interval in ("1m", "2m", "5m", "15m", "30m") else 20

        # Try historical database first for daily data (no API calls needed)
        remaining_symbols = list(symbols)
        if interval in ("1d", "1wk", "1mo"):
            try:
                if not hasattr(self, "_historical_dl"):
                    from src.data.historical_downloader import HistoricalDownloader
                    self._historical_dl = HistoricalDownloader()

                db_hits = 0
                still_need = []
                for symbol in symbols:
                    try:
                        df = self._historical_dl.get_historical(symbol, days=lookback)
                        if df is not None and len(df) >= min_rows:
                            # Already has indicators from the database
                            data[symbol] = df
                            db_hits += 1
                        else:
                            still_need.append(symbol)
                    except Exception:
                        still_need.append(symbol)

                remaining_symbols = still_need
                if db_hits > 0:
                    logger.info(
                        f"[auto] Historical DB: {db_hits} symbols loaded instantly, "
                        f"{len(still_need)} need API fetch"
                    )
            except Exception as e:
                logger.debug(f"[auto] Historical DB not available: {e}")

        # Fetch remaining symbols from yfinance
        for symbol in remaining_symbols:
            try:
                df = self._fetcher.get_historical(
                    symbol=symbol,
                    interval=interval,
                    start=start,
                    end=end,
                )
                if df is not None and len(df) >= min_rows:
                    df = add_all_indicators(df)
                    data[symbol] = df
                else:
                    rows = len(df) if df is not None else 0
                    logger.debug(f"[auto] {symbol}: utilstrækkelig data ({rows} rækker)")
            except Exception as e:
                logger.debug(f"[auto] {symbol}: data fejl -- {e}")

        logger.info(
            f"[auto] Data hentet for {len(data)}/{len(symbols)} symboler "
            f"(interval={interval}, lookback={lookback}d)"
        )
        return data

    # ── Exit signals ──────────────────────────────────────

    def _check_exits(self, data: dict[str, pd.DataFrame]) -> list[TradeAction]:
        actions = []
        if self._risk_manager is None:
            return actions

        prices = {}
        for sym, df in data.items():
            if len(df) > 0:
                prices[sym] = float(df["Close"].iloc[-1])

        try:
            # Batch check med default stop-loss (muterer IKKE delt state)
            exit_signals = self._risk_manager.check_positions(prices)

            # Per-symbol override: check ALLE positioner mod exchange-specifik stop-loss
            if self._exchange_stop_loss:
                exit_set = {s.symbol for s in exit_signals}
                for sym, pos in self._risk_manager.portfolio.positions.items():
                    if sym in exit_set:
                        continue  # allerede flagget af batch check
                    override = self.get_stop_loss_for_symbol(sym)
                    if override is not None and pos.unrealized_pnl_pct <= -override:
                        from src.risk.risk_manager import ExitSignal
                        exit_signals.append(ExitSignal(
                            symbol=sym,
                            reason="stop_loss",
                            message=f"Exchange stop-loss: {pos.unrealized_pnl_pct:.2%} <= -{override:.2%}",
                            trigger_price=prices.get(sym, pos.current_price),
                        ))

            # Deduplicate within this batch AND apply last-exit cooldown
            # (30s) so a retriggered scan cannot fire the same exit twice.
            # Fixes exit-signal idempotency (2026-04-17).
            seen_syms: set[str] = set()
            now = _now_cet()
            for sig in exit_signals:
                if sig.symbol in seen_syms:
                    continue
                last = self._last_exit.get(sig.symbol)
                if last is not None and (now - last).total_seconds() < 30:
                    logger.debug(
                        f"[auto] Skip duplicate exit for {sig.symbol} "
                        f"(last exit {(now - last).total_seconds():.1f}s ago)"
                    )
                    continue
                if sig.symbol in self._in_flight:
                    logger.debug(f"[auto] Skip exit for {sig.symbol}: already in flight")
                    continue
                qty = self._get_position_qty(sig.symbol)
                if qty > 0:
                    action = TradeAction(
                        symbol=sig.symbol,
                        side="SELL",
                        qty=qty,
                        reason=f"EXIT: {sig.reason} — {sig.message}",
                        signal_confidence=100.0,
                        risk_approved=True,
                        risk_message="Exit signal — auto-approved",
                    )
                    actions.append(action)
                    seen_syms.add(sig.symbol)
                    logger.warning(f"[auto] EXIT: SELL {qty} {sig.symbol} — {sig.reason}")
        except Exception as e:
            logger.error(f"[auto] Exit check fejl: {e}")

        return actions

    def _get_position_qty(self, symbol: str) -> float:
        try:
            positions = self.router.get_positions()
            for pos in positions:
                sym = getattr(pos, "symbol", "")
                if sym == symbol:
                    return float(getattr(pos, "qty", 0))
        except Exception:
            pass
        return 0.0

    # ── Entry signals ─────────────────────────────────────

    def _process_entry_signals(
        self,
        signals: list[SymbolSignal],
        data: dict[str, pd.DataFrame],
        size_multiplier: float = 1.0,
        risk_off: bool = False,
    ) -> list[TradeAction]:
        actions    = []
        actionable = [
            s for s in signals
            if s.signal != Signal.HOLD and s.confidence >= self.min_confidence
        ]
        actionable.sort(key=lambda s: s.confidence, reverse=True)

        logger.info(
            f"[auto] {len(actionable)} signaler over {self.min_confidence}% confidence"
        )

        for sig in actionable:
            if self._in_cooldown(sig.symbol):
                continue

            alpha = self._get_alpha_score(sig.symbol, data.get(sig.symbol))
            if alpha is not None and alpha < self.min_alpha_score:
                continue

            # Smart money confidence adjustment (-15 to +15)
            try:
                if not hasattr(self, "_insider_tracker"):
                    from src.data.insider_tracking import InsiderTracker
                    self._insider_tracker = InsiderTracker()
                _conf_adj = self._insider_tracker.get_confidence_adjustment(sig.symbol)
                if _conf_adj != 0:
                    sig.confidence = max(0, min(100, sig.confidence + _conf_adj))
                    logger.debug(
                        f"[smart-money] {sig.symbol}: confidence {_conf_adj:+d} "
                        f"→ {sig.confidence:.0f}%"
                    )
            except Exception:
                pass

            # Options flow confidence adjustment (-10 to +10)
            try:
                if not hasattr(self, "_options_tracker"):
                    from src.data.options_flow import OptionsFlowTracker
                    self._options_tracker = OptionsFlowTracker()
                _opt_adj = self._options_tracker.get_confidence_adjustment(sig.symbol)
                if _opt_adj != 0:
                    sig.confidence = max(0, min(100, sig.confidence + _opt_adj))
                    logger.debug(
                        f"[options-flow] {sig.symbol}: confidence {_opt_adj:+d} "
                        f"→ {sig.confidence:.0f}%"
                    )
            except Exception:
                pass

            # Alt data confidence adjustment (-10 to +10)
            try:
                if not hasattr(self, "_alt_data_tracker"):
                    from src.data.alternative_data import AlternativeDataTracker
                    self._alt_data_tracker = AlternativeDataTracker()
                _alt_adj = self._alt_data_tracker.get_confidence_adjustment(sig.symbol)
                if _alt_adj != 0:
                    sig.confidence = max(0, min(100, sig.confidence + _alt_adj))
                    logger.debug(
                        f"[alt-data] {sig.symbol}: confidence {_alt_adj:+d} "
                        f"→ {sig.confidence:.0f}%"
                    )
            except Exception:
                pass

            # News sentiment confidence adjustment (-10 to +10)
            try:
                if not hasattr(self, "_news_sentiment_score"):
                    self._news_sentiment_score = {}
                    self._news_sentiment_ts = 0
                import time as _time
                # Reload from DB at most once per 5 minutes
                if _time.time() - self._news_sentiment_ts > 300:
                    import sqlite3
                    from pathlib import Path
                    _ns_db = Path("data_cache/news_sentiment.db")
                    if _ns_db.exists():
                        with sqlite3.connect(str(_ns_db)) as _ns_conn:
                            _ns_rows = _ns_conn.execute(
                                "SELECT symbol, sentiment_avg FROM daily_sentiment "
                                "WHERE date = (SELECT MAX(date) FROM daily_sentiment)"
                            ).fetchall()
                        self._news_sentiment_score = {r[0]: r[1] for r in _ns_rows}
                    self._news_sentiment_ts = _time.time()
                _sent = self._news_sentiment_score.get(sig.symbol)
                if _sent is not None:
                    # Map sentiment_avg (roughly -1 to +1) to confidence adjustment (-10 to +10)
                    _sent_adj = int(max(-10, min(10, _sent * 10)))
                    if _sent_adj != 0:
                        sig.confidence = max(0, min(100, sig.confidence + _sent_adj))
                        logger.debug(
                            f"[news-sentiment] {sig.symbol}: sentiment={_sent:.2f} "
                            f"confidence {_sent_adj:+d} → {sig.confidence:.0f}%"
                        )
            except Exception:
                pass

            side  = "BUY" if sig.signal == Signal.BUY else "SELL"
            price = self._get_current_price(sig.symbol, data)
            if price <= 0:
                continue

            # Apply handoff multiplier to position size
            position_usd = self._calculate_position_size(
                sig.confidence, price, size_multiplier
            )
            qty = position_usd / price
            if qty < 0.01:
                continue

            is_fractional = any(sig.symbol.endswith(s) for s in ("-USD", "=F", "=X"))
            if not is_fractional:
                qty          = max(1.0, round(qty))
                position_usd = qty * price

            risk_approved = True
            risk_message  = "No risk manager"
            if self._risk_manager is not None:
                try:
                    decision      = self._risk_manager.check_order(
                        symbol=sig.symbol,
                        side="long" if side == "BUY" else "short",
                        requested_usd=position_usd,
                        price=price,
                    )
                    risk_approved = decision.approved
                    risk_message  = decision.message
                    if decision.approved and decision.adjusted_qty > 0:
                        qty = decision.adjusted_qty
                except Exception as e:
                    risk_message  = f"Risk check error: {e}"
                    risk_approved = False

            action = TradeAction(
                symbol=sig.symbol,
                side=side,
                qty=qty,
                reason=sig.reason,
                signal_confidence=sig.confidence,
                alpha_score=alpha,
                risk_approved=risk_approved,
                risk_message=risk_message,
            )
            actions.append(action)

        return actions

    def _get_alpha_score(self, symbol: str, df: pd.DataFrame | None) -> float | None:
        if self._alpha_engine is None:
            return None
        try:
            score = self._alpha_engine.calculate(symbol, df)
            return score.total
        except Exception:
            return None

    def _get_current_price(self, symbol: str, data: dict[str, pd.DataFrame]) -> float:
        df = data.get(symbol)
        if df is not None and len(df) > 0:
            return float(df["Close"].iloc[-1])
        return 0.0

    def _calculate_position_size(
        self, confidence: float, price: float, size_multiplier: float = 1.0
    ) -> float:
        try:
            account = self.router.get_account()
            equity  = account.equity
        except Exception:
            equity = 100_000

        # Floor på 0.85 sikrer at selv lave-confidence signaler udnytter mindst 85% af
        # den allokerede plads — målet er at 90-100% af kapitalen er ude og arbejde.
        confidence_factor = max(min(confidence / 100.0, 1.0), 0.85)
        max_usd           = equity * self.position_size_pct * size_multiplier
        position_usd      = max_usd * confidence_factor

        # Apply absolute DKK cap per symbol (convert DKK→USD using ~6.90 rate)
        if self.max_dkk_per_symbol and self.max_dkk_per_symbol > 0:
            usd_dkk = 6.90
            try:
                import yfinance as yf
                fx = yf.Ticker("DKK=X")
                rate = getattr(fx.fast_info, "last_price", None)
                if rate and rate > 0:
                    usd_dkk = rate
            except Exception:
                pass
            cap_usd = self.max_dkk_per_symbol / usd_dkk
            position_usd = min(position_usd, cap_usd)

        return max(position_usd, 0)

    # ── Execution ─────────────────────────────────────────

    def _execute_action(self, action: TradeAction) -> None:
        if not action.risk_approved:
            logger.info(f"[auto] {action.side} {action.symbol}: AFVIST — {action.risk_message}")
            return

        # Guard against duplicate concurrent orders for the same symbol.
        with self._claim_symbol(action.symbol) as claimed:
            if not claimed:
                action.error = "duplicate_in_flight"
                action.executed = False
                logger.warning(
                    f"[auto] Skip {action.side} {action.symbol}: order already in flight"
                )
                return
            self._execute_action_locked(action)

    def _execute_action_locked(self, action: TradeAction) -> None:
        try:
            if action.side == "BUY":
                order = self.router.buy(symbol=action.symbol, qty=action.qty)
            else:
                has_long = False
                try:
                    positions = self.router.get_positions()
                    for pos in positions:
                        sym  = getattr(pos, "symbol", "")
                        side = getattr(pos, "side", "long")
                        if sym == action.symbol and side == "long":
                            has_long = True
                            break
                except Exception:
                    pass

                if has_long:
                    order = self.router.sell(symbol=action.symbol, qty=action.qty)
                else:
                    # Kun åbn short hvis det IKKE er et exit-signal
                    if action.reason.startswith("EXIT"):
                        logger.info(f"[auto] Skip short for exit: {action.symbol}")
                        return
                    order        = self.router.sell(symbol=action.symbol, qty=action.qty, short=True)
                    action.reason = f"SHORT: {action.reason}"

            action.order   = order
            action.executed = True
            self._total_trades                 += 1
            self._last_trade[action.symbol]     = _now_cet()
            # Record exit timestamp so the cooldown in _check_exits works.
            if action.reason.startswith("EXIT"):
                self._last_exit[action.symbol] = _now_cet()

            # Sync to risk manager portfolio (kun hvis RM har sin EGEN portfolio)
            # Hvis broker og RM deler portfolio-instans, er den allerede opdateret
            if self._risk_manager and hasattr(self._risk_manager, "portfolio"):
                rm_port = self._risk_manager.portfolio
                broker_port = getattr(self, "_portfolio", None)
                if rm_port is not broker_port:
                    try:
                        price = getattr(order, "filled_avg_price", 0) or getattr(order, "price", 0) or 0
                        if action.side == "BUY":
                            rm_port.open_position(
                                symbol=action.symbol,
                                qty=action.qty,
                                price=price,
                                side="long",
                            )
                        else:
                            # SELL = enten luk eksisterende long ELLER åbn ny short
                            if action.symbol in rm_port.positions:
                                rm_port.close_position(
                                    symbol=action.symbol,
                                    price=price,
                                    reason=action.reason or "signal",
                                )
                            else:
                                rm_port.open_position(
                                    symbol=action.symbol,
                                    qty=action.qty,
                                    price=price,
                                    side="short",
                                )
                    except Exception as e:
                        logger.error(f"[auto] RM portfolio sync FEJL — positioner IKKE tracket: {e}", exc_info=True)

            logger.info(
                f"[auto] ✓ {action.side} {action.qty:.1f} {action.symbol} "
                f"→ ordre {order.order_id} ({action.reason})"
            )

        except Exception as e:
            action.error    = str(e)
            action.executed = False
            logger.error(f"[auto] ✗ {action.side} {action.symbol} fejlede: {e}")

    # ── Cooldown ──────────────────────────────────────────

    def _in_cooldown(self, symbol: str) -> bool:
        last = self._last_trade.get(symbol)
        if last is None:
            return False
        elapsed = (_now_cet() - last).total_seconds() / 60
        # Cleanup: remove entries older than 24h to prevent unbounded growth
        if elapsed > 1440:
            del self._last_trade[symbol]
            return False
        return elapsed < self.cooldown_minutes

    def _cleanup_last_trade(self) -> None:
        """Remove stale entries from _last_trade dict (>24h old)."""
        now = _now_cet()
        stale = [s for s, t in self._last_trade.items()
                 if (now - t).total_seconds() > 86400]
        for s in stale:
            del self._last_trade[s]

    def _prune_databases(self) -> None:
        """Prune old scan/trade/signal logs to prevent unbounded DB growth."""
        try:
            with self._db_connect() as conn:
                # Keep last 7 days of scans, last 30 days of trades
                conn.execute(
                    "DELETE FROM scans WHERE timestamp < datetime('now', '-7 days')"
                )
                conn.execute(
                    "DELETE FROM trades WHERE timestamp < datetime('now', '-30 days')"
                )
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            logger.debug("[auto] Database pruned (scans 7d, trades 30d)")
        except Exception as e:
            logger.debug(f"[auto] DB prune error: {e}")

    # ── Feedback Loop: Adaptive Thresholds ───────────────

    def _apply_feedback_adjustments(self) -> None:
        """
        Adjust min_confidence and position_size_pct based on
        ContinuousLearner performance analysis.

        Rules:
          - If recent win rate < 40%: raise min_confidence by 10, reduce position size by 30%
          - If recent win rate > 60%: lower min_confidence by 5 (floor at base)
          - If concept drift detected: raise min_confidence by 15, halve position size
          - Recalculate ensemble weights every 20 scans
        """
        if self._learner is None:
            return

        try:
            summary = self._learner.get_learning_summary()

            # Check drift status
            drift = summary.get("drift_status", "OK")
            if drift == "CRITICAL":
                self.min_confidence = min(self._base_min_confidence + 15, 80.0)
                self.position_size_pct = self._base_position_size_pct * 0.5
                logger.warning(
                    f"[feedback] DRIFT CRITICAL: confidence={self.min_confidence:.0f}, "
                    f"position_size={self.position_size_pct:.1%}"
                )
                return
            elif drift == "WARNING":
                self.min_confidence = min(self._base_min_confidence + 8, 70.0)
                self.position_size_pct = self._base_position_size_pct * 0.75
                logger.info(
                    f"[feedback] DRIFT WARNING: confidence={self.min_confidence:.0f}, "
                    f"position_size={self.position_size_pct:.1%}"
                )
                return

            # Analyze recent trade performance from the learner's analytics
            analytics = self._learner.get_trade_analytics()
            total_analyzed = analytics.get("total_trades_analyzed", 0)

            if total_analyzed < 5:
                # Not enough data yet, keep defaults
                return

            win_rate = analytics.get("win_rate", 0.5)
            avg_pnl = analytics.get("avg_pnl_pct", 0.0)

            # Adjust based on win rate
            if win_rate < 0.35:
                # Poor performance: be more selective
                self.min_confidence = min(self._base_min_confidence + 10, 75.0)
                self.position_size_pct = self._base_position_size_pct * 0.7
                logger.info(
                    f"[feedback] Low win rate ({win_rate:.0%}): "
                    f"tightened confidence={self.min_confidence:.0f}, "
                    f"size={self.position_size_pct:.1%}"
                )
            elif win_rate < 0.45:
                # Below average: slight tightening
                self.min_confidence = min(self._base_min_confidence + 5, 65.0)
                self.position_size_pct = self._base_position_size_pct * 0.85
            elif win_rate > 0.60 and avg_pnl > 0:
                # Good performance: can relax slightly
                self.min_confidence = max(self._base_min_confidence - 5, 25.0)
                self.position_size_pct = min(
                    self._base_position_size_pct * 1.1,
                    0.15,  # hard cap: never more than 15% per position
                )
                logger.info(
                    f"[feedback] High win rate ({win_rate:.0%}): "
                    f"relaxed confidence={self.min_confidence:.0f}, "
                    f"size={self.position_size_pct:.1%}"
                )
            else:
                # Normal performance: use base values
                self.min_confidence = self._base_min_confidence
                self.position_size_pct = self._base_position_size_pct

            # Recalculate ensemble weights
            new_weights = self._learner.recalculate_ensemble_weights()
            if new_weights:
                logger.debug(f"[feedback] Ensemble weights updated: {new_weights}")

            # ── Advanced feedback: auto-apply report recommendations ──
            if self._advanced_feedback_enabled:
                self._apply_advanced_feedback(analytics, summary)

        except Exception as e:
            logger.debug(f"[feedback] Adjustment error: {e}")

    def _apply_advanced_feedback(self, analytics: dict, learner_summary: dict) -> None:
        """
        Auto-apply the 6 recommendation types from the performance report
        that are not covered by the basic feedback loop:
          1. Per-exchange stop-loss tightening for underperforming regions
          2. Drawdown-based exposure reduction
          3. Sharpe-based stop-loss tightening
          4. Benchmark-relative exposure reduction
          5. Drawdown-based position closing (critical only — flags, doesn't force-sell)
          6. Session loss — tighten stops on losing positions
        """
        import json

        try:
            # ── Get portfolio state ──
            positions = []
            drawdown = 0.0
            sharpe = 0.0
            try:
                positions = self.router.get_positions()
                if self._risk_manager and hasattr(self._risk_manager, "portfolio"):
                    portfolio = self._risk_manager.portfolio
                    drawdown = getattr(portfolio, "current_drawdown_pct", 0.0)
                    sharpe = getattr(portfolio, "sharpe_ratio", 0.0) or 0.0
            except Exception:
                pass

            config_dir = Path(__file__).resolve().parent.parent.parent / "config"

            # ── 1. Per-exchange stop-loss auto-tightening ──
            # If positions in a region underperform their benchmark by >2%,
            # tighten that exchange's stop-loss by 1% (min 1.5%)
            if positions:
                exchange_pnls: dict[str, list[float]] = {}
                for p in positions:
                    sym = getattr(p, "symbol", "")
                    pnl_pct = getattr(p, "unrealized_pnl_pct", 0) or 0
                    for suffix, exchange_key in self._SUFFIX_EXCHANGE.items():
                        if sym.endswith(suffix):
                            exchange_pnls.setdefault(exchange_key, []).append(pnl_pct)
                            break
                    else:
                        if sym.endswith("-USD"):
                            exchange_pnls.setdefault("crypto", []).append(pnl_pct)
                        else:
                            exchange_pnls.setdefault("us_stocks", []).append(pnl_pct)

                sl_changed = False
                for exch, pnls in exchange_pnls.items():
                    avg_pnl = sum(pnls) / len(pnls) if pnls else 0
                    if avg_pnl < -0.02:  # underperforming by >2%
                        current_sl = self._exchange_stop_loss.get(exch, 5.0)
                        new_sl = max(current_sl - 1.0, 1.5)
                        if new_sl < current_sl:
                            self._exchange_stop_loss[exch] = new_sl
                            sl_changed = True
                            logger.info(
                                f"[adv-feedback] {exch} avg P&L {avg_pnl:.1%} — "
                                f"tightened stop-loss {current_sl:.1f}% → {new_sl:.1f}%"
                            )
                    elif avg_pnl > 0.03:  # performing well, relax back toward default
                        current_sl = self._exchange_stop_loss.get(exch)
                        if current_sl is not None:
                            default_sl = 5.0
                            new_sl = min(current_sl + 0.5, default_sl)
                            if new_sl > current_sl:
                                self._exchange_stop_loss[exch] = new_sl
                                sl_changed = True
                                logger.info(
                                    f"[adv-feedback] {exch} avg P&L {avg_pnl:+.1%} — "
                                    f"relaxed stop-loss {current_sl:.1f}% → {new_sl:.1f}%"
                                )

                if sl_changed:
                    try:
                        sl_path = config_dir / "exchange_stop_loss.json"
                        sl_path.write_text(json.dumps(self._exchange_stop_loss, indent=2))
                    except Exception:
                        pass

            # ── 2. Drawdown-based exposure reduction ──
            # >10% critical: halve max exposure. >5% warning: reduce by 25%
            if self._risk_manager and drawdown > 0.05:
                try:
                    rs_path = config_dir / "risk_sizing.json"
                    rs = json.loads(rs_path.read_text()) if rs_path.exists() else {}
                    base_exposure = rs.get("_base_max_exposure_pct", rs.get("max_exposure_pct", 95.0))

                    if drawdown > 0.10:
                        new_exposure = max(base_exposure * 0.5, 20.0)
                        logger.warning(
                            f"[adv-feedback] CRITICAL drawdown {drawdown:.1%} — "
                            f"exposure reduced to {new_exposure:.0f}%"
                        )
                    else:
                        new_exposure = max(base_exposure * 0.75, 30.0)
                        logger.info(
                            f"[adv-feedback] Elevated drawdown {drawdown:.1%} — "
                            f"exposure reduced to {new_exposure:.0f}%"
                        )

                    rs["max_exposure_pct"] = round(new_exposure, 1)
                    if "_base_max_exposure_pct" not in rs:
                        rs["_base_max_exposure_pct"] = base_exposure
                    rs_path.write_text(json.dumps(rs, indent=2))

                    # Apply to running DynamicRiskManager
                    drm = getattr(self, "_dynamic_risk_manager", None)
                    if drm and hasattr(drm, "_current_params"):
                        drm._current_params["max_exposure_pct"] = new_exposure / 100.0
                except Exception as e:
                    logger.debug(f"[adv-feedback] Exposure adjustment error: {e}")

            elif self._risk_manager and drawdown < 0.02:
                # Drawdown recovered — restore base exposure
                try:
                    rs_path = config_dir / "risk_sizing.json"
                    if rs_path.exists():
                        rs = json.loads(rs_path.read_text())
                        base = rs.pop("_base_max_exposure_pct", None)
                        if base and rs.get("max_exposure_pct", 95.0) < base:
                            rs["max_exposure_pct"] = base
                            rs_path.write_text(json.dumps(rs, indent=2))
                            logger.info(
                                f"[adv-feedback] Drawdown recovered ({drawdown:.1%}) — "
                                f"exposure restored to {base:.0f}%"
                            )
                except Exception:
                    pass

            # ── 3. Sharpe-based stop-loss tightening ──
            # Sharpe < 0.5: tighten global stop-loss by 1%. Sharpe > 1.5: relax by 0.5%
            if self._risk_manager and sharpe != 0:
                try:
                    gsl_path = config_dir / "global_stop_loss.json"
                    gsl = json.loads(gsl_path.read_text()) if gsl_path.exists() else {}
                    current_sl = gsl.get("stop_loss_pct", 5.0)
                    base_sl = gsl.get("_base_stop_loss_pct", current_sl)

                    if sharpe < 0.5:
                        new_sl = max(current_sl - 1.0, 2.0)
                        if new_sl < current_sl:
                            gsl["stop_loss_pct"] = new_sl
                            if "_base_stop_loss_pct" not in gsl:
                                gsl["_base_stop_loss_pct"] = current_sl
                            gsl_path.write_text(json.dumps(gsl, indent=2))
                            self._risk_manager.stop_loss_pct = new_sl / 100.0
                            logger.info(
                                f"[adv-feedback] Low Sharpe {sharpe:.2f} — "
                                f"tightened global SL {current_sl:.1f}% → {new_sl:.1f}%"
                            )
                    elif sharpe > 1.5 and current_sl < base_sl:
                        new_sl = min(current_sl + 0.5, base_sl)
                        gsl["stop_loss_pct"] = new_sl
                        gsl_path.write_text(json.dumps(gsl, indent=2))
                        self._risk_manager.stop_loss_pct = new_sl / 100.0
                        logger.info(
                            f"[adv-feedback] Strong Sharpe {sharpe:.2f} — "
                            f"relaxed global SL to {new_sl:.1f}%"
                        )
                except Exception as e:
                    logger.debug(f"[adv-feedback] Sharpe SL adjustment error: {e}")

            # ── 4. Benchmark underperformance — reduce new position sizing ──
            total_return = analytics.get("total_return_pct", 0) or 0
            try:
                import yfinance as yf
                spy = yf.Ticker("SPY")
                hist = spy.history(period="30d")
                if len(hist) >= 2:
                    spy_30d = (hist["Close"].iloc[-1] / hist["Close"].iloc[0]) - 1
                    gap = spy_30d - (total_return / 100.0 if abs(total_return) > 1 else total_return)
                    if gap > 0.02:
                        # Underperforming SPY by >2%: reduce position sizing by 20%
                        self.position_size_pct = max(
                            self._base_position_size_pct * 0.8, 0.02
                        )
                        logger.info(
                            f"[adv-feedback] Trailing SPY by {gap:.1%} — "
                            f"position size reduced to {self.position_size_pct:.1%}"
                        )
            except Exception:
                pass

            # ── 5 & 6. Drawdown critical alert + session loss logging ──
            # These are informational — logged for the next report cycle
            if drawdown > 0.10:
                logger.warning(
                    f"[adv-feedback] CRITICAL: Drawdown {drawdown:.1%} approaching risk limit. "
                    f"Exposure and position sizes already reduced. "
                    f"Manual review of worst performers recommended."
                )

            logger.debug("[adv-feedback] Advanced feedback cycle complete")
        except Exception as e:
            logger.warning(f"[adv-feedback] Advanced feedback error: {e}")

    # ── SQLite logging ────────────────────────────────────

    def _db_connect(self) -> sqlite3.Connection:
        """Tuned connection to auto_trader_log.db.

        The dashboard reads scans/trades while the trader thread appends
        to them — WAL + NORMAL gives the dashboard consistent reads
        without blocking writers. busy_timeout absorbs the occasional
        dashboard-side long query.

        journal_mode=WAL is persistent (stored in the DB file header) and
        is applied once in _init_db(). synchronous and busy_timeout are
        per-connection settings and stay here.
        """
        conn = sqlite3.connect(self._db_path, timeout=30.0)
        conn.execute("PRAGMA synchronous=NORMAL").fetchone()
        conn.execute("PRAGMA busy_timeout=30000").fetchone()
        return conn

    def _init_db(self) -> None:
        # One-time WAL activation. PRAGMA journal_mode=WAL is persistent
        # in the file header; running it on every _db_connect() caused
        # lock contention between simultaneous connections.
        with sqlite3.connect(self._db_path, timeout=30.0) as _setup:
            _setup.execute("PRAGMA journal_mode=WAL").fetchone()
        with self._db_connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS scans (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbols_scanned INTEGER,
                    signals_generated INTEGER,
                    trades_executed INTEGER,
                    trades_rejected INTEGER,
                    exit_signals INTEGER,
                    open_markets TEXT,
                    duration_sec REAL,
                    summary TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    scan_id INTEGER,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    qty REAL,
                    reason TEXT,
                    confidence REAL,
                    alpha_score REAL,
                    risk_approved INTEGER,
                    risk_message TEXT,
                    order_id TEXT,
                    executed INTEGER,
                    error TEXT
                )
            """)
            # Dashboard queries sort by timestamp DESC and filter by
            # symbol; these indexes keep the scan-history page fast
            # even at millions of rows.
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_scans_ts ON scans(timestamp)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_trades_ts ON trades(timestamp)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_trades_symbol_ts "
                "ON trades(symbol, timestamp)"
            )

    def _log_scan(self, result: ScanResult) -> None:
        try:
            with self._db_connect() as conn:
                cursor = conn.execute(
                    """INSERT INTO scans
                       (timestamp, symbols_scanned, signals_generated,
                        trades_executed, trades_rejected, exit_signals,
                        open_markets, duration_sec, summary)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        result.timestamp.isoformat(),
                        result.symbols_scanned,
                        result.signals_generated,
                        result.trades_executed,
                        result.trades_rejected,
                        result.exit_signals,
                        ", ".join(result.open_markets),
                        result.duration_sec,
                        result.summary(),
                    ),
                )
                scan_id = cursor.lastrowid
                for action in result.actions:
                    conn.execute(
                        """INSERT INTO trades
                           (timestamp, scan_id, symbol, side, qty, reason,
                            confidence, alpha_score, risk_approved,
                            risk_message, order_id, executed, error)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            result.timestamp.isoformat(),
                            scan_id,
                            action.symbol,
                            action.side,
                            action.qty,
                            action.reason,
                            action.signal_confidence,
                            action.alpha_score,
                            1 if action.risk_approved else 0,
                            action.risk_message,
                            action.order.order_id if action.order else None,
                            1 if action.executed else 0,
                            action.error,
                        ),
                    )
        except Exception as e:
            logger.error(f"[auto] DB log fejl: {e}")

    # ── Status ────────────────────────────────────────────

    def status(self) -> dict:
        open_markets = []
        if self._calendar is not None:
            open_markets = self._calendar.get_open_markets()
        return {
            "paper_mode":               self.paper,
            "watchlist":                len(self.watchlist),
            "total_scans":              self._total_scans,
            "total_trades":             self._total_trades,
            "open_markets":             open_markets,
            "min_confidence":           self.min_confidence,
            "max_new_positions_per_scan": self.max_new_positions,
            "position_size_pct":        f"{self.position_size_pct:.1%}",
            "cooldown_minutes":         self.cooldown_minutes,
            "risk_manager":             self._risk_manager is not None,
            "alpha_engine":             self._alpha_engine is not None,
            "market_calendar":          self._calendar is not None,
            "market_handoff":           self._handoff is not None,
            "strategies":               len(self._engine._strategies) if hasattr(self._engine, "_strategies") else 0,
        }

    def get_trade_history(self, days: int = 7) -> list[dict]:
        try:
            since = (datetime.now() - timedelta(days=days)).isoformat()
            with self._db_connect() as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT * FROM trades WHERE timestamp > ? ORDER BY timestamp DESC",
                    (since,),
                ).fetchall()
                return [dict(r) for r in rows]
        except Exception:
            return []

    def get_scan_history(self, limit: int = 20) -> list[dict]:
        try:
            with self._db_connect() as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT * FROM scans ORDER BY timestamp DESC LIMIT ?",
                    (limit,),
                ).fetchall()
                return [dict(r) for r in rows]
        except Exception:
            return []
