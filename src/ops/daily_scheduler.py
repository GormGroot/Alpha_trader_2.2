"""
Daily Scheduler — automatisk daglig rutine for Alpha Trader.

Tidsplan (CET) — alle markeder:
  00:30  Pre-market: New Zealand, Australia, Japan, Hong Kong
  01:00  New Zealand, Australia, Tokyo åbner
  02:00  Hong Kong åbner
  04:15  Pre-market: Mumbai
  04:45  Mumbai åbner
  07:30  Morgen-check (broker connections, portfolio snapshot)
  08:00  Pre-market: EU, Nordic, London
  09:00  EU, Nordic, London åbner
  10:00  Pre-market: US
  11:15  Mumbai lukker
  15:30  US Regular åbner
  17:30  EU, Nordic, London lukker
  22:00  US Regular lukker → US Post-market starter
  22:00  New Zealand pre-market starter
  23:00  Vedligeholdelse (backup, logs)

Features:
  - Fuld 24/7 dækning alle ugedage
  - Pre/post market support
  - Multi-timezone market calendar integration
  - Weekend/helligdag-detection
  - Graceful shutdown
  - Event callbacks
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from enum import Enum
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

from loguru import logger

# ── Timezones ──────────────────────────────────────────────
TZ_CET = ZoneInfo("Europe/Copenhagen")

# ── Heartbeat ──────────────────────────────────────────────
# The scheduler writes the current epoch seconds to this file on every
# loop iteration (~30s cadence). A Docker HEALTHCHECK, systemd watchdog,
# or external monitor reads it back and alerts if the file is missing
# or stale (>3 min by default). The path is env-overridable so tests
# and containers can point at their own tmpfs location.
HEARTBEAT_FILE = Path(os.environ.get("ALPHA_TRADER_HEARTBEAT", "/tmp/alpha_trader_heartbeat"))
HEARTBEAT_STALE_SECONDS = 180  # 3 min — covers one missed 30s tick + margin


def _write_heartbeat(now: datetime | None = None) -> None:
    """Persist a unix-epoch timestamp so an external watchdog can verify
    the scheduler loop is alive. Best-effort — filesystem failures must
    never crash the loop."""
    try:
        ts = (now or _now_cet()).timestamp()
        HEARTBEAT_FILE.parent.mkdir(parents=True, exist_ok=True)
        # Atomic-ish write via replace — avoids a torn read from the watchdog.
        tmp = HEARTBEAT_FILE.with_suffix(HEARTBEAT_FILE.suffix + ".tmp")
        tmp.write_text(f"{ts:.3f}\n")
        tmp.replace(HEARTBEAT_FILE)
    except Exception as exc:  # pragma: no cover — defensive
        logger.debug(f"[scheduler] heartbeat write failed: {exc}")


def is_scheduler_alive(
    max_age_seconds: int = HEARTBEAT_STALE_SECONDS,
    path: Path | str | None = None,
) -> bool:
    """True if the heartbeat file exists and is younger than max_age.
    Used by Docker HEALTHCHECK and the dashboard watchdog widget."""
    target = Path(path) if path is not None else HEARTBEAT_FILE
    try:
        if not target.exists():
            return False
        raw = target.read_text().strip()
        last_ts = float(raw)
        age = time.time() - last_ts
        return age <= max_age_seconds
    except Exception:
        return False


def _now_cet() -> datetime:
    """Get CET time from web-synced time service, fallback to local clock."""
    try:
        from src.ops.time_service import now_cet
        return now_cet()
    except Exception:
        return datetime.now(TZ_CET)
TZ_ET  = ZoneInfo("America/New_York")
TZ_UTC = ZoneInfo("UTC")


class TaskPriority(Enum):
    CRITICAL = "critical"
    HIGH     = "high"
    NORMAL   = "normal"
    LOW      = "low"


class TaskStatus(Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    COMPLETED = "completed"
    FAILED    = "failed"
    SKIPPED   = "skipped"


@dataclass
class ScheduledTask:
    name:                str
    hour:                int
    minute:              int
    func:                Callable
    priority:            TaskPriority = TaskPriority.NORMAL
    requires_market_day: bool = True
    timeout_seconds:     int = 300
    retry_count:         int = 1
    last_run:            datetime | None = None
    last_status:         TaskStatus = TaskStatus.PENDING
    last_error:          str = ""
    enabled:             bool = True


@dataclass
class TaskResult:
    task_name:        str
    status:           TaskStatus
    started_at:       datetime
    finished_at:      datetime
    duration_seconds: float
    error:            str = ""
    details:          dict = field(default_factory=dict)


# ── Holiday calendar ───────────────────────────────────────

def _dk_holidays(year: int) -> set[date]:
    from datetime import date as d
    holidays = {
        d(year, 1, 1),
        d(year, 6, 5),
        d(year, 12, 24),
        d(year, 12, 25),
        d(year, 12, 26),
        d(year, 12, 31),
    }
    try:
        from dateutil.easter import easter
        e = easter(year)
        holidays.update({
            e - timedelta(days=3),
            e - timedelta(days=2),
            e,
            e + timedelta(days=1),
            e + timedelta(days=39),
            e + timedelta(days=49),
            e + timedelta(days=50),
        })
    except ImportError:
        pass
    return holidays


def _us_holidays(year: int) -> set[date]:
    from datetime import date as d
    return {
        d(year, 1, 1),
        d(year, 7, 4),
        d(year, 12, 25),
    }


def is_market_day(d: date | None = None) -> bool:
    if d is None:
        d = _now_cet().date()
    if d.weekday() >= 5:
        return False
    year = d.year
    if d in _dk_holidays(year) or d in _us_holidays(year):
        return False
    return True


# ── Task functions ─────────────────────────────────────────

def _asia_pacific_open() -> dict:
    """00:30 CET — Pre-market Asia/Pacific opens."""
    logger.info("[scheduler] 🌏 Asia/Pacific pre-market starting (NZ, AU, JP, HK)")
    try:
        from src.ops.market_calendar import get_calendar
        cal = get_calendar()
        symbols = cal.get_symbols_to_scan()
        logger.info(f"[scheduler] Asia/Pacific: {len(symbols)} symbols queued for scan")
        return {"symbols": len(symbols), "markets": ["new_zealand", "australia", "japan", "hong_kong"]}
    except Exception as e:
        logger.warning(f"[scheduler] Asia/Pacific open failed: {e}")
        return {}


def _india_pre_market() -> dict:
    """04:15 CET — India pre-market."""
    logger.info("[scheduler] 🇮🇳 India pre-market starting (NSE)")
    try:
        from src.ops.market_calendar import MARKET_SYMBOLS
        symbols = MARKET_SYMBOLS.get("india", [])
        logger.info(f"[scheduler] India pre-market: {len(symbols)} symbols")
        return {"symbols": len(symbols)}
    except Exception as e:
        logger.warning(f"[scheduler] India pre-market failed: {e}")
        return {}


def _morning_check() -> dict:
    """07:30 CET — Morning check."""
    results = {"brokers": {}, "portfolio_value": 0}
    logger.info("[scheduler] ☀️  Morning check starting")

    try:
        from src.broker.connection_manager import ConnectionManager
        cm = ConnectionManager()
        status = cm.check_all()
        results["brokers"] = status
        logger.info(f"[scheduler] Broker status: {status}")
    except Exception as e:
        logger.warning(f"[scheduler] Broker check failed: {e}")

    try:
        from src.broker.broker_router import BrokerRouter
        from src.broker.aggregated_portfolio import AggregatedPortfolio
        router = BrokerRouter()
        portfolio = AggregatedPortfolio(router)
        summary = portfolio.get_total_value("DKK")
        results["portfolio_value"] = summary.total_value_dkk
        logger.info(f"[scheduler] Portfolio value: {summary.total_value_dkk:,.0f} DKK")
    except Exception as e:
        logger.warning(f"[scheduler] Portfolio fetch failed: {e}")

    # Print market status overview
    try:
        from src.ops.market_calendar import get_calendar
        get_calendar().print_status()
    except Exception:
        pass

    return results


def _eu_pre_market() -> dict:
    """08:00 CET — EU/Nordic/London pre-market."""
    logger.info("[scheduler] 🇪🇺 EU/Nordic/London pre-market starting")
    try:
        from src.ops.market_calendar import MARKET_SYMBOLS
        count = len(MARKET_SYMBOLS.get("eu_nordic", [])) + len(MARKET_SYMBOLS.get("london", []))
        logger.info(f"[scheduler] EU pre-market: {count} symbols")
        return {"symbols": count}
    except Exception as e:
        logger.warning(f"[scheduler] EU pre-market failed: {e}")
        return {}


def _eu_market_open() -> dict:
    """09:00 CET — EU market opens."""
    logger.info("[scheduler] 🇪🇺 EU/Nordic/London market OPEN")
    results = {"signals": [], "strategies_run": 0}
    try:
        from src.strategy.sma_crossover import SMACrossoverStrategy
        from src.strategy.rsi_strategy import RSIStrategy
        strategies = [
            SMACrossoverStrategy(short_window=20, long_window=50),
            RSIStrategy(oversold=30, overbought=70),
        ]
        results["strategies_run"] = len(strategies)
    except Exception as e:
        logger.warning(f"[scheduler] EU open routine failed: {e}")
    return results


def _india_market_close() -> dict:
    """11:15 CET — India market closes."""
    logger.info("[scheduler] 🇮🇳 India (NSE) market CLOSED")
    try:
        from src.ops.market_handoff import get_handoff_engine
        engine = get_handoff_engine()
        from src.ops.market_handoff import SessionResult
        from datetime import date
        result = SessionResult(
            market="india",
            date=date.today(),
            change_pct=0.0,
            volatility=0.0,
            volume_ratio=1.0,
            breadth=0.5,
            sentiment_score=0.0,
            regime="sideways",
        )
        engine.record_session(result)
        logger.info(f"[scheduler] India handoff recorded")
    except Exception as e:
        logger.warning(f"[scheduler] India close handoff failed: {e}")
    return {}


def _us_pre_market() -> dict:
    """10:00 CET — US pre-market opens."""
    logger.info("[scheduler] 🇺🇸 US pre-market starting (10:00 CET)")
    try:
        from src.ops.market_calendar import MARKET_SYMBOLS
        count = len(MARKET_SYMBOLS.get("us_stocks", [])) + len(MARKET_SYMBOLS.get("chicago", []))
        logger.info(f"[scheduler] US pre-market: {count} symbols")
        return {"symbols": count}
    except Exception as e:
        logger.warning(f"[scheduler] US pre-market failed: {e}")
        return {}


def _us_market_open() -> dict:
    """15:30 CET — US regular market opens."""
    logger.info("[scheduler] 🇺🇸 US market OPEN (15:30 CET)")
    results = {"cross_market": {}}
    try:
        from src.broker.aggregated_portfolio import AggregatedPortfolio
        from src.broker.broker_router import BrokerRouter
        router = BrokerRouter()
        portfolio = AggregatedPortfolio(router)
        portfolio.invalidate_cache()
        logger.info("[scheduler] US market open — cache invalidated")

        # Get handoff signal from EU session
        try:
            from src.ops.market_handoff import get_handoff_engine
            signal = get_handoff_engine().get_handoff_signal("us")
            logger.info(
                f"[scheduler] US handoff signal: size_multiplier={signal.position_size_multiplier:.2f}, "
                f"risk_off={signal.risk_off}"
            )
            results["handoff"] = {
                "multiplier": signal.position_size_multiplier,
                "risk_off": signal.risk_off,
                "notes": signal.notes,
            }
        except Exception:
            pass
    except Exception as e:
        logger.warning(f"[scheduler] US open routine failed: {e}")
    return results


def _eu_market_close() -> dict:
    """17:30 CET — EU market closes."""
    logger.info("[scheduler] 🇪🇺 EU/Nordic/London market CLOSED")
    results = {"snapshot": {}, "tax_updated": False}

    try:
        from src.broker.broker_router import BrokerRouter
        from src.broker.aggregated_portfolio import AggregatedPortfolio
        router = BrokerRouter()
        portfolio = AggregatedPortfolio(router)
        summary = portfolio.get_total_value("DKK")
        results["snapshot"]["eu_value"] = summary.total_value_dkk
    except Exception as e:
        logger.warning(f"[scheduler] EU close snapshot failed: {e}")

    try:
        from src.tax.mark_to_market import MarkToMarketEngine
        mtm = MarkToMarketEngine()
        results["tax_updated"] = True
        logger.info("[scheduler] Tax calculation updated at EU close")
    except Exception as e:
        logger.warning(f"[scheduler] Tax update failed: {e}")

    # Record EU session for US handoff
    try:
        from src.ops.market_handoff import get_handoff_engine, SessionResult
        from datetime import date
        result = SessionResult(
            market="eu",
            date=date.today(),
            change_pct=0.0,
            volatility=0.0,
            volume_ratio=1.0,
            breadth=0.5,
            sentiment_score=0.0,
            regime="sideways",
        )
        get_handoff_engine().record_session(result)
    except Exception:
        pass

    return results


def _us_market_close() -> dict:
    """22:00 CET — US regular market closes, post-market starts."""
    logger.info("[scheduler] 🇺🇸 US market CLOSED — post-market starting")
    results = {"daily_pnl": 0, "portfolio_value": 0, "tax_suggestions": []}

    try:
        from src.broker.broker_router import BrokerRouter
        from src.broker.aggregated_portfolio import AggregatedPortfolio
        router = BrokerRouter()
        portfolio = AggregatedPortfolio(router)
        summary = portfolio.get_total_value("DKK")
        results["portfolio_value"] = summary.total_value_dkk
        results["daily_pnl"] = summary.total_unrealized_pnl_dkk
        logger.info(f"[scheduler] End of day portfolio: {summary.total_value_dkk:,.0f} DKK")
    except Exception as e:
        logger.warning(f"[scheduler] US close snapshot failed: {e}")

    try:
        from src.tax.corporate_tax import CorporateTaxCalculator
        calc = CorporateTaxCalculator()
        suggestions = calc.suggest_tax_optimization([])
        results["tax_suggestions"] = [s.description for s in suggestions[:5]]
    except Exception as e:
        logger.warning(f"[scheduler] Tax optimization check failed: {e}")

    # Record US session for NZ/Asia handoff next day
    try:
        from src.ops.market_handoff import get_handoff_engine, SessionResult
        from datetime import date
        result = SessionResult(
            market="us",
            date=date.today(),
            change_pct=0.0,
            volatility=0.0,
            volume_ratio=1.0,
            breadth=0.5,
            sentiment_score=0.0,
            regime="sideways",
        )
        get_handoff_engine().record_session(result)
    except Exception:
        pass

    return results


def _nz_pre_market() -> dict:
    """22:00 CET — New Zealand pre-market (same time as US close)."""
    logger.info("[scheduler] 🇳🇿 New Zealand pre-market starting (22:00 CET)")
    try:
        from src.ops.market_calendar import MARKET_SYMBOLS
        symbols = MARKET_SYMBOLS.get("new_zealand", [])
        logger.info(f"[scheduler] NZ pre-market: {len(symbols)} symbols")
        return {"symbols": len(symbols)}
    except Exception as e:
        logger.warning(f"[scheduler] NZ pre-market failed: {e}")
        return {}


def _maintenance() -> dict:
    """23:00 CET — Maintenance."""
    results = {"backup_ok": False, "disk_ok": False, "logs_archived": False}
    logger.info("[scheduler] 🔧 Maintenance starting")

    try:
        from src.ops.backup import BackupManager
        bm = BackupManager()
        bm.run_daily_backup()
        results["backup_ok"] = True
    except Exception as e:
        logger.warning(f"[scheduler] Backup failed: {e}")

    try:
        import shutil
        usage = shutil.disk_usage("/")
        free_gb = usage.free / (1024 ** 3)
        results["disk_free_gb"] = round(free_gb, 1)
        results["disk_ok"] = free_gb > 5
        if free_gb < 5:
            logger.warning(f"[scheduler] Low disk space: {free_gb:.1f} GB free")

        # Also check trading data drive
        trading_path = "/mnt/trading"
        import os
        if os.path.exists(trading_path):
            usage2 = shutil.disk_usage(trading_path)
            free_gb2 = usage2.free / (1024 ** 3)
            results["trading_disk_free_gb"] = round(free_gb2, 1)
            logger.info(f"[scheduler] Trading disk: {free_gb2:.1f} GB free")
    except Exception as e:
        logger.warning(f"[scheduler] Disk check failed: {e}")

    try:
        import glob, os, gzip
        log_dir = "logs"
        if os.path.isdir(log_dir):
            old_logs = [
                f for f in glob.glob(os.path.join(log_dir, "*.log"))
                if os.path.getmtime(f) < time.time() - 7 * 86400
            ]
            for lf in old_logs:
                with open(lf, "rb") as f_in:
                    with gzip.open(f"{lf}.gz", "wb") as f_out:
                        f_out.write(f_in.read())
                os.remove(lf)
            results["logs_archived"] = True
            logger.info(f"[scheduler] Archived {len(old_logs)} old log files")
    except Exception as e:
        logger.warning(f"[scheduler] Log archival failed: {e}")

    # News sentiment: handled by ContinuousNewsFetcher (every 5 min)
    # Nightly maintenance just triggers a re-aggregation of daily scores
    try:
        from src.data.news_sentiment_downloader import _aggregate_daily_sentiment, _DB_PATH
        import sqlite3
        with sqlite3.connect(str(_DB_PATH), timeout=60) as _ns_conn:
            _aggregate_daily_sentiment(_ns_conn)
        results["news_sentiment_ok"] = True
        logger.info("[scheduler] News sentiment daily aggregation done")
    except Exception as e:
        logger.warning(f"News sentiment aggregation failed: {e}")
        results["news_sentiment_ok"] = False

    # Reset daily handoff data
    try:
        from src.ops.market_handoff import get_handoff_engine
        get_handoff_engine().reset_daily()
    except Exception:
        pass

    # Daily historical data update (download last 24h for all universe symbols)
    # The downloader automatically triggers the NPU/GPU data processor
    # after download to rebuild the processed data block.
    try:
        from src.data.historical_downloader import HistoricalDownloader
        dl = HistoricalDownloader()
        dl_stats = dl.run_daily_update(run_processor=True)
        results["historical_updated"] = dl_stats.get("updated", 0)
        results["historical_bars"] = dl_stats.get("bars", 0)
        # Capture processor results if available
        if "processor" in dl_stats:
            results["data_processor"] = dl_stats["processor"]
        logger.info(
            f"[scheduler] Historical data: {dl_stats['updated']} symbols updated, "
            f"{dl_stats['bars']:,} bars"
        )
    except Exception as e:
        logger.warning(f"[scheduler] Historical data update failed: {e}")
        results["historical_updated"] = 0

    return results


def _weekend_rotation_check() -> dict:
    """Exchange-aligned weekend/holiday rotation — runs every scheduled tick.

    Works for **any** non-trading stretch: regular weekends, national
    holidays, long weekends (e.g. Good Friday → Easter Monday = 4 days).

    **Activate** when ALL non-crypto exchanges have closed AND the next
    calendar day is not a trading day for at least one exchange.  This is
    detected per-market via ``is_last_trading_day_before_break()``.

    **Deactivate** 30 minutes before the first exchange reopens after the
    break, determined by ``get_earliest_reopen()`` which respects per-market
    holiday calendars (e.g. NZ may reopen while US stays closed).
    """
    import json as _json
    from pathlib import Path

    _cfg_path = Path(__file__).resolve().parent.parent.parent / "config" / "weekend_rotation.json"
    try:
        cfg = _json.loads(_cfg_path.read_text()) if _cfg_path.exists() else {}
    except Exception:
        cfg = {}

    if not cfg.get("enabled", False):
        return {"skipped": True, "reason": "disabled"}

    now = _now_cet()
    today = now.date()

    try:
        from src.broker.registry import get_auto_trader
        trader = get_auto_trader()
        if not trader:
            return {"skipped": True, "reason": "no_trader"}

        weekend_active = getattr(trader, "_weekend_mode", False)
    except Exception:
        return {"skipped": True, "reason": "no_trader"}

    from src.ops.market_calendar import (
        get_friday_close_schedule,
        get_earliest_reopen,
        is_last_trading_day_before_break,
        MARKET_SYMBOLS,
    )

    # ── ACTIVATE: last trading day, all exchanges closed ─────────
    if not weekend_active:
        # Check if ANY non-crypto market considers today its last
        # trading day before a break (weekend or holiday).
        non_crypto = [m for m in MARKET_SYMBOLS if m != "crypto"]
        any_closing = any(is_last_trading_day_before_break(m, today) for m in non_crypto)

        if not any_closing:
            return {"skipped": True, "reason": "not_pre_break_day"}

        # Today is the last trading day for at least one market.
        # Wait until all exchanges that ARE open today have closed.
        schedule = get_friday_close_schedule(today)
        if not schedule:
            # No exchanges open today at all (full holiday) — activate now
            pass
        else:
            still_open = [
                m for close_t, markets in schedule
                for m in markets if now.time() < close_t
            ]
            if still_open:
                logger.debug(
                    f"[scheduler] Weekend rotation — waiting for: {still_open}"
                )
                return {"skipped": True, "reason": "exchanges_still_open", "open": still_open}

        # All done — request user approval before activating
        crypto_alloc_pct = cfg.get("crypto_allocation_pct", 60)
        close_stocks = cfg.get("close_stocks", True)
        close_futures = cfg.get("close_futures", True)

        reopen_date, reopen_time, first_market = get_earliest_reopen(today)
        reopen_info = f"{first_market} {reopen_date} {reopen_time.strftime('%H:%M')} CET"

        from src.ops.weekend_approval import (
            get_approval_state, request_approval, is_pending, is_approved, is_rejected,
        )

        approval = get_approval_state()
        status = approval.get("status")

        if status == "approved":
            # User approved — activate weekend mode
            trader.set_weekend_mode(True, crypto_alloc_pct, close_stocks, close_futures)
            logger.info(
                f"[scheduler] Weekend rotation ACTIVATED (user approved) — "
                f"crypto target={crypto_alloc_pct}%, "
                f"first reopen: {reopen_info}"
            )
            return {
                "weekend_mode": True,
                "trigger": "user_approved",
                "reopen": reopen_info,
            }

        elif status == "rejected":
            logger.info("[scheduler] Weekend rotation SKIPPED — user rejected")
            return {"skipped": True, "reason": "user_rejected"}

        elif status == "pending":
            # Still waiting for user response
            return {"skipped": True, "reason": "awaiting_approval"}

        else:
            # No request yet — calculate fees and request approval
            from src.fees.fee_calculator import FeeCalculator, get_exchange_for_symbol

            fee_calc = FeeCalculator(broker="paper")
            positions_to_close = []
            estimated_close_fees = 0.0

            if hasattr(trader, "_portfolio") and hasattr(trader._portfolio, "positions"):
                for sym, pos in trader._portfolio.positions.items():
                    is_crypto = sym.endswith("-USD")
                    is_future = "=F" in sym
                    if is_crypto:
                        continue
                    if is_future and not close_futures:
                        continue
                    if not is_future and not close_stocks:
                        continue
                    price = getattr(pos, "current_price", None) or getattr(pos, "entry_price", 0)
                    qty = getattr(pos, "qty", 0)
                    fee = fee_calc.calculate(sym, "sell", qty, price)
                    estimated_close_fees += fee.total
                    positions_to_close.append({
                        "symbol": sym,
                        "qty": qty,
                        "price": price,
                        "close_fee": round(fee.total, 2),
                        "exchange": get_exchange_for_symbol(sym),
                    })

            # Estimate crypto entry fees (based on allocation)
            portfolio_value = getattr(trader._portfolio, "total_equity", 100000)
            crypto_budget = portfolio_value * (crypto_alloc_pct / 100.0)
            crypto_symbols = [s for s in MARKET_SYMBOLS.get("crypto", []) if s.endswith("-USD")][:4]
            estimated_crypto_entry = 0.0
            per_crypto = crypto_budget / max(1, len(crypto_symbols))
            for sym in crypto_symbols:
                fee = fee_calc.calculate(sym, "buy", 1, per_crypto)
                # Scale: fee was for qty=1 at full budget, recalc properly
                fee = fee_calc.calculate(sym, "buy", per_crypto / 100, 100)  # approximate
                estimated_crypto_entry += fee.total

            # Better estimate: use pct-based fee directly
            crypto_fee_schedule = fee_calc.get_fee_schedule("BTC-USD")
            crypto_comm_pct = crypto_fee_schedule.get("commission_pct", 0.001)
            crypto_spread_pct = crypto_fee_schedule.get("spread_pct", 0.001)
            estimated_crypto_entry = crypto_budget * (crypto_comm_pct + crypto_spread_pct)
            # Same fees for exit on Monday
            estimated_crypto_exit = estimated_crypto_entry

            total_estimated = estimated_close_fees + estimated_crypto_entry + estimated_crypto_exit

            request_approval(
                estimated_fees=round(total_estimated, 2),
                crypto_allocation_pct=crypto_alloc_pct,
                positions_to_close=positions_to_close,
                crypto_symbols=crypto_symbols,
                reopen_info=reopen_info,
            )

            # Send notification
            try:
                from src.notifications.notifier import Notifier
                notifier = Notifier()
                notifier.send(
                    severity="WARNING",
                    title="Weekend Crypto Rollover — godkendelse kræves",
                    message=(
                        f"Alle børser er lukket. Weekend crypto-rotation er klar.\n\n"
                        f"Estimerede ekstra gebyrer: ${total_estimated:,.2f}\n"
                        f"  - Lukning af {len(positions_to_close)} positioner: ${estimated_close_fees:,.2f}\n"
                        f"  - Crypto entry (køb): ${estimated_crypto_entry:,.2f}\n"
                        f"  - Crypto exit (mandag): ${estimated_crypto_exit:,.2f}\n\n"
                        f"Crypto allokering: {crypto_alloc_pct}% af portefølje\n"
                        f"Næste åbning: {reopen_info}\n\n"
                        f"Godkend eller afvis i dashboardet under Rapporter."
                    ),
                    category="weekend_rotation",
                )
            except Exception as e:
                logger.warning(f"[scheduler] Could not send weekend approval notification: {e}")

            logger.info(
                f"[scheduler] Weekend rotation PENDING — estimated fees: ${total_estimated:,.2f}, "
                f"awaiting user approval in dashboard"
            )
            return {
                "skipped": True,
                "reason": "approval_requested",
                "estimated_fees": total_estimated,
                "positions_to_close": len(positions_to_close),
            }

    # ── DEACTIVATE: before first exchange reopens ────────────────
    if weekend_active:
        # Find when the next exchange opens (accounts for per-market holidays)
        # Use yesterday as anchor so we catch today's opens
        yesterday = today - timedelta(days=1)
        reopen_date, reopen_time, first_market = get_earliest_reopen(yesterday)

        # Restore 30 min before first open
        from datetime import datetime as _dt
        reopen_dt = _dt.combine(reopen_date, reopen_time, tzinfo=TZ_CET)
        restore_dt = reopen_dt - timedelta(minutes=30)

        if now >= restore_dt:
            trader.set_weekend_mode(False)
            # Clear approval state for next weekend
            from src.ops.weekend_approval import clear as _clear_approval
            _clear_approval()
            logger.info(
                f"[scheduler] Weekend rotation DEACTIVATED — "
                f"{first_market} opens {reopen_date} at {reopen_time.strftime('%H:%M')} CET"
            )
            return {"weekend_mode": False, "trigger": f"before_{first_market}_open"}
        else:
            remaining = (restore_dt - now).total_seconds() / 60
            return {
                "skipped": True,
                "reason": "waiting_for_restore",
                "restore_at": restore_dt.strftime("%a %H:%M CET"),
                "minutes_remaining": int(remaining),
            }

    return {"skipped": True, "reason": "no_action_needed"}


def _data_processor_retrain() -> dict:
    """23:30 CET — Full model retrain on processed data block (weekly)."""
    logger.info("[scheduler] NPU/GPU data processor — weekly full retrain")
    now = _now_cet()

    # Only do full retrain on Sundays (weekly)
    if now.weekday() != 6:
        logger.info("[scheduler] Skipping full retrain (not Sunday)")
        return {"skipped": True, "reason": "not_sunday"}

    try:
        from src.ops.data_processor import DataProcessor
        dp = DataProcessor()
        result = dp.run(retrain=True)
        logger.info(
            f"[scheduler] Full retrain complete: "
            f"{result.models_trained} models trained, "
            f"{result.predictions_written} predictions "
            f"({result.device}, {result.duration_seconds:.1f}s)"
        )
        return {
            "models_trained": result.models_trained,
            "predictions": result.predictions_written,
            "device": result.device,
            "duration_s": round(result.duration_seconds, 1),
            "model_results": {
                k: {"accuracy": v.get("accuracy", 0), "auc": v.get("auc", 0)}
                for k, v in result.model_results.items()
            },
        }
    except Exception as e:
        logger.error(f"[scheduler] Data processor retrain failed: {e}")
        return {"error": str(e)}


def _circuit_breaker_reset() -> dict:
    """08:45 CET — Auto-reset daglige circuit breakers ved ny handelsdag."""
    logger.info("[scheduler] Circuit breaker daily reset")
    try:
        from src.broker.registry import get_auto_trader
        trader = get_auto_trader()
        if trader is None:
            return {"skipped": True, "reason": "no_auto_trader"}

        rm = getattr(trader, '_risk_manager', None)
        if rm and rm.is_trading_halted:
            rm.resume_trading()
            logger.info("[scheduler] RiskManager circuit breaker RESET — handel genoptaget")

        drm = getattr(trader, '_dynamic_risk_manager', None)
        if drm and hasattr(drm, 'resume_trading') and getattr(drm, 'is_trading_halted', False):
            drm.resume_trading()
            logger.info("[scheduler] DynamicRiskManager circuit breaker RESET")

        return {"reset": True}
    except Exception as e:
        logger.warning(f"[scheduler] Circuit breaker reset failed: {e}")
        return {"error": str(e)}


def _db_maintenance() -> dict:
    """03:00 CET Saturdays — prune + VACUUM hot SQLite DBs.

    signals.db and learning.db grow by ~1 GB/day in production; without
    this hook the DBs blow through disk and the dashboard queries slow
    to a crawl. Runs Saturday only — low-traffic window so the file
    locks held during VACUUM don't interfere with active scanning.
    """
    now = _now_cet()
    if now.weekday() != 5:  # Saturday = 5
        return {"skipped": True, "reason": "not_saturday"}

    result: dict = {}
    # signals.db — prune >90d, then VACUUM
    try:
        from pathlib import Path
        from src.strategy.signal_engine import SignalStore

        signals_path = Path("data_cache/signals.db")
        if signals_path.exists():
            store = SignalStore(signals_path)
            before = store.count()
            store.prune(keep_days=90)
            after = store.count()
            store.vacuum()
            size_mb = signals_path.stat().st_size / 1024 / 1024
            result["signals"] = {
                "rows_pruned": before - after,
                "rows_remaining": after,
                "size_mb": round(size_mb, 1),
            }
            logger.info(
                f"[scheduler] signals.db pruned: {before - after} rows removed, "
                f"{after} remain, {size_mb:.1f} MB"
            )
    except Exception as e:
        logger.warning(f"[scheduler] signals.db maintenance failed: {e}")
        result["signals_error"] = str(e)

    # learning.db — prune + VACUUM
    try:
        import sqlite3
        from pathlib import Path
        from src.learning.continuous_learner import ContinuousLearner

        learning_path = Path("data_cache/learning.db")
        if learning_path.exists():
            learner = ContinuousLearner(db_path=str(learning_path))
            learner._prune_learning_db()
            # VACUUM must run on a fresh connection with no open tx
            conn = sqlite3.connect(str(learning_path), timeout=60.0)
            try:
                conn.execute("VACUUM")
            finally:
                conn.close()
            size_mb = learning_path.stat().st_size / 1024 / 1024
            result["learning"] = {"size_mb": round(size_mb, 1)}
            logger.info(f"[scheduler] learning.db pruned + vacuumed, {size_mb:.1f} MB")
    except Exception as e:
        logger.warning(f"[scheduler] learning.db maintenance failed: {e}")
        result["learning_error"] = str(e)

    return result


# ── Scheduler ──────────────────────────────────────────────

class DailyScheduler:
    """
    Runs daily routines covering ALL global markets 24/7.
    All times are CET (Copenhagen timezone).
    """

    DEFAULT_TASKS = [
        # ── Asia/Pacific ──────────────────────────────────
        ScheduledTask("asia_pacific_pre",  0, 30, _asia_pacific_open,  TaskPriority.HIGH,     True,  120, 1),
        ScheduledTask("india_pre_market",  4, 15, _india_pre_market,   TaskPriority.NORMAL,   True,  60,  1),
        # ── Morning ───────────────────────────────────────
        ScheduledTask("morning_check",     7, 30, _morning_check,      TaskPriority.CRITICAL, True,  120, 2),
        # ── EU ────────────────────────────────────────────
        ScheduledTask("eu_pre_market",     8,  0, _eu_pre_market,      TaskPriority.NORMAL,   True,  60,  1),
        ScheduledTask("circuit_breaker_reset", 8, 45, _circuit_breaker_reset, TaskPriority.CRITICAL, True, 30, 1),
        ScheduledTask("eu_market_open",    9,  0, _eu_market_open,     TaskPriority.HIGH,     True,  180, 1),
        # ── US Pre ────────────────────────────────────────
        ScheduledTask("us_pre_market",    10,  0, _us_pre_market,      TaskPriority.NORMAL,   True,  60,  1),
        # ── India close ───────────────────────────────────
        ScheduledTask("india_close",      11, 15, _india_market_close, TaskPriority.NORMAL,   True,  60,  1),
        # ── US Regular ────────────────────────────────────
        ScheduledTask("us_market_open",   15, 30, _us_market_open,     TaskPriority.HIGH,     True,  180, 1),
        # ── EU close ──────────────────────────────────────
        ScheduledTask("eu_market_close",  17, 30, _eu_market_close,    TaskPriority.HIGH,     True,  300, 2),
        # ── US close + NZ pre ─────────────────────────────
        ScheduledTask("us_market_close",  22,  0, _us_market_close,    TaskPriority.CRITICAL, True,  300, 2),
        ScheduledTask("nz_pre_market",    22,  0, _nz_pre_market,      TaskPriority.NORMAL,   True,  60,  1),
        # ── Weekend Rotation (exchange-aligned) ───────────
        # Checked at key times: after last Friday close, before first Sunday/Monday open
        # The task itself decides whether to activate/deactivate based on exchange hours.
        # 22:30 = 30 min after US close (last Friday exchange to close)
        ScheduledTask("weekend_rotation_fri", 22, 30, _weekend_rotation_check, TaskPriority.HIGH, False, 120, 1),
        # 21:00 = 30 min before NZ pre-market (earliest Sunday opener)
        ScheduledTask("weekend_rotation_sun", 21,  0, _weekend_rotation_check, TaskPriority.HIGH, False, 120, 1),
        # 00:15 = fallback early Monday if Sunday check was missed
        ScheduledTask("weekend_rotation_mon",  0, 15, _weekend_rotation_check, TaskPriority.HIGH, False, 120, 1),
        # ── Maintenance ───────────────────────────────────
        ScheduledTask("maintenance",      23,  0, _maintenance,        TaskPriority.NORMAL,   False, 600, 1),
        # ── NPU/GPU Data Processor (weekly full retrain) ─
        ScheduledTask("data_processor_retrain", 23, 30, _data_processor_retrain, TaskPriority.LOW, False, 1800, 1),
        # ── DB Maintenance (Saturday 03:00 CET) ──────────
        # Prune signals.db (>90d) + learning.db + VACUUM both. Fires daily,
        # no-ops on non-Saturdays. Low-traffic window chosen so VACUUM's
        # exclusive lock doesn't interfere with live scanning.
        ScheduledTask("db_maintenance", 3, 0, _db_maintenance, TaskPriority.LOW, False, 1800, 1),
    ]

    def __init__(self, tasks: list[ScheduledTask] | None = None):
        self._tasks       = tasks or list(self.DEFAULT_TASKS)
        self._running     = False
        self._thread      = None
        self._stop_event  = threading.Event()
        self._results     = deque(maxlen=500)
        self._callbacks   = []
        self._lock        = threading.Lock()

    def start(self) -> None:
        if self._running:
            logger.warning("[scheduler] Already running")
            return
        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="DailyScheduler",
            daemon=True,
        )
        self._thread.start()
        logger.info("[scheduler] Started — 24/7 global market coverage active")

    def stop(self) -> None:
        self._running = False
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10)
        logger.info("[scheduler] Stopped")

    def on_task_complete(self, cb: Callable[[TaskResult], None]) -> None:
        self._callbacks.append(cb)

    def get_schedule(self) -> list[dict]:
        return [
            {
                "name":                t.name,
                "time":                f"{t.hour:02d}:{t.minute:02d} CET",
                "priority":            t.priority.value,
                "enabled":             t.enabled,
                "requires_market_day": t.requires_market_day,
                "last_run":            t.last_run.isoformat() if t.last_run else None,
                "last_status":         t.last_status.value,
            }
            for t in self._tasks
        ]

    def get_results(self, limit: int = 50) -> list[TaskResult]:
        with self._lock:
            return list(self._results)[-limit:]

    def run_task_now(self, task_name: str) -> TaskResult | None:
        for task in self._tasks:
            if task.name == task_name:
                return self._execute_task(task)
        logger.warning(f"[scheduler] Task '{task_name}' not found")
        return None

    def enable_task(self, task_name: str, enabled: bool = True) -> None:
        for task in self._tasks:
            if task.name == task_name:
                task.enabled = enabled
                return

    def _run_loop(self) -> None:
        logger.info("[scheduler] Run loop started")
        # Prime heartbeat so the watchdog doesn't flag us stale during the
        # first 30s wait before the first tick.
        _write_heartbeat()
        while not self._stop_event.is_set():
            try:
                now_cet = _now_cet()
                # Heartbeat BEFORE task execution — if a task hangs, the
                # timestamp reflects when the loop last reached this line.
                _write_heartbeat(now_cet)
                for task in self._tasks:
                    if not task.enabled:
                        continue
                    if task.requires_market_day and not is_market_day(now_cet.date()):
                        continue
                    if now_cet.hour == task.hour and now_cet.minute == task.minute:
                        if task.last_run and task.last_run.date() == now_cet.date():
                            continue
                        logger.info(f"[scheduler] Running task: {task.name}")
                        self._execute_task(task)
            except Exception as e:
                logger.error(f"[scheduler] Loop error: {e}")
            self._stop_event.wait(30)

    def _execute_task(self, task: ScheduledTask) -> TaskResult:
        started = _now_cet()
        error   = ""
        status  = TaskStatus.COMPLETED
        details = {}

        for attempt in range(task.retry_count):
            try:
                result  = task.func()
                details = result if isinstance(result, dict) else {}
                status  = TaskStatus.COMPLETED
                error   = ""
                break
            except Exception as e:
                error = f"Attempt {attempt + 1}: {e}"
                logger.warning(f"[scheduler] Task '{task.name}' attempt {attempt + 1} failed: {e}")
                if attempt < task.retry_count - 1:
                    time.sleep(5)
                else:
                    status = TaskStatus.FAILED

        finished = _now_cet()
        duration = (finished - started).total_seconds()

        task.last_run    = finished
        task.last_status = status
        task.last_error  = error

        result = TaskResult(
            task_name=task.name,
            status=status,
            started_at=started,
            finished_at=finished,
            duration_seconds=duration,
            error=error,
            details=details,
        )

        with self._lock:
            self._results.append(result)

        for cb in self._callbacks:
            try:
                cb(result)
            except Exception as e:
                logger.warning(f"[scheduler] Callback error: {e}")

        log_level = "info" if status == TaskStatus.COMPLETED else "warning"
        getattr(logger, log_level)(
            f"[scheduler] Task '{task.name}' {status.value} in {duration:.1f}s"
        )
        return result
