#!/usr/bin/env python3
"""
Alpha Trading Platform — Main Entry Point.

Starter hele platformen baseret på --mode flag:
  --mode trader    → Multi-broker trading med 4 brokers + skat (Ole's maskine)
  --mode research  → GPU-accelerated research (Gorm's maskine)
  --mode dashboard → Kun dashboard (ingen trading)

Usage:
  python main.py --mode trader
  python main.py --mode trader --paper        # Paper trading only
  python main.py --mode dashboard             # Dashboard only
  python main.py --mode trader --no-scheduler # Skip daglig scheduler
"""

from __future__ import annotations

import argparse
import os
import sys
import signal
import threading
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from loguru import logger

# Load .env (API keys etc.) fra projektets rod
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# ── Paths ──────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

TZ_CET = ZoneInfo("Europe/Copenhagen")


# ── Logging Setup ──────────────────────────────────────────

def setup_logging(level: str = "DEBUG", log_file: str = "logs/trading.log") -> None:
    """Konfigurér loguru med rotation og retention."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger.remove()
    logger.add(sys.stderr, level=level, colorize=True,
               format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | <cyan>{name}</cyan> — <level>{message}</level>")
    logger.add(log_file, level=level, rotation="1 day", retention="30 days",
               format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name} — {message}")


# ── Broker Setup (Trader Mode) ─────────────────────────────

def setup_brokers(paper: bool = False) -> object:
    """
    Initialisér alle brokers og BrokerRouter.
    Returns BrokerRouter med alle aktive brokers registreret.

    Altid registrerer PaperBroker som fallback, så platformen
    kan køre helt lokalt uden eksterne API-forbindelser.
    """
    from src.broker.broker_router import BrokerRouter

    router = BrokerRouter()
    active_brokers = []

    # 0. PaperBroker (altid tilgængelig — lokal simulator)
    try:
        from src.broker.paper_broker import PaperBroker
        paper_broker = PaperBroker(initial_capital=100_000)
        router.register("paper", paper_broker)
        active_brokers.append("paper")
        logger.info("[startup] PaperBroker registered (local simulator, $100k)")
    except Exception as e:
        logger.error(f"[startup] PaperBroker failed: {e}")

    # 1. Alpaca (kun hvis API-nøgler er sat OG vi ikke er i paper-only mode)
    try:
        from src.broker.alpaca_broker import AlpacaBroker
        alpaca_key = os.getenv("ALPACA_API_KEY", "")
        alpaca_secret = os.getenv("ALPACA_SECRET_KEY", "")
        if alpaca_key and alpaca_secret:
            base_url = "https://paper-api.alpaca.markets" if paper else "https://api.alpaca.markets"
            alpaca = AlpacaBroker(api_key=alpaca_key, secret_key=alpaca_secret, base_url=base_url)
            if hasattr(alpaca, "connect"):
                alpaca.connect()
            router.register("alpaca", alpaca)
            active_brokers.append("alpaca")
            logger.info(f"[startup] Alpaca connected ({'paper' if paper else 'live'})")
        else:
            logger.info("[startup] Alpaca: No API keys — using PaperBroker as fallback")
    except Exception as e:
        logger.warning(f"[startup] Alpaca failed: {e} — PaperBroker available as fallback")

    # 2. Saxo Bank
    try:
        from src.broker.saxo_broker import SaxoBroker
        from src.broker.saxo_auth import SaxoAuthManager, SaxoConfig
        saxo_key = os.getenv("SAXO_APP_KEY", "")
        if saxo_key:
            env = "sim" if paper else "live"
            config = SaxoConfig(
                app_key=saxo_key,
                app_secret=os.getenv("SAXO_APP_SECRET", ""),
                redirect_uri=os.getenv("SAXO_REDIRECT_URI", "http://localhost:8080/callback"),
                environment=env,
            )
            auth = SaxoAuthManager(config)
            saxo = SaxoBroker(auth_manager=auth)
            saxo.connect()
            router.register("saxo", saxo)
            active_brokers.append("saxo")
            logger.info(f"[startup] Saxo Bank connected ({env})")
        else:
            logger.debug("[startup] Saxo: No app key — skipped")
    except Exception as e:
        logger.warning(f"[startup] Saxo failed: {e}")

    # 3. IBKR (only if explicitly enabled via IBKR_ENABLED env var)
    ibkr_enabled = os.getenv("IBKR_ENABLED", "false").lower() in ("true", "1", "yes")
    if ibkr_enabled:
        try:
            from src.broker.ibkr_broker import IBKRBroker
            ibkr_host = os.getenv("IBKR_HOST", "127.0.0.1")
            ibkr_port = int(os.getenv("IBKR_PORT", "7497" if paper else "7496"))
            ibkr_client = int(os.getenv("IBKR_CLIENT_ID", "1"))
            ibkr = IBKRBroker(host=ibkr_host, port=ibkr_port, client_id=ibkr_client)
            ibkr.connect(timeout=10)
            router.register("ibkr", ibkr)
            active_brokers.append("ibkr")
            logger.info(f"[startup] IBKR connected ({ibkr_host}:{ibkr_port})")
        except Exception as e:
            logger.debug(f"[startup] IBKR not available: {e}")
    else:
        logger.debug("[startup] IBKR: not enabled (set IBKR_ENABLED=true to connect)")

    # 4. Nordnet
    try:
        from src.broker.nordnet_broker import NordnetBroker
        from src.broker.nordnet_auth import NordnetSession, NordnetConfig
        nordnet_user = os.getenv("NORDNET_USER", "")
        nordnet_pass = os.getenv("NORDNET_PASS", "")
        if nordnet_user and nordnet_pass:
            country = os.getenv("NORDNET_COUNTRY", "dk")
            config = NordnetConfig(country=country)
            session = NordnetSession(config)
            session.login(nordnet_user, nordnet_pass)
            nordnet = NordnetBroker(session=session)
            nordnet.connect()
            router.register("nordnet", nordnet)
            active_brokers.append("nordnet")
            logger.info(f"[startup] Nordnet connected ({country})")
        else:
            logger.debug("[startup] Nordnet: No credentials — skipped")
    except Exception as e:
        logger.warning(f"[startup] Nordnet failed: {e}")

    logger.info(f"[startup] Active brokers: {active_brokers}")
    if active_brokers == ["paper"]:
        logger.info("[startup] Running in LOCAL MODE — PaperBroker only (no external APIs needed)")

    return router


# ── Connection Monitoring ──────────────────────────────────

def setup_connection_monitor(router: object) -> object | None:
    """Start ConnectionManager med health checks."""
    try:
        from src.broker.connection_manager import ConnectionManager
        cm = ConnectionManager()
        for name in router.available_brokers:
            cm.register(name, router.get_broker(name))

        # Alert callback
        from src.ops.email_reports import EmailReportRunner
        reporter = EmailReportRunner()

        def on_status_change(change) -> None:
            if change.new_status.value == "disconnected":
                reporter.alarm.broker_disconnected(change.broker_name)

        cm.on_status_change(on_status_change)
        cm.start(interval=60)
        logger.info("[startup] ConnectionManager started (60s interval)")
        return cm
    except Exception as e:
        logger.warning(f"[startup] ConnectionManager failed: {e}")
        return None


# ── Tax Setup ──────────────────────────────────────────────

def setup_tax() -> dict:
    """Initialisér skattemodulerne."""
    modules = {}

    try:
        from src.tax.corporate_tax import CorporateTaxCalculator
        modules["calculator"] = CorporateTaxCalculator()
    except Exception as e:
        logger.warning(f"[startup] CorporateTaxCalculator failed: {e}")

    try:
        from src.tax.tax_credit_tracker import TaxCreditTracker
        modules["credit_tracker"] = TaxCreditTracker()
    except Exception as e:
        logger.warning(f"[startup] TaxCreditTracker failed: {e}")

    try:
        from src.tax.mark_to_market import MarkToMarketEngine
        modules["mtm"] = MarkToMarketEngine()
    except Exception as e:
        logger.warning(f"[startup] MarkToMarketEngine failed: {e}")

    try:
        from src.tax.dividend_tracker import DividendTracker
        modules["dividends"] = DividendTracker()
    except Exception as e:
        logger.warning(f"[startup] DividendTracker failed: {e}")

    try:
        from src.tax.currency_pnl import CurrencyPnLTracker
        modules["fx_pnl"] = CurrencyPnLTracker()
    except Exception as e:
        logger.warning(f"[startup] CurrencyPnLTracker failed: {e}")

    logger.info(f"[startup] Tax modules loaded: {list(modules.keys())}")
    return modules


# ── Scheduler Setup ────────────────────────────────────────

def setup_scheduler() -> object | None:
    """Start daglig scheduler."""
    try:
        from src.ops.daily_scheduler import DailyScheduler
        from src.ops.email_reports import EmailReportRunner

        scheduler = DailyScheduler()
        reporter = EmailReportRunner()

        # Hook email reports into scheduler results
        def on_task_result(result):
            if result.task_name == "morning_check" and result.status.value == "completed":
                reporter.send_morning_report()
            elif result.task_name == "us_market_close" and result.status.value == "completed":
                reporter.send_evening_report()

        scheduler.on_task_complete(on_task_result)
        scheduler.start()
        logger.info("[startup] DailyScheduler started")
        return scheduler
    except Exception as e:
        logger.warning(f"[startup] Scheduler failed: {e}")
        return None


# ── Dashboard ──────────────────────────────────────────────

def run_dashboard(host: str = "127.0.0.1", port: int = 8050, debug: bool = False) -> None:
    """Start Dash dashboard."""
    try:
        from src.dashboard.app import app
        logger.info(f"[startup] Dashboard starting on http://{host}:{port}")
        app.run(host=host, port=port, debug=debug)
    except Exception as e:
        logger.error(f"[startup] Dashboard failed: {e}")
        raise


# ── Mode Runners ───────────────────────────────────────────

def setup_auto_trader(router, paper: bool = True):
    """Initialisér AutoTrader med SignalEngine + RiskManager."""
    try:
        from src.trader.auto_trader import AutoTrader

        auto = AutoTrader(router=router, paper=paper)

        # RiskManager for trade checks + DynamicRiskManager for circuit breakers
        try:
            from src.risk.risk_manager import RiskManager
            from src.risk.portfolio_tracker import PortfolioTracker
            tracker = PortfolioTracker()
            rm = RiskManager(portfolio=tracker)
            auto.set_risk_manager(rm)

            # Add DynamicRiskManager for circuit breakers + regime adaptation
            try:
                from src.risk.dynamic_risk import DynamicRiskManager
                drm = DynamicRiskManager(portfolio=tracker)
                auto._dynamic_risk_manager = drm
                logger.info("[startup] RiskManager + DynamicRiskManager aktiv — circuit breakers enabled")
            except Exception as e2:
                logger.info(f"[startup] RiskManager aktiv (DynamicRisk not available: {e2})")
        except Exception as e:
            logger.warning(f"[startup] RiskManager ikke tilgængelig: {e}")

        # Register globally so dashboard settings page can access it
        from src.broker.registry import set_auto_trader
        set_auto_trader(auto)

        logger.info(f"[startup] AutoTrader klar — {len(auto.watchlist)} symboler, {'PAPER' if paper else 'LIVE'}")
        return auto
    except Exception as e:
        logger.error(f"[startup] AutoTrader failed: {e}")
        return None


def setup_scheduler_with_auto_trader(auto_trader=None) -> object | None:
    """Start daglig scheduler + continuous AutoTrader scan loop."""
    try:
        from src.ops.daily_scheduler import DailyScheduler
        from src.ops.email_reports import EmailReportRunner

        scheduler = DailyScheduler()
        reporter = EmailReportRunner()

        # Hook email reports into scheduler results
        def on_task_result(result):
            if result.task_name == "morning_check" and result.status.value == "completed":
                reporter.send_morning_report()
            elif result.task_name == "us_market_close" and result.status.value == "completed":
                reporter.send_evening_report()

        scheduler.on_task_complete(on_task_result)
        scheduler.start()
        logger.info(f"[startup] DailyScheduler started — {len(scheduler._tasks)} tasks")

        # Start continuous scan loop (hvert 15. minut under markedstid)
        if auto_trader is not None:
            scan_thread, scan_stop_event = start_continuous_scanner(auto_trader, interval_minutes=10)
            # Store thread + stop_event on scheduler for graceful shutdown (H-19)
            scheduler._scan_thread = scan_thread
            scheduler._scan_stop_event = scan_stop_event
            logger.info("[startup] Continuous scanner started (10 min interval)")

        return scheduler
    except Exception as e:
        logger.warning(f"[startup] Scheduler failed: {e}")
        return None


def start_continuous_scanner(auto_trader, interval_minutes: int = 1) -> tuple[threading.Thread, threading.Event]:
    """
    Kør AutoTrader scan hvert N. minut — 24/7 global market coverage.

    Returns (thread, stop_event) so the caller can stop the scanner
    gracefully by calling stop_event.set().

    Bruger MarketCalendar til at bestemme hvilke markeder der er åbne.
    Scanner kører altid på handelsdage — MarketCalendar filtrerer symboler.

    Markedsdækning (CET):
      22:00 - 03:00  New Zealand
      01:00 - 07:30  Tokyo + Sydney
      02:00 - 08:00  Hong Kong
      04:45 - 11:15  Mumbai
      09:00 - 17:30  EU + Nordic + London
      10:00 - 15:30  US Pre-market
      15:30 - 22:00  US Regular
      22:00 - 02:00  US Post-market
      24/7           Crypto
    """
    import threading
    from src.ops.daily_scheduler import is_market_day
    from src.broker.broker_router import _CRYPTO_PATTERN

    _stop = threading.Event()

    def _scanner_loop():
        while not _stop.is_set():
            now = datetime.now(TZ_CET)
            today = now.date()

            # Scanner kører alle handelsdage — MarketCalendar håndterer hvilke symboler
            if is_market_day(today):
                wait_time = interval_minutes * 60
                try:
                    # Check hvilke markeder er åbne nu
                    open_markets = []
                    try:
                        from src.ops.market_calendar import get_calendar
                        open_markets = get_calendar().get_open_markets(now)
                    except Exception:
                        pass

                    logger.info(
                        f"[scanner] ── Scan kl. {now:%H:%M} CET ──"
                        f" Åbne markeder: {open_markets or ['alle']}"
                    )
                    scan_start = time.time()
                    result = auto_trader.scan_and_trade()
                    scan_duration = time.time() - scan_start
                    trades = result.trades_executed
                    if trades > 0:
                        logger.info(f"[scanner] {trades} handler udført!")
                    # Wait mindst interval_minutes, men længere hvis scan tog lang tid
                    wait_time = max(interval_minutes * 60, scan_duration * 1.5)
                    logger.info(f"[scanner] Scan tog {scan_duration:.0f}s — næste om {wait_time:.0f}s")
                except Exception as e:
                    logger.error(f"[scanner] Scan fejl: {e}")

                _stop.wait(timeout=wait_time)

            else:
                # Weekend — kun crypto scanner hvert 5. minut
                try:
                    logger.debug(f"[scanner] Weekend — kun crypto scan {now:%H:%M}")
                    # Filter to crypto-only symbols for weekend scanning
                    crypto_symbols = [
                        s for s in auto_trader.watchlist
                        if _CRYPTO_PATTERN.match(s.upper())
                    ]
                    if crypto_symbols:
                        # scan_and_trade() har ingen symbols-parameter — filtrér watchlist midlertidigt
                        original_watchlist = auto_trader.watchlist[:]
                        auto_trader.watchlist = crypto_symbols
                        try:
                            result = auto_trader.scan_and_trade()
                            if result.trades_executed > 0:
                                logger.info(f"[scanner] Crypto: {result.trades_executed} handler")
                        finally:
                            auto_trader.watchlist = original_watchlist
                    else:
                        logger.debug("[scanner] Weekend — ingen crypto symboler i watchlist")
                except Exception as e:
                    logger.error(f"[scanner] Weekend scan fejl: {e}")
                _stop.wait(timeout=300)

    thread = threading.Thread(
        target=_scanner_loop,
        name="ContinuousScanner",
        daemon=True,
    )
    thread.start()
    return thread, _stop


def run_trader(args: argparse.Namespace) -> None:
    """Start Alpha Trader — full multi-broker trading platform."""
    # Start web time sync service (syncs at startup + nightly 23:00 CET)
    try:
        from src.ops.time_service import start as start_time_service, now_cet as _now_cet
        start_time_service()
        _time_str = _now_cet().strftime('%Y-%m-%d %H:%M CET')
    except Exception as _te:
        logger.warning(f"[startup] Time service failed: {_te} — using local clock")
        _time_str = datetime.now(TZ_CET).strftime('%Y-%m-%d %H:%M CET')

    logger.info("═══════════════════════════════════════════")
    logger.info("  ALPHA TRADER — Starting")
    logger.info(f"  Mode: {'paper' if args.paper else 'LIVE'}")
    logger.info(f"  Time: {_time_str} (web-synced)")
    logger.info("═══════════════════════════════════════════")

    # SIKKERHED: Kræv eksplicit bekræftelse for live trading
    if not args.paper:
        print("\n" + "=" * 50)
        print("  ⚠️  LIVE TRADING MODE  ⚠️")
        print("  Du er ved at handle med RIGTIGE PENGE.")
        print("  Skriv LIVE for at bekræfte, eller Ctrl+C for at stoppe.")
        print("=" * 50)
        try:
            confirmation = input("\n  Bekræft: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nAfbrudt.")
            logger.info("[startup] Live trading afbrudt af bruger")
            return
        if confirmation != "LIVE":
            print("Forkert bekræftelse. Brug --paper for paper trading.")
            logger.info(f"[startup] Live trading afvist (svar: '{confirmation}')")
            return
        logger.info("[startup] Live trading bekræftet af bruger")

    # 1. Brokers
    router = setup_brokers(paper=args.paper)

    # Registrer globalt så dashboard kan tilgå samme router
    from src.broker.registry import set_router
    set_router(router)

    # 2. Connection monitor
    cm = setup_connection_monitor(router)

    # 3. Tax modules
    tax = setup_tax()

    # 4. AutoTrader (signaler → ordrer)
    auto_trader = None
    if not getattr(args, "no_auto_trade", False):
        auto_trader = setup_auto_trader(router, paper=args.paper)
    else:
        logger.info("[startup] AutoTrader disabled (--no-auto-trade)")

    # 4a. Wire ConnectionManager → RiskManager (afvis ordrer ved broker disconnect)
    if auto_trader is not None and cm is not None:
        rm = getattr(auto_trader, '_risk_manager', None)
        if rm is not None and hasattr(rm, 'set_connection_manager'):
            rm.set_connection_manager(cm)
            logger.info("[startup] ConnectionManager → RiskManager wired (broker disconnect = ordre afvist)")

    # 4b. Position-reconciliation — synkronisér broker-positioner med lokal portfolio
    if not args.paper and auto_trader is not None:
        try:
            rm = getattr(auto_trader, '_risk_manager', None)
            if rm and hasattr(rm, 'portfolio'):
                broker_positions = router.get_positions()
                synced = 0
                for pos in broker_positions:
                    if pos.symbol not in rm.portfolio.positions:
                        rm.portfolio.open_position(
                            symbol=pos.symbol,
                            qty=pos.qty,
                            price=pos.current_price or pos.entry_price,
                            side=pos.side,
                        )
                        synced += 1
                        logger.info(f"[reconcile] Synkroniseret: {pos.symbol} ({pos.side}, {pos.qty} @ ${pos.entry_price:.2f})")
                if synced > 0:
                    logger.info(f"[reconcile] {synced} broker-positioner synkroniseret til lokal portfolio")
                else:
                    logger.info("[reconcile] Ingen åbne broker-positioner at synkronisere")
        except Exception as e:
            logger.warning(f"[reconcile] Position-reconciliation fejlede: {e}")

    # 5. Scheduler med AutoTrader integreret
    scheduler = None
    if not args.no_scheduler:
        scheduler = setup_scheduler_with_auto_trader(auto_trader)

    # 5b. Continuous Learning Engine — already started inside AutoTrader.__init__
    # (removed duplicate instance that was leaking memory)

    # 6. Order Manager — register globally so dashboard can access it (H-20)
    try:
        from src.broker.order_manager import OrderManager
        from src.broker.registry import set_order_manager
        order_mgr = OrderManager(router=router)
        set_order_manager(order_mgr)
        logger.info("[startup] OrderManager ready (registered in global registry)")
    except Exception as e:
        logger.warning(f"[startup] OrderManager failed: {e}")

    # Startup summary
    logger.info("═══════════════════════════════════════════")
    logger.info("  ✓ Alpha Trader RUNNING")
    if auto_trader:
        status = auto_trader.status()
        logger.info(f"  ✓ AutoTrader: {status['watchlist']} symboler, {status['strategies']} strategier")
        logger.info(f"  ✓ Scanner: hvert 1. minut, 24/7 global market coverage (MarketCalendar)")
    if auto_trader and getattr(auto_trader, '_learner', None):
        logger.info(f"  ✓ Learning Engine: feedback loop + drift detection + krise-scanning")
    logger.info("═══════════════════════════════════════════")

    # 7. Mobile API (port 8051)
    try:
        from src.api.server import start_api_server
        start_api_server(host="0.0.0.0", port=8051, background=True)
        logger.info("[startup] Mobile API started on http://0.0.0.0:8051")
    except Exception as _api_err:
        logger.warning(f"[startup] Mobile API ikke startet: {_api_err}")

    # 8. Dashboard
    if not args.no_dashboard:
        run_dashboard(
            host=args.host,
            port=args.port,
            debug=args.debug,
        )
    else:
        # Keep running without dashboard (headless mode)
        shutdown_event = getattr(args, '_shutdown_event', None)
        logger.info("[startup] Running in headless mode (no dashboard)")
        logger.info("[startup] Press Ctrl+C to stop")
        try:
            if shutdown_event:
                shutdown_event.wait()
            else:
                while True:
                    time.sleep(1)
        except KeyboardInterrupt:
            pass

    # Cleanup (always runs — shutdown_handler sets event instead of sys.exit)
    if auto_trader and getattr(auto_trader, '_learner', None):
        auto_trader._learner.stop()
    if scheduler:
        # Stop the continuous scanner gracefully if running
        scan_stop = getattr(scheduler, '_scan_stop_event', None)
        if scan_stop:
            scan_stop.set()
            logger.info("[shutdown] Continuous scanner stop signal sent")
            # Wait for scanner thread to finish (H-19)
            scan_thread = getattr(scheduler, '_scan_thread', None)
            if scan_thread and scan_thread.is_alive():
                scan_thread.join(timeout=10)
                logger.info("[shutdown] Continuous scanner thread joined")
        scheduler.stop()
    if cm:
        cm.stop()
    logger.info("[shutdown] Alpha Trader stopped")


def run_research(args: argparse.Namespace) -> None:
    """Start Alpha Research — GPU-accelerated research platform."""
    logger.info("═══════════════════════════════════════════")
    logger.info("  ALPHA RESEARCH — Starting")
    logger.info("═══════════════════════════════════════════")

    # Research mode uses the existing analysis infrastructure
    # without multi-broker trading features
    if not args.no_dashboard:
        run_dashboard(
            host=args.host,
            port=args.port,
            debug=args.debug,
        )
    else:
        logger.info("[startup] Research mode — headless")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass

    logger.info("[shutdown] Alpha Research stopped")


def run_dashboard_only(args: argparse.Namespace) -> None:
    """Start kun dashboard — ingen trading."""
    logger.info("═══════════════════════════════════════════")
    logger.info("  ALPHA DASHBOARD — View-only mode")
    logger.info("═══════════════════════════════════════════")
    run_dashboard(
        host=args.host,
        port=args.port,
        debug=args.debug,
    )


# ── CLI ────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Alpha Trading Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode trader --paper      # Paper trading med alle brokers
  python main.py --mode trader              # Live trading
  python main.py --mode research            # Research platform
  python main.py --mode dashboard           # Dashboard only
  python main.py --mode trader --headless   # Trader uden dashboard
        """,
    )
    parser.add_argument(
        "--mode", choices=["trader", "research", "dashboard"],
        default="trader",
        help="Platform mode (default: trader)",
    )
    parser.add_argument("--paper", action="store_true", help="Paper trading (Alpaca paper, Saxo sim, IBKR paper)")
    parser.add_argument("--no-scheduler", action="store_true", help="Skip daily scheduler")
    parser.add_argument("--no-auto-trade", action="store_true", help="Disable automatic trading (signals only, no orders)")
    parser.add_argument("--no-dashboard", "--headless", action="store_true", dest="no_dashboard", help="Run without dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="Dashboard host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8050, help="Dashboard port (default: 8050)")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--log-level", default="DEBUG", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Log level")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Logging
    setup_logging(level=args.log_level)

    # Graceful shutdown via event flag (avoids sys.exit skipping cleanup)
    shutdown_event = threading.Event()

    def shutdown_handler(signum, frame):
        logger.info(f"[shutdown] Signal {signum} received — shutting down")
        shutdown_event.set()

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    # Make shutdown_event available to run_trader for its main loop
    args._shutdown_event = shutdown_event

    # Route to mode
    if args.mode == "trader":
        run_trader(args)
    elif args.mode == "research":
        run_research(args)
    elif args.mode == "dashboard":
        run_dashboard_only(args)


if __name__ == "__main__":
    main()
