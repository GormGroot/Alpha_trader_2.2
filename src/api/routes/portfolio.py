"""
Portfolio endpoints — summary, positions, trades, equity curve.

PRIMARY SOURCE: Live broker via BrokerRouter (Alpaca paper or live).
This guarantees PWA shows the SAME positions/equity as the broker has.

FALLBACK: Local SQLite (data_cache/paper_portfolio.db) for historical
trades and equity-curve when broker is unavailable.

Phase A1 fix (2026-05-10): Previously /positions, /summary read from
local SQLite which drifted from Alpaca (showed 19 positions while Alpaca
had 11; AAPL long in PWA but SHORT at Alpaca). This file now reads
positions/account directly from the broker.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends
from loguru import logger

from src.api.auth import get_current_user

router = APIRouter()
DB_PATH = Path("data_cache/paper_portfolio.db")


def _db():
    return sqlite3.connect(DB_PATH)


_lazy_alpaca = None  # cached AlpacaBroker instance for subprocess access


def _get_alpaca_lazy():
    """Build an AlpacaBroker on-demand when running in a subprocess.

    The mobile API runs as a subprocess (see src/api/server.py:start_api_server)
    so it can't share the in-memory BrokerRouter from the main process. Each
    subprocess request lazily constructs its own AlpacaBroker from the same
    `.env` credentials. Alpaca calls are stateless so this is safe.
    """
    global _lazy_alpaca
    if _lazy_alpaca is not None:
        return _lazy_alpaca

    import os
    api_key = os.getenv("ALPACA_API_KEY", "")
    api_secret = os.getenv("ALPACA_SECRET_KEY", "")
    if not (api_key and api_secret):
        return None
    try:
        from src.broker.alpaca_broker import AlpacaBroker
        # Default to paper unless explicit live flag is present
        live = os.getenv("ALPACA_LIVE", "").strip().lower() in ("1", "true", "yes")
        base = "https://api.alpaca.markets" if live else "https://paper-api.alpaca.markets"
        _lazy_alpaca = AlpacaBroker(
            api_key=api_key, secret_key=api_secret, base_url=base
        )
        logger.info(f"[portfolio] AlpacaBroker constructed lazily ({base})")
        return _lazy_alpaca
    except Exception as exc:
        logger.warning(f"[portfolio] Lazy Alpaca-init fejlede: {exc}")
        return None


def _get_router_or_none():
    """Return a broker capable of serving positions/account.

    Tries in order:
      1. Global BrokerRouter from registry (when API runs in main process)
      2. Lazy AlpacaBroker (when API runs in subprocess)
    """
    try:
        from src.broker.registry import get_router
        r = get_router()
        if r is not None:
            return r
    except Exception as exc:
        logger.debug(f"[portfolio] registry not ready: {exc}")
    return _get_alpaca_lazy()


def _account_summary_from_broker(router_obj) -> dict[str, Any] | None:
    """Pull live equity/cash from the broker. Returns None on failure."""
    try:
        acc = router_obj.get_account()
        if acc and acc.equity is not None:
            return {
                "total_equity": float(acc.equity),
                "cash": float(acc.cash),
                "buying_power": float(acc.buying_power),
                "currency": acc.currency,
                "source": "broker",
            }
    except Exception as exc:
        logger.warning(f"[portfolio] router.get_account() fejlede: {exc}")
    return None


def _pnl_metrics_from_db() -> dict[str, Any]:
    """Read P&L history (realized, win-rate, drawdown) from local SQLite.

    These metrics require historical trade data that the broker API
    doesn't expose directly. Fall back to zeros if DB is empty.
    """
    metrics = {
        "realized_pnl": 0.0,
        "daily_pnl": 0.0,
        "daily_pnl_pct": 0.0,
        "total_return_pct": 0.0,
        "current_drawdown_pct": 0.0,
        "max_drawdown_pct": 0.0,
        "win_rate": 0.0,
        "profit_factor": 0.0,
        "sharpe_ratio": 0.0,
        "closed_trades": 0,
    }
    try:
        from src.risk.portfolio_tracker import PortfolioTracker
        pt = PortfolioTracker()
        summary = pt.summary() or {}
        for k in metrics:
            if k in summary and summary[k] is not None:
                metrics[k] = summary[k]
    except Exception as exc:
        logger.debug(f"[portfolio] PnL fallback to zeros: {exc}")
    return metrics


@router.get("/summary")
@router.get("/summary/")
async def portfolio_summary(_: str = Depends(get_current_user)):
    """
    Total equity, cash, P&L, key metrics.

    EQUITY/CASH from broker (Alpaca-direct).
    P&L METRICS from local SQLite (historical trade log).
    """
    router_obj = _get_router_or_none()
    broker_acc = _account_summary_from_broker(router_obj) if router_obj else None
    pnl = _pnl_metrics_from_db()

    if broker_acc is not None:
        positions_count = 0
        unrealized = 0.0
        try:
            for p in router_obj.get_positions():
                positions_count += 1
                unrealized += getattr(p, "unrealized_pnl", 0.0) or 0.0
        except Exception:
            pass

        return {
            "total_equity": broker_acc["total_equity"],
            "cash": broker_acc["cash"],
            "buying_power": broker_acc["buying_power"],
            "currency": broker_acc["currency"],
            "positions": positions_count,
            "unrealized_pnl": unrealized,
            **pnl,
            "source": "broker",
        }

    # Fallback: pure SQLite if broker unavailable
    logger.warning("[portfolio] /summary bruger SQLite-fallback (broker utilgængelig)")
    if not DB_PATH.exists():
        return {"error": "Ingen data endnu", **pnl}
    with _db() as conn:
        rows = conn.execute("SELECT key, value FROM portfolio_state").fetchall()
        state = {r[0]: r[1] for r in rows}
    return {
        "total_equity": state.get("cash", 0),
        "cash": state.get("cash", 0),
        "positions": 0,
        "unrealized_pnl": 0,
        **pnl,
        "source": "sqlite_fallback",
    }


@router.get("/positions")
@router.get("/positions/")
async def portfolio_positions(_: str = Depends(get_current_user)):
    """
    Open positions list — direct from broker.

    Phase A1 fix: previously read from data_cache/paper_portfolio.db which
    drifted from broker reality. Now calls router.get_positions() so PWA
    always shows the SAME positions that exist at Alpaca.
    """
    router_obj = _get_router_or_none()
    if router_obj is None:
        logger.warning("[portfolio] /positions: ingen broker — returnerer tom liste")
        return []

    try:
        broker_positions = router_obj.get_positions()
    except Exception as exc:
        logger.error(f"[portfolio] router.get_positions() fejlede: {exc}")
        return []

    out = []
    for p in broker_positions:
        try:
            qty = float(getattr(p, "qty", 0) or 0)
            entry = float(getattr(p, "entry_price", 0) or 0)
            current = float(getattr(p, "current_price", entry) or entry)
            side = getattr(p, "side", "long")

            # Calculate unrealized P&L respecting side (short = inverse)
            if side == "short":
                pnl = (entry - current) * qty
            else:
                pnl = (current - entry) * qty
            pnl_pct = ((current - entry) / entry * 100) if entry else 0
            if side == "short":
                pnl_pct = -pnl_pct

            out.append({
                "symbol": getattr(p, "symbol", ""),
                "side": side,
                "qty": qty,
                "entry_price": entry,
                "current_price": current,
                "market_value": current * qty,
                "unrealized_pnl": pnl,
                "unrealized_pnl_pct": pnl_pct,
                "entry_time": getattr(p, "entry_time", ""),
            })
        except Exception as exc:
            logger.warning(f"[portfolio] kunne ikke serialisere position: {exc}")
    return out


@router.get("/trades")
@router.get("/trades/")
async def portfolio_trades(limit: int = 50, _: str = Depends(get_current_user)):
    """Recent closed trades (from local SQLite — broker API doesn't expose history)."""
    if not DB_PATH.exists():
        return []
    with _db() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM closed_trades ORDER BY exit_time DESC LIMIT ?",
            (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


@router.post("/close/{symbol}")
@router.post("/close/{symbol}/")
async def portfolio_close(symbol: str, user: str = Depends(get_current_user)):
    """Phase A5: Manuel lukning af en åben position.

    1. Slå position op hos broker
    2. Send modsat order via router.sell() / router.buy()
    3. Sync til lokal portfolio + Telegram-notifikation
    """
    from fastapi import HTTPException, status
    sym = symbol.upper()
    router_obj = _get_router_or_none()
    if router_obj is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Broker ikke tilgængelig",
        )

    # Find positionen
    target = None
    try:
        for p in router_obj.get_positions():
            if getattr(p, "symbol", "").upper() == sym:
                target = p
                break
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Kunne ikke hente positioner: {exc}")

    if target is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Ingen åben position i {sym}",
        )

    qty = abs(float(getattr(target, "qty", 0) or 0))
    side = getattr(target, "side", "long")
    if qty <= 0:
        raise HTTPException(status_code=400, detail="Position har 0 qty")

    # Send modsat order
    try:
        if side == "long":
            order = router_obj.sell(symbol=sym, qty=qty)
        else:  # short → cover med BUY
            order = router_obj.buy(symbol=sym, qty=qty)
    except Exception as exc:
        logger.error(f"[portfolio] Manuel close fejlede {sym}: {exc}")
        raise HTTPException(status_code=500, detail=f"Order-fejl: {exc}")

    logger.warning(f"[portfolio] MANUAL CLOSE: {user} solgte {qty} {sym} ({side})")

    # Telegram-notifikation
    try:
        from src.api.security_notify import _send_telegram
        _send_telegram(
            f"💼 *Manuel salg*\n\n"
            f"Symbol: `{sym}`\n"
            f"Qty: {qty}\n"
            f"Side: {side}\n"
            f"Udført af: `{user}` via PWA"
        )
    except Exception:
        pass

    return {
        "symbol": sym,
        "qty": qty,
        "side": side,
        "order_id": getattr(order, "order_id", None),
        "status": "submitted",
        "executed_by": user,
    }


@router.get("/equity")
@router.get("/equity/")
async def portfolio_equity(days: int = 30, _: str = Depends(get_current_user)):
    """Equity curve for the last N days (from local SQLite)."""
    if not DB_PATH.exists():
        return []
    with _db() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """SELECT timestamp, equity FROM equity_history
               ORDER BY timestamp DESC LIMIT ?""",
            (days * 24,)  # up to 24 snapshots/day
        ).fetchall()
    data = [{"timestamp": r["timestamp"], "equity": r["equity"]} for r in rows]
    return list(reversed(data))  # chronological order
