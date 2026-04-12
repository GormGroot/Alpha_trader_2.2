"""
Portfolio endpoints — summary, positions, trades, equity curve.
Reads from PortfolioTracker (in-memory) + SQLite.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from fastapi import APIRouter, Depends

from src.api.auth import get_current_user

router = APIRouter()
DB_PATH = Path("data_cache/paper_portfolio.db")


def _db():
    return sqlite3.connect(DB_PATH)


@router.get("/summary")
async def portfolio_summary(_: str = Depends(get_current_user)):
    """Total equity, cash, P&L, key metrics."""
    try:
        from src.risk.portfolio_tracker import PortfolioTracker
        pt = PortfolioTracker()
        return pt.summary()
    except Exception as e:
        # Fallback: read directly from SQLite
        if not DB_PATH.exists():
            return {"error": "Ingen data endnu"}
        with _db() as conn:
            rows = conn.execute("SELECT key, value FROM portfolio_state").fetchall()
            state = {r[0]: r[1] for r in rows}
        return {
            "total_equity": state.get("cash", 0),
            "cash": state.get("cash", 0),
            "positions": 0,
            "unrealized_pnl": 0,
            "realized_pnl": 0,
            "daily_pnl": 0,
            "daily_pnl_pct": 0,
            "total_return_pct": 0,
            "current_drawdown_pct": 0,
            "max_drawdown_pct": 0,
            "win_rate": 0,
            "profit_factor": 0,
            "sharpe_ratio": 0,
            "closed_trades": 0,
            "note": str(e),
        }


@router.get("/positions")
async def portfolio_positions(_: str = Depends(get_current_user)):
    """Open positions list."""
    if not DB_PATH.exists():
        return []
    with _db() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM open_positions ORDER BY symbol"
        ).fetchall()
    positions = []
    for r in rows:
        d = dict(r)
        entry = d.get("entry_price", 0) or 1
        current = d.get("current_price", 0) or 0
        qty = d.get("qty", 0) or 0
        pnl = (current - entry) * qty
        pnl_pct = (current - entry) / entry * 100 if entry else 0
        positions.append({
            "symbol": d["symbol"],
            "side": d.get("side", "long"),
            "qty": qty,
            "entry_price": entry,
            "current_price": current,
            "market_value": current * qty,
            "unrealized_pnl": pnl,
            "unrealized_pnl_pct": pnl_pct,
            "entry_time": d.get("entry_time", ""),
        })
    return positions


@router.get("/trades")
async def portfolio_trades(limit: int = 50, _: str = Depends(get_current_user)):
    """Recent closed trades."""
    if not DB_PATH.exists():
        return []
    with _db() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM closed_trades ORDER BY exit_time DESC LIMIT ?",
            (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


@router.get("/equity")
async def portfolio_equity(days: int = 30, _: str = Depends(get_current_user)):
    """Equity curve for the last N days."""
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
