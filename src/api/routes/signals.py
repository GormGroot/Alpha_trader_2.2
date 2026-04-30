"""
Signal endpoints — active trading signals with confidence scores.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from fastapi import APIRouter, Depends

from src.api.auth import get_current_user

router = APIRouter()
SIGNAL_DB = Path("data_cache/signal_log.db")


@router.get("")
@router.get("/")
async def get_signals(limit: int = 30, _: str = Depends(get_current_user)):
    """Latest trading signals."""
    if not SIGNAL_DB.exists():
        return []
    try:
        with sqlite3.connect(SIGNAL_DB) as conn:
            conn.row_factory = sqlite3.Row
            # Try common table names
            tables = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()]
            if not tables:
                return []
            table = tables[0]
            rows = conn.execute(
                f"SELECT * FROM {table} ORDER BY rowid DESC LIMIT ?",
                (limit,)
            ).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        return []
