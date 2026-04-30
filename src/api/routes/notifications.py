"""
Notification endpoints — recent alerts from notifications.db.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from fastapi import APIRouter, Depends

from src.api.auth import get_current_user

router = APIRouter()
NOTIF_DB = Path("data_cache/notifications.db")


@router.get("")
@router.get("/")
async def get_notifications(limit: int = 20, _: str = Depends(get_current_user)):
    """Recent notifications/alerts."""
    if not NOTIF_DB.exists():
        return []
    with sqlite3.connect(NOTIF_DB) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """SELECT id, timestamp, severity, title, message, category
               FROM notification_history
               ORDER BY timestamp DESC LIMIT ?""",
            (limit,)
        ).fetchall()
    return [dict(r) for r in rows]
