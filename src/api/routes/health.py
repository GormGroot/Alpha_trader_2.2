"""
Health/status endpoint — broker connections + platform uptime.
"""
from __future__ import annotations

import os
from datetime import datetime

from fastapi import APIRouter, Depends

from src.api.auth import get_current_user

router = APIRouter()
_START_TIME = datetime.utcnow()


@router.get("")
async def get_health(_: str = Depends(get_current_user)):
    """Platform health and broker status."""
    uptime_seconds = int((datetime.utcnow() - _START_TIME).total_seconds())
    hours, remainder = divmod(uptime_seconds, 3600)
    minutes = remainder // 60

    brokers = []
    # Alpaca
    alpaca_key = os.getenv("ALPACA_API_KEY", "")
    brokers.append({
        "name": "Alpaca",
        "status": "connected" if alpaca_key else "not_configured",
        "mode": "paper",
    })
    # Saxo
    saxo_key = os.getenv("SAXO_APP_KEY", "")
    brokers.append({
        "name": "Saxo",
        "status": "connected" if saxo_key else "not_configured",
        "mode": "live",
    })

    return {
        "status": "running",
        "uptime": f"{hours}t {minutes}m",
        "uptime_seconds": uptime_seconds,
        "started_at": _START_TIME.isoformat(),
        "brokers": brokers,
        "timestamp": datetime.utcnow().isoformat(),
    }
