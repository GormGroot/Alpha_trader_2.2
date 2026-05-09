"""
Phase A5: Control endpoints — pause/resume trading + close position.

API-serveren kører i en subprocess (separat fra AutoTrader), så vi bruger
en fil-baseret signal (data_cache/trading_paused.flag) til IPC. AutoTrader
tjekker flagget ved hvert scan via auto_trader.is_paused property.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger

from src.api.auth import get_current_user

router = APIRouter()

PAUSE_FLAG = Path("data_cache/trading_paused.flag")


@router.get("/status")
@router.get("/status/")
async def control_status(_: str = Depends(get_current_user)):
    """Hent pause-status fra fil-baseret flag."""
    if PAUSE_FLAG.exists():
        try:
            lines = PAUSE_FLAG.read_text().strip().split("\n")
            since = lines[0] if lines else None
            return {"paused": True, "paused_since": since}
        except Exception:
            return {"paused": True, "paused_since": None}
    return {"paused": False, "paused_since": None}


@router.post("/pause")
@router.post("/pause/")
async def control_pause(user: str = Depends(get_current_user)):
    """Sæt pause-flag → AutoTrader stopper med nye entries ved næste scan."""
    PAUSE_FLAG.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now().isoformat()
    PAUSE_FLAG.write_text(f"{now}\nreason=PWA user={user}\n")
    logger.warning(f"[control] Pause-flag sat af {user}")

    # Også prøv at notifyere AutoTrader-instansen direkte hvis i samme proces
    try:
        from src.broker.registry import get_auto_trader
        trader = get_auto_trader()
        if trader and hasattr(trader, "set_paused"):
            trader.set_paused(True, reason=f"PWA user={user}")
    except Exception:
        pass

    try:
        from src.api.security_notify import _send_telegram
        _send_telegram(f"⏸️ *Trading PAUSED*\n\nPauset af: `{user}` via PWA\nTidspunkt: {now}")
    except Exception:
        pass

    return {"paused": True, "paused_since": now, "reason": f"PWA user={user}"}


@router.post("/resume")
@router.post("/resume/")
async def control_resume(user: str = Depends(get_current_user)):
    """Slet pause-flag → AutoTrader genoptager nye entries."""
    if PAUSE_FLAG.exists():
        try:
            PAUSE_FLAG.unlink()
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Kunne ikke fjerne pause-flag: {e}",
            )
    logger.info(f"[control] Pause-flag fjernet af {user}")

    try:
        from src.broker.registry import get_auto_trader
        trader = get_auto_trader()
        if trader and hasattr(trader, "set_paused"):
            trader.set_paused(False, reason=f"PWA user={user}")
    except Exception:
        pass

    try:
        from src.api.security_notify import _send_telegram
        _send_telegram(f"▶️ *Trading RESUMED*\n\nGenoptaget af: `{user}` via PWA")
    except Exception:
        pass

    return {"paused": False, "paused_since": None}
