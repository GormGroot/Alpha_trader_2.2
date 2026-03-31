"""
Centralised time service — fetches authoritative time from the web.

All platform code should use `now_cet()` from this module instead of
`datetime.now(TZ_CET)` to ensure accurate timestamps regardless of
local clock drift.

The offset between local clock and web time is computed at startup
and refreshed nightly at 23:00 CET.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from loguru import logger

TZ_CET = ZoneInfo("Europe/Copenhagen")

# Offset = (web_time - local_time) in seconds.
# Positive means local clock is behind, negative means ahead.
_offset_seconds: float = 0.0
_offset_lock = threading.Lock()
_last_sync: float = 0.0  # monotonic timestamp of last successful sync


def _fetch_web_time() -> datetime | None:
    """Fetch current UTC time from a web API (tries multiple sources)."""
    import urllib.request
    import json

    sources = [
        ("http://worldtimeapi.org/api/timezone/Etc/UTC", "utc_datetime"),
        ("http://worldtimeapi.org/api/timezone/Europe/Copenhagen", "utc_datetime"),
    ]

    for url, key in sources:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "AlphaTrader/2.2"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
                dt_str = data[key]
                # Parse ISO format: "2026-03-31T05:14:23.456789+00:00"
                # Strip microseconds beyond 6 digits if present
                dt = datetime.fromisoformat(dt_str)
                return dt.astimezone(timezone.utc)
        except Exception as exc:
            logger.debug(f"[time-service] {url} failed: {exc}")
            continue

    # Fallback: use HTTP Date header from a reliable server
    try:
        req = urllib.request.Request("https://www.google.com", method="HEAD",
                                     headers={"User-Agent": "AlphaTrader/2.2"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            date_header = resp.headers.get("Date", "")
            if date_header:
                from email.utils import parsedate_to_datetime
                return parsedate_to_datetime(date_header).astimezone(timezone.utc)
    except Exception as exc:
        logger.debug(f"[time-service] Google HEAD failed: {exc}")

    return None


def sync_time() -> bool:
    """Synchronise local offset with web time. Returns True on success."""
    global _offset_seconds, _last_sync

    web_utc = _fetch_web_time()
    if web_utc is None:
        logger.warning("[time-service] Could not fetch web time — using local clock")
        return False

    local_utc = datetime.now(timezone.utc)
    offset = (web_utc - local_utc).total_seconds()

    with _offset_lock:
        old_offset = _offset_seconds
        _offset_seconds = offset
        _last_sync = time.monotonic()

    drift_ms = offset * 1000
    change_ms = (offset - old_offset) * 1000
    logger.info(
        f"[time-service] Synced — drift: {drift_ms:+.0f}ms "
        f"(change: {change_ms:+.0f}ms from last sync)"
    )
    return True


def now_utc() -> datetime:
    """Get current UTC time corrected by web offset."""
    with _offset_lock:
        offset = _offset_seconds
    return datetime.now(timezone.utc) + timedelta(seconds=offset)


def now_cet() -> datetime:
    """Get current CET time corrected by web offset."""
    return now_utc().astimezone(TZ_CET)


def get_offset_ms() -> float:
    """Get the current offset in milliseconds (for diagnostics)."""
    with _offset_lock:
        return _offset_seconds * 1000


def get_last_sync_age() -> float:
    """Seconds since last successful sync."""
    if _last_sync == 0:
        return float("inf")
    return time.monotonic() - _last_sync


# ── Nightly resync thread ──────────────────────────────────

def _nightly_sync_worker():
    """Background thread: resync at 23:00 CET every night."""
    while True:
        try:
            now = now_cet()
            # Calculate seconds until next 23:00 CET
            target = now.replace(hour=23, minute=0, second=0, microsecond=0)
            if now >= target:
                # Already past 23:00 today, aim for tomorrow
                target += timedelta(days=1)
            wait_seconds = (target - now).total_seconds()
            logger.debug(
                f"[time-service] Next sync at 23:00 CET "
                f"(in {wait_seconds / 3600:.1f}h)"
            )
            time.sleep(wait_seconds)

            # Resync
            logger.info("[time-service] Nightly 23:00 CET resync starting...")
            sync_time()

        except Exception as exc:
            logger.error(f"[time-service] Nightly sync error: {exc}")
            time.sleep(3600)  # retry in 1 hour


def start():
    """Initial sync + start the nightly resync thread."""
    logger.info("[time-service] Initial time sync...")
    sync_time()

    t = threading.Thread(target=_nightly_sync_worker, daemon=True, name="time-sync")
    t.start()
    logger.info("[time-service] Nightly 23:00 CET resync thread started")
