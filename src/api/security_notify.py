"""
Sikkerhedsnotifikationer til Telegram.

Sender besked til Ole når:
  - Login lykkes (med tidspunkt, IP, by/land)
  - Login afvises efter 2FA-fejl
  - Geo-lock blokerer login fra ukendt land
"""
from __future__ import annotations

import os
import ssl
import urllib.parse
import urllib.request
from datetime import datetime
from zoneinfo import ZoneInfo

from loguru import logger

try:
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except Exception:
    _SSL_CTX = ssl.create_default_context()

TZ_CET = ZoneInfo("Europe/Copenhagen")


def _send_telegram(text: str) -> bool:
    """Send markdown-formateret besked til konfigureret Telegram-bot."""
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not (token and chat_id):
        return False

    payload = urllib.parse.urlencode({
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown",
    }).encode()

    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        req = urllib.request.Request(url, data=payload, method="POST")
        with urllib.request.urlopen(req, timeout=5, context=_SSL_CTX) as resp:
            return resp.status == 200
    except Exception as exc:
        logger.warning(f"[security] Telegram-besked fejlede: {exc}")
        return False


def notify_login_success(username: str, ip: str, location: str | None = None) -> None:
    """Send notifikation til Ole når login lykkes."""
    now = datetime.now(TZ_CET).strftime("%d/%m %H:%M")
    loc = f" fra {location}" if location else ""
    text = (
        f"🟢 *Login godkendt*\n\n"
        f"👤 Bruger: `{username}`\n"
        f"🕒 Tid: {now} CET\n"
        f"📍 IP: `{ip}`{loc}\n\n"
        f"_Hvis det ikke var dig — skift adgangskode straks._"
    )
    if _send_telegram(text):
        logger.info(f"[security] Login-notifikation sendt til Telegram for {username}")


def notify_login_failed(username: str, ip: str, reason: str, location: str | None = None) -> None:
    """Send notifikation når login fejler efter password er bekræftet (2FA-fejl)."""
    now = datetime.now(TZ_CET).strftime("%d/%m %H:%M")
    loc = f" fra {location}" if location else ""
    text = (
        f"🟡 *Mislykket login-forsøg*\n\n"
        f"👤 Bruger: `{username}`\n"
        f"🕒 Tid: {now} CET\n"
        f"📍 IP: `{ip}`{loc}\n"
        f"❗ Årsag: {reason}\n\n"
        f"_Hvis det ikke var dig — overvej at skifte adgangskode._"
    )
    _send_telegram(text)


def notify_geo_blocked(username: str, ip: str, location: str | None = None) -> None:
    """Send notifikation når login blokeres af geo-lock."""
    now = datetime.now(TZ_CET).strftime("%d/%m %H:%M")
    loc = f" ({location})" if location else ""
    text = (
        f"🔴 *Login BLOKERET af geo-lock*\n\n"
        f"👤 Bruger: `{username}`\n"
        f"🕒 Tid: {now} CET\n"
        f"📍 IP: `{ip}`{loc}\n\n"
        f"_Login forsøgt fra ukendt land. "
        f"Hvis det var dig — brug GEO_LOCK_BYPASS-koden i .env._"
    )
    _send_telegram(text)
