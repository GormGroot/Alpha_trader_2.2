"""
Two-Factor Authentication (2FA) via Telegram.

Flow:
  1. Bruger logger ind med username + password → modtager login_token (5 min levetid)
  2. Kode på 6 cifre sendes til brugerens Telegram-bot
  3. Bruger indtaster koden → modtager access_token (24 timers levetid)

Sikkerhed:
  - Koden er gyldig i 5 minutter
  - Maks 3 forsøg per kode
  - Rate-limit: 1 kode per 30 sekunder
  - login_token er ENGANGSBRUG og kasseres efter validering
"""
from __future__ import annotations

import os
import secrets
import ssl
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import RLock

from loguru import logger

try:
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except Exception:
    _SSL_CTX = ssl.create_default_context()

# ── Konfiguration ─────────────────────────────────────────────
CODE_LENGTH = 6
CODE_TTL_SECONDS = 300  # 5 minutter
MAX_ATTEMPTS = 3
RATE_LIMIT_SECONDS = 30  # min interval mellem koder
LOGIN_TOKEN_TTL_SECONDS = 300


@dataclass
class PendingLogin:
    """Et login der venter på 2FA-bekræftelse."""

    username: str
    code: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    attempts: int = 0
    consumed: bool = False

    @property
    def is_expired(self) -> bool:
        return datetime.utcnow() > self.created_at + timedelta(seconds=CODE_TTL_SECONDS)


class TwoFactorService:
    """In-memory 2FA-service. Trådsikker."""

    def __init__(self) -> None:
        self._pending: dict[str, PendingLogin] = {}  # login_token → PendingLogin
        self._last_code_time: dict[str, float] = {}  # username → unix-tid
        self._lock = RLock()

    @property
    def is_telegram_configured(self) -> bool:
        return bool(os.getenv("TELEGRAM_BOT_TOKEN") and os.getenv("TELEGRAM_CHAT_ID"))

    def request_code(self, username: str) -> tuple[str, str] | None:
        """
        Opret en ny 2FA-kode og send den via Telegram.

        Returns:
            (login_token, masked_destination) hvis koden blev sendt.
            None hvis Telegram ikke er konfigureret eller rate-limit ramt.
        """
        with self._lock:
            # Rate limit
            now = time.time()
            last = self._last_code_time.get(username, 0)
            if now - last < RATE_LIMIT_SECONDS:
                wait = int(RATE_LIMIT_SECONDS - (now - last))
                logger.warning(f"[2fa] Rate limit for {username}: vent {wait}s")
                return None

            if not self.is_telegram_configured:
                logger.error("[2fa] Telegram ikke konfigureret — kan ikke sende kode")
                return None

            # Generer kode + login_token
            code = "".join(secrets.choice("0123456789") for _ in range(CODE_LENGTH))
            login_token = secrets.token_urlsafe(32)
            self._pending[login_token] = PendingLogin(username=username, code=code)
            self._last_code_time[username] = now

            # Ryd op i udløbne tokens
            self._cleanup_expired()

            # Send via Telegram
            if self._send_telegram_code(code):
                masked = self._mask_chat_id(os.getenv("TELEGRAM_CHAT_ID", ""))
                logger.info(f"[2fa] Kode sendt til Telegram for {username}")
                return login_token, masked

            # Hvis Telegram fejlede — fjern det pending-token
            del self._pending[login_token]
            return None

    def verify_code(self, login_token: str, code: str) -> str | None:
        """
        Verificér en 2FA-kode.

        Returns:
            username hvis koden er korrekt og ikke udløbet.
            None hvis ugyldig.
        """
        with self._lock:
            pending = self._pending.get(login_token)
            if pending is None:
                logger.warning("[2fa] Ukendt login_token")
                return None

            if pending.consumed:
                logger.warning(f"[2fa] Token allerede brugt for {pending.username}")
                return None

            if pending.is_expired:
                logger.warning(f"[2fa] Kode udløbet for {pending.username}")
                del self._pending[login_token]
                return None

            pending.attempts += 1
            if pending.attempts > MAX_ATTEMPTS:
                logger.warning(f"[2fa] Max forsøg overskredet for {pending.username}")
                del self._pending[login_token]
                return None

            if not secrets.compare_digest(pending.code, code.strip()):
                remaining = MAX_ATTEMPTS - pending.attempts
                logger.warning(
                    f"[2fa] Forkert kode for {pending.username} "
                    f"(forsøg {pending.attempts}/{MAX_ATTEMPTS}, {remaining} tilbage)"
                )
                return None

            # Korrekt — markér som brugt og returnér
            pending.consumed = True
            del self._pending[login_token]
            logger.info(f"[2fa] Login bekræftet for {pending.username}")
            return pending.username

    def _send_telegram_code(self, code: str) -> bool:
        """Send 6-cifret kode via Telegram-bot."""
        token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
        if not (token and chat_id):
            return False

        text = (
            f"🔐 *Alpha Trader login*\n\n"
            f"Din kode: `{code}`\n\n"
            f"_Gyldig i 5 minutter — del aldrig denne kode._"
        )
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
            logger.error(f"[2fa] Telegram send fejlede: {exc}")
            return False

    def _cleanup_expired(self) -> None:
        """Fjern udløbne pending logins (kaldes med lock holdt)."""
        expired = [t for t, p in self._pending.items() if p.is_expired]
        for t in expired:
            del self._pending[t]

    @staticmethod
    def _mask_chat_id(chat_id: str) -> str:
        """Returnér en maskeret version af chat-id (f.eks. '••••6183')."""
        if len(chat_id) < 4:
            return "••••"
        return "•" * (len(chat_id) - 4) + chat_id[-4:]


# Module-level singleton
_service: TwoFactorService | None = None


def get_two_factor_service() -> TwoFactorService:
    global _service
    if _service is None:
        _service = TwoFactorService()
    return _service
