"""
Saxo Bank OAuth2 Token Manager.

Håndterer:
  - OAuth2 authorization code flow
  - Access token refresh (udløber efter 20 min)
  - Refresh token tracking (udløber efter 24 timer)
  - Encrypted token storage (.saxo_tokens)
  - Background refresh scheduler
  - Alerts ved token-udløb
"""

from __future__ import annotations

import base64
import json
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlencode

from loguru import logger


# ── Configuration ───────────────────────────────────────────

@dataclass
class SaxoConfig:
    """Saxo OpenAPI konfiguration — læses fra .env."""
    app_key: str = ""
    app_secret: str = ""
    redirect_uri: str = "http://localhost:8080/callback"
    environment: str = "sim"  # "sim" eller "live"
    account_key: str = ""

    # Auto-populated
    @property
    def base_url(self) -> str:
        if self.environment == "live":
            return "https://gateway.saxobank.com/openapi"
        return "https://gateway.saxobank.com/sim/openapi"

    @property
    def auth_url(self) -> str:
        if self.environment == "live":
            return "https://live.logonvalidation.net/authorize"
        return "https://sim.logonvalidation.net/authorize"

    @property
    def token_url(self) -> str:
        if self.environment == "live":
            return "https://live.logonvalidation.net/token"
        return "https://sim.logonvalidation.net/token"

    @classmethod
    def from_env(cls) -> SaxoConfig:
        """Læs konfiguration fra environment variables."""
        return cls(
            app_key=os.environ.get("SAXO_APP_KEY", ""),
            app_secret=os.environ.get("SAXO_APP_SECRET", ""),
            redirect_uri=os.environ.get(
                "SAXO_REDIRECT_URI", "http://localhost:8080/callback"
            ),
            environment=os.environ.get("SAXO_ENVIRONMENT", "sim"),
            account_key=os.environ.get("SAXO_ACCOUNT_KEY", ""),
        )


# ── Token Storage ───────────────────────────────────────────

@dataclass
class TokenData:
    """Saxo API tokens med metadata."""
    access_token: str = ""
    refresh_token: str = ""
    token_type: str = "Bearer"
    access_expires_at: str = ""     # ISO timestamp
    refresh_expires_at: str = ""    # ISO timestamp
    account_key: str = ""
    client_key: str = ""
    last_refresh: str = ""

    def is_access_expired(self) -> bool:
        """Er access token udløbet?"""
        if not self.access_expires_at:
            return True
        try:
            expires = datetime.fromisoformat(self.access_expires_at)
            # Refresh 2 minutter før udløb for safety margin
            return datetime.now() >= (expires - timedelta(minutes=2))
        except ValueError:
            return True

    def is_refresh_expired(self) -> bool:
        """Er refresh token udløbet?"""
        if not self.refresh_expires_at:
            return True
        try:
            expires = datetime.fromisoformat(self.refresh_expires_at)
            return datetime.now() >= expires
        except ValueError:
            return True

    def access_token_remaining(self) -> timedelta:
        """Tid til access token udløber."""
        if not self.access_expires_at:
            return timedelta(0)
        try:
            expires = datetime.fromisoformat(self.access_expires_at)
            remaining = expires - datetime.now()
            return max(remaining, timedelta(0))
        except ValueError:
            return timedelta(0)

    def refresh_token_remaining(self) -> timedelta:
        """Tid til refresh token udløber."""
        if not self.refresh_expires_at:
            return timedelta(0)
        try:
            expires = datetime.fromisoformat(self.refresh_expires_at)
            remaining = expires - datetime.now()
            return max(remaining, timedelta(0))
        except ValueError:
            return timedelta(0)


try:
    from cryptography.fernet import Fernet
    _HAS_FERNET = True
except ImportError:
    _HAS_FERNET = False
    logger.warning(
        "[saxo-auth] cryptography package not installed — "
        "token encryption disabled. Install with: pip install cryptography"
    )


def _derive_fernet_key(secret: str) -> bytes:
    """Derive a Fernet-compatible 32-byte key from an arbitrary secret string."""
    import hashlib
    digest = hashlib.sha256(secret.encode()).digest()
    return base64.urlsafe_b64encode(digest)


class TokenStore:
    """
    Encrypted token storage i lokal fil.

    Uses cryptography.fernet.Fernet with a key derived from the app secret
    (or machine-id as fallback). Falls back to base64-only if cryptography
    is not installed.
    """

    def __init__(self, path: str = ".saxo_tokens", key: str = "") -> None:
        self._path = Path(path)
        secret = key or self._get_machine_id() or "alpha-vision-saxo-default-key"
        self._fernet_key = _derive_fernet_key(secret)
        if _HAS_FERNET:
            self._fernet = Fernet(self._fernet_key)
        else:
            self._fernet = None

    @staticmethod
    def _get_machine_id() -> str:
        """Try to read a stable machine identifier."""
        for path in ("/etc/machine-id", "/var/lib/dbus/machine-id"):
            try:
                return Path(path).read_text().strip()
            except OSError:
                continue
        return ""

    def save(self, tokens: TokenData) -> None:
        """Gem tokens til disk (encrypted)."""
        data = json.dumps({
            "access_token": tokens.access_token,
            "refresh_token": tokens.refresh_token,
            "token_type": tokens.token_type,
            "access_expires_at": tokens.access_expires_at,
            "refresh_expires_at": tokens.refresh_expires_at,
            "account_key": tokens.account_key,
            "client_key": tokens.client_key,
            "last_refresh": tokens.last_refresh,
        })
        encoded = self._encrypt(data)
        self._path.write_text(encoded)
        # chmod 600 — owner read/write only
        self._path.chmod(0o600)
        logger.debug(f"[saxo-auth] Tokens gemt til {self._path}")

    def load(self) -> TokenData | None:
        """Load tokens fra disk."""
        if not self._path.exists():
            return None
        try:
            encoded = self._path.read_text()
            data = json.loads(self._decrypt(encoded))
            return TokenData(**data)
        except Exception as exc:
            logger.warning(f"[saxo-auth] Kunne ikke læse tokens: {exc}")
            return None

    def delete(self) -> None:
        """Slet token-fil."""
        if self._path.exists():
            self._path.unlink()

    def _encrypt(self, data: str) -> str:
        """Encrypt data using Fernet (falls back to base64 if unavailable)."""
        if self._fernet is not None:
            return self._fernet.encrypt(data.encode()).decode()
        # Fallback: base64 only — WARNING: tokens er IKKE krypteret!
        logger.warning("[saxo-auth] Fernet ikke tilgængelig — tokens gemmes UKRYPTERET (base64). Installér cryptography-pakken.")
        return base64.b64encode(data.encode()).decode()

    def _decrypt(self, encoded: str) -> str:
        """Decrypt data using Fernet (falls back to base64 if unavailable)."""
        if self._fernet is not None:
            return self._fernet.decrypt(encoded.encode()).decode()
        # Fallback: base64 only — WARNING
        return base64.b64decode(encoded.encode()).decode()


# ── Saxo Auth Manager ──────────────────────────────────────

class SaxoAuthManager:
    """
    Saxo Bank OAuth2 authentication manager.

    Workflow:
        1. auth = SaxoAuthManager(config)
        2. Første gang: url = auth.get_authorization_url()
           → Åbn URL i browser, login, kopier code
        3. auth.exchange_code(code)
        4. Herefter: auth.get_access_token() auto-refresher

    Brug:
        config = SaxoConfig.from_env()
        auth = SaxoAuthManager(config)

        # Check om vi har gyldige tokens
        if auth.is_authenticated():
            token = auth.get_access_token()
            headers = {"Authorization": f"Bearer {token}"}
        else:
            print(f"Login required: {auth.get_authorization_url()}")
    """

    # Refresh access token 3 minutter før udløb
    REFRESH_MARGIN_SECONDS = 180
    # Alert når refresh token har < 2 timer
    REFRESH_TOKEN_ALERT_HOURS = 2

    def __init__(
        self,
        config: SaxoConfig | None = None,
        token_path: str = ".saxo_tokens",
    ) -> None:
        self._config = config or SaxoConfig.from_env()
        self._store = TokenStore(
            path=token_path,
            key=self._config.app_secret or "default",
        )
        self._tokens: TokenData | None = self._store.load()
        self._on_alert_callbacks: list[Callable[[str], None]] = []
        self._refresh_thread: threading.Thread | None = None
        self._running = False

    # ── OAuth2 Flow ─────────────────────────────────────────

    def get_authorization_url(self) -> str:
        """
        Generér Saxo login URL.

        Brugeren skal åbne denne i en browser, logge ind,
        og returnere authorization code.
        """
        params = urlencode({
            "response_type": "code",
            "client_id": self._config.app_key,
            "redirect_uri": self._config.redirect_uri,
            "state": f"alpha-vision-{int(time.time())}",
        })
        return f"{self._config.auth_url}?{params}"

    def exchange_code(self, authorization_code: str) -> TokenData:
        """
        Exchange authorization code for tokens.

        Args:
            authorization_code: Code fra OAuth2 callback.

        Returns:
            TokenData med access og refresh tokens.
        """
        import requests

        response = requests.post(
            self._config.token_url,
            data={
                "grant_type": "authorization_code",
                "code": authorization_code,
                "redirect_uri": self._config.redirect_uri,
                "client_id": self._config.app_key,
                "client_secret": self._config.app_secret,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        return self._process_token_response(data)

    def refresh_access_token(self) -> TokenData:
        """
        Refresh access token via refresh token.

        Returns:
            Opdateret TokenData.

        Raises:
            AuthError: Hvis refresh fejler.
        """
        if not self._tokens or not self._tokens.refresh_token:
            raise AuthError("Ingen refresh token — login påkrævet")

        if self._tokens.is_refresh_expired():
            raise AuthError(
                "Refresh token er udløbet — ny login påkrævet. "
                f"Brug: {self.get_authorization_url()}"
            )

        import requests

        try:
            response = requests.post(
                self._config.token_url,
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": self._tokens.refresh_token,
                    "client_id": self._config.app_key,
                    "client_secret": self._config.app_secret,
                    "redirect_uri": self._config.redirect_uri,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            return self._process_token_response(data)

        except Exception as exc:
            logger.error(f"[saxo-auth] Token refresh fejlet: {exc}")
            raise AuthError(f"Token refresh fejlet: {exc}") from exc

    def _process_token_response(self, data: dict) -> TokenData:
        """Process token response fra Saxo."""
        now = datetime.now()

        # Access token expires efter ~20 min
        access_expires = data.get("expires_in", 1200)  # default 20 min
        # Refresh token expires efter ~24 timer
        refresh_expires = data.get("refresh_token_expires_in", 86400)

        tokens = TokenData(
            access_token=data["access_token"],
            refresh_token=data.get("refresh_token", ""),
            token_type=data.get("token_type", "Bearer"),
            access_expires_at=(
                now + timedelta(seconds=access_expires)
            ).isoformat(),
            refresh_expires_at=(
                now + timedelta(seconds=refresh_expires)
            ).isoformat(),
            account_key=self._config.account_key,
            last_refresh=now.isoformat(),
        )

        self._tokens = tokens
        self._store.save(tokens)

        logger.info(
            f"[saxo-auth] Tokens opdateret — "
            f"access expires om {access_expires // 60} min, "
            f"refresh expires om {refresh_expires // 3600} timer"
        )

        return tokens

    # ── Token Access ────────────────────────────────────────

    def get_access_token(self) -> str:
        """
        Hent gyldig access token (refresher automatisk hvis nødvendigt).

        Returns:
            Access token string.

        Raises:
            AuthError: Hvis ikke authenticated.
        """
        if not self._tokens:
            raise AuthError("Ikke authenticated — kør login flow først")

        if self._tokens.is_access_expired():
            logger.info("[saxo-auth] Access token expired — refresher...")
            self.refresh_access_token()

        return self._tokens.access_token

    def get_headers(self) -> dict[str, str]:
        """Hent HTTP headers med auth token."""
        token = self.get_access_token()
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

    def is_authenticated(self) -> bool:
        """Er vi authenticated med gyldige tokens?"""
        if not self._tokens:
            return False
        return not self._tokens.is_refresh_expired()

    # ── Background Refresh ──────────────────────────────────

    def start_auto_refresh(self) -> None:
        """Start baggrunds-thread der auto-refresher access token."""
        if self._running:
            return

        self._running = True

        def _refresh_loop() -> None:
            logger.info("[saxo-auth] Auto-refresh startet")
            while self._running:
                try:
                    if self._tokens and not self._tokens.is_refresh_expired():
                        # Refresh access token hvis det nærmer sig udløb
                        remaining = self._tokens.access_token_remaining()
                        if remaining.total_seconds() < self.REFRESH_MARGIN_SECONDS:
                            self.refresh_access_token()

                        # Alert hvis refresh token nærmer sig udløb
                        refresh_remaining = self._tokens.refresh_token_remaining()
                        if (refresh_remaining.total_seconds()
                                < self.REFRESH_TOKEN_ALERT_HOURS * 3600):
                            self._alert(
                                f"Saxo refresh token udløber om "
                                f"{refresh_remaining.total_seconds() / 3600:.1f} timer. "
                                f"Ny login påkrævet snart!"
                            )
                except Exception as exc:
                    logger.error(f"[saxo-auth] Auto-refresh fejl: {exc}")

                # Check hvert 60. sekund
                for _ in range(60):
                    if not self._running:
                        break
                    time.sleep(1)

            logger.info("[saxo-auth] Auto-refresh stoppet")

        self._refresh_thread = threading.Thread(
            target=_refresh_loop,
            name="saxo-token-refresh",
            daemon=True,
        )
        self._refresh_thread.start()

    def stop_auto_refresh(self) -> None:
        """Stop auto-refresh thread."""
        self._running = False
        if self._refresh_thread:
            self._refresh_thread.join(timeout=5)
            self._refresh_thread = None

    # ── Alerts ──────────────────────────────────────────────

    def on_alert(self, callback: Callable[[str], None]) -> None:
        """Registrér alert callback."""
        self._on_alert_callbacks.append(callback)

    def _alert(self, message: str) -> None:
        logger.warning(f"[saxo-auth] ALERT: {message}")
        for cb in self._on_alert_callbacks:
            try:
                cb(message)
            except Exception:
                pass

    # ── Status ──────────────────────────────────────────────

    def status(self) -> dict[str, Any]:
        """Auth status til dashboard."""
        if not self._tokens:
            return {
                "authenticated": False,
                "login_url": self.get_authorization_url(),
            }

        return {
            "authenticated": self.is_authenticated(),
            "access_token_remaining": str(self._tokens.access_token_remaining()),
            "refresh_token_remaining": str(self._tokens.refresh_token_remaining()),
            "access_expired": self._tokens.is_access_expired(),
            "refresh_expired": self._tokens.is_refresh_expired(),
            "last_refresh": self._tokens.last_refresh,
            "auto_refresh_active": self._running,
        }


# ── Exceptions ──────────────────────────────────────────────

class AuthError(Exception):
    """Saxo authentication fejl."""
