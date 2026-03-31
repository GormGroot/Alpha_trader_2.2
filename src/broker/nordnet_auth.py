"""
Nordnet Session Manager — uofficiel API auth.

Nordnet har INGEN officiel API for private kunder.
Denne integration bruger Nordnet's interne web-API med session cookies.

VIGTIGT:
  - "Best effort" integration — kan bryde ved Nordnet platform-opdateringer
  - Saxo Bank er backup for nordiske aktier
  - Max 1 request/sec for at undgå ban
  - User-Agent header der ligner browser
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from loguru import logger


# ── Configuration ───────────────────────────────────────────

@dataclass
class NordnetConfig:
    """Nordnet konfiguration fra .env."""
    username: str = ""
    password: str = ""
    market: str = "dk"  # dk, se, no, fi

    @property
    def base_url(self) -> str:
        return f"https://www.nordnet.{self.market}"

    @property
    def api_base(self) -> str:
        return f"{self.base_url}/api/2"

    @classmethod
    def from_env(cls) -> NordnetConfig:
        return cls(
            username=os.environ.get("NORDNET_USERNAME", ""),
            password=os.environ.get("NORDNET_PASSWORD", ""),
            market=os.environ.get("NORDNET_MARKET", "dk"),
        )


# ── Rate Limiter ────────────────────────────────────────────

class _NordnetRateLimiter:
    """Aggressiv rate limiter: max 1 req/sec."""

    def __init__(self, min_interval: float = 1.0) -> None:
        self._min_interval = min_interval
        self._last_request = 0.0

    def wait(self) -> None:
        now = time.time()
        elapsed = now - self._last_request
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request = time.time()


# ── Session Manager ─────────────────────────────────────────

class NordnetSession:
    """
    Nordnet session-baseret authentication.

    Login via Nordnet's interne login endpoint.
    Holder session-cookie alive med periodic pings.

    Brug:
        config = NordnetConfig.from_env()
        session = NordnetSession(config)
        session.login()

        # Alle requests via session
        data = session.get("/accounts")
    """

    # Browser-lignende headers
    _DEFAULT_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json",
        "Accept-Language": "da-DK,da;q=0.9,en-US;q=0.8,en;q=0.7",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "client-id": "NEXT",
    }

    # Session timeout (Nordnet sessioner udløber efter ~30 min inaktivitet)
    SESSION_TIMEOUT_MINUTES = 25
    # Keepalive interval
    KEEPALIVE_MINUTES = 5

    def __init__(self, config: NordnetConfig | None = None) -> None:
        self._config = config or NordnetConfig.from_env()
        self._rate_limiter = _NordnetRateLimiter(min_interval=1.0)
        self._session: Any = None
        self._account_id: str = ""
        self._logged_in = False
        self._login_time: datetime | None = None
        self._last_activity: datetime | None = None

    # ── Login ───────────────────────────────────────────────

    def login(self) -> dict:
        """
        Log ind på Nordnet.

        Returns:
            Dict med session info og account ID.

        Raises:
            NordnetAuthError: Ved login-fejl.
        """
        import requests

        if not self._config.username or not self._config.password:
            raise NordnetAuthError(
                "NORDNET_USERNAME og NORDNET_PASSWORD skal sættes i .env"
            )

        self._session = requests.Session()
        self._session.headers.update(self._DEFAULT_HEADERS)

        # Login endpoint
        login_url = f"{self._config.api_base}/authentication/basic/login"

        self._rate_limiter.wait()
        try:
            # NB: form-data sendes over HTTPS. Sæt IKKE urllib3 debug-logging i produktion.
            response = self._session.post(
                login_url,
                data={
                    "username": self._config.username,
                    "password": self._config.password,
                },
                timeout=15,
            )

            if response.status_code == 401:
                raise NordnetAuthError("Forkert brugernavn/adgangskode")
            if response.status_code >= 400:
                raise NordnetAuthError(
                    f"Login fejl ({response.status_code}): {response.text[:200]}"
                )

            self._logged_in = True
            self._login_time = datetime.now()
            self._last_activity = datetime.now()

            logger.info("[nordnet] Logged in successfully")

            # Hent accounts
            accounts = self._fetch_accounts()
            if accounts:
                self._account_id = str(accounts[0].get("accid", ""))

            return {
                "logged_in": True,
                "account_id": self._account_id,
                "market": self._config.market,
            }

        except NordnetAuthError:
            raise
        except Exception as exc:
            raise NordnetAuthError(f"Login fejl: {exc}") from exc

    def _fetch_accounts(self) -> list[dict]:
        """Hent accounts efter login."""
        data = self.get("/accounts")
        if isinstance(data, list):
            return data
        return data.get("Data", data.get("accounts", []))

    # ── Session Management ──────────────────────────────────

    def is_logged_in(self) -> bool:
        """Er sessionen aktiv?"""
        if not self._logged_in or not self._login_time:
            return False

        # Check session timeout
        if self._last_activity:
            elapsed = datetime.now() - self._last_activity
            if elapsed > timedelta(minutes=self.SESSION_TIMEOUT_MINUTES):
                logger.info("[nordnet] Session timeout — re-login påkrævet")
                self._logged_in = False
                return False

        return True

    def ensure_logged_in(self) -> None:
        """Sikr at vi er logget ind (re-login ved behov)."""
        if not self.is_logged_in():
            logger.info("[nordnet] Session expired — logger ind igen...")
            self.login()

    def keepalive(self) -> None:
        """Ping for at holde session alive."""
        if self.is_logged_in():
            try:
                self.get("/accounts")
            except Exception:
                pass

    # ── HTTP Methods ────────────────────────────────────────

    def get(self, endpoint: str, params: dict | None = None) -> Any:
        """GET request til Nordnet API."""
        return self._request("GET", endpoint, params=params)

    def post(self, endpoint: str, data: dict | None = None) -> Any:
        """POST request til Nordnet API."""
        return self._request("POST", endpoint, json_data=data)

    def delete(self, endpoint: str) -> Any:
        """DELETE request til Nordnet API."""
        return self._request("DELETE", endpoint)

    def _request(
        self,
        method: str,
        endpoint: str,
        params: dict | None = None,
        json_data: dict | None = None,
    ) -> Any:
        """Authenticated request til Nordnet."""
        self.ensure_logged_in()
        self._rate_limiter.wait()

        if not self._session:
            raise NordnetAuthError("Ingen session — login påkrævet")

        url = f"{self._config.api_base}{endpoint}"

        try:
            response = self._session.request(
                method=method,
                url=url,
                params=params,
                json=json_data,
                timeout=10,
            )

            self._last_activity = datetime.now()

            if response.status_code == 401:
                # Session expired — re-login (with guard against infinite recursion)
                now = datetime.now()
                last_relogin = getattr(self, '_last_relogin_attempt', None)
                if last_relogin and (now - last_relogin) < timedelta(seconds=30):
                    raise NordnetAuthError(
                        "Nordnet 401 re-login failed — last attempt was less than 30s ago"
                    )
                self._last_relogin_attempt = now
                logger.info("[nordnet] 401 — re-logger ind...")
                self._logged_in = False
                self.login()
                return self._request(method, endpoint, params, json_data)

            if response.status_code >= 400:
                from src.broker.models import BrokerError
                raise BrokerError(
                    f"Nordnet API {response.status_code}: "
                    f"{response.text[:300]}"
                )

            if response.status_code == 204:
                return {}

            return response.json()

        except NordnetAuthError:
            raise
        except Exception as exc:
            from src.broker.models import BrokerError
            raise BrokerError(f"Nordnet request fejl: {exc}") from exc

    # ── Properties ──────────────────────────────────────────

    @property
    def account_id(self) -> str:
        return self._account_id

    def status(self) -> dict[str, Any]:
        return {
            "logged_in": self.is_logged_in(),
            "account_id": self._account_id,
            "market": self._config.market,
            "login_time": (
                self._login_time.isoformat() if self._login_time else None
            ),
            "last_activity": (
                self._last_activity.isoformat() if self._last_activity else None
            ),
        }

    def logout(self) -> None:
        """Log ud."""
        if self._session and self._logged_in:
            try:
                self._rate_limiter.wait()
                self._session.delete(
                    f"{self._config.api_base}/authentication"
                )
            except Exception:
                pass
        self._logged_in = False
        self._session = None
        logger.info("[nordnet] Logged out")


# ── Exception ───────────────────────────────────────────────

class NordnetAuthError(Exception):
    """Nordnet authentication fejl."""
