"""
Geo-lock for Alpha Trader login.

Blokerer login fra IP-adresser udenfor godkendte lande (default: kun DK).

Bypass-mekanisme: en hemmelig kode i `.env` kan bruges til at logge ind
fra hvor som helst (f.eks. når Ole rejser). Bypassen logges + sendes til
Telegram så han altid ved hvis nogen bruger den.

Konfiguration via .env:
    GEO_LOCK_ENABLED=true        # eller false for at slå fra
    GEO_LOCK_COUNTRIES=DK,SE,NO  # ISO-koder, komma-separeret
    GEO_LOCK_BYPASS=hemmelig-kode-min-32-chars
"""
from __future__ import annotations

import json
import os
import ssl
import urllib.parse
import urllib.request
from dataclasses import dataclass
from threading import RLock

from loguru import logger

try:
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except Exception:
    _SSL_CTX = ssl.create_default_context()


@dataclass
class GeoInfo:
    """Resultat af IP-lookup."""

    ip: str
    country_code: str | None  # ISO-2 (DK, SE, …) eller None hvis lookup fejlede
    country_name: str | None
    city: str | None

    @property
    def location_str(self) -> str:
        parts = [p for p in (self.city, self.country_name) if p]
        return ", ".join(parts) if parts else "ukendt sted"


class GeoLockService:
    """In-process IP-til-land cache med konfigurerbar policy."""

    def __init__(self) -> None:
        self._cache: dict[str, GeoInfo] = {}
        self._lock = RLock()

    @property
    def enabled(self) -> bool:
        return os.getenv("GEO_LOCK_ENABLED", "false").strip().lower() in ("1", "true", "yes")

    @property
    def allowed_countries(self) -> list[str]:
        raw = os.getenv("GEO_LOCK_COUNTRIES", "DK").strip()
        return [c.strip().upper() for c in raw.split(",") if c.strip()]

    @property
    def bypass_code(self) -> str:
        return os.getenv("GEO_LOCK_BYPASS", "").strip()

    def lookup(self, ip: str) -> GeoInfo:
        """
        Slå et IP-til-land op via ip-api.com (gratis, ingen API-key, 45 req/min).

        Loopback (127.x, ::1) og private IP'er behandles som lokal/DK.
        """
        if not ip or self._is_local(ip):
            return GeoInfo(ip=ip, country_code="DK", country_name="Lokal", city=None)

        with self._lock:
            cached = self._cache.get(ip)
            if cached is not None:
                return cached

        info = self._do_lookup(ip)
        with self._lock:
            self._cache[ip] = info
        return info

    def is_allowed(self, ip: str, bypass_attempt: str | None = None) -> tuple[bool, GeoInfo, bool]:
        """
        Tjek om en IP-adresse må logge ind.

        Returns:
            (allowed, geo_info, used_bypass)
        """
        info = self.lookup(ip)

        # Ikke-aktiveret: tillad alt
        if not self.enabled:
            return True, info, False

        # Bypass-kode brugt og match
        bypass = self.bypass_code
        if bypass and bypass_attempt and bypass_attempt == bypass:
            return True, info, True

        # Hvis vi ikke kunne lookup'e, så fail-OPEN (ellers låser vi os ude
        # ved netværksproblemer). Logges dog som warning.
        if info.country_code is None:
            logger.warning(f"[geo] IP-lookup fejlede for {ip} — fail-open")
            return True, info, False

        return info.country_code in self.allowed_countries, info, False

    @staticmethod
    def _is_local(ip: str) -> bool:
        if ip in ("127.0.0.1", "::1", "localhost"):
            return True
        if ip.startswith("192.168.") or ip.startswith("10."):
            return True
        if ip.startswith("172."):
            try:
                second = int(ip.split(".")[1])
                return 16 <= second <= 31
            except Exception:
                return False
        return False

    def _do_lookup(self, ip: str) -> GeoInfo:
        """Gør den faktiske HTTP-call. Bruges også af tests (kan mockes)."""
        try:
            url = f"http://ip-api.com/json/{urllib.parse.quote(ip)}?fields=status,country,countryCode,city"
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=3, context=_SSL_CTX) as resp:
                data = json.loads(resp.read())
            if data.get("status") == "success":
                return GeoInfo(
                    ip=ip,
                    country_code=data.get("countryCode"),
                    country_name=data.get("country"),
                    city=data.get("city"),
                )
        except Exception as exc:
            logger.debug(f"[geo] Lookup-fejl for {ip}: {exc}")
        return GeoInfo(ip=ip, country_code=None, country_name=None, city=None)


_service: GeoLockService | None = None


def get_geo_lock_service() -> GeoLockService:
    global _service
    if _service is None:
        _service = GeoLockService()
    return _service


def extract_client_ip(request) -> str:
    """
    Hent klient-IP fra FastAPI Request-objektet.

    Respekterer X-Forwarded-For (når der er en reverse proxy foran), men
    falder tilbage til socket-IP hvis headeren ikke er sat.
    """
    fwd = request.headers.get("x-forwarded-for", "")
    if fwd:
        # Første IP i kæden er den oprindelige klient
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"
