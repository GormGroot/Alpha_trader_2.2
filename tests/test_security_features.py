"""
Integrationstests for nye sikkerhedsfeatures:
  - Geo-lock (allowed/blocked countries + bypass)
  - Login-notifikationer (Telegram)
  - Auto-logout via JWT TTL (token expires_in)
"""
from __future__ import annotations

import os
import time

import pytest
from fastapi.testclient import TestClient


os.environ.setdefault("APP_USERNAME", "testuser")
os.environ.setdefault("APP_PASSWORD", "testpass123")
os.environ.setdefault("APP_SECRET_KEY", "test-secret-key-32-chars-minimum-x" * 2)


@pytest.fixture(autouse=True)
def isolate_env(monkeypatch):
    """Standard test-state: ingen Telegram, ingen geo-lock."""
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    monkeypatch.delenv("GEO_LOCK_ENABLED", raising=False)
    monkeypatch.delenv("GEO_LOCK_COUNTRIES", raising=False)
    monkeypatch.delenv("GEO_LOCK_BYPASS", raising=False)
    monkeypatch.delenv("APP_TOKEN_TTL_MINUTES", raising=False)
    # Reset 2FA service så env-vars re-evalueres
    from src.api import two_factor
    two_factor._service = None
    # Reset geo-service
    from src.api import geo_lock
    geo_lock._service = None
    yield


@pytest.fixture
def client():
    from src.api import server as server_mod
    return TestClient(server_mod.app)


@pytest.fixture
def disable_notifications(monkeypatch):
    """Mock alle notifikationsfunktioner så de ikke prøver at nå Telegram."""
    sent = {"success": [], "failed": [], "geo_blocked": []}

    def fake_success(u, ip, loc=None):
        sent["success"].append({"user": u, "ip": ip, "loc": loc})

    def fake_failed(u, ip, reason, loc=None):
        sent["failed"].append({"user": u, "ip": ip, "reason": reason, "loc": loc})

    def fake_geo(u, ip, loc=None):
        sent["geo_blocked"].append({"user": u, "ip": ip, "loc": loc})

    from src.api import server as server_mod
    monkeypatch.setattr(server_mod, "notify_login_success", fake_success)
    monkeypatch.setattr(server_mod, "notify_login_failed", fake_failed)
    monkeypatch.setattr(server_mod, "notify_geo_blocked", fake_geo)
    return sent


# ── Geo-lock ───────────────────────────────────────────────────


class TestGeoLock:
    def test_disabled_allows_all_ips(self, client, disable_notifications, monkeypatch):
        """Geo-lock OFF → alle IPs tilladt."""
        monkeypatch.delenv("GEO_LOCK_ENABLED", raising=False)
        from src.api import geo_lock
        geo_lock._service = None

        res = client.post(
            "/api/auth/login",
            json={"username": "testuser", "password": "testpass123"},
        )
        assert res.status_code == 200

    def test_localhost_always_allowed(self, client, disable_notifications, monkeypatch):
        """TestClient bruger 127.0.0.1 → altid tilladt selv med geo-lock ON."""
        monkeypatch.setenv("GEO_LOCK_ENABLED", "true")
        monkeypatch.setenv("GEO_LOCK_COUNTRIES", "DK")
        from src.api import geo_lock
        geo_lock._service = None

        res = client.post(
            "/api/auth/login",
            json={"username": "testuser", "password": "testpass123"},
        )
        assert res.status_code == 200

    def test_foreign_ip_blocked(self, client, disable_notifications, monkeypatch):
        """Mock geo-lookup til at sige 'US' → blokeres når policy er DK."""
        monkeypatch.setenv("GEO_LOCK_ENABLED", "true")
        monkeypatch.setenv("GEO_LOCK_COUNTRIES", "DK")

        from src.api import geo_lock
        geo_lock._service = None

        # Mock _do_lookup til at returnere US uanset IP
        from src.api.geo_lock import GeoInfo
        def fake_lookup(self, ip):
            return GeoInfo(ip=ip, country_code="US", country_name="United States", city="New York")

        monkeypatch.setattr(geo_lock.GeoLockService, "_do_lookup", fake_lookup)

        # Send via X-Forwarded-For så IP'en ikke er local
        res = client.post(
            "/api/auth/login",
            headers={"X-Forwarded-For": "8.8.8.8"},
            json={"username": "testuser", "password": "testpass123"},
        )
        assert res.status_code == 403
        assert "blokeret" in res.json()["detail"].lower()
        assert len(disable_notifications["geo_blocked"]) == 1

    def test_bypass_code_unlocks(self, client, disable_notifications, monkeypatch):
        """Korrekt bypass-kode → tillader login fra blokeret land."""
        monkeypatch.setenv("GEO_LOCK_ENABLED", "true")
        monkeypatch.setenv("GEO_LOCK_COUNTRIES", "DK")
        monkeypatch.setenv("GEO_LOCK_BYPASS", "rejser-2026")

        from src.api import geo_lock
        from src.api.geo_lock import GeoInfo
        geo_lock._service = None

        def fake_lookup(self, ip):
            return GeoInfo(ip=ip, country_code="ES", country_name="Spain", city="Barcelona")

        monkeypatch.setattr(geo_lock.GeoLockService, "_do_lookup", fake_lookup)

        # Uden bypass: blokeret
        res1 = client.post(
            "/api/auth/login",
            headers={"X-Forwarded-For": "8.8.8.8"},
            json={"username": "testuser", "password": "testpass123"},
        )
        assert res1.status_code == 403

        # Med korrekt bypass: tilladt
        res2 = client.post(
            "/api/auth/login",
            headers={"X-Forwarded-For": "8.8.8.8"},
            json={
                "username": "testuser",
                "password": "testpass123",
                "geo_bypass": "rejser-2026",
            },
        )
        assert res2.status_code == 200

    def test_wrong_bypass_still_blocked(self, client, disable_notifications, monkeypatch):
        monkeypatch.setenv("GEO_LOCK_ENABLED", "true")
        monkeypatch.setenv("GEO_LOCK_COUNTRIES", "DK")
        monkeypatch.setenv("GEO_LOCK_BYPASS", "korrekt-kode")

        from src.api import geo_lock
        from src.api.geo_lock import GeoInfo
        geo_lock._service = None

        def fake_lookup(self, ip):
            return GeoInfo(ip=ip, country_code="RU", country_name="Russia", city="Moscow")

        monkeypatch.setattr(geo_lock.GeoLockService, "_do_lookup", fake_lookup)

        res = client.post(
            "/api/auth/login",
            headers={"X-Forwarded-For": "8.8.8.8"},
            json={
                "username": "testuser",
                "password": "testpass123",
                "geo_bypass": "forkert-kode",
            },
        )
        assert res.status_code == 403


# ── Login-notifikationer ───────────────────────────────────────


class TestLoginNotifications:
    def test_success_notification_sent(self, client, disable_notifications):
        """Når login lykkes (uden 2FA), skal success-notifikation sendes."""
        res = client.post(
            "/api/auth/login",
            json={"username": "testuser", "password": "testpass123"},
        )
        assert res.status_code == 200
        assert len(disable_notifications["success"]) == 1
        assert disable_notifications["success"][0]["user"] == "testuser"

    def test_no_notification_on_wrong_password(self, client, disable_notifications):
        """Forkert password → ingen success-notifikation."""
        res = client.post(
            "/api/auth/login",
            json={"username": "testuser", "password": "wrong"},
        )
        assert res.status_code == 401
        assert len(disable_notifications["success"]) == 0


# ── Auto-logout (JWT TTL) ──────────────────────────────────────


class TestAutoLogout:
    def test_default_ttl_is_30_minutes(self, client, disable_notifications, monkeypatch):
        """Default TTL: 30 minutter = 1800 sekunder."""
        monkeypatch.delenv("APP_TOKEN_TTL_MINUTES", raising=False)
        res = client.post(
            "/api/auth/login",
            json={"username": "testuser", "password": "testpass123"},
        )
        assert res.status_code == 200
        assert res.json()["expires_in"] == 1800

    def test_custom_ttl_respected(self, client, disable_notifications, monkeypatch):
        """APP_TOKEN_TTL_MINUTES=60 → expires_in = 3600."""
        monkeypatch.setenv("APP_TOKEN_TTL_MINUTES", "60")
        res = client.post(
            "/api/auth/login",
            json={"username": "testuser", "password": "testpass123"},
        )
        assert res.json()["expires_in"] == 3600

    def test_ttl_clamped_to_min_15(self, client, disable_notifications, monkeypatch):
        """TTL <15 min skal clamp'es op til 15."""
        monkeypatch.setenv("APP_TOKEN_TTL_MINUTES", "5")
        res = client.post(
            "/api/auth/login",
            json={"username": "testuser", "password": "testpass123"},
        )
        assert res.json()["expires_in"] == 15 * 60

    def test_ttl_clamped_to_max_1440(self, client, disable_notifications, monkeypatch):
        """TTL >24t skal clamp'es ned til 24t."""
        monkeypatch.setenv("APP_TOKEN_TTL_MINUTES", "9999")
        res = client.post(
            "/api/auth/login",
            json={"username": "testuser", "password": "testpass123"},
        )
        assert res.json()["expires_in"] == 1440 * 60

    def test_ttl_invalid_falls_back_to_default(self, client, disable_notifications, monkeypatch):
        """Ugyldigt TTL → fallback til 30 min."""
        monkeypatch.setenv("APP_TOKEN_TTL_MINUTES", "not-a-number")
        res = client.post(
            "/api/auth/login",
            json={"username": "testuser", "password": "testpass123"},
        )
        assert res.json()["expires_in"] == 1800


# ── Geo-lock service unit tests ────────────────────────────────


class TestGeoLockService:
    def test_local_ips_treated_as_dk(self, monkeypatch):
        from src.api.geo_lock import GeoLockService
        svc = GeoLockService()
        for ip in ["127.0.0.1", "::1", "localhost", "192.168.1.5", "10.0.0.1"]:
            info = svc.lookup(ip)
            assert info.country_code == "DK", f"{ip} should be local"

    def test_lookup_caches_result(self, monkeypatch):
        from src.api import geo_lock
        from src.api.geo_lock import GeoLockService, GeoInfo
        calls = []

        def fake_lookup(self, ip):
            calls.append(ip)
            return GeoInfo(ip=ip, country_code="DE", country_name="Germany", city="Berlin")

        monkeypatch.setattr(GeoLockService, "_do_lookup", fake_lookup)
        svc = GeoLockService()
        svc.lookup("1.2.3.4")
        svc.lookup("1.2.3.4")
        svc.lookup("1.2.3.4")
        assert len(calls) == 1  # cached

    def test_failed_lookup_fails_open(self, monkeypatch):
        """Hvis IP-lookup fejler, så tillad login (fail-open)."""
        from src.api import geo_lock
        from src.api.geo_lock import GeoLockService, GeoInfo

        def fake_lookup(self, ip):
            return GeoInfo(ip=ip, country_code=None, country_name=None, city=None)

        monkeypatch.setattr(GeoLockService, "_do_lookup", fake_lookup)
        monkeypatch.setenv("GEO_LOCK_ENABLED", "true")
        monkeypatch.setenv("GEO_LOCK_COUNTRIES", "DK")

        svc = GeoLockService()
        allowed, info, bypass = svc.is_allowed("1.2.3.4")
        assert allowed is True  # fail-open

    def test_extract_client_ip_from_header(self):
        from unittest.mock import MagicMock
        from src.api.geo_lock import extract_client_ip

        req = MagicMock()
        req.headers = {"x-forwarded-for": "8.8.8.8, 10.0.0.1"}
        req.client.host = "127.0.0.1"
        assert extract_client_ip(req) == "8.8.8.8"

    def test_extract_client_ip_fallback(self):
        from unittest.mock import MagicMock
        from src.api.geo_lock import extract_client_ip

        req = MagicMock()
        req.headers = {}
        req.client.host = "1.2.3.4"
        assert extract_client_ip(req) == "1.2.3.4"
