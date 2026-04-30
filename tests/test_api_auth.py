"""
Integration tests for mobile API auth (login + 2FA).

Dækker:
  - POST /api/auth/login med korrekt + forkert password
  - POST /api/auth/verify med korrekt + forkert kode
  - POST /api/auth/resend (rate-limit + cred-validering)
  - 2FA-flow ende-til-ende
  - PWA-routing skal IKKE catche /api/* paths
  - Health-route med og uden trailing slash
  - JWT-token validering på beskyttede endpoints
"""
from __future__ import annotations

import os
import time
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


# Set required env vars before importing server
os.environ.setdefault("APP_USERNAME", "testuser")
os.environ.setdefault("APP_PASSWORD", "testpass123")
os.environ.setdefault("APP_SECRET_KEY", "test-secret-key-32-chars-minimum-x" * 2)


@pytest.fixture(autouse=True)
def isolate_telegram_env(monkeypatch):
    """
    Sørg for at TELEGRAM_BOT_TOKEN/CHAT_ID IKKE lækker mellem tests.
    Standard er "ikke konfigureret" — tests der har brug for Telegram
    bruger mock_telegram-fixturen som sætter dem eksplicit.
    """
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    yield


@pytest.fixture
def client():
    """FastAPI TestClient. Re-importer modul for ren state."""
    from src.api import server as server_mod
    from src.api.two_factor import _service
    # Reset 2FA state mellem tests
    if _service is not None:
        _service._pending.clear()
        _service._last_code_time.clear()
    return TestClient(server_mod.app)


@pytest.fixture
def mock_telegram(monkeypatch):
    """Mock Telegram-API til altid succes."""
    sent_codes = []

    def fake_send(self, code):
        sent_codes.append(code)
        return True

    from src.api import two_factor
    monkeypatch.setattr(two_factor.TwoFactorService, "_send_telegram_code", fake_send)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345678")
    return sent_codes


# ── /api/auth/login ───────────────────────────────────────────


class TestLogin:
    def test_login_correct_password_no_telegram(self, client, monkeypatch):
        """Uden Telegram konfigureret → access_token returneres direkte."""
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
        # Reset 2FA service så is_telegram_configured re-evalueres
        from src.api import two_factor
        two_factor._service = None

        res = client.post(
            "/api/auth/login",
            json={"username": "testuser", "password": "testpass123"},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["two_factor"] is False
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert len(data["access_token"]) > 50

    def test_login_correct_password_with_telegram(self, client, mock_telegram):
        """Med Telegram → 2FA krævet, login_token returneres."""
        from src.api import two_factor
        two_factor._service = None

        res = client.post(
            "/api/auth/login",
            json={"username": "testuser", "password": "testpass123"},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["two_factor"] is True
        assert "login_token" in data
        assert data["login_token"]
        assert "destination" in data
        assert "Telegram" in data["destination"]
        # Telegram skal være kaldt
        assert len(mock_telegram) == 1
        assert len(mock_telegram[0]) == 6
        assert mock_telegram[0].isdigit()

    def test_login_wrong_password(self, client):
        res = client.post(
            "/api/auth/login",
            json={"username": "testuser", "password": "wrong"},
        )
        assert res.status_code == 401
        assert res.json()["detail"] == "Forkert brugernavn eller adgangskode"

    def test_login_wrong_username(self, client):
        res = client.post(
            "/api/auth/login",
            json={"username": "admin", "password": "testpass123"},
        )
        assert res.status_code == 401

    def test_login_empty_credentials(self, client):
        res = client.post(
            "/api/auth/login",
            json={"username": "", "password": ""},
        )
        assert res.status_code == 401

    def test_login_missing_field(self, client):
        res = client.post("/api/auth/login", json={"username": "testuser"})
        assert res.status_code == 422  # validation error


# ── /api/auth/verify ──────────────────────────────────────────


class TestVerify:
    def test_verify_correct_code_grants_token(self, client, mock_telegram):
        from src.api import two_factor
        two_factor._service = None

        # Step 1: login
        res1 = client.post(
            "/api/auth/login",
            json={"username": "testuser", "password": "testpass123"},
        )
        login_token = res1.json()["login_token"]
        code = mock_telegram[0]

        # Step 2: verify
        res2 = client.post(
            "/api/auth/verify",
            json={"login_token": login_token, "code": code},
        )
        assert res2.status_code == 200
        data = res2.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

    def test_verify_wrong_code(self, client, mock_telegram):
        from src.api import two_factor
        two_factor._service = None

        res1 = client.post(
            "/api/auth/login",
            json={"username": "testuser", "password": "testpass123"},
        )
        login_token = res1.json()["login_token"]

        res2 = client.post(
            "/api/auth/verify",
            json={"login_token": login_token, "code": "000000"},
        )
        assert res2.status_code == 401
        assert "Ugyldig" in res2.json()["detail"]

    def test_verify_unknown_login_token(self, client, mock_telegram):
        from src.api import two_factor
        two_factor._service = None

        res = client.post(
            "/api/auth/verify",
            json={"login_token": "fake-token", "code": "123456"},
        )
        assert res.status_code == 401

    def test_verify_token_consumed_once(self, client, mock_telegram):
        """Login_token må ikke kunne bruges to gange."""
        from src.api import two_factor
        two_factor._service = None

        res1 = client.post(
            "/api/auth/login",
            json={"username": "testuser", "password": "testpass123"},
        )
        login_token = res1.json()["login_token"]
        code = mock_telegram[0]

        # Første verify: succes
        ok = client.post(
            "/api/auth/verify",
            json={"login_token": login_token, "code": code},
        )
        assert ok.status_code == 200

        # Andet forsøg med samme token: skal afvises
        again = client.post(
            "/api/auth/verify",
            json={"login_token": login_token, "code": code},
        )
        assert again.status_code == 401

    def test_verify_max_attempts(self, client, mock_telegram):
        """Efter 3 forkerte forsøg skal token ugyldiggøres."""
        from src.api import two_factor
        two_factor._service = None

        res1 = client.post(
            "/api/auth/login",
            json={"username": "testuser", "password": "testpass123"},
        )
        login_token = res1.json()["login_token"]

        # 3 forkerte forsøg
        for _ in range(3):
            client.post(
                "/api/auth/verify",
                json={"login_token": login_token, "code": "000000"},
            )

        # Selv den korrekte kode skal nu fejle (token slettet efter MAX_ATTEMPTS overskredet)
        correct_code = mock_telegram[0]
        res = client.post(
            "/api/auth/verify",
            json={"login_token": login_token, "code": correct_code},
        )
        assert res.status_code == 401


# ── /api/auth/resend ──────────────────────────────────────────


class TestResend:
    def test_resend_requires_credentials(self, client, mock_telegram):
        from src.api import two_factor
        two_factor._service = None

        # Forkert password ved resend skal afvises
        res = client.post(
            "/api/auth/resend",
            json={"username": "testuser", "password": "wrong"},
        )
        assert res.status_code == 401

    def test_resend_rate_limited(self, client, mock_telegram):
        """Anden resend inden 30s skal returnere 429."""
        from src.api import two_factor
        two_factor._service = None

        # Første: login (sender kode)
        client.post(
            "/api/auth/login",
            json={"username": "testuser", "password": "testpass123"},
        )
        # Anden inden 30s
        res = client.post(
            "/api/auth/resend",
            json={"username": "testuser", "password": "testpass123"},
        )
        assert res.status_code == 429


# ── PWA routing ───────────────────────────────────────────────


class TestRouting:
    def test_root_serves_pwa(self, client):
        res = client.get("/")
        assert res.status_code == 200
        assert "text/html" in res.headers["content-type"]

    def test_unknown_api_path_returns_json_404(self, client):
        """Ukendt /api/* skal returnere JSON-404, IKKE PWA-HTML."""
        res = client.get("/api/nonexistent")
        assert res.status_code == 404
        assert "application/json" in res.headers["content-type"]

    def test_pwa_path_serves_html(self, client):
        """Random PWA-path skal stadig serve index.html (SPA-routing)."""
        res = client.get("/positions")
        assert res.status_code == 200
        assert "text/html" in res.headers["content-type"]


# ── Beskyttede endpoints ──────────────────────────────────────


class TestProtectedEndpoints:
    def test_health_requires_token(self, client):
        res = client.get("/api/health")
        assert res.status_code == 401

    def test_health_with_valid_token(self, client, monkeypatch):
        """Health endpoint med token skal returnere status."""
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
        from src.api import two_factor
        two_factor._service = None

        # Get token via direct login (no 2FA)
        login = client.post(
            "/api/auth/login",
            json={"username": "testuser", "password": "testpass123"},
        )
        token = login.json()["access_token"]

        res = client.get("/api/health", headers={"Authorization": f"Bearer {token}"})
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "running"
        assert "uptime" in data
        assert "brokers" in data

    def test_health_with_invalid_token(self, client):
        res = client.get(
            "/api/health",
            headers={"Authorization": "Bearer invalid.jwt.token"},
        )
        assert res.status_code == 401

    def test_health_trailing_slash_works(self, client, monkeypatch):
        """/api/health/ (med slash) skal ramme samme route."""
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
        from src.api import two_factor
        two_factor._service = None

        login = client.post(
            "/api/auth/login",
            json={"username": "testuser", "password": "testpass123"},
        )
        token = login.json()["access_token"]

        res = client.get("/api/health/", headers={"Authorization": f"Bearer {token}"})
        assert res.status_code == 200
        assert "application/json" in res.headers["content-type"]
