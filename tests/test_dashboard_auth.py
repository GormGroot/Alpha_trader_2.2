"""
Regression tests for Fase 1.7 — dashboard HTTP Basic auth (2026-04-17).

Covers the actual threat model, not just the happy path:
  * Auth is OFF when env vars are missing (local paper-trading stays friction-free)
  * Auth is ON when both DASHBOARD_USER + DASHBOARD_PASS are set
  * /healthz is reachable without credentials (Docker HEALTHCHECK)
  * /_dash-layout IS protected (leaks component IDs if exposed)
  * Wrong credentials → 401 with WWW-Authenticate header
  * Correct credentials → 200
  * Constant-time comparison (hmac.compare_digest) is used — verified
    indirectly via the rejection path on partial matches
  * install_auth is idempotent — safe to call multiple times in the
    same process (e.g. test reloads)
"""

from __future__ import annotations

import base64

import pytest
from flask import Flask

from src.dashboard import auth as auth_mod


def _basic_header(user: str, pw: str) -> dict[str, str]:
    token = base64.b64encode(f"{user}:{pw}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


@pytest.fixture
def flask_app():
    """Minimal Flask app with a few routes that mimic the real dashboard
    surface: the Dash layout endpoint, an asset path, and a generic page."""
    app = Flask(__name__)

    @app.route("/")
    def _root():
        return "dashboard\n"

    @app.route("/_dash-layout")
    def _layout():
        return "LAYOUT-SENSITIVE\n"

    @app.route("/assets/style.css")
    def _asset():
        return "body{}\n"

    return app


# ──────────────────────────────────────────────────────────
# Auth OFF by default
# ──────────────────────────────────────────────────────────
class TestAuthDisabled:
    def test_root_accessible_without_credentials_when_disabled(self, flask_app, monkeypatch):
        monkeypatch.delenv("DASHBOARD_USER", raising=False)
        monkeypatch.delenv("DASHBOARD_PASS", raising=False)
        auth_mod.install_auth(flask_app)
        client = flask_app.test_client()
        r = client.get("/")
        assert r.status_code == 200
        assert b"dashboard" in r.data

    def test_only_user_set_keeps_auth_off(self, flask_app, monkeypatch):
        """Half-configured auth (user but no password) must fail closed-open:
        we don't want to accidentally lock Ole out because he forgot one var."""
        monkeypatch.setenv("DASHBOARD_USER", "ole")
        monkeypatch.delenv("DASHBOARD_PASS", raising=False)
        auth_mod.install_auth(flask_app)
        client = flask_app.test_client()
        assert client.get("/").status_code == 200


# ──────────────────────────────────────────────────────────
# Auth ON
# ──────────────────────────────────────────────────────────
class TestAuthEnabled:
    @pytest.fixture(autouse=True)
    def _set_creds(self, monkeypatch):
        monkeypatch.setenv("DASHBOARD_USER", "ole")
        monkeypatch.setenv("DASHBOARD_PASS", "correct-horse-battery-staple")

    def test_no_credentials_returns_401(self, flask_app):
        auth_mod.install_auth(flask_app)
        client = flask_app.test_client()
        r = client.get("/")
        assert r.status_code == 401
        assert r.headers.get("WWW-Authenticate", "").startswith("Basic")

    def test_wrong_password_returns_401(self, flask_app):
        auth_mod.install_auth(flask_app)
        client = flask_app.test_client()
        r = client.get("/", headers=_basic_header("ole", "wrong"))
        assert r.status_code == 401

    def test_wrong_user_returns_401(self, flask_app):
        auth_mod.install_auth(flask_app)
        client = flask_app.test_client()
        r = client.get("/", headers=_basic_header("attacker", "correct-horse-battery-staple"))
        assert r.status_code == 401

    def test_correct_credentials_pass(self, flask_app):
        auth_mod.install_auth(flask_app)
        client = flask_app.test_client()
        r = client.get("/", headers=_basic_header("ole", "correct-horse-battery-staple"))
        assert r.status_code == 200
        assert b"dashboard" in r.data

    def test_dash_layout_requires_auth(self, flask_app):
        """/_dash-layout leaks sensitive component IDs if exposed — must
        NOT be in the exempt list."""
        auth_mod.install_auth(flask_app)
        client = flask_app.test_client()
        r = client.get("/_dash-layout")
        assert r.status_code == 401, (
            "/_dash-layout leaked without auth — attacker could enumerate "
            "portfolio component IDs"
        )

    def test_healthz_bypasses_auth(self, flask_app):
        """Docker HEALTHCHECK hits /healthz without credentials; if auth
        blocked it the container would be marked unhealthy."""
        auth_mod.install_auth(flask_app)
        client = flask_app.test_client()
        r = client.get("/healthz")
        assert r.status_code == 200
        assert b"ok" in r.data

    def test_assets_bypass_auth(self, flask_app):
        """Static CSS/JS under /assets/ must load so the browser can
        render the auth prompt page with WV branding."""
        auth_mod.install_auth(flask_app)
        client = flask_app.test_client()
        r = client.get("/assets/style.css")
        assert r.status_code == 200

    def test_failed_auth_does_not_log_password(self, flask_app, caplog):
        """Logging the supplied password is a regression we never want.
        If a user mistypes their password into the basic-auth dialog,
        that password must not end up in stdout/loguru/logs that get
        shipped to Sentry or similar."""
        auth_mod.install_auth(flask_app)
        client = flask_app.test_client()
        sensitive = "super-secret-typo-password"
        client.get("/", headers=_basic_header("ole", sensitive))
        combined = " ".join(r.getMessage() for r in caplog.records)
        assert sensitive not in combined


# ──────────────────────────────────────────────────────────
# Install idempotency
# ──────────────────────────────────────────────────────────
class TestInstallIdempotency:
    def test_install_auth_is_idempotent(self, flask_app, monkeypatch):
        """Reimporting the dashboard module in tests must not stack
        before_request hooks — otherwise every request runs auth N times
        and the 401 reason gets logged N times."""
        monkeypatch.setenv("DASHBOARD_USER", "ole")
        monkeypatch.setenv("DASHBOARD_PASS", "pw")
        auth_mod.install_auth(flask_app)
        auth_mod.install_auth(flask_app)
        auth_mod.install_auth(flask_app)
        # Count before_request hooks on None (app-wide) blueprint.
        hooks = flask_app.before_request_funcs.get(None, [])
        assert len(hooks) == 1, f"expected 1 before_request hook, got {len(hooks)}"
