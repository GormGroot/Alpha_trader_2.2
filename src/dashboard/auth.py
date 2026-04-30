"""
HTTP Basic authentication for the dashboard (Fase 1.7, 2026-04-17).

Design:
  * Opt-in — auth is OFF unless BOTH DASHBOARD_USER and DASHBOARD_PASS are
    set. Local `python main.py --mode trader --paper` stays friction-free.
  * Constant-time comparison (hmac.compare_digest) so attackers can't
    extract credentials via timing side-channels.
  * /healthz is always exempt — Docker HEALTHCHECK must work without
    shipping credentials into the container layer. The endpoint is a
    liveness probe only (no sensitive data).
  * Static assets (Dash's /_dash-* and /assets/*) are also exempt so
    the browser can render the login-prompted page.

This is intentionally minimal — the threat model is "keep curious
network neighbours off an internal dashboard", NOT "withstand a
determined attacker". For prod exposure, front with a reverse proxy
(nginx + Let's Encrypt) and restrict DASHBOARD_USER to a break-glass
admin account.
"""

from __future__ import annotations

import hmac
import os
from typing import Iterable

from flask import Response, request
from loguru import logger

# Paths that never require auth — Docker healthcheck + favicon + any
# user-dropped static assets. Prefix match.
# NOTE: /_dash-layout and /_dash-dependencies are deliberately NOT exempt;
# an unauthenticated layout fetch would leak component IDs that reveal
# portfolio state, broker names, and callback topology.
_AUTH_EXEMPT_PREFIXES: tuple[str, ...] = ("/healthz", "/_favicon.ico", "/assets/")


def _get_expected_credentials() -> tuple[str, str] | None:
    """Return (user, pass) if auth is configured, else None (auth off)."""
    user = os.environ.get("DASHBOARD_USER")
    pw = os.environ.get("DASHBOARD_PASS")
    if not user or not pw:
        return None
    return user, pw


def _is_exempt(path: str) -> bool:
    return any(path.startswith(prefix) for prefix in _AUTH_EXEMPT_PREFIXES)


def _check_credentials(supplied_user: str, supplied_pw: str) -> bool:
    expected = _get_expected_credentials()
    if expected is None:
        return True  # auth disabled — allow everything
    exp_user, exp_pw = expected
    # hmac.compare_digest is constant-time; both operands must be bytes or str.
    user_ok = hmac.compare_digest(supplied_user.encode(), exp_user.encode())
    pw_ok = hmac.compare_digest(supplied_pw.encode(), exp_pw.encode())
    return user_ok and pw_ok


def _auth_required_response() -> Response:
    return Response(
        "Authentication required\n",
        status=401,
        headers={"WWW-Authenticate": 'Basic realm="Alpha Trader Dashboard"'},
    )


def _before_request_auth():
    """Flask before_request hook. Return None to let the request through,
    or a Response to short-circuit with 401."""
    if _get_expected_credentials() is None:
        return None  # auth disabled
    if _is_exempt(request.path):
        return None
    auth = request.authorization
    if auth is None or auth.type != "basic":
        return _auth_required_response()
    if not _check_credentials(auth.username or "", auth.password or ""):
        # Log the rejected attempt — but NEVER the supplied credentials
        # (could be a password typo the user doesn't want in logs).
        logger.warning(
            f"[dashboard] auth failed from {request.remote_addr} "
            f"user={auth.username!r} path={request.path}"
        )
        return _auth_required_response()
    return None


def install_auth(flask_server) -> bool:
    """Wire Basic auth into the given Flask server. Also registers
    /healthz — a cheap liveness endpoint used by Docker HEALTHCHECK.

    Returns True if auth was activated, False if it stayed disabled
    (env vars absent). Idempotent: calling twice only installs one hook.
    """
    # Idempotency guard — prevents double-registration if the app is
    # re-imported in a test session.
    if getattr(flask_server, "_alpha_auth_installed", False):
        return bool(_get_expected_credentials())

    @flask_server.route("/healthz")
    def _healthz():  # pragma: no cover — thin wrapper
        # Intentionally minimal: just confirms the WSGI layer is alive.
        # Scheduler heartbeat is checked separately in Dockerfile HEALTHCHECK.
        return ("ok\n", 200, {"Content-Type": "text/plain"})

    flask_server.before_request(_before_request_auth)
    flask_server._alpha_auth_installed = True

    activated = _get_expected_credentials() is not None
    if activated:
        logger.info("[dashboard] HTTP Basic auth ENABLED (DASHBOARD_USER set)")
    else:
        logger.debug(
            "[dashboard] HTTP Basic auth disabled "
            "(set DASHBOARD_USER + DASHBOARD_PASS to enable)"
        )
    return activated
