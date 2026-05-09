"""
Alpha Trader Mobile API — FastAPI server on port 8051.
Serves REST endpoints + PWA frontend.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

# CRITICAL: load .env FØR auth modulet importeres (auth læser env-vars ved import)
from dotenv import load_dotenv
_REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_REPO_ROOT / ".env")

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel

from src.api.auth import authenticate, create_token, token_ttl_seconds
from src.api.geo_lock import extract_client_ip, get_geo_lock_service
from src.api.routes import control, health, notifications, portfolio, signals
from src.api.security_notify import (
    notify_geo_blocked,
    notify_login_failed,
    notify_login_success,
)
from src.api.two_factor import get_two_factor_service

# ── App ───────────────────────────────────────────────────────
app = FastAPI(title="Alpha Trader Mobile API", version="1.0.0", docs_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_no_cache_headers(request: Request, call_next):
    """Tving browseren til ALDRIG at cache /api/* — undgår at gamle responses
    sidder fast i browser-cache (især på POST-requests som login)."""
    response = await call_next(request)
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

# ── Auth ──────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    username: str
    password: str
    geo_bypass: str | None = None  # bypass-kode hvis Ole rejser


class VerifyCodeRequest(BaseModel):
    login_token: str
    code: str


def _check_geo(request: Request, username: str, geo_bypass: str | None):
    """
    Tjek geo-lock før login fortsættes.

    Returnerer (geo_info, used_bypass). Raiser 403 hvis blokeret.
    """
    geo = get_geo_lock_service()
    ip = extract_client_ip(request)
    allowed, info, bypassed = geo.is_allowed(ip, geo_bypass)
    if not allowed:
        notify_geo_blocked(username, ip, info.location_str)
        logger.warning(
            f"[geo] BLOKERET login for {username} fra {ip} ({info.location_str})"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                f"Login blokeret fra {info.location_str}. "
                f"Brug bypass-kode hvis du rejser."
            ),
        )
    return info, bypassed


@app.post("/api/auth/login")
@app.post("/api/auth/login/")
async def login(req: LoginRequest, request: Request):
    """
    Trin 1: bekræft brugernavn + password + geo-lock.

    Hvis Telegram er konfigureret, sendes en 6-cifret kode og der returneres
    et midlertidigt login_token til /api/auth/verify.

    Hvis Telegram ikke er konfigureret (single-user dev mode), gives
    access_token direkte (bagudkompatibilitet).
    """
    if not authenticate(req.username, req.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Forkert brugernavn eller adgangskode",
        )

    # Geo-lock før vi sender kode (sparer Telegram-kald hvis blokeret)
    geo_info, used_bypass = _check_geo(request, req.username, req.geo_bypass)
    ip = extract_client_ip(request)

    tfa = get_two_factor_service()
    if not tfa.is_telegram_configured:
        # Fallback: ingen 2FA tilgængelig
        notify_login_success(req.username, ip, geo_info.location_str)
        return {
            "access_token": create_token(req.username),
            "token_type": "bearer",
            "two_factor": False,
            "expires_in": token_ttl_seconds(),
        }

    result = tfa.request_code(req.username)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Vent venligst 30 sekunder før du beder om en ny kode",
        )

    login_token, masked = result
    return {
        "two_factor": True,
        "login_token": login_token,
        "destination": f"Telegram {masked}",
        "message": "Tjek Telegram for din 6-cifrede kode",
        "bypass_used": used_bypass,
    }


@app.post("/api/auth/verify")
@app.post("/api/auth/verify/")
async def verify_code(req: VerifyCodeRequest, request: Request):
    """Trin 2: bekræft 6-cifret Telegram-kode → modtag access_token."""
    tfa = get_two_factor_service()
    username = tfa.verify_code(req.login_token, req.code)
    ip = extract_client_ip(request)

    if username is None:
        # Send notifikation hvis vi kender brugernavnet via login_token
        # (her kan vi ikke, så vi sender bare en generisk advarsel)
        notify_login_failed("ukendt", ip, "Forkert/udløbet 2FA-kode")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Ugyldig eller udløbet kode",
        )

    # Login lykkedes — send notifikation
    geo = get_geo_lock_service()
    info = geo.lookup(ip)
    notify_login_success(username, ip, info.location_str)

    return {
        "access_token": create_token(username),
        "token_type": "bearer",
        "expires_in": token_ttl_seconds(),
    }


@app.post("/api/auth/resend")
@app.post("/api/auth/resend/")
async def resend_code(req: LoginRequest, request: Request):
    """Bed om en ny 2FA-kode (kræver username + password igen for sikkerhed)."""
    if not authenticate(req.username, req.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Forkert brugernavn eller adgangskode",
        )
    _check_geo(request, req.username, req.geo_bypass)
    tfa = get_two_factor_service()
    if not tfa.is_telegram_configured:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Telegram er ikke konfigureret",
        )
    result = tfa.request_code(req.username)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Vent 30 sekunder mellem koder",
        )
    login_token, masked = result
    return {
        "login_token": login_token,
        "destination": f"Telegram {masked}",
        "message": "Ny kode sendt",
    }


# ── Routes ────────────────────────────────────────────────────
app.include_router(portfolio.router, prefix="/api/portfolio", tags=["portfolio"])
app.include_router(signals.router,   prefix="/api/signals",   tags=["signals"])
app.include_router(notifications.router, prefix="/api/notifications", tags=["notifications"])
app.include_router(health.router,    prefix="/api/health",    tags=["health"])
app.include_router(control.router,   prefix="/api/control",   tags=["control"])

# ── Static / PWA ──────────────────────────────────────────────
STATIC_DIR = Path(__file__).parent / "static"

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/manifest.json")
    async def manifest():
        return FileResponse(STATIC_DIR / "manifest.json")

    @app.get("/")
    @app.get("/{path:path}")
    async def serve_pwa(path: str = ""):
        # KRITISK: API-paths skal IKKE caches af PWA — returnér 404 så de korrekte
        # routere kan matche, eller giver klar fejl hvis ruten ikke eksisterer.
        if path.startswith("api/") or path == "api":
            return JSONResponse(
                status_code=404,
                content={"detail": f"API endpoint not found: /{path}"},
            )
        index = STATIC_DIR / "index.html"
        if index.exists():
            # Tving browseren til altid at hente nyeste index.html — undgår
            # at gamle UI-versioner sidder fast i browser-cachen efter deploy.
            return FileResponse(
                index,
                headers={
                    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                    "Pragma": "no-cache",
                    "Expires": "0",
                },
            )
        return {"message": "Alpha Trader Mobile API", "docs": "/api"}


# ── Startup ───────────────────────────────────────────────────

def start_api_server(host: str = "0.0.0.0", port: int = 8051, background: bool = True):
    """Start the API server as a subprocess (avoids asyncio conflict with Dash)."""
    import subprocess
    import sys

    cmd = [
        sys.executable, "-m", "uvicorn",
        "src.api.server:app",
        "--host", host,
        "--port", str(port),
        "--log-level", "warning",
    ]

    if background:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return proc
    else:
        subprocess.run(cmd)
