"""
Alpha Trader Mobile API — FastAPI server on port 8051.
Serves REST endpoints + PWA frontend.
"""
from __future__ import annotations

import threading
from pathlib import Path

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.api.auth import authenticate, create_token
from src.api.routes import health, notifications, portfolio, signals

# ── App ───────────────────────────────────────────────────────
app = FastAPI(title="Alpha Trader Mobile API", version="1.0.0", docs_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Auth ──────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    username: str
    password: str


@app.post("/api/auth/login")
async def login(req: LoginRequest):
    if not authenticate(req.username, req.password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Forkert brugernavn eller adgangskode")
    return {"access_token": create_token(req.username), "token_type": "bearer"}


# ── Routes ────────────────────────────────────────────────────
app.include_router(portfolio.router, prefix="/api/portfolio", tags=["portfolio"])
app.include_router(signals.router,   prefix="/api/signals",   tags=["signals"])
app.include_router(notifications.router, prefix="/api/notifications", tags=["notifications"])
app.include_router(health.router,    prefix="/api/health",    tags=["health"])

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
        index = STATIC_DIR / "index.html"
        if index.exists():
            return FileResponse(index)
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
