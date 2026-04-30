"""
JWT authentication for Alpha Trader mobile API.
Single-user: credentials read from .env (APP_USERNAME, APP_PASSWORD).
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv

# Sørg for at .env er loaded uanset hvordan dette modul importeres.
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext

# ── Config ────────────────────────────────────────────────────
ALGORITHM    = "HS256"
# Default 30 minutter — brugeren får ny session ved at logge ind igen.
# Kan justeres via APP_TOKEN_TTL_MINUTES i .env (15-1440 = 15 min til 24t).
DEFAULT_TTL_MINUTES = 30


def _ttl_minutes() -> int:
    raw = os.getenv("APP_TOKEN_TTL_MINUTES", str(DEFAULT_TTL_MINUTES))
    try:
        v = int(raw)
        return max(15, min(1440, v))
    except ValueError:
        return DEFAULT_TTL_MINUTES


def _secret_key() -> str:
    """Læs secret key dynamisk så miljø-ændringer respekteres (vigtigt for tests)."""
    return os.getenv("APP_SECRET_KEY", "change-me-in-dotenv-32chars-min!")


def _username() -> str:
    return os.getenv("APP_USERNAME", "ole")


def _password() -> str:
    return os.getenv("APP_PASSWORD", "")


# Bagud-kompatible konstanter (læses ved import, men brug funktioner i runtime)
SECRET_KEY  = _secret_key()
APP_USERNAME = _username()
APP_PASSWORD = _password()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


# ── Helpers ───────────────────────────────────────────────────

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def create_token(username: str) -> str:
    expire = datetime.utcnow() + timedelta(minutes=_ttl_minutes())
    return jwt.encode({"sub": username, "exp": expire}, _secret_key(), algorithm=ALGORITHM)


def token_ttl_seconds() -> int:
    """Returnerer token-levetid i sekunder (til frontend)."""
    return _ttl_minutes() * 60


def authenticate(username: str, password: str) -> bool:
    expected_user = _username()
    expected_pass = _password()
    if not username or username != expected_user:
        return False
    if not password:
        return False
    # Support both plain-text and bcrypt-hashed passwords in .env
    if expected_pass.startswith("$2b$"):
        return verify_password(password, expected_pass)
    # Konstant-tids sammenligning for at undgå timing attacks
    import hmac
    return hmac.compare_digest(password, expected_pass)


async def get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    credentials_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Ugyldigt login",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, _secret_key(), algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if not username:
            raise credentials_exc
        return username
    except JWTError:
        raise credentials_exc
