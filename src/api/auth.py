"""
JWT authentication for Alpha Trader mobile API.
Single-user: credentials read from .env (APP_USERNAME, APP_PASSWORD).
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext

# ── Config ────────────────────────────────────────────────────
SECRET_KEY  = os.getenv("APP_SECRET_KEY", "change-me-in-dotenv-32chars-min!")
ALGORITHM   = "HS256"
EXPIRE_HOURS = 24

APP_USERNAME = os.getenv("APP_USERNAME", "ole")
APP_PASSWORD = os.getenv("APP_PASSWORD", "")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


# ── Helpers ───────────────────────────────────────────────────

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def create_token(username: str) -> str:
    expire = datetime.utcnow() + timedelta(hours=EXPIRE_HOURS)
    return jwt.encode({"sub": username, "exp": expire}, SECRET_KEY, algorithm=ALGORITHM)


def authenticate(username: str, password: str) -> bool:
    if username != APP_USERNAME:
        return False
    # Support both plain-text and bcrypt-hashed passwords in .env
    if APP_PASSWORD.startswith("$2b$"):
        return verify_password(password, APP_PASSWORD)
    return password == APP_PASSWORD


async def get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    credentials_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Ugyldigt login",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if not username:
            raise credentials_exc
        return username
    except JWTError:
        raise credentials_exc
