#!/usr/bin/env python3
"""
Manuel verifikations-test for Alpha Trader login-systemet.

Bruges efter platformen er startet (python3 main.py --mode trader --paper).
Tjekker:
  1. API server svarer på port 8051
  2. Login med korrekt credentials trigger 2FA-kode på Telegram
  3. Forkert password afvises korrekt
  4. PWA serveres på root-path
  5. Beskyttede endpoints kræver token

Kør med:
    python3 scripts/test_login.py
"""
from __future__ import annotations

import os
import sys
import time
import urllib.parse
import urllib.request
import json
from pathlib import Path

# Load .env
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


BASE = "http://127.0.0.1:8051"
USERNAME = os.getenv("APP_USERNAME", "ole")
PASSWORD = os.getenv("APP_PASSWORD", "")

# ANSI colors
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
RESET = "\033[0m"
BOLD = "\033[1m"


def post(path: str, body: dict, timeout: int = 5) -> tuple[int, dict]:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        BASE + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read()) if e.read() else {}
    except urllib.error.URLError as e:
        return 0, {"error": str(e.reason)}


def get(path: str, token: str | None = None, timeout: int = 5) -> tuple[int, dict | str]:
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(BASE + path, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            ct = resp.headers.get("content-type", "")
            if "json" in ct:
                return resp.status, json.loads(body)
            return resp.status, body.decode()[:200]
    except urllib.error.HTTPError as e:
        return e.code, {}
    except urllib.error.URLError as e:
        return 0, {"error": str(e.reason)}


def check(name: str, ok: bool, detail: str = "") -> bool:
    icon = f"{GREEN}✓{RESET}" if ok else f"{RED}✗{RESET}"
    print(f"  {icon} {name}", end="")
    if detail:
        print(f" — {detail}")
    else:
        print()
    return ok


def main():
    print(f"{BOLD}{BLUE}═══ Alpha Trader Login Test ══════════════════════════{RESET}\n")
    print(f"  Server: {BASE}")
    print(f"  Bruger: {USERNAME}")
    print(f"  .env loaded: {bool(PASSWORD)}\n")

    if not PASSWORD:
        print(f"{RED}✗ APP_PASSWORD ikke sat i .env — kan ikke teste{RESET}")
        sys.exit(1)

    results = []

    # 1. Server svarer
    print(f"{BOLD}1. Server reachability{RESET}")
    code, body = get("/")
    results.append(check("Server svarer på port 8051", code == 200,
                         f"HTTP {code}" if code != 200 else "OK"))
    results.append(check("PWA serveres som HTML", code == 200 and "html" in str(body).lower()[:50]))

    # 2. Login afviser forkert password
    print(f"\n{BOLD}2. Forkerte credentials{RESET}")
    code, body = post("/api/auth/login", {"username": USERNAME, "password": "wrongpass"})
    results.append(check("Forkert password afvises", code == 401,
                         f"HTTP {code}, detail: {body.get('detail', 'n/a')}"))

    code, body = post("/api/auth/login", {"username": "wronguser", "password": PASSWORD})
    results.append(check("Forkert username afvises", code == 401))

    # 3. Korrekt login starter 2FA
    print(f"\n{BOLD}3. Korrekt login (2FA flow){RESET}")
    time.sleep(31)  # rate-limit reset
    code, body = post("/api/auth/login", {"username": USERNAME, "password": PASSWORD})
    print(f"     Response: {body}")

    if code == 200 and body.get("two_factor"):
        results.append(check("2FA aktiveret", True,
                             f"login_token udstedt → {body.get('destination')}"))
        login_token = body.get("login_token")

        # Verify forkert kode
        code, body = post("/api/auth/verify",
                          {"login_token": login_token, "code": "000000"})
        results.append(check("Forkert 2FA-kode afvises", code == 401))

        print(f"\n  {YELLOW}ℹ Tjek Telegram for den 6-cifrede kode!{RESET}")
        print(f"  {YELLOW}  (Vi kan ikke teste verify-flowet automatisk uden den){RESET}")

    elif code == 200 and body.get("access_token"):
        results.append(check("Direkte login (ingen Telegram)", True,
                             "access_token udstedt — Telegram ikke konfigureret"))
        token = body["access_token"]

        # Test beskyttet endpoint
        print(f"\n{BOLD}4. Beskyttede endpoints{RESET}")
        code, body = get("/api/health", token=token)
        results.append(check("/api/health med token", code == 200))

        code, body = get("/api/health", token="invalid.token")
        results.append(check("/api/health med ugyldigt token afvises", code == 401))

        code, body = get("/api/health")
        results.append(check("/api/health uden token afvises", code == 401))

    elif code == 429:
        results.append(check("Login (rate-limited)", False,
                             "Vent 30 sek mellem koder — prøv igen om lidt"))
    else:
        results.append(check("Korrekt login", False,
                             f"HTTP {code}: {body}"))

    # 4. Routing
    print(f"\n{BOLD}5. API routing (PWA-catchall fix){RESET}")
    code, body = get("/api/nonexistent")
    results.append(check("Ukendt /api/* returnerer JSON-404",
                         code == 404 and isinstance(body, dict)))

    # Sammendrag
    passed = sum(results)
    total = len(results)
    print(f"\n{BOLD}{BLUE}═══════════════════════════════════════════════════════{RESET}")
    if passed == total:
        print(f"{GREEN}{BOLD}  ✓ Alle {total} tests passerede!{RESET}")
    else:
        print(f"{RED}{BOLD}  {passed}/{total} tests passerede ({total - passed} fejl){RESET}")
    print(f"{BOLD}{BLUE}═══════════════════════════════════════════════════════{RESET}\n")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
