#!/usr/bin/env bash
# =============================================================================
# Alpha Trader 2.2 — Startup script
# Bruges af launchd (auto-restart) og til manuel start.
#
# Features:
#   - Sourcer .env (API keys etc.)
#   - Aktiverer Python 3.14 venv
#   - Logger til logs/ med rotation
#   - Sætter arbejdsmappe korrekt
#   - Exitkode 0 = clean stop, !=0 = crash (launchd genstarter)
# =============================================================================

set -euo pipefail

# ── Stier ────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_PYTHON="$PROJECT_DIR/.venv/bin/python"
ENV_FILE="$PROJECT_DIR/.env"
LOG_DIR="$PROJECT_DIR/logs"

# ── Skift til projektmappe ───────────────────────────────────────────────────
cd "$PROJECT_DIR"

# ── Opret log-mappe ──────────────────────────────────────────────────────────
mkdir -p "$LOG_DIR"

# ── Source .env ──────────────────────────────────────────────────────────────
if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck source=/dev/null
    source "$ENV_FILE"
    set +a
else
    echo "[start_trader] ADVARSEL: .env ikke fundet på $ENV_FILE" >&2
fi

# ── Log startup ──────────────────────────────────────────────────────────────
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
echo "[$TIMESTAMP] Alpha Trader starter (PID: $$, Mode: paper)" >> "$LOG_DIR/startup.log"

# ── Start platform ───────────────────────────────────────────────────────────
exec "$VENV_PYTHON" -m src.main trader \
    --paper \
    2>&1
