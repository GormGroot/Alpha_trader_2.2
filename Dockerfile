# ═══════════════════════════════════════════════════════════
#  Alpha Trading Platform — Dockerfile
#  Python 3.13: Bumps from 3.12 (code imports from 3.14 std-lib
#  path-compat still works on 3.13; switch to 3.14-slim once
#  the official image is published on Docker Hub).
# ═══════════════════════════════════════════════════════════

FROM python:3.13-slim

# System deps for psycopg2, TA-Lib, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libpq-dev curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps
COPY requirements.txt requirements-trader.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -r requirements-trader.txt

# Application code
COPY . .

# Create data directories
RUN mkdir -p data logs backups

# Non-root user
RUN useradd -m -r trader && chown -R trader:trader /app
USER trader

# Healthcheck — returns non-zero when EITHER the dashboard is unreachable
# OR the scheduler heartbeat is stale (>3 min old). The previous
# `curl ... || python -c 'sys.exit(0)'` swallowed all failures and
# reported HEALTHY on a dead container (fixed 2026-04-17). This variant
# also catches the case where Flask responds but the scheduler thread
# has wedged — see src/ops/daily_scheduler.py:is_scheduler_alive.
#
# We hit /healthz (always unauthenticated) so this works even after
# DASHBOARD_USER/DASHBOARD_PASS activate HTTP Basic auth on the app.
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -fsS http://localhost:8050/healthz >/dev/null \
        && python -c "import sys; from src.ops.daily_scheduler import is_scheduler_alive; sys.exit(0 if is_scheduler_alive() else 1)" \
        || exit 1

EXPOSE 8050

# Default: trader mode with paper trading
CMD ["python", "main.py", "--mode", "trader", "--paper", "--host", "0.0.0.0"]
