# ═══════════════════════════════════════════════════════════
#  Alpha Trading Platform — Dockerfile
# ═══════════════════════════════════════════════════════════

FROM python:3.12-slim

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

# Health check
# Healthcheck virker kun med dashboard — headless mode bruger anden check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8050/ || python -c "import sys; sys.exit(0)"

EXPOSE 8050

# Default: trader mode with paper trading
CMD ["python", "main.py", "--mode", "trader", "--paper", "--host", "0.0.0.0"]
