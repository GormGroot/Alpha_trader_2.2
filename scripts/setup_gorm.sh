#!/usr/bin/env bash
# =============================================================================
# Alpha Trader — Setup script til Gorms Linux-maskine
# =============================================================================
# Kør dette script ÉN gang på Gorms maskine for at sætte alt op.
#
# Brug:
#   chmod +x setup_gorm.sh
#   ./setup_gorm.sh
#
# Hvad gør det?
#   1. Tjekker Python 3.9+
#   2. Opretter Python virtual environment (~/alpha_trader_venv)
#   3. Installerer alle nødvendige pakker (inkl. XGBoost med GPU-support)
#   4. Opretter models/ og logs/ mapper
#   5. Laver en test-kørsel for at verificere alt virker
#   6. Sætter cron job op (valgfri) til ugentlig auto-retræning
#
# =============================================================================

set -e  # Stop ved fejl

# Farver til output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log()     { echo -e "${BLUE}[setup]${NC} $1"; }
ok()      { echo -e "${GREEN}[OK]${NC} $1"; }
warn()    { echo -e "${YELLOW}[ADVARSEL]${NC} $1"; }
err()     { echo -e "${RED}[FEJL]${NC} $1"; exit 1; }

# Arbejdsmappe = scriptets placering
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$HOME/alpha_trader_venv"
MODELS_DIR="$PROJECT_DIR/models"
LOGS_DIR="$PROJECT_DIR/logs"

echo ""
echo "============================================================"
echo "  ALPHA TRADER — REMOTE TRAINING SETUP"
echo "  Projekt:  $PROJECT_DIR"
echo "  Venv:     $VENV_DIR"
echo "============================================================"
echo ""

# ── 1. Tjek Python version ─────────────────────────────────────
log "Tjekker Python version..."

PYTHON_CMD=""
for cmd in python3.11 python3.10 python3.9 python3; do
    if command -v "$cmd" &>/dev/null; then
        VER=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        MAJOR=$(echo "$VER" | cut -d. -f1)
        MINOR=$(echo "$VER" | cut -d. -f2)
        if [ "$MAJOR" -ge 3 ] && [ "$MINOR" -ge 9 ]; then
            PYTHON_CMD="$cmd"
            ok "Python $VER fundet ($cmd)"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    err "Python 3.9+ ikke fundet! Installer med: sudo apt install python3.11 python3.11-venv python3-pip"
fi

# ── 2. Tjek CUDA / GPU ─────────────────────────────────────────
log "Tjekker GPU/CUDA..."
if command -v nvidia-smi &>/dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)
    ok "GPU fundet: $GPU_INFO"
    HAS_GPU=true
else
    warn "Ingen NVIDIA GPU fundet — kører på CPU (langsommere, men virker)"
    HAS_GPU=false
fi

# ── 3. Opret virtual environment ───────────────────────────────
log "Opretter virtual environment..."
if [ -d "$VENV_DIR" ]; then
    warn "Venv findes allerede ($VENV_DIR) — genbruger"
else
    $PYTHON_CMD -m venv "$VENV_DIR"
    ok "Venv oprettet: $VENV_DIR"
fi

# Aktiver venv
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
ok "Venv aktiveret"

# Opgrader pip
pip install --upgrade pip --quiet
ok "pip opgraderet"

# ── 4. Installer dependencies ──────────────────────────────────
log "Installerer Python pakker..."

# Core pakker
pip install --quiet \
    "yfinance>=0.2.31" \
    "pandas>=2.1.0" \
    "numpy>=1.24.0" \
    "scikit-learn>=1.3.0" \
    "joblib>=1.3.0" \
    "loguru>=0.7.0"

ok "Core pakker installeret"

# XGBoost — med GPU support hvis tilgængeligt
log "Installerer XGBoost..."
if [ "$HAS_GPU" = true ]; then
    log "  Prøver XGBoost med GPU-support..."
    pip install --quiet "xgboost>=2.0.0" && ok "XGBoost (GPU) installeret" || {
        warn "XGBoost GPU fejlede — installerer CPU version"
        pip install --quiet "xgboost" && ok "XGBoost (CPU) installeret" || warn "XGBoost fejlede — bruger ExtraTrees"
    }
else
    pip install --quiet "xgboost>=2.0.0" && ok "XGBoost (CPU) installeret" || warn "XGBoost fejlede — bruger ExtraTrees"
fi

# Optuna til hyperparameter-søgning (valgfri men anbefalet)
log "Installerer Optuna (hyperparameter optimering)..."
pip install --quiet "optuna>=3.0.0" && ok "Optuna installeret" || warn "Optuna ikke installeret — bruger grid search"

# Øvrige nyttige pakker
pip install --quiet \
    "pyarrow>=14.0.0" \
    "requests>=2.32.0" \
    "python-dateutil>=2.8.2" \
    --quiet

ok "Alle pakker installeret"

# ── 5. Opret mapper ────────────────────────────────────────────
log "Opretter mapper..."
mkdir -p "$MODELS_DIR" "$LOGS_DIR"
ok "models/ og logs/ oprettet"

# ── 6. Skriv aktiverings-script ────────────────────────────────
ACTIVATE_SCRIPT="$PROJECT_DIR/scripts/activate_and_train.sh"
cat > "$ACTIVATE_SCRIPT" << EOF
#!/usr/bin/env bash
# Auto-genereret af setup_gorm.sh
# Aktiverer venv og kører træning
source "$VENV_DIR/bin/activate"
cd "$PROJECT_DIR"
python train_remote.py "\$@" 2>&1 | tee "$LOGS_DIR/training_\$(date +%Y%m%d_%H%M%S).log"
EOF
chmod +x "$ACTIVATE_SCRIPT"
ok "Aktiverings-script oprettet: $ACTIVATE_SCRIPT"

# ── 7. Test installation ───────────────────────────────────────
log "Tester installation..."

$VENV_DIR/bin/python -c "
import sys
pkgs = ['yfinance', 'sklearn', 'joblib', 'numpy', 'pandas']
ok = []
missing = []
for p in pkgs:
    try:
        __import__(p)
        ok.append(p)
    except ImportError:
        missing.append(p)

print(f'OK: {ok}')
if missing:
    print(f'MANGLER: {missing}')
    sys.exit(1)

try:
    import xgboost
    print(f'XGBoost: {xgboost.__version__}')
except ImportError:
    print('XGBoost: ikke installeret (bruger ExtraTrees)')

try:
    import optuna
    print(f'Optuna: {optuna.__version__}')
except ImportError:
    print('Optuna: ikke installeret (ok)')

print('Alle core pakker OK!')
"
ok "Installation verificeret"

# ── 8. Cron job (valgfri) ──────────────────────────────────────
echo ""
echo "------------------------------------------------------------"
echo "  VALGFRI: Opsæt ugentlig auto-retræning"
echo "------------------------------------------------------------"
echo ""
read -rp "Vil du opsætte ugentlig auto-retræning (søndag kl. 02:00)? [j/n]: " SETUP_CRON

if [[ "$SETUP_CRON" =~ ^[jJyY]$ ]]; then
    CRON_CMD="0 2 * * 0 $ACTIVATE_SCRIPT >> $LOGS_DIR/cron.log 2>&1"

    # Fjern eventuelt eksisterende alpha_trader cron job
    (crontab -l 2>/dev/null | grep -v "alpha_trader\|train_remote"; echo "$CRON_CMD") | crontab -
    ok "Cron job oprettet: kører søndag kl. 02:00"
    echo "  Se: crontab -l"
else
    log "Cron job ikke oprettet — kør manuelt med: ./scripts/activate_and_train.sh"
fi

# ── Færdig ────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo -e "  ${GREEN}SETUP KOMPLET!${NC}"
echo "============================================================"
echo ""
echo "  Kør træning med:"
echo "    cd $PROJECT_DIR"
echo "    ./scripts/activate_and_train.sh"
echo ""
echo "  Eller direkte:"
echo "    source $VENV_DIR/bin/activate"
echo "    python train_remote.py"
echo ""
echo "  Når Ole's Mac Mini kører sync_to_gorm.sh hentes modellerne"
echo "  automatisk og platformen bruger dem ved næste scan."
echo ""
echo "  Logfiler: $LOGS_DIR/"
echo "============================================================"
