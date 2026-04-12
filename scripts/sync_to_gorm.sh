#!/usr/bin/env bash
# =============================================================================
# Alpha Trader — Sync script (kører på Ole's Mac Mini)
# =============================================================================
# Synkroniserer data til Gorms Linux-maskine og henter trænede modeller tilbage.
#
# Brug:
#   ./scripts/sync_to_gorm.sh                  # Push data + start træning + pull model
#   ./scripts/sync_to_gorm.sh --pull-only       # Kun hent modeller
#   ./scripts/sync_to_gorm.sh --push-only       # Kun send data
#   ./scripts/sync_to_gorm.sh --no-train        # Push + pull men start ikke træning
#
# Opsætning (én gang):
#   1. Rediger GORM_HOST nedenfor med Gorms server IP/hostname
#   2. Sørg for at SSH-nøgle er sat op: ssh-copy-id gorm@<server>
#   3. Kør setup_gorm.sh på Gorms maskine én gang
#
# =============================================================================

set -e

# ─── KONFIGURATION (ret disse) ─────────────────────────────────
GORM_HOST="gorm-server"          # SSH hostname eller IP til Gorms maskine
GORM_USER="gorm"                  # SSH brugernavn på Gorms maskine
GORM_PROJECT="/home/gorm/alpha_trader_2.2"  # Sti til projektet på Gorms maskine
GORM_VENV="/home/gorm/alpha_trader_venv"    # Venv på Gorms maskine
SSH_KEY="$HOME/.ssh/id_ed25519"  # SSH nøgle (skift hvis du bruger anden)
# ───────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$PROJECT_DIR/models"

# Farver
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log()  { echo -e "${BLUE}[sync]${NC} $1"; }
ok()   { echo -e "${GREEN}[OK]${NC} $1"; }
warn() { echo -e "${YELLOW}[!]${NC} $1"; }
err()  { echo -e "${RED}[FEJL]${NC} $1"; exit 1; }

SSH_OPTS="-i $SSH_KEY -o ConnectTimeout=10 -o StrictHostKeyChecking=no"
SSH_CMD="ssh $SSH_OPTS $GORM_USER@$GORM_HOST"
RSYNC_OPTS="-az --progress --exclude=__pycache__ --exclude='*.pyc' --exclude='.env'"

# Argumenter
PULL_ONLY=false
PUSH_ONLY=false
NO_TRAIN=false

for arg in "$@"; do
    case $arg in
        --pull-only)  PULL_ONLY=true ;;
        --push-only)  PUSH_ONLY=true ;;
        --no-train)   NO_TRAIN=true ;;
        --help|-h)
            echo "Brug: $0 [--pull-only|--push-only|--no-train]"
            exit 0 ;;
    esac
done

echo ""
echo "============================================================"
echo "  ALPHA TRADER — SYNC TIL GORMS MASKINE"
echo "  Host: $GORM_USER@$GORM_HOST"
echo "  Projekt: $PROJECT_DIR"
echo "============================================================"
echo ""

# ── Tjek SSH forbindelse ───────────────────────────────────────
log "Tjekker SSH forbindelse til $GORM_HOST..."
if ! $SSH_CMD "echo connected" &>/dev/null; then
    err "Kan ikke forbinde til $GORM_HOST

  Tjek:
  1. Er Gorms maskine tændt og online?
  2. Er SSH-nøgle sat op? Kør: ssh-copy-id -i $SSH_KEY $GORM_USER@$GORM_HOST
  3. Er GORM_HOST korrekt? (nuværende: $GORM_HOST)
  4. Prøv manuelt: ssh $GORM_USER@$GORM_HOST"
fi
ok "SSH forbindelse OK"

# ── Opret mappe på Gorms maskine ──────────────────────────────
$SSH_CMD "mkdir -p $GORM_PROJECT/models $GORM_PROJECT/logs $GORM_PROJECT/data_cache $GORM_PROJECT/scripts"

# ── PUSH: Send scripts og kode til Gorm ───────────────────────
if [ "$PULL_ONLY" = false ]; then
    log "Sender kode og scripts til Gorms maskine..."

    rsync $RSYNC_OPTS \
        -e "ssh $SSH_OPTS" \
        "$PROJECT_DIR/train_remote.py" \
        "$PROJECT_DIR/scripts/setup_gorm.sh" \
        "$PROJECT_DIR/scripts/activate_and_train.sh" \
        "$GORM_USER@$GORM_HOST:$GORM_PROJECT/" \
        2>/dev/null || true

    # Send src/ mapper der bruges af train_remote (det er standalone, men lad os sende hele src/)
    # train_remote.py er standalone og behøver ikke src/ — men send den alligevel som backup
    rsync $RSYNC_OPTS \
        -e "ssh $SSH_OPTS" \
        --exclude="data_cache/" \
        --exclude="models/" \
        --exclude="logs/" \
        --exclude=".env" \
        --exclude=".git/" \
        --exclude="*.log" \
        "$PROJECT_DIR/requirements.txt" \
        "$GORM_USER@$GORM_HOST:$GORM_PROJECT/"

    ok "Kode sendt"

    # Sørg for scripts er eksekverbare på Gorms maskine
    $SSH_CMD "chmod +x $GORM_PROJECT/scripts/*.sh 2>/dev/null || true; chmod +x $GORM_PROJECT/train_remote.py 2>/dev/null || true"
fi

if [ "$PUSH_ONLY" = true ]; then
    ok "Push komplet (--push-only valgt)"
    exit 0
fi

# ── TRÆNING: Start træning på Gorms maskine ───────────────────
if [ "$NO_TRAIN" = false ] && [ "$PULL_ONLY" = false ]; then
    log "Starter model-træning på Gorms maskine..."
    log "(Dette tager typisk 20-60 minutter afhængig af CPU/GPU)"
    echo ""

    # Kør træning i baggrunden med nohup så det fortsætter selv ved SSH-disconnect
    $SSH_CMD "
        cd $GORM_PROJECT
        source $GORM_VENV/bin/activate 2>/dev/null || true
        nohup python train_remote.py > logs/training_\$(date +%Y%m%d_%H%M%S).log 2>&1 &
        echo \"Træning startet med PID \$!\"
        echo \$! > /tmp/alpha_trainer.pid
    "

    echo ""
    log "Træning startet i baggrunden på Gorms maskine."
    log "Følg progress med:"
    echo "  ssh $GORM_USER@$GORM_HOST 'tail -f $GORM_PROJECT/logs/*.log'"
    echo ""
    log "Vent på at træningen er færdig, kør derefter:"
    echo "  $0 --pull-only"
    echo ""
    exit 0
fi

# ── PULL: Hent trænede modeller ───────────────────────────────
log "Henter trænede modeller fra Gorms maskine..."
mkdir -p "$MODELS_DIR"

# Tjek om der er modeller klar
MODEL_COUNT=$($SSH_CMD "ls $GORM_PROJECT/models/*.joblib 2>/dev/null | wc -l" || echo "0")
MODEL_COUNT=$(echo "$MODEL_COUNT" | tr -d ' ')

if [ "$MODEL_COUNT" -eq 0 ]; then
    warn "Ingen modeller fundet på Gorms maskine endnu."
    warn "Start træning med: $0"
    exit 1
fi

log "Fandt $MODEL_COUNT model-filer — henter..."

rsync $RSYNC_OPTS \
    -e "ssh $SSH_OPTS" \
    "$GORM_USER@$GORM_HOST:$GORM_PROJECT/models/" \
    "$MODELS_DIR/"

ok "Modeller hentet til $MODELS_DIR/"

# Vis hvad vi fik
echo ""
echo "  Modeller i $MODELS_DIR/:"
ls -lh "$MODELS_DIR/"*.joblib 2>/dev/null | awk '{print "    "$NF": "$5}' || echo "  (ingen endnu)"
echo ""

# Tjek om latest metrics findes og print dem
METRICS_FILE="$MODELS_DIR/latest_metrics.json"
if [ -f "$METRICS_FILE" ]; then
    log "Seneste trænings-metrics:"
    python3 -c "
import json, sys
with open('$METRICS_FILE') as f:
    m = json.load(f)
print(f\"  Trænet: {m.get('trained_at', 'ukendt')}\")
ml = m.get('ml')
if ml:
    print(f\"  MLStrategy:  AUC={ml.get('auc_roc',0):.4f}  Accuracy={ml.get('accuracy',0):.1%}  F1={ml.get('f1',0):.1%}\")
ens = m.get('ensemble')
if ens:
    print(f\"  Ensemble:    RF={ens.get('rf_auc',0):.4f}  XGB={ens.get('xgb_auc',0):.4f}  Acc={ens.get('ensemble_acc',0):.1%}\")
" 2>/dev/null || true
fi

echo ""
echo "============================================================"
echo -e "  ${GREEN}SYNC KOMPLET!${NC}"
echo "============================================================"
echo ""
echo "  Platformen vil automatisk bruge de nye modeller ved næste"
echo "  genstart. Genstart nu med:"
echo ""
echo "    lsof -ti:8050,8051 | xargs kill -9 2>/dev/null"
echo "    python3 main.py --mode trader --paper"
echo ""
echo "============================================================"
