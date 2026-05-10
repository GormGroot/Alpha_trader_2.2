# Alpha Trader 2.2 — Claude Kontekst

## Projekt
Fuldautomatisk trading-platform i Python 3.14. Multi-broker, multi-strategi, 24/7 global markedsdækning.

**Repo:** https://github.com/GormGroot/Alpha_trader_2.2 (branch: main)
**Lokation:** `/Users/olekjaergaardschroter/Desktop/Claude/Alpha_trader_2.2/`

## Team
- **Ole** — kommerciel/strategisk, ejer dette projekt
- **Gorm** — primær Python-udvikler, ejer GitHub-repo
- **HC** — Ole's far, del af trading-teamet

## Platform Status (10. maj 2026 — Vej A komplet)
- **Tests:** 2055+ grønne (1737 baseline + 19 fra Vej A circuit/duplikat)
- **Broker:** Alpaca paper trading aktiv ($99k aktuel, 11 positioner)
- **API keys:** `.env` (ikke i git)
- **Fase:** Vej A.7 (parameter-optimering) komplet, klar til 5-7 dages paper
- **Mobile API:** http://localhost:8051 (PWA + REST + 2FA + pause/sælg-knapper)
- **Dashboard:** http://localhost:8050 (forenklet til 6 hovedsider)

### Strategi (efter param-sweep)
- **SMA 50/200** (vægt 0.45) — Golden/Death Cross, PF 17.86, MaxDD 3.5%, Sharpe 2.10
- **RSI 14, 35/65** (vægt 0.35) — Loose tærskler, PF 5.20, WR 77%
- **Combined** (vægt 0.20) — vægtet konsensus af de to ovenstående
- **Inaktive:** ML, EnsembleML (utrænede modeller — afventer Gorm)

### Vej A audit-fixes (commit 6a7592a + 7c0866a)
1. **A1:** PWA viser direkte Alpaca-positioner (ikke lokal SQLite drift)
2. **A2:** Utrænede ML-strategier deaktiveret (var 60% død vægt)
3. **A3:** Circuit breakers verificeret + equity sync fra broker
4. **A4:** Duplikat-handler blokeret (fase 3/4 isolation + DB-idempotency)
5. **A5:** PWA Pause-knap + ✕-knap pr. position + Telegram-notifikation
6. **A6:** Dashboard 20 → 6 hovedsider
7. **A7:** Strategi-parametre optimeret efter 30-symbol param-sweep

### Sikkerhed (login)
- **Lag 1:** Username + password (konstant-tids sammenligning, bcrypt-kompatibel)
- **Lag 2:** Telegram 2FA (6-cifret kode, 5 min levetid, max 3 forsøg, 30s rate-limit)
- **Lag 3:** JWT-token (30 min auto-logout, justerbar via `APP_TOKEN_TTL_MINUTES`)
- **Lag 4:** Geo-lock (kun DK, bypass-kode for rejser, fail-open ved netværksfejl)
- **Lag 5:** Telegram-notifikationer ved login/blokering/2FA-fejl

### Konfiguration (.env)
- `APP_USERNAME`, `APP_PASSWORD`, `APP_SECRET_KEY` — basis auth
- `APP_TOKEN_TTL_MINUTES=30` — auto-logout (15-1440 min)
- `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` — 2FA + notifikationer
- `GEO_LOCK_ENABLED=true`, `GEO_LOCK_COUNTRIES=DK`, `GEO_LOCK_BYPASS=...` — geo-lock

## Arkitektur
```
main.py                  — entry point (--mode trader/research/dashboard)
src/
  broker/                — AlpacaBroker, PaperBroker, BrokerRouter, ConnectionManager
  risk/                  — RiskManager, PortfolioTracker, DynamicRiskManager
  trader/                — AutoTrader, SignalEngine, AlphaScore
  strategy/              — CombinedStrategy, RSI, SMA, ML, Pattern
  data/                  — MarketData, OptionsFlow, AlternativeData
  ops/                   — DailyScheduler, NPUAccelerator, TimeService
  sentiment/             — SentimentAnalyzer
  learning/              — ContinuousLearner
  tax/                   — dansk skat + FX
config/
  global_stop_loss.json  — 3% stop-loss
  risk_sizing.json       — max_position_pct: 5%, max_dkk_per_symbol: 50000
  max_positions.json     — max_open_positions: 20
  default_config.yaml    — master config
```

## Vigtige konfigurationer (april 2026)
| Fil | Nøgleværdi | Note |
|-----|-----------|------|
| `config/global_stop_loss.json` | 3% | Hævet fra 1.5% (Gorms original — afklar grund) |
| `config/risk_sizing.json` | max_position_pct: 5%, max_dkk: 50k | Hævet fra 10%/5k til paper trading |
| `config/max_positions.json` | 20 | Hævet fra 8 |
| `.env` | ALPACA_API_KEY, ALPACA_SECRET_KEY | Paper trading keys |

## Kendte issues / TODO
- **FinBERT/NPU:** `transformers` ikke installeret → keyword-fallback. Installer `pip3 install transformers` for bedre sentiment
- **Options flow NaN:** `[options] UOA fejl: cannot convert float NaN to integer` — yfinance data-problem, ikke kritisk
- **MSFT SELL 403:** Alpaca afviser short-sell uden short-tilladelse — forventet i paper mode
- **Gorm:** Spørg om grunden til 1.5% stop-loss (vi har hævet til 3%)
- **SSL cert:** time-service kan ikke nå worldtimeapi.org — lokalt certifikat-problem, harmløst

## Rettede bugs (april 2026)
1. Position stacking — AutoTrader åbnede duplikat-positioner (parameter-navn fejl)
2. sqlite3.Row crash — ContinuousLearner manglede `dict()` konvertering
3. AlphaScore clustering — alle scores 48-54 → confidence-weighted averaging
4. NPU log spam — 328x → 1x (singleton + class-level cache)
5. Options flow NaN — tidlig exit for symboler uden options
6. Scan interval — dynamisk wait baseret på faktisk scan-varighed

## Sikkerhedshardening (Fase 1, april 2026)
- Live-mode kræver "LIVE" input ved opstart
- Position-reconciliation med broker ved live-opstart
- ConnectionManager → RiskManager: ordrer afvises ved broker-disconnect
- Circuit breaker auto-reset kl. 08:45 CET dagligt
- urllib3 ≥2.0.0 (CVE-fix)
- `load_dotenv()` tilføjet i main.py

## Live-trading plan
Se `/Users/olekjaergaardschroter/.claude/plans/elegant-snuggling-globe.md`

| Fase | Status |
|------|--------|
| 1: Sikkerhedshardening | ✅ Færdig |
| 2: Paper trading verification | ✅ Færdig |
| 3: Alpaca API-opsætning | ✅ Færdig |
| 4: 1-2 ugers paper trading (Mac Mini) | 🔄 Næste |
| 5: Første live-handel | ⏳ Venter |
| 6: Løbende overvågning | ⏳ Venter |

## Kør platformen
```bash
# Paper trading med dashboard
python3 main.py --mode trader --paper

# Dashboard: http://localhost:8050
# Stop: Ctrl+C
```

## Tests
```bash
python3 -m pytest tests/ -q
# Forventet: 1737 passed
```

---

## Mac Mini deployment (april 2026)

**Status:** Skal deployeres på Oles Mac Mini hjemme. Pull seneste fra GitHub:
```bash
cd ~/Desktop/Claude/Alpha_trader_2.2
git pull            # henter v2.3.1 (commit a0b0c5a)
pip install -r requirements.txt
python3 main.py --mode trader --paper
```

**Login-test:**
- http://localhost:8051 (mobile API + 2FA)
- http://localhost:8050 (dashboard)
- Credentials i `.env` (ole / alphavision2026)

**Kendt issue (kun MacBook):** Oles MacBook kan ikke nå localhost via browser
(Chrome/Safari/incognito). Curl virker fint. Mistanke: proxy-extension eller
VPN-software. Mac Mini har ingen sådan så det BØR virke direkte.

**Hvis login fejler igen:** Tjek `/tmp/api_running.log` for at se om requests
kommer ind. Hvis ja men ingen Telegram-kode: tjek SSL/Telegram-API direkte
med `scripts/test_login.py`.

---

## Sessionslog

### 30. april 2026 — Login-system v2 + Sikkerhedshardening (commit a0b0c5a)

**Hvad blev lavet:**

**Sikkerhedshardening (6 ændringer):**
1. Live-mode bekræftelsesprompt i `main.py`
2. Stop-loss config 1.5% → 3%
3. Position-reconciliation ved live-opstart
4. ConnectionManager → RiskManager wired
5. Circuit breaker auto-reset 08:45 CET
6. urllib3 ≥2.0.0 (CVE-fix)

**Login-system v2 (5 sikkerhedslag):**
1. Username + password (konstant-tids sammenligning)
2. Telegram 2FA (6-cifret kode, 5 min, max 3 forsøg)
3. JWT 30 min auto-logout (justerbar)
4. Geo-lock kun DK (bypass-kode for rejser)
5. Telegram-notifikationer (godkendt/fejl/blokering)

**Lækker login-UI:**
- 2-trins flow med glassmorphism
- Auto-submit ved 6 cifre
- 5 min countdown-timer
- Send ny kode-knap

**Bug-fixes (6):**
- Position stacking
- sqlite3.Row crash
- AlphaScore clustering
- NPU log spam (328x → 1x)
- Options flow NaN
- Scan interval

**Frontend/backend infrastruktur:**
- `load_dotenv()` i main.py + server.py + auth.py
- Trailing-slash kompatibilitet på alle /api/* routes
- PWA-routing fanger ikke /api/* paths
- No-cache middleware på /api/*-svar
- SSL cert fix til macOS (certifi)
- Cache-busting på alle frontend fetch-kald

**Tests:** 1658 baseline → **1737 passed** (20 nye API auth + 17 sikkerhedsfeatures + 42 andre)

**Konfiguration justeret for paper trading:**
- max_position_pct: 10% → 5%
- max_dkk_per_symbol: 5000 → 50000
- max_open_positions: 8 → 20

**Nye filer:**
- `src/api/two_factor.py` — Telegram 2FA-service
- `src/api/geo_lock.py` — geo-lock med IP-lookup via ip-api.com
- `src/api/security_notify.py` — Telegram login-notifikationer
- `tests/test_api_auth.py` — 20 integrationstests
- `tests/test_security_features.py` — 17 sikkerhedstests
- `scripts/test_login.py` — manuel verifikations-script

**Eksterne integrations:**
- Telegram bot: @alphatraderok_bot ("Alpha Trader Ole")
- IP geo-lookup: ip-api.com (gratis, 45 req/min)

**Browser-issue på Oles MacBook:**
Trods alt fungerer login ikke fra Chrome/Safari/incognito på MacBook. Curl virker
fint. Server logger ingen requests fra browser. Mistanke: proxy-extension eller
VPN-software. Plan: deploy på Mac Mini der ikke har det issue.
