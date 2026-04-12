# Alpha Trader 2.2 — Claude Kontekst

## Projekt
Fuldautomatisk trading-platform i Python 3.14. Multi-broker, multi-strategi, 24/7 global markedsdækning.

**Repo:** https://github.com/GormGroot/Alpha_trader_2.2 (branch: main)
**Lokation:** `/Users/olekjaergaardschroter/Desktop/Claude/Alpha_trader_2.2/`

## Team
- **Ole** — kommerciel/strategisk, ejer dette projekt
- **Gorm** — primær Python-udvikler, ejer GitHub-repo
- **HC** — Ole's far, del af trading-teamet

## Platform Status (april 2026)
- **Tests:** 1658/1658 grønne
- **Broker:** Alpaca paper trading aktiv ($100k paper, ACTIVE)
- **API keys:** `.env` (ikke i git)
- **Fase:** Fase 2 af live-trading-plan gennemført ✅

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
# Forventet: 1658 passed
```
