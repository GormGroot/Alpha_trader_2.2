# Changelog

All notable changes to Alpha Trading Platform will be documented in this file.

Format follows [Semantic Versioning](https://semver.org/):
- **MAJOR** (v2.0, v3.0): Breaking changes — requires Gorm to update config/setup
- **MINOR** (v1.1, v1.2): New features or improvements — backwards compatible
- **PATCH** (v1.0.1, v1.0.2): Bug fixes — safe to update immediately

---

## [2.2.1] - 2026-03-31
### Code Review — 118 fixes across 52 files (1658 tests green)

3-runde code review med Claude Opus 4.6. Alle fixes er bagudkompatible.

#### KRITISKE fixes (22)
- **Short-selling:** Nordnet sell() crash fix, fee-beregning for shorts, 150% margin-reservation + korrekt frigørelse ved close
- **Risk manager:** Exit-bypass virker nu for ALLE positioner (ikke kun shorts), JSON config valideres med grænser
- **Paper broker:** NoneType crash på limit-ordrer, fee refunderes ved fejlet short-åbning
- **Threading:** Deadlock i ConnectionManager (Lock → RLock)
- **auto_trader:** exit_price → price (stille fejl der forhindrede RM sync), dobbelt-booking prevention
- **Docker:** PostgreSQL/Redis bundet til localhost, password kræves

#### HØJE fixes (33)
- SQLite thread-safety (timeout=10), Sharpe ratio ddof=1
- Circuit breaker nulstilles ikke længere som sideeffekt af property-access
- Email rate limiting (5 min cooldown per kategori)
- Signal prune kører periodisk (ikke hvert scan), SQL fix
- SL/TP justeret fra crypto-only (1.5%/2.5%) til multi-asset (3%/5%)
- position_size_pct gendannes nu fra base ved normalt regime
- stop_loss muterer ikke længere delt state under _check_exits
- Netto-eksponering korrekt for short-positioner
- Dividend dashboard bruger korrekte attributnavne

#### MEDIUM + LAV fixes (63)
- 4 manglende dependencies: scikit-learn, pyarrow, feedparser, pyyaml
- Backup-stier rettet til data_cache/
- HTML escape i email notifikationer
- VIX fjernet fra handelsliste (indeks, ikke handelbart)
- Duplikat Airbus (AIR.DE) fjernet
- ECB CSV parser robusthed
- Corporate tax: commit parameter forhindrer sideeffekter ved simulering
- Exchange stop-loss for alle 23 børser (ikke kun 4)
- risk_sizing.json auto-detect format (procent vs. decimal)
- 148 test-fejl løst via conftest isolation
- Docker: Python 3.12, fjernet deprecated version field

#### Gorm-specifikke fixes (fundet i v2.2 merge)
- `_now_cet()` uendelig rekursion rettet i daily_scheduler
- `router._brokers` → public API i setup_connection_monitor

---

## [2.2.0] - 2026-03-30
### Gorm — Sell routing, currency, UI
- Sell routing: automatisk position-aware broker selection
- NZ exchange (NZX) tilføjet
- Web-synced time service (nightly 23:00 CET resync)
- ContinuousNewsFetcher (5 min interval)
- Dashboard UI forbedringer (trading, portfolio, performance)
- Currency standardisering
- Sprogfiler opdateret (dansk, engelsk)

---

## [2.1.0] - 2026-03-28
### Trading fees, weekend approval, risk updates
- Trading fee calculator med YAML config
- Weekend approval system
- Risk & strategy parameter updates
- NPU accelerator setup
- Smoke test script

---

## [1.0.0] - 2026-03-19
### Initial Release
- Core trading engine with signal generation
- Multi-strategy support (momentum, mean reversion, breakout)
- Real-time data integration
- Risk management system
- Backtesting framework
- Docker deployment setup
- Delivered to Gorm as initial partner version
