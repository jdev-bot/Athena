# Athena Project Context

## Overview
ML-powered autonomous strategy generator and execution engine for crypto futures trading via **Freqtrade**.

## Stack
- **Backend:** Python 3.12, FastAPI, SQLAlchemy + PostgreSQL/SQLite
- **Genetic Engine:** Custom GA with DNA encoding
- **Backtesting:** Freqtrade framework with real market data via ccxt
- **Live Execution:** Freqtrade bot subprocess in paper mode
- **Live Feed:** ccxt Pro WebSocket streaming (Binance)
- **Tests:** pytest, 272 tests across 22 files (all passing)
- **Repo:** https://github.com/jdev-bot/Athena

## Git
- **Remote:** https://github.com/jdev-bot/Athena.git
- **Branch:** main
- **Latest commit:** b6c301d (comprehensive dashboard test suite)

## Phase Status

### Phase 1 ✅ — Generator + Backtest
- Strategy DNA generator + GA engine
- FastAPI service with 30+ endpoints
- Freqtrade `Backtesting` API with real market data
- 22 E2E tests covering API, DNA, GA, scorer, engine, wrapper

### Phase 2 ✅ — LiveFeed + Forward-test
- `LiveFeed` — async ccxt Pro WebSocket streaming 1m candles
- `ForwardRunner` — dry-run evaluator (signal-only, no real orders)
- Kill-switch circuit breakers (max_drawdown 15%, daily_loss_limit 10%)
- `live_sessions` DB table with runtime stats
- Forward-test scheduler with drift detection

### Phase 3 ✅ — Freqtrade Live Execution
- `BotManager` spawns `freqtrade trade` subprocess in paper mode
- `Deployer` writes strategy + config into Freqtrade `user_data` directory
- `LiveRunner` public API wrapping BotManager
- `/live/start`, `/live/stop`, `/live/status` endpoints
- Kill-switch monitor polls Freqtrade API every 30s
- Graceful SIGTERM shutdown + deploy dir cleanup

### Phase 4 🔲 — Production Hardening
- Docker containerization
- Alembic database migrations
- API authentication / rate limiting
- systemd service files
- Structured logging + log rotation

### Phase 5 🔲 — Multi-Exchange
- Config templates for Bybit, OKX, Kraken
- Exchange-specific fee structures + leverage limits
- Per-exchange data downloaders

### Phase 6 🔲 — Real-time Dashboard
- WebSocket push from live sessions
- Live PnL chart updates
- Signal stream visualization

## Endpoints (30+)

| Method | Path | Description |
|---|---|---|
| GET | `/` | Dashboard HTML |
| GET | `/health` | Service status |
| POST | `/strategies/generate` | Generate strategies |
| GET | `/strategies` | List/filter |
| GET | `/strategies/{id}` | Detail |
| POST | `/evolve` | GA evolution |
| POST | `/backtests/run` | Backtest |
| GET | `/backtests` | Results list |
| GET | `/stats` | Aggregates |
| POST | `/promote` | Run gates |
| POST | `/deploy` | Export `.py` |
| POST | `/hyperopt` | Parameter tuning |
| POST | `/live/start` | Start bot |
| POST | `/live/stop` | Stop bot |
| GET | `/live/status` | Session stats |
| POST | `/forward/run` | Dry-run test |
| GET | `/forward/pnl` | PnL series |
| GET/POST | `/forward/scheduler/*` | Scheduler control |
| GET/POST | `/portfolio/*` | Portfolio ops |
| GET/POST | `/drift/*` | Drift monitoring |
| POST/GET | `/dna/*` | DNA versioning |
| GET | `/metrics` | Prometheus metrics |

## Architecture

```
FastAPI / Uvicorn
├── /strategies    → Generator (DNA / GA)
├── /backtests     → FreqtradeWrapper
├── /live          → LiveRunner → BotManager → Freqtrade process
├── /forward       → ForwardRunner + ForwardScheduler
├── /portfolio     → PortfolioManager
├── /drift         → FeedbackCollector + AdaptiveLoop
├── /dna           → DNA versioning
├── /scheduler     → AutonomousScheduler
└── /metrics       → TelemetryCollector
```

## Key Files

| File | Purpose |
|---|---|
| `athena/services/api.py` | FastAPI app |
| `athena/core/freqtrade_wrapper.py` | Programmatic backtesting |
| `athena/core/engine.py` | `AthenaEngine` |
| `athena/live/bot_manager.py` | Bot process lifecycle |
| `athena/live/deployer.py` | Deploy directory builder |
| `athena/live/freqtrade_config.py` | Freqtrade `config.json` generator |
| `athena/live/runner.py` | `LiveRunner` |
| `athena/live/scheduler.py` | Autonomous evolution |
| `athena/live/forward_scheduler.py` | Forward-test + drift |
| `athena/live/feed.py` | WebSocket candle stream |
| `athena/portfolio/manager.py` | Capital allocation |
| `athena/generator/dna.py` | DNA encoder |
| `athena/generator/ga_engine.py` | GA engine |
| `athena/generator/templates.py` | Strategy templates |
| `athena/evaluator/scorer.py` | Scoring + gates |
| `athena/market/provider.py` | ccxt data |
| `athena/services/ui/index.html` | Dashboard |
| `tests/` | 22 test files, 272 tests |

## Environment
- Python venv at `.venv/`
- Freqtrade installed in venv
- ccxt 4.x installed
- PostgreSQL on `localhost:5435` with `athena_db` (SQLite default for dev)

## Testing
```bash
pytest tests/ -q
```
