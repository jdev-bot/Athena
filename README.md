# Athena

ML-powered autonomous strategy generator and execution engine for crypto futures trading via **Freqtrade**.

## Overview

Athena discovers, backtests, promotes, forward-tests, and optionally deploys algorithmic trading strategies through a fully automated genetic pipeline. All strategies run against **real market data** (ccxt → Binance) — no synthetic candles.

## Capabilities

| Capability | Status |
|---|---|
| Strategy DNA encoding + random generation | ✅ |
| Genetic algorithm evolution with crossover/mutation | ✅ |
| Real-market backtesting via Freqtrade API | ✅ |
| Performance scoring with robustness gates | ✅ |
| Strategy promotion / demotion lifecycle | ✅ |
| Portfolio manager with risk-budget sizing | ✅ |
| Market regime detection + template filtering | ✅ |
| ML predictor seeding into GA population | ✅ |
| Post-promotion hyperopt parameter tuning | ✅ |
| DNA versioning with immutable snapshots | ✅ |
| Forward-test dry-run with signal persistence | ✅ |
| Drift detection + auto-demote + mini-GA restart | ✅ |
| Live paper trading via Freqtrade bot spawn | ✅ |
| Kill-switch circuit breakers (drawdown / daily loss) | ✅ |
| Autonomous evolution scheduler | ✅ |
| Forward-test scheduler for promoted strategies | ✅ |
| Prometheus-compatible `/metrics` telemetry | ✅ |
| Web dashboard with real-time panels | ✅ |
| 272 automated tests | ✅ |

## Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                        FastAPI / Uvicorn                             │
│  /health  /strategies  /backtests  /stats  /live/*  /portfolio/*   │
│  /forward/*  /drift/*  /dna/*  /scheduler/*  /metrics  /           │
└──────────────┬─────────────────────┬──────────────┬────────────────┘
               │                     │              │
        ┌──────▼──────┐      ┌─────▼──────┐  ┌──▼──────────┐
        │  Generator  │      │  Backtest  │  │  LiveRunner │
        │  DNA / GA   │      │  Freqtrade │  │  BotManager │
        └─────────────┘      └─────┬──────┘  └──────┬──────┘
                                   │                  │
                          ┌────────▼─────────┐  ┌────▼──────┐
                          │ MarketDataProvider│  │  LiveFeed │
                          │  ccxt / Binance   │  │  WS 1m    │
                          └───────────────────┘  └───────────┘
```

## Quick Start

```bash
# 1. Install
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 2. Start PostgreSQL (or SQLite works for local dev)
# Default: DATABASE_URL=sqlite:///athena.db

# 3. Run server
python -m athena.services.api
# → http://127.0.0.1:8000/health

# 4. Run tests
pytest tests/ -q
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/` | Dashboard HTML |
| GET | `/health` | Service status |
| POST | `/strategies/generate` | Generate strategies by DNA |
| GET | `/strategies` | List/filter strategies |
| GET | `/strategies/{id}` | Strategy detail |
| POST | `/evolve` | Full GA evolution (background) |
| POST | `/backtests/run` | Run Freqtrade backtest |
| GET | `/backtests` | Completed backtest list |
| GET | `/stats` | Aggregate counts |
| POST | `/promote` | Run gates + promote |
| POST | `/deploy` | Export strategy to `.py` file |
| POST | `/hyperopt` | Post-promotion hyperopt |
| POST | `/live/start` | Start paper/live bot |
| POST | `/live/stop` | Stop bot session |
| GET | `/live/status` | Session stats |
| POST | `/forward/run` | Dry-run forward test |
| GET | `/forward/pnl` | Cumulative PnL series |
| GET | `/forward/scheduler/status` | Forward scheduler status |
| POST | `/portfolio/add` | Add promoted strategy |
| POST | `/portfolio/remove` | Remove from portfolio |
| POST | `/portfolio/kill` | Emergency stop all |
| GET | `/portfolio/status` | Portfolio snapshot |
| GET | `/drift/status` | Drift classification |
| GET | `/drift/history` | Drift time-series |
| POST | `/drift/demote` | Manual demote |
| POST | `/dna/snapshot` | Save DNA version |
| GET | `/dna/versions` | List DNA versions |
| GET | `/dna/restore` | Restore DNA version |
| GET | `/metrics` | Prometheus metrics |

## Key Components

| File | Purpose |
|---|---|
| `athena/services/api.py` | FastAPI app (30+ endpoints) |
| `athena/core/freqtrade_wrapper.py` | Programmatic Freqtrade backtesting |
| `athena/core/engine.py` | `AthenaEngine` — GA loop + evaluation |
| `athena/live/bot_manager.py` | Spawn/monitor Freqtrade bot processes |
| `athena/live/deployer.py` | Build Freqtrade `user_data` directory |
| `athena/live/freqtrade_config.py` | Generate valid `config.json` |
| `athena/live/runner.py` | `LiveRunner` — public API over `BotManager` |
| `athena/live/scheduler.py` | Autonomous evolution scheduler |
| `athena/live/forward_scheduler.py` | Forward-test scheduler with drift checks |
| `athena/live/feed.py` | ccxt Pro WebSocket candle stream |
| `athena/portfolio/manager.py` | Multi-strategy capital allocation |
| `athena/generator/dna.py` | DNA encoding/decoding |
| `athena/generator/ga_engine.py` | Genetic algorithm engine |
| `athena/generator/templates.py` | Strategy template library |
| `athena/evaluator/scorer.py` | Performance scoring + gates |
| `athena/market/provider.py` | Market data via ccxt |
| `athena/services/ui/index.html` | Standalone dashboard |

## Configuration

Environment variables (see `athena/common/config.py`):

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `sqlite:///athena.db` | PostgreSQL or SQLite |
| `AUTO_SCHEDULER` | `false` | Start evolution scheduler on boot |
| `FORWARD_SCHEDULER_AUTO` | `false` | Start forward scheduler on boot |
| `BACKTEST_FRAMEWORK` | `freqtrade` | Engine backend |

## Testing

```bash
pytest tests/ -q
```

**Test breakdown (272 total)**

| File | Tests | Focus |
|---|---|---|
| `test_e2e.py` | 22 | Full API round-trip |
| `test_integration.py` | 10 | Engine + API integration |
| `test_bot_manager.py` | 4 | Bot spawn, kill-switch, API proxy |
| `test_bridge.py` | 8 | Dry-run signal logic |
| `test_dashboard.py` | 51 | HTML, CSS, JS, API contracts |
| `test_dna_versioning.py` | 10 | Snapshot/restore API |
| `test_drift.py` | 13 | Degradation classification |
| `test_feedback.py` | 10 | Drift severity rules |
| `test_forward_drift.py` | 5 | Ratio-based drift checks |
| `test_forward_runner.py` | 2 | Forward test execution |
| `test_forward_scheduler.py` | 13 | Scheduler cycles + API |
| `test_portfolio.py` | 27 | Capital allocation, kill-switch |
| `test_regime.py` | 20 | Regime detection + template mapping |
| `test_regime_basket.py` | 9 | Multi-pair regime baskets |
| `test_risk_budget_sizing.py` | 14 | Position sizing math |
| `test_ml_predictor.py` | 13 | ML seeding in GA |
| `test_hyperopt.py` | 5 | Hyperopt deployment |
| `test_hyperopt_expand.py` | 5 | DNA range expansion |
| `test_downloader.py` | 5 | Market data caching |
| `test_telemetry.py` | 7 | Metrics collection |
| `test_api_endpoints.py` | 5 | Signals, forward run |
| `test_api_forward_pnl.py` | 3 | PnL time-series |

## Phases

| Phase | Description | Status |
|---|---|---|
| 1 | Strategy generator + Freqtrade backtest + API | ✅ |
| 2 | LiveFeed WebSocket + forward-test + kill-switch | ✅ |
| 3 | Freqtrade dry-run bot spawn + deployer + proxy | ✅ |
| 4 | Production hardening (Docker, migrations, auth) | 🔲 |
| 5 | Multi-exchange support (Bybit, OKX, Kraken) | 🔲 |
| 6 | Dashboard WebSocket real-time push | 🔲 |

## License

MIT
