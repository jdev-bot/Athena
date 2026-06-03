# Athena Architecture

ML-powered autonomous strategy generator and execution engine for crypto futures trading via **Freqtrade**.

---

## Design Philosophy

Athena wraps Freqtrade as a programmable library and layers a fully autonomous genetic pipeline on top. The design goals are:

1. **One strategy format** — a single `.py` file runs in backtest, paper, and live without rewrites.
2. **Real data only** — all backtests and forward tests use live Binance OHLCV via ccxt.
3. **Signal-only by default** — forward tests persist signals and PnL to DB, never submit real orders unless explicitly promoted to live mode.
4. **Kill-switch always on** — max drawdown (15%) and daily loss limit (10%) circuit breakers at portfolio, strategy, and bot levels.
5. **Autonomous evolution** — background schedulers run micro-GA cycles, forward-test promoted strategies, and auto-demote on drift.

---

## System Layers

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          FastAPI / Uvicorn                                │
│  Dashboard  Health  Strategies  Backtests  Live  Forward  Portfolio     │
│  Drift  DNA  Scheduler  Metrics                                         │
└──────────────┬─────────────────────┬──────────────┬──────────────────────┘
               │                     │              │
        ┌──────▼──────┐      ┌─────▼──────┐  ┌──▼──────────┐
        │  Generator  │      │  Backtest  │  │  LiveRunner │
        │  DNA / GA   │      │  Freqtrade │  │  BotManager │
        │  Templates  │      │  Wrapper   │  │  Deployer   │
        └─────────────┘      └─────┬──────┘  └──────┬──────┘
                                   │                  │
                          ┌────────▼─────────┐  ┌────▼──────────┐
                          │ MarketDataProvider│  │  ForwardRunner │
                          │  ccxt / Binance  │  │  (signal-only)│
                          └───────────────────┘  └───────────────┘
```

---

## Modules

### `athena/core` — Engine + Backtest

| File | Role |
|---|---|
| `engine.py` | `AthenaEngine` — orchestrates GA evolution, evaluation, promotion, deployment |
| `freqtrade_wrapper.py` | Programmatic Freqtrade backtesting in a temp project with real data |
| `hyperopt.py` | `HyperoptFinisher` — post-promotion parameter tuning via Freqtrade hyperopt |
| `dna_versioning.py` | Immutable DNA snapshots with versioned restore |
| `strategy_loader.py` | Dynamic strategy class loading (legacy) |

### `athena/generator` — Strategy Creation

| File | Role |
|---|---|
| `dna.py` | `DNAEncoder` — random generation, crossover, mutation, parameter mapping |
| `ga_engine.py` | `GAEngine` — population initialization, fitness evaluation, elitism |
| `templates.py` | 6 parametric Freqtrade `IStrategy` templates (trend, mean-reversion, breakout, etc.) |

### `athena/evaluator` — Scoring + Gates

| File | Role |
|---|---|
| `scorer.py` | `Scorer` — composite scoring (sharpe, sortino, calmar, win_rate) with promote/demote thresholds |
| `monte_carlo.py` | Trade-level Monte Carlo shuffling for robustness |

### `athena/live` — Execution Infrastructure

| File | Role |
|---|---|
| `bot_manager.py` | `BotManager` — spawn `freqtrade trade` subprocess, monitor via REST API, kill-switch |
| `deployer.py` | `Deployer` — build Freqtrade `user_data` directory (config + strategy + historical data) |
| `freqtrade_config.py` | Generate valid Freqtrade `config.json` from DNA + risk params |
| `runner.py` | `LiveRunner` — public async API over `BotManager` (start/stop/stats) |
| `feed.py` | `LiveFeed` — ccxt Pro WebSocket 1m candle streaming |
| `scheduler.py` | `AutonomousScheduler` — periodic background GA evolution with adaptive expansion |
| `forward_scheduler.py` | `ForwardScheduler` — dry-run forward-test of promoted strategies with drift checks |
| `data_downloader.py` | `download_pair_data` — Freqtrade CLI data download wrapper |
| `feedback.py` | `FeedbackCollector` + `AdaptiveLoop` — drift classification + auto-demote + mini-GA restart |
| `bridge.py` | `DryRunTrader` — forward-test signal evaluation without real exchange orders |

### `athena/market` — Market Data

| File | Role |
|---|---|
| `provider.py` | `MarketDataProvider` — ccxt OHLCV fetcher with caching |

### `athena/portfolio` — Capital Management

| File | Role |
|---|---|
| `manager.py` | `PortfolioManager` — multi-strategy allocation, risk budget sizing, kill-switch, correlation matrix |

### `athena/services` — API + Persistence

| File | Role |
|---|---|
| `api.py` | FastAPI app (30+ endpoints) |
| `models.py` | SQLAlchemy models: `StrategyModel`, `LiveSessionModel`, `Signal`, `LiveSnapshot`, `DnaSnapshot` |
| `telemetry.py` | `TelemetryCollector` — Prometheus-compatible `/metrics` |
| `forward_runner.py` | `run_forward()` — single-shot forward test |
| `ui/index.html` | Standalone dashboard (no external deps) |

### `athena/common` — Shared Models + Config

| File | Role |
|---|---|
| `models.py` | Pydantic models: `StrategyTemplate`, `StrategyStatus`, `PerformanceMetrics`, `ScoreResult`, `GenerationConfig` |
| `config.py` | Environment-based configuration (`DATABASE_URL`, `AUTO_SCHEDULER`, etc.) |

---

## Strategy Lifecycle

```
                    ┌─────────────┐
                    │    DRAFT    │
                    └──────┬──────┘
                           │ generate
                    ┌──────▼──────┐
                    │  GENERATED  │
                    └──────┬──────┘
                           │ backtest
              ┌────────────┼────────────┐
              ↓            ↓            ↓
        ┌─────────┐ ┌──────────┐ ┌───────────┐
        │  DONE   │ │  FAILED  │ │ DEMOTED   │
        └────┬────┘ └──────────┘ └───────────┘
             │
        ┌────▼────┐
        │ PROMOTE │ ← run gates (sharpe, drawdown, trades)
        └────┬────┘
             │
        ┌────▼────┐
        │ PROMOTED│
        └────┬────┘
             │ forward-test (scheduled)
    ┌────────┼────────┐
    ↓        ↓        ↓
┌───────┐ ┌──────┐ ┌────────┐
│ PASS  │ │ DRIFT│ │ KILL   │
└───┬───┘ └──┬───┘ └───┬────┘
    │        │         │
┌───▼───┐ ┌──▼────┐ ┌─▼─────┐
│ STAY  │ │DEMOTE │ │RETIRED│
│PROMOTED│ │+ mini- │ │       │
│        │ │ GA     │ │       │
└────────┘ └───────┘ └───────┘
```

**Status enum:** `DRAFT` → `GENERATED` → `BACKTEST_QUEUED` → `BACKTEST_RUNNING` → `BACKTEST_DONE` / `BACKTEST_FAILED` → `PROMOTED` / `RETIRED`

---

## Generator Architecture

1. **Template Library** — 6 Freqtrade `IStrategy` templates with parametric holes
2. **DNA Encoding** — Each strategy's free parameters as a typed vector
3. **Population** — Maintains N candidates per generation
4. **Fitness** — Backtest score via `FreqtradeWrapper.run_backtest()`
5. **Selection** — Rank-based selection
6. **Crossover** — Two-point crossover on DNA vectors (same template only)
7. **Mutation** — Gaussian noise on continuous params, random swap on discrete
8. **Elitism** — Top K survive unmodified
9. **ML Seed** — Trained regressor predicts fitness from DNA to seed promising candidates (optional, ratio-controlled)
10. **Regime Filter** — Market regime detection restricts templates to regime-appropriate types

---

## Evaluator Architecture

1. **Backtest** — Full historical backtest via `freqtrade.optimize.backtesting.Backtesting`
2. **Walk-Forward** — Implicit via Freqtrade's timerange splitting
3. **Monte Carlo** — Trade-level shuffling in `monte_carlo.py`
4. **Scoring** — Composite: `sharpe(40%) + sortino(30%) + calmar(20%) + win_rate(10%)`
5. **Gates** — Promote: sharpe > 1.0, drawdown < 15%, trades > 10. Demote: any gate fails.

---

## Live Execution Architecture

When `POST /live/start` is called:

1. `LiveRunner.start()` → `BotManager.start()`
2. `Deployer.deploy()` writes:
   - `strategies/AthenaStrategy.py` (compiled from DNA)
   - `config.json` (Freqtrade-valid with API server creds)
   - `data/` (3 days historical warmup via `download_pair_data`)
3. `subprocess.Popen(["python", "-m", "freqtrade", "trade", ...])`
4. Health check: `proc.poll()` after 500ms
5. `asyncio.create_task(_kill_switch_monitor())` polls Freqtrade `/api/v1/profit` every 30s
6. Athena proxies status via `GET /live/status` → queries Freqtrade API

When `POST /live/stop`:
1. `proc.send_signal(SIGTERM)` → Freqtrade catches and exits cleanly
2. `Deployer.cleanup()` removes temp directory
3. DB record updated with `stopped_at`

---

## Data Flow

```
Binance (ccxt)
    │
    ▼
MarketDataProvider.fetch_ohlcv()
    │
    ├──► FreqtradeWrapper._write_candle_data() → backtest
    │
    ├──► LiveFeed.watch_ohlcv() → ForwardRunner.on_candle()
    │
    └──► download_pair_data() → Freqtrade bot warmup
```

---

## Technology Stack

| Layer | Technology |
|---|---|
| Engine | Freqtrade (backtest, live, hyperopt) |
| Generator | Custom GA + scikit-learn (ML seeding) |
| Evaluator | Freqtrade backtest + custom scoring |
| API | FastAPI + Uvicorn |
| DB | PostgreSQL (production) / SQLite (dev) |
| Market Data | ccxt Pro (Binance) |
| Tests | pytest + fastapi.testclient.TestClient |
| Observability | Prometheus-compatible `/metrics` |

---

## Project Structure

```
athena/
├── core/
│   ├── engine.py               # AthenaEngine — GA + evaluation + promotion
│   ├── freqtrade_wrapper.py    # Freqtrade backtest runner
│   ├── hyperopt.py             # HyperoptFinisher
│   ├── dna_versioning.py       # Immutable DNA snapshots
│   └── strategy_loader.py      # Dynamic loading
├── generator/
│   ├── dna.py                  # DNAEncoder
│   ├── ga_engine.py            # GAEngine
│   └── templates.py            # IStrategy templates
├── evaluator/
│   ├── scorer.py               # Composite scoring + gates
│   └── monte_carlo.py          # Trade shuffling
├── live/
│   ├── bot_manager.py          # Freqtrade subprocess lifecycle
│   ├── deployer.py             # user_data directory builder
│   ├── freqtrade_config.py     # config.json generator
│   ├── runner.py               # LiveRunner
│   ├── feed.py                 # WebSocket candle stream
│   ├── scheduler.py            # AutonomousScheduler
│   ├── forward_scheduler.py    # ForwardScheduler + drift
│   ├── data_downloader.py      # Freqtrade CLI data download
│   ├── feedback.py             # Drift detection + auto-demote
│   └── bridge.py               # DryRunTrader (signal-only)
├── market/
│   └── provider.py             # ccxt OHLCV fetcher
├── portfolio/
│   └── manager.py              # Multi-strategy allocation
├── services/
│   ├── api.py                  # FastAPI app (30+ endpoints)
│   ├── models.py               # SQLAlchemy DB models
│   ├── telemetry.py            # Prometheus metrics
│   ├── forward_runner.py       # Single-shot forward test
│   └── ui/index.html           # Dashboard
├── common/
│   ├── models.py               # Pydantic models
│   └── config.py               # Environment config
tests/
├── test_e2e.py                 # 22 API round-trip tests
├── test_integration.py         # 10 engine + API tests
├── test_bot_manager.py         # 4 bot lifecycle tests
├── test_bridge.py              # 8 dry-run signal tests
├── test_dashboard.py           # 51 UI / API contract tests
├── test_portfolio.py           # 27 allocation tests
├── test_regime.py              # 20 regime detection tests
├── test_drift.py               # 13 drift classification tests
├── test_forward_scheduler.py   # 13 scheduler tests
├── test_dna_versioning.py      # 10 snapshot tests
├── test_ml_predictor.py        # 13 ML seeding tests
├── test_hyperopt.py            # 5 hyperopt tests
├── test_risk_budget_sizing.py  # 14 position sizing tests
├── test_feedback.py            # 10 feedback tests
├── test_forward_drift.py       # 5 forward drift tests
├── test_forward_runner.py      # 2 forward execution tests
├── test_downloader.py          # 5 data download tests
├── test_telemetry.py           # 7 metrics tests
├── test_api_endpoints.py       # 5 signal + run tests
└── test_api_forward_pnl.py     # 3 PnL series tests
```

---

## Development

```bash
# Install
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Start server
python -m athena.services.api
# → http://127.0.0.1:8000/health

# Run all tests
pytest tests/ -q

# Run e2e only
pytest tests/test_e2e.py -v
```
