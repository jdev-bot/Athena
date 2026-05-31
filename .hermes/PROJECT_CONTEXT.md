# Athena Project Context

## Overview
ML-powered strategy generator for crypto trading via the Jesse framework.

## Stack
- **Backend:** Python 3.12, FastAPI, SQLAlchemy + PostgreSQL
- **Genetic Engine:** Custom GA with DNA encoding
- **Backtesting:** Jesse framework
- **Tests:** pytest with 20 E2E tests (all passing)
- **Repo:** https://github.com/jdev-bot/Athena

## Git
- **Remote:** https://github.com/jdev-bot/Athena.git
- **Branch:** main
- **Latest commit:** 32c8729 (20/20 E2E tests passing)

## Phase 1 Complete ✅
- Strategy DNA generator + GA engine
- FastAPI service with 7 endpoints
- Jesse `research.backtest()` integration via isolated temp projects
- 20 E2E tests covering API, DNA, GA, scorer, orchestrator, Jesse wrapper

## Known Issues / Notes
- Jesse `is_unit_testing()` looks at `PYTEST_CURRENT_TEST` env var; we pop it before backtest to ensure strategies are found in the temp project dir rather than the pytest package dir
- Backtest returns zero trades on synthetic `fake_candle` data — expected for random-noise candles
- `datetime.utcnow()` deprecation warnings throughout (SQLAlchemy + our code)
- No remote push configured in early commits — fixed, `origin/main` now tracking

## Next Phase
Phase 2: Forward testing / paper trading — wire live exchange WebSocket data into a live runner loop, add `live_runner` process supervisor, and build a `/live` API endpoint for starting/stopping paper trades.

## Architecture
```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   FastAPI    │────▶│  Generation  │────▶│   Backtest   │
│   /api       │     │  /engine     │     │   /runner    │
└──────────────┘     └──────────────┘     └──────────────┘
       │                    │                    │
       ▼                    ▼                    ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Postgres   │     │   DNA / GA   │     │   Jesse FW   │
│   athena_db  │     │   Templates  │     │   research   │
└──────────────┘     └──────────────┘     └──────────────┘
```

## Sensitive Config
Postgres passwords and any API/secret credentials must be redacted in command output.

## Key Files
| File | Purpose |
|---|---|
| `athena/services/api.py` | FastAPI app (7 endpoints) |
| `athena/core/jesse_wrapper.py` | `JesseWrapper` — isolated backtest runner |
| `athena/orchestrator.py` | `AthenaOrchestrator` — GA generation loop |
| `tests/test_e2e.py` | 20 end-to-end tests |
| `pyproject.toml` | Dependencies + pytest config |
| `README.md` | Project documentation |

## Environment
- Python venv at `.venv/`
- Jesse installed in venv with editable or standard install
- PostgreSQL running on `localhost:5435` with `athena_db`
