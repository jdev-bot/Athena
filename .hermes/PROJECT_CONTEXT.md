# Athena Project Context

## Overview
ML-powered strategy generator for crypto trading via the Jesse framework.

## Stack
- **Backend:** Python 3.12, FastAPI, SQLAlchemy + PostgreSQL
- **Genetic Engine:** Custom GA with DNA encoding
- **Backtesting:** Jesse framework with real market data via ccxt
- **Live Feed:** ccxt Pro WebSocket streaming (Binance)
- **Tests:** pytest with 22 E2E tests (all passing)
- **Repo:** https://github.com/jdev-bot/Athena

## Git
- **Remote:** https://github.com/jdev-bot/Athena.git
- **Branch:** main
- **Latest commit:** df3a222 (22/22 E2E tests, Phase 2 complete)

## Phase 1 Complete ✅
- Strategy DNA generator + GA engine
- FastAPI service with 7 endpoints
- Jesse `research.backtest()` integration via isolated temp projects
- Real market data via ccxt / Binance OHLCV
- 22 E2E tests covering API, DNA, GA, scorer, orchestrator, Jesse wrapper, live runner

## Phase 2 Complete ✅
- **LiveFeed** — async ccxt Pro WebSocket streaming real 1m BTC/USDT candles
- **LiveRunner** — forward-test evaluator (signal-only, no trade execution)
- **FastAPI live endpoints** — `POST /live/start`, `POST /live/stop`, `GET /live/status`
- **Kill-switch** — max drawdown (15%) + daily loss limit (10%) circuit breakers
- **`live_sessions`** DB table with runtime stats (equity, positions, trades, signals)

## Available Endpoints (10 total)
| Method | Path | Description |
|---|---|---|
| GET | `/health` | Service status |
| POST | `/strategies/generate` | Generate strategies by DNA |
| GET | `/strategies` | List/filter strategies |
| GET | `/strategies/{id}` | Strategy detail |
| POST | `/backtests/run` | Run Jesse backtest on real data |
| GET | `/backtests` | Completed backtest list |
| GET | `/stats` | Aggregate counts |
| POST | `/live/start` | Start forward-test session |
| POST | `/live/stop` | Stop session |
| GET | `/live/status` | Session stats + signals |

## Architecture
```
┌────────────────────────────────────────────────────────────────────┐
│                         FastAPI / Uvicorn                          │
│    /strategies    /backtests    /stats    /live/start|stop|status  │
└──────────────┬─────────────────────┬──────────────┬────────────────┘
               │                     │              │
        ┌──────▼──────┐      ┌─────▼──────┐  ┌──▼──────────┐
        │  Generator  │      │  Backtest  │  │  LiveRunner │
        │  DNA / GA   │      │  Jesse FW  │  │  Signal-only│
        └─────────────┘      └─────┬──────┘  └──────┬──────┘
                                   │                  │
                          ┌────────▼─────────┐  ┌────▼──────┐
                          │ MarketDataProvider│  │  LiveFeed │
                          │  ccxt / Binance   │  │  WS 1m    │
                          └───────────────────┘  └───────────┘
```

## Key Files
| File | Purpose |
|---|---|
| `athena/services/api.py` | FastAPI app (10 endpoints) |
| `athena/core/jesse_wrapper.py` | `JesseWrapper` — real-data backtest runner |
| `athena/market/provider.py` | `MarketDataProvider` — ccxt OHLCV fetcher |
| `athena/live/feed.py` | `LiveFeed` — ccxt Pro WebSocket streamer |
| `athena/live/runner.py` | `ForwardRunner` + `LiveRunner` — forward-test |
| `athena/orchestrator.py` | `AthenaOrchestrator` — GA generation loop |
| `tests/test_e2e.py` | 22 end-to-end tests |
| `pyproject.toml` | Dependencies (includes ccxt) + pytest config |

## Environment
- Python venv at `.venv/`
- Jesse installed in venv with editable or standard install
- ccxt 4.x installed in venv
- PostgreSQL running on `localhost:5435` with `athena_db`
