# Athena — AI Strategy Generator on Freqtrade

## Architecture

Athena wraps Freqtrade as a library and adds three layers on top:

1. **Core (`athena/core`)** — Freqtrade wrapper, strategy loader
2. **Generator (`athena/generator`)** — ML-based strategy generator + mutation engine
3. **Evaluator (`athena/evaluator`)** — Backtest orchestration, scoring
4. **Services (`athena/services`)** — FastAPI orchestration, lifecycle management
5. **Common (`athena/common`)** — Shared Pydantic models, config, utilities
6. **Live (`athena/live`)** — Forward-test runner with live WebSocket candles
7. **Market (`athena/market`)** — Real-time OHLCV via ccxt (Binance public API)

## Strategy Model

A strategy in Athena is a **Freqtrade-compatible Python class** produced by the generator.

Each strategy has:
- `template_id` — base template (e.g. trend_following, mean_reversion, breakout)
- `dna` — hyperparameter vector (indicator periods, thresholds, etc.)
- `objective` — what it optimizes for (sharpe, sortino, win_rate, max_drawdown)
- `performance` — backtest results (sharpe, total_return, max_dd, trades, etc.)
- `metadata` — created_at, generation, parent_id (for lineage)
- `status` — draft → generated → backtest → optimize → paper → live → retired

## Generator Architecture

The generator uses a **Genetic Algorithm + ML ensemble**:

1. **Template Library** — Pre-defined strategy templates (Freqtrade `IStrategy` subclasses with parametric holes)
2. **DNA Encoding** — Each strategy's free parameters encoded as a vector
3. **Population** — Maintains N candidate strategies
4. **Fitness** — Backtest score (sharpe × sortino × (1 - max_dd)) normalized
5. **Selection** — Tournament selection
6. **Crossover** — Two-point crossover on DNA vectors
7. **Mutation** — Gaussian noise on continuous params, random swap on discrete
8. **Elitism** — Top K survive unmodified
9. **ML Boost** — Trained regressor predicts fitness from DNA to seed promising candidates

## Evaluator Architecture

1. **Backtest** — Full historical backtest via Freqtrade vectorized engine
2. **Walk-Forward** — Train on first 70%, validate on last 30% (prevent overfit)
3. **Monte Carlo** — Shuffle trade order, resample candles
4. **Scoring** — Composite: sharpe(40%) + sortino(30%) + calmar(20%) + win_rate(10%)
5. **Thresholds** — promote ≥ 0.6, demote < 0.3, retire after 3 demotions

## Lifecycle

draft → generated → backtest → [score < 0.3] → retire
              ↓ [score ≥ 0.6]
         optimize (GA fine-tuning) → paper_trade (30 days)
                                              ↓ [score ≥ 0.6]
                                         live_trade
                                              ↓ [score < 0.3 × 3 days]
                                         kill_switch → retire
```

## Technology Stack

- **Engine:** Freqtrade (backtest, live, optimization)
- **Generator:** Python + DEAP (genetic algorithm) + scikit-learn (ML)
- **Evaluator:** Freqtrade backtest + custom scoring
- **API:** FastAPI + WebSocket
- **DB:** PostgreSQL (production), SQLite (local/tests)
- **Observability:** Prometheus + Grafana (optional)
- **Infra:** Docker Compose

## Local Development

```bash
# Start infrastructure
make up

# Run generator
python -m athena.generator.ga_engine --mode evolve --symbols BTC-USD --timeframe 1h

# Run evaluator
python -m athena.evaluator.scorer --strategy-id <id>

# Run API server
uvicorn athena.services.api:app --reload

# Run tests
pytest tests/test_e2e.py -v
```

## Project Structure

```
athena/
├── core/
│   ├── __init__.py
│   ├── freqtrade_wrapper.py   # Freqtrade temp-project backtest runner
│   └── strategy_loader.py     # Load strategies dynamically
├── generator/
│   ├── __init__.py
│   ├── templates.py            # Strategy template library (IStrategy subclasses)
│   ├── dna.py                  # DNA encoding/decoding
│   ├── ga_engine.py            # Genetic algorithm
│   └── templates/              # (future) Template variant directory
├── evaluator/
│   ├── __init__.py
│   ├── scorer.py               # Composite scoring
│   └── backtest_runner.py      # (future) Batch backtest runner
├── services/
│   ├── __init__.py
│   ├── api.py                  # FastAPI app (10 endpoints)
│   └── models.py               # DB models (SQLAlchemy)
├── live/
│   ├── __init__.py
│   ├── feed.py                 # ccxt Pro WebSocket streamer
│   └── runner.py               # ForwardRunner + LiveRunner (signal-only)
├── market/
│   └── provider.py             # MarketDataProvider — Binance OHLCV via ccxt
├── common/
│   ├── __init__.py
│   ├── models.py               # Pydantic models (StrategyStatus, ScoreResult, etc.)
│   └── config.py               # App configuration
├── tests/
│   ├── __init__.py
│   └── test_e2e.py             # 22 end-to-end tests (all passing)
├── scripts/
│   └── setup.sh                # One-time setup
├── infra/
│   └── docker-compose.yml      # Local infrastructure
│   ├── postgres.yml
│── .hermes/
│   └── PROJECT_CONTEXT.md      # Runtime project knowledge
├── Makefile
├── pyproject.toml              # Dependencies (freqtrade, ccxt, pandas-ta)
└── README.md
```
