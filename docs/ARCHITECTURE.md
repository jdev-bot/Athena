# Athena — AI Strategy Generator on Jesse

## Architecture

Athena wraps Jesse as a library and adds three layers on top:

1. **Core (`athena/core`)** — Jesse wrapper, strategy loader, execution runner
2. **Generator (`athena/generator`)** — ML-based strategy generator + mutation engine
3. **Evaluator (`athena/evaluator`)** — Backtest orchestration, walk-forward analysis, scoring
4. **Services (`athena/services`)** — FastAPI orchestration, lifecycle management, WebSocket telemetry
5. **Common (`athena/common`)** — Shared Pydantic models, config, utilities

## Strategy Model

A strategy in Athena is a **Jesse-compatible Python class** produced by the generator.

Each strategy has:
- `template_id` — base template (e.g. trend_following, mean_reversion, breakout)
- `dna` — hyperparameter vector (indicator periods, thresholds, etc.)
- `objective` — what it optimizes for (sharpe, sortino, win_rate, max_drawdown)
- `performance` — backtest results (sharpe, total_return, max_dd, trades, etc.)
- `metadata` — created_at, generation, parent_id (for lineage)
- `status` — draft → backtest → optimize → paper → live → retired

## Generator Architecture

The generator uses a **Genetic Algorithm + ML ensemble**:

1. **Template Library** — Pre-defined strategy templates (Jesse classes with parametric holes)
2. **DNA Encoding** — Each strategy's free parameters encoded as a vector
3. **Population** — Maintains N candidate strategies
4. **Fitness** — Backtest score (sharpe × sortino × (1 - max_dd)) normalized
5. **Selection** — Tournament selection
6. **Crossover** — Two-point crossover on DNA vectors
7. **Mutation** — Gaussian noise on continuous params, random swap on discrete
8. **Elitism** — Top K survive unmodified
9. **ML Boost** — Trained regressor predicts fitness from DNA to seed promising candidates

## Evaluator Architecture

1. **Backtest** — Full historical backtest via Jesse engine
2. **Walk-Forward** — Train on first 70%, validate on last 30% (prevent overfit)
3. **Monte Carlo** — Shuffle trade order, resample candles (Jesse built-in)
4. **Scoring** — Composite: sharpe(40%) + sortino(30%) + calmar(20%) + win_rate(10%)
5. **Thresholds** — promote ≥ 0.6, demote < 0.3, retire after 3 demotions

## Lifecycle

```
draft → backtest → [score < 0.3] → retire
              ↓ [score ≥ 0.6]
         optimize (GA fine-tuning) → paper_trade (30 days)
                                              ↓ [score ≥ 0.6]
                                         live_trade
                                              ↓ [score < 0.3 × 3 days]
                                         kill_switch → retire
```

## Technology Stack

- **Engine:** Jesse (backtest, live, optimization)
- **Generator:** Python + DEAP (genetic algorithm) + scikit-learn (ML)
- **Evaluator:** Jesse backtest + custom scoring
- **API:** FastAPI + WebSocket
- **DB:** SQLite (local) / PostgreSQL (production)
- **Observability:** Prometheus + Grafana (optional)
- **Infra:** Docker Compose

## Local Development

```bash
# Start infrastructure
make up

# Run generator
cd athena && python -m generator.main --mode evolve --symbols BTC-USD --timeframe 1h

# Run evaluator
cd athena && python -m evaluator.main --strategy-id <id>

# Run API server
cd athena && uvicorn services.api:app --reload
```

## Project Structure

```
athena/
├── core/
│   ├── __init__.py
│   ├── jesse_wrapper.py        # Initialize Jesse, run backtests
│   ├── strategy_loader.py      # Load strategies dynamically
│   ├── execution_runner.py     # Run backtest/live/paper
│   └── config.py               # Jesse config management
├── generator/
│   ├── __init__.py
│   ├── templates/              # Strategy template library
│   ├── dna.py                  # DNA encoding/decoding
│   ├── population.py           # Population management
│   ├── fitness.py              # Fitness evaluation
│   ├── ga_engine.py            # Genetic algorithm
│   ├── ml_predictor.py         # ML fitness predictor
│   ├── mutator.py              # Mutation operators
│   ├── crossover.py            # Crossover operators
│   └── main.py                 # Generator CLI entry
├── evaluator/
│   ├── __init__.py
│   ├── backtest_runner.py      # Run backtests
│   ├── walk_forward.py         # Walk-forward analysis
│   ├── monte_carlo.py          # Monte Carlo stress test
│   ├── scorer.py               # Composite scoring
│   └── main.py                 # Evaluator CLI entry
├── services/
│   ├── __init__.py
│   ├── api.py                  # FastAPI app
│   ├── lifecycle.py            # Strategy lifecycle management
│   ├── websocket.py            # Real-time telemetry
│   └── models.py               # DB models (SQLAlchemy)
├── common/
│   ├── __init__.py
│   ├── models.py               # Pydantic models
│   ├── config.py               # App configuration
│   └── utils.py                # Utilities
├── tests/
│   ├── __init__.py
│   ├── test_generator.py
│   ├── test_evaluator.py
│   └── test_core.py
├── scripts/
│   └── setup.sh                # One-time setup
├── infra/
│   └── docker-compose.yml      # Local infrastructure
├── config/
│   ├── jesse_config.py         # Jesse configuration
│   └── athena.yaml             # Athena configuration
├── Makefile
├── pyproject.toml
└── README.md
```
