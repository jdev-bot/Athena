# Changelog

## [0.3.0] — 2026-06-03

### Added
- Dashboard: standalone HTML/CSS/JS with 51 automated tests
- DNA versioning: immutable snapshots, restore, list endpoints
- Drift detection: forward vs backtest comparison with auto-demote
- Mini-GA restart on severe drift
- Forward-test scheduler with kill-switch integration
- Telemetry: Prometheus-compatible `/metrics` endpoint
- Scheduler control endpoints (`/scheduler/*`, `/forward/scheduler/*`)

## [0.2.0] — 2026-06-01

### Added
- Portfolio manager: multi-strategy capital allocation, risk budget, rebalance
- Market regime detection: trend/range/volatility classifier
- ML predictor seeding into GA population
- Hyperopt finisher: post-promotion parameter tuning
- Forward dry-run bridge with signal persistence
- Live paper trading via Freqtrade bot spawn
- Kill-switch circuit breakers (drawdown, daily loss)
- Autonomous evolution scheduler
- Real-market data downloader (ccxt → Binance)

## [0.1.0] — 2026-05-28

### Added
- Strategy DNA encoding + random generation
- Genetic algorithm engine with crossover / mutation
- Freqtrade backtesting integration with real market data
- Performance scoring with robustness gates
- Promotion / demotion lifecycle
- FastAPI service with CRUD endpoints
- SQLite / PostgreSQL persistence
