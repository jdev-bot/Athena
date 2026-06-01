"""Application configuration."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Config:
    # Paths
    STRATEGIES_DIR = PROJECT_ROOT / "strategies"
    TEMPLATES_DIR = PROJECT_ROOT / "strategies" / "templates"
    GENERATED_DIR = PROJECT_ROOT / "strategies" / "generated"
    DATA_DIR = PROJECT_ROOT / "data"
    
    # Database
    DATA_DIR = PROJECT_ROOT / "data"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATA_DIR / 'athena.db'}")
    
    # Backtesting framework
    BACKTEST_FRAMEWORK = "freqtrade"
    
    # Generator
    DEFAULT_POPULATION_SIZE = int(os.getenv("POPULATION_SIZE", "30"))
    DEFAULT_GENERATIONS = int(os.getenv("GENERATIONS", "20"))
    MUTATION_RATE = float(os.getenv("MUTATION_RATE", "0.2"))
    CROSSOVER_RATE = float(os.getenv("CROSSOVER_RATE", "0.7"))
    ELITISM_COUNT = int(os.getenv("ELITISM_COUNT", "3"))
    
    # Evaluator
    WALK_FORWARD_TRAIN_RATIO = 0.7
    PROMOTE_THRESHOLD = 0.25
    DEMOTE_THRESHOLD = 0.08
    RETIRE_AFTER_DEMOTIONS = 3

    # Scoring weights (tuned for short-term data)
    SHARPE_WEIGHT = 0.15
    SORTINO_WEIGHT = 0.15
    CALMAR_WEIGHT = 0.10
    WIN_RATE_WEIGHT = 0.25
    PROFIT_FACTOR_WEIGHT = 0.20
    TOTAL_RETURN_WEIGHT = 0.15
    
    # Server
    API_PORT = int(os.getenv("API_PORT", "8000"))
    WS_PORT = int(os.getenv("WS_PORT", "8001"))
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


config = Config()
