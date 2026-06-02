"""Shared Pydantic models for Athena."""
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class StrategyStatus(str, Enum):
    DRAFT = "draft"
    GENERATED = "generated"
    BACKTEST_QUEUED = "backtest_queued"
    BACKTEST_RUNNING = "backtest_running"
    BACKTEST_DONE = "backtest_done"
    BACKTEST_FAILED = "backtest_failed"
    OPTIMIZE = "optimize"
    PAPER = "paper"
    LIVE = "live"
    PROMOTED = "promoted"
    RETIRED = "retired"
    FAILED = "failed"


class StrategyTemplate(str, Enum):
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"


class DNASpec(BaseModel):
    """Specification for a single DNA parameter."""
    name: str
    type: str  # "int", "float", "bool", "choice"
    min: Optional[float] = None
    max: Optional[float] = None
    choices: Optional[List[Any]] = None
    default: Any


class StrategyDNA(BaseModel):
    """Encoded strategy parameters."""
    template: StrategyTemplate
    vector: Dict[str, Any] = Field(default_factory=dict)
    spec: List[DNASpec] = Field(default_factory=list)


class PerformanceMetrics(BaseModel):
    """Backtest / live performance metrics."""
    total_return: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    calmar: float = 0.0
    win_rate: float = 0.0
    max_drawdown: float = 0.0
    total_trades: int = 0
    avg_trade: float = 0.0
    profit_factor: float = 0.0


class ScoreResult(BaseModel):
    """Composite score from evaluator."""
    raw_score: float = 0.0  # 0.0 - 1.0
    sharpe_contrib: float = 0.0
    sortino_contrib: float = 0.0
    calmar_contrib: float = 0.0
    win_rate_contrib: float = 0.0
    verdict: str = ""  # "promote", "demote", "hold"


class StrategyRecord(BaseModel):
    """Full strategy record."""
    id: str
    name: str
    template: StrategyTemplate
    dna: StrategyDNA
    objective: str = "sharpe"
    status: StrategyStatus = StrategyStatus.DRAFT
    generation: int = 0
    parent_id: Optional[str] = None
    performance: PerformanceMetrics = Field(default_factory=PerformanceMetrics)
    score: ScoreResult = Field(default_factory=ScoreResult)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict)


class WalkForwardResult(BaseModel):
    """Walk-forward analysis result."""
    in_sample_metrics: PerformanceMetrics
    out_sample_metrics: PerformanceMetrics
    degradation_ratio: float  # out / in_sample
    is_robust: bool


class MonteCarloResult(BaseModel):
    """Monte Carlo stress test result."""
    original_sharpe: float
    shuffled_sharpe_mean: float
    shuffled_sharpe_std: float
    p_value: float
    is_significant: bool


class GenerationConfig(BaseModel):
    """Configuration for a generator run."""
    symbols: List[str] = Field(default=["BTC-USD"])
    timeframe: str = "1h"
    start_date: str = "2026-02-01"
    end_date: str = "2026-06-01"
    exchange: str = "Sandbox"
    population_size: int = 30
    generations: int = 20
    mutation_rate: float = 0.2
    crossover_rate: float = 0.7
    elitism_count: int = 3
    ml_boost: bool = True
    parallel_workers: int = 4


class PortfolioPosition(BaseModel):
    """A single strategy allocation within the portfolio."""
    strategy_id: str
    weight: float = Field(ge=0.0, le=1.0)          # capital allocation fraction
    notional: float = 0.0                             # current allocated capital
    open_pnl: float = 0.0                             # unrealized PnL
    closed_pnl: float = 0.0                           # realized PnL
    max_drawdown: float = 0.0                         # peak-to-trough drawdown
    sharpe_30d: float = 0.0                         # rolling 30-day Sharpe
    correlation_to_portfolio: float = 0.0             # correlation with rest of book
    status: str = "active"                            # active, paused, stopped
    started_at: datetime = Field(default_factory=datetime.utcnow)
    last_rebalanced_at: Optional[datetime] = None


class PortfolioSnapshot(BaseModel):
    """Full portfolio state at a point in time."""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    total_capital: float
    allocated_capital: float
    free_cash: float
    total_closed_pnl: float
    total_open_pnl: float
    portfolio_max_drawdown: float
    portfolio_sharpe: float
    positions: List[PortfolioPosition]
    active_strategies: int
    paused_strategies: int
    killed_strategies: int


class PortfolioConfig(BaseModel):
    """Portfolio-level risk settings."""
    total_capital: float = 10_000.0
    max_per_strategy_weight: float = 0.35             # no strategy >35%
    min_per_strategy_weight: float = 0.05             # trim below 5%
    max_correlation: float = 0.80                     # flag if pairwise corr >0.8
    rebalance_interval_hours: int = 24
    portfolio_max_drawdown_kill: float = 0.15         # kill all at 15% portfolio DD
    target_volatility_annual: float = 0.25            # 25% annual vol target
    allocation_method: str = "inverse_vol"            # inverse_vol, equal_risk, equal_weight
