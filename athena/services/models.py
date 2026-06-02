"""Database models for Athena."""
from datetime import datetime, timezone
from typing import Optional
from sqlalchemy import create_engine, Column, String, Float, Integer, DateTime, JSON, Index
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import declarative_base
from athena.common.config import config

Base = declarative_base()

# Timezone-aware UTC factory for SQLAlchemy defaults
_utcnow = lambda: datetime.now(timezone.utc)


class StrategyModel(Base):
    __tablename__ = "strategies"
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    template = Column(String, nullable=False)
    dna = Column(JSON, nullable=False)
    objective = Column(String, default="sharpe")
    status = Column(String, default="draft")
    generation = Column(Integer, default=0)
    parent_id = Column(String, nullable=True)
    
    # Performance
    total_return = Column(Float, default=0.0)
    sharpe = Column(Float, default=0.0)
    sortino = Column(Float, default=0.0)
    calmar = Column(Float, default=0.0)
    win_rate = Column(Float, default=0.0)
    max_drawdown = Column(Float, default=0.0)
    total_trades = Column(Integer, default=0)
    avg_trade = Column(Float, default=0.0)
    profit_factor = Column(Float, default=0.0)
    
    # Score
    raw_score = Column(Float, default=0.0)
    verdict = Column(String, default="")
    
    # Metadata
    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow)
    metadata_json = Column(JSON, default=dict)


class LiveSessionModel(Base):
    __tablename__ = "live_sessions"

    id = Column(String, primary_key=True)
    strategy_id = Column(String, nullable=False)
    started_at = Column(DateTime, default=_utcnow)
    stopped_at = Column(DateTime, nullable=True)
    status = Column(String, default="running")
    mode = Column(String, default="paper")

    equity = Column(Float, default=50.0)
    open_positions = Column(Integer, default=0)
    unrealized_pnl = Column(Float, default=0.0)
    total_trades_taken = Column(Integer, default=0)
    max_drawdown_seen = Column(Float, default=0.0)
    last_signals = Column(JSON, default=list)
    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow)


class LiveSnapshot(Base):
    """Time-series snapshots from Freqtrade bot status for drift detection."""
    __tablename__ = "live_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, nullable=False)
    strategy_id = Column(String, nullable=False)
    timestamp = Column(DateTime, default=_utcnow)

    # Raw from bot
    equity = Column(Float, default=0.0)
    total_trades = Column(Integer, default=0)
    profit_closed_pct = Column(Float, default=0.0)
    profit_all_pct = Column(Float, default=0.0)

    # Computed rolling metrics
    sharpe_estimate = Column(Float, default=0.0)        # rolling
    max_drawdown = Column(Float, default=0.0)          # peak-to-trough
    win_rate = Column(Float, default=0.0)                # closed wins / total closed

    # Drift detection (against backtest baseline from strategies table)
    backtest_sharpe = Column(Float, default=0.0)
    backtest_max_drawdown = Column(Float, default=0.0)
    sharpe_ratio = Column(Float, default=0.0)            # live / backtest
    drawdown_ratio = Column(Float, default=0.0)          # live / backtest
    is_degraded = Column(String, default="")             # "" | "mild" | "severe"

    created_at = Column(DateTime, default=_utcnow)

    # Index for fast queries on latest snapshots
    __table_args__ = (Index("ix_snapshots_session_time", "session_id", "timestamp"),)


class Signal(Base):
    """Dry-run / forward-test signals persisted for PnL tracking and post-analysis."""
    __tablename__ = "signals"

    id = Column(String, primary_key=True)
    strategy_id = Column(String, nullable=False, index=True)
    symbol = Column(String, nullable=False, index=True)
    direction = Column(String, nullable=False)   # long / short
    confidence = Column(Float, default=1.0)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=True)
    timestamp = Column(DateTime, default=_utcnow)
    exit_time = Column(DateTime, nullable=True)
    pnl_pct = Column(Float, default=0.0)
    pnl_abs = Column(Float, default=0.0)
    status = Column(String, default="open")        # open | closed
    triggered_kill_switch = Column(String, default="")
    __table_args__ = (Index("ix_signals_strategy", "strategy_id", "timestamp"),)


# Setup
_DB_ENGINE = None

def get_engine():
    global _DB_ENGINE
    if _DB_ENGINE is None:
        _DB_ENGINE = create_engine(
            config.DATABASE_URL,
            pool_pre_ping=True,
            connect_args={"timeout": 30.0, "check_same_thread": False}
            if config.DATABASE_URL.startswith("sqlite") else {},
        )
    return _DB_ENGINE
def get_session():
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    return Session()

def init_db():
    engine = get_engine()
    Base.metadata.create_all(engine)
