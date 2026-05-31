"""Database models for Athena."""
from datetime import datetime
from typing import Optional
from sqlalchemy import create_engine, Column, String, Float, Integer, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from athena.common.config import config

Base = declarative_base()


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
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    metadata_json = Column(JSON, default=dict)


# Setup
def get_engine():
    return create_engine(config.DATABASE_URL)

def get_session():
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    return Session()

def init_db():
    engine = get_engine()
    Base.metadata.create_all(engine)
