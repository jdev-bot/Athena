"""FastAPI service for Athena — Strategy generation + backtest runner."""
import os
import uuid
import tempfile
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from athena.common.models import StrategyTemplate, StrategyStatus
from athena.services.models import init_db, get_session, StrategyModel
from athena.generator.dna import DNAEncoder
from athena.generator.templates import TEMPLATE_MAP, TEMPLATE_SPECS
from athena.common.config import config
from athena.core.jesse_wrapper import JesseWrapper


# ── startup / shutdown ────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="Athena", version="0.2.0", lifespan=lifespan)


# ── request / response schemas ────────────────────────────────────
class GenerateRequest(BaseModel):
    template: str = "trend_following"
    count: int = Field(default=1, ge=1, le=20)


class GenerateResponse(BaseModel):
    strategies: list


class BacktestRequest(BaseModel):
    strategy_id: str
    start_date: str = "2024-01-01"
    end_date: str = "2025-01-01"


class BacktestResponse(BaseModel):
    strategy_id: str
    status: str
    metrics: Optional[dict] = None


class StrategyListItem(BaseModel):
    id: str
    name: str
    template: str
    status: str
    generation: int
    raw_score: float
    total_return: float
    sharpe: float
    total_trades: int
    created_at: str


class BacktestListItem(BaseModel):
    strategy_id: str
    name: str
    total_return: float
    sharpe: float
    max_drawdown: float
    total_trades: int
    win_rate: float
    created_at: str


# ── helpers ────────────────────────────────────────────────────────
def _compile_strategy(record) -> str:
    """Render Jesse strategy source from DB record."""
    encoder = DNAEncoder()
    params = encoder.to_strategy_params(record.dna, StrategyTemplate(record.template))
    params["class_name"] = "AthenaStrategy"
    template = TEMPLATE_MAP.get(StrategyTemplate(record.template))
    return template.format(**params)


def _run_jesse_backtest(record, start_date: str, end_date: str) -> dict:
    """Run Jesse backtest with real market data via JesseWrapper."""
    code = _compile_strategy(record)
    wrapper = JesseWrapper()
    metrics = wrapper.run_backtest(
        code, start_date=start_date, end_date=end_date,
        exchange="binance", symbol="BTC-USD", timeframe="1h",
    )
    return metrics


# ── endpoints ──────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "service": "athena", "version": "0.2.0"}


@app.post("/strategies/generate", response_model=GenerateResponse)
async def generate_strategies(req: GenerateRequest):
    """Generate new random strategies and store them."""
    encoder = DNAEncoder()
    try:
        tpl = StrategyTemplate(req.template)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unknown template: {req.template}")

    session = get_session()
    created = []
    for _ in range(req.count):
        dna = encoder.random_dna(tpl)
        sid = f"strat_{uuid.uuid4().hex[:12]}"
        record = StrategyModel(
            id=sid,
            name=f"{tpl.value}_{sid[-6:]}",
            template=tpl.value,
            dna=dna,
            status=StrategyStatus.GENERATED.value,
            generation=0,
        )
        session.add(record)
        created.append({
            "id": sid,
            "name": record.name,
            "template": tpl.value,
            "dna": dna,
        })

    session.commit()
    return {"strategies": created}


@app.get("/strategies", response_model=List[StrategyListItem])
async def list_strategies(status: Optional[str] = None, limit: int = 50):
    session = get_session()
    q = session.query(StrategyModel)
    if status:
        q = q.filter(StrategyModel.status == status)
    rows = q.order_by(StrategyModel.created_at.desc()).limit(limit).all()
    return [
        StrategyListItem(
            id=r.id, name=r.name, template=r.template, status=r.status,
            generation=r.generation, raw_score=r.raw_score,
            total_return=r.total_return, sharpe=r.sharpe,
            total_trades=r.total_trades,
            created_at=r.created_at.isoformat(),
        )
        for r in rows
    ]


@app.get("/strategies/{strategy_id}")
async def get_strategy(strategy_id: str):
    session = get_session()
    row = session.query(StrategyModel).filter(StrategyModel.id == strategy_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Strategy not found")
    return {
        "id": row.id,
        "name": row.name,
        "template": row.template,
        "dna": row.dna,
        "status": row.status,
        "generation": row.generation,
        "performance": {
            "total_return": row.total_return,
            "sharpe": row.sharpe,
            "sortino": row.sortino,
            "max_drawdown": row.max_drawdown,
            "total_trades": row.total_trades,
        },
        "score": {"raw_score": row.raw_score, "verdict": row.verdict},
        "created_at": row.created_at.isoformat(),
    }


@app.post("/backtests/run", response_model=BacktestResponse)
async def run_backtest(req: BacktestRequest):
    """Run a Jesse backtest for a stored strategy using real market data."""
    session = get_session()
    row = session.query(StrategyModel).filter(StrategyModel.id == req.strategy_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Strategy not found")

    row.status = StrategyStatus.BACKTEST_RUNNING.value
    session.commit()

    metrics = _run_jesse_backtest(row, req.start_date, req.end_date)

    if "error" in metrics:
        row.status = StrategyStatus.BACKTEST_FAILED.value
        row.metadata_json = {"error": metrics["error"]}
    else:
        row.status = StrategyStatus.BACKTEST_DONE.value
        row.total_return = metrics.get("total_return", 0.0)
        row.sharpe = metrics.get("sharpe", 0.0)
        row.sortino = metrics.get("sortino", 0.0)
        row.calmar = metrics.get("calmar", 0.0)
        row.win_rate = metrics.get("win_rate", 0.0)
        row.max_drawdown = metrics.get("max_drawdown", 0.0)
        row.total_trades = metrics.get("total_trades", 0)
        row.avg_trade = metrics.get("avg_trade", 0.0)
        row.profit_factor = metrics.get("profit_factor", 0.0)
        row.raw_score = metrics.get("sharpe", 0.0) / 2.0  # simple scoring

    row.updated_at = datetime.utcnow()
    session.commit()

    return BacktestResponse(
        strategy_id=req.strategy_id,
        status=row.status,
        metrics=metrics,
    )


@app.get("/backtests", response_model=List[BacktestListItem])
async def list_backtests(limit: int = 50):
    session = get_session()
    rows = (
        session.query(StrategyModel)
        .filter(StrategyModel.status == StrategyStatus.BACKTEST_DONE.value)
        .order_by(StrategyModel.created_at.desc())
        .limit(limit)
        .all()
    )
    return [
        BacktestListItem(
            strategy_id=r.id,
            name=r.name,
            total_return=r.total_return,
            sharpe=r.sharpe,
            max_drawdown=r.max_drawdown,
            total_trades=r.total_trades,
            win_rate=r.win_rate,
            created_at=r.created_at.isoformat(),
        )
        for r in rows
    ]


@app.get("/stats")
async def get_stats():
    session = get_session()
    total = session.query(StrategyModel).count()
    by_status = {}
    for s in StrategyStatus:
        by_status[s.value] = session.query(StrategyModel).filter(
            StrategyModel.status == s.value
        ).count()
    best = (
        session.query(StrategyModel)
        .order_by(StrategyModel.raw_score.desc())
        .first()
    )
    return {
        "total_strategies": total,
        "by_status": by_status,
        "best_strategy": {
            "id": best.id if best else None,
            "name": best.name if best else None,
            "score": best.raw_score if best else 0.0,
        },
    }
