"""FastAPI service for Athena."""
from contextlib import asynccontextmanager
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from athena.services.models import init_db, get_session, StrategyModel
from athena.common.models import StrategyRecord, StrategyStatus


# Startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield

app = FastAPI(title="Athena", version="0.1.0", lifespan=lifespan)


# Request/Response models
class StrategyCreate(BaseModel):
    name: str
    template: str
    dna: dict

class StrategyResponse(BaseModel):
    id: str
    name: str
    template: str
    status: str
    generation: int
    raw_score: float
    created_at: str

# Endpoints
@app.get("/health")
async def health():
    return {"status": "ok", "service": "athena"}

@app.get("/strategies", response_model=List[StrategyResponse])
async def list_strategies(status: Optional[str] = None, limit: int = 50):
    session = get_session()
    query = session.query(StrategyModel)
    if status:
        query = query.filter(StrategyModel.status == status)
    rows = query.order_by(StrategyModel.created_at.desc()).limit(limit).all()
    return [
        StrategyResponse(
            id=r.id,
            name=r.name,
            template=r.template,
            status=r.status,
            generation=r.generation,
            raw_score=r.raw_score,
            created_at=r.created_at.isoformat(),
        )
        for r in rows
    ]

@app.post("/strategies", response_model=StrategyResponse)
async def create_strategy(req: StrategyCreate):
    import uuid
    session = get_session()
    strat = StrategyModel(
        id=f"strat_{uuid.uuid4().hex[:12]}",
        name=req.name,
        template=req.template,
        dna=req.dna,
        status="draft",
    )
    session.add(strat)
    session.commit()
    return StrategyResponse(
        id=strat.id,
        name=strat.name,
        template=strat.template,
        status=strat.status,
        generation=strat.generation,
        raw_score=strat.raw_score,
        created_at=strat.created_at.isoformat(),
    )

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
        "score": {
            "raw_score": row.raw_score,
            "verdict": row.verdict,
        },
        "created_at": row.created_at.isoformat(),
    }

@app.post("/strategies/{strategy_id}/run")
async def run_strategy(strategy_id: str):
    """Trigger backtest/evaluation for a strategy."""
    session = get_session()
    row = session.query(StrategyModel).filter(StrategyModel.id == strategy_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Strategy not found")
    
    # Update status
    row.status = "backtest"
    session.commit()
    
    # TODO: Queue backtest job
    return {"id": strategy_id, "status": "backtest_queued"}

@app.get("/stats")
async def get_stats():
    session = get_session()
    total = session.query(StrategyModel).count()
    by_status = {}
    for s in ["draft", "backtest", "optimize", "paper", "live", "retired"]:
        by_status[s] = session.query(StrategyModel).filter(StrategyModel.status == s).count()
    
    best = session.query(StrategyModel).order_by(StrategyModel.raw_score.desc()).first()
    
    return {
        "total_strategies": total,
        "by_status": by_status,
        "best_strategy": {
            "id": best.id if best else None,
            "name": best.name if best else None,
            "score": best.raw_score if best else 0.0,
        }
    }
