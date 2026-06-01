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
from athena.core.freqtrade_wrapper import FreqtradeWrapper
from athena.live.runner import LiveRunner
from athena.live.feedback import AdaptiveLoop


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


class LiveStartRequest(BaseModel):
    strategy_id: str
    mode: str = "paper"  # paper | live
    max_drawdown: float = 0.15
    daily_loss_limit: float = 0.10
    exchange_key: str = ""
    exchange_secret: str = ""
    sandbox: bool = False


class LiveStartResponse(BaseModel):
    session_id: str
    strategy_id: str
    status: str


class LiveStatusResponse(BaseModel):
    session_id: str
    status: str
    mode: Optional[str] = None
    equity: float
    open_positions: int
    total_trades: int
    max_drawdown: float
    started_at: Optional[str] = None
    stopped_at: Optional[str] = None
    last_signals: list = []
    profit_closed_pct: float = 0.0
    profit_all_pct: float = 0.0


# ── live runner registry (in-memory, holds async tasks) ──────────
_runners: dict[str, LiveRunner] = {}

# ── adaptive loop singleton (shared across sessions) ───────────────
_adaptive = AdaptiveLoop()


@app.post("/live/adaptive_watch")
async def adaptive_watch(session_id: str):
    """Start the continuous feedback collector + adaptive loop for a session."""
    runner = _runners.get(session_id)
    if not runner:
        raise HTTPException(status_code=404, detail="Session not found")
    # Also start drift monitoring + adaptive re-optimization
    await _adaptive.watch_session(session_id)
    return {"session_id": session_id, "adaptive": True}


@app.get("/live/snapshots")
async def live_snapshots(session_id: str, limit: int = 50):
    """Retrieve recent snapshots with drift metrics for a session."""
    rows = _adaptive.collector.get_recent_snapshots(session_id, limit=limit)
    return [
        {
            "timestamp": r.timestamp.isoformat(),
            "equity": r.equity,
            "total_trades": r.total_trades,
            "profit_closed_pct": r.profit_closed_pct,
            "sharpe_estimate": r.sharpe_estimate,
            "max_drawdown": r.max_drawdown,
            "win_rate": r.win_rate,
            "sharpe_ratio": r.sharpe_ratio,
            "drawdown_ratio": r.drawdown_ratio,
            "is_degraded": r.is_degraded,
        }
        for r in rows
    ]


@app.get("/live/analyze")
async def live_analyze(session_id: str):
    """High-level drift report for a session vs its backtest baseline."""
    snapshots = _adaptive.collector.get_recent_snapshots(session_id, limit=10)
    if not snapshots:
        raise HTTPException(status_code=404, detail="No snapshots yet")

    latest = snapshots[0]
    trend = "stable"
    if all(s.is_degraded in ("mild", "severe") for s in snapshots[:5]):
        trend = "degrading"
    elif latest.is_degraded == "":
        trend = "stable"

    return {
        "session_id": session_id,
        "strategy_id": latest.strategy_id,
        "latest": {
            "timestamp": latest.timestamp.isoformat(),
            "equity": latest.equity,
            "total_trades": latest.total_trades,
            "profit_closed_pct": latest.profit_closed_pct,
            "sharpe_estimate": latest.sharpe_estimate,
            "max_drawdown": latest.max_drawdown,
            "win_rate": latest.win_rate,
        },
        "baseline": {
            "backtest_sharpe": latest.backtest_sharpe,
            "backtest_max_drawdown": latest.backtest_max_drawdown,
        },
        "ratios": {
            "sharpe_ratio": latest.sharpe_ratio,
            "drawdown_ratio": latest.drawdown_ratio,
        },
        "drift": {
            "is_degraded": latest.is_degraded,
            "trend": trend,
        },
        "snapshots_count": len(snapshots),
    }


# ── helpers ────────────────────────────────────────────────────────
def _compile_strategy(record) -> str:
    """Render Freqtrade strategy source from DB record."""
    encoder = DNAEncoder()
    params = encoder.to_strategy_params(record.dna, StrategyTemplate(record.template))
    params["class_name"] = "AthenaStrategy"
    params["template_name"] = StrategyTemplate(record.template).value
    params["timeframe"] = getattr(record, "timeframe", "1h")
    template = TEMPLATE_MAP.get(StrategyTemplate(record.template))
    return template.format(**params)


def _run_freqtrade_backtest(record, start_date: str, end_date: str) -> dict:
    """Run Freqtrade backtest with real market data via FreqtradeWrapper."""
    code = _compile_strategy(record)
    wrapper = FreqtradeWrapper()
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

    metrics = _run_freqtrade_backtest(row, req.start_date, req.end_date)

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


@app.post("/live/start", response_model=LiveStartResponse)
async def live_start(req: LiveStartRequest):
    """Start a forward-test paper/live session for a strategy."""
    session = get_session()
    strategy = session.query(StrategyModel).filter_by(id=req.strategy_id).first()
    if not strategy:
        raise HTTPException(status_code=404, detail="Strategy not found")
    session.close()

    if req.mode not in ("paper", "live"):
        raise HTTPException(status_code=400, detail="mode must be 'paper' or 'live'")

    # Gate: live mode requires API keys
    if req.mode == "live" and (not req.exchange_key or not req.exchange_secret):
        raise HTTPException(status_code=400, detail="live mode requires exchange_key and exchange_secret")

    # Gate: default sandbox to True for safety (testnet first)
    sandbox = req.sandbox if req.mode == "live" else True

    runner = LiveRunner(
        strategy_id=req.strategy_id,
        mode=req.mode,
        risk={"max_drawdown": req.max_drawdown, "daily_loss_limit": req.daily_loss_limit},
        exchange_key=req.exchange_key,
        exchange_secret=req.exchange_secret,
        sandbox=sandbox,
    )
    await runner.start()
    _runners[runner.session_id] = runner
    # Auto-start adaptive drift monitoring on every session
    try:
        await _adaptive.watch_session(runner.session_id)
    except Exception:
        pass  # best-effort; don't block start on feedback issues
    return LiveStartResponse(
        session_id=runner.session_id,
        strategy_id=req.strategy_id,
        status="running",
    )


@app.post("/live/stop")
async def live_stop(session_id: str):
    """Stop a running forward-test session."""
    runner = _runners.pop(session_id, None)
    if not runner:
        raise HTTPException(status_code=404, detail="Session not found")
    await runner.stop("stopped_by_user")
    return {"session_id": session_id, "status": "stopped"}


@app.get("/live/status", response_model=LiveStatusResponse)
async def live_status(session_id: str):
    """Get current stats of a live session."""
    runner = _runners.get(session_id)
    if not runner:
        raise HTTPException(status_code=404, detail="Session not found")
    stats = runner.stats
    if not stats:
        raise HTTPException(status_code=404, detail="Session record missing")
    return LiveStatusResponse(**stats)
