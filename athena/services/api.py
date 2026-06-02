"""FastAPI service for Athena — unified API over AthenaEngine."""
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import List, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks, Response
from pydantic import BaseModel, Field

from athena.common.models import StrategyTemplate, StrategyStatus
from athena.services.models import init_db, get_session, StrategyModel
from athena.generator.dna import DNAEncoder
from athena.generator.templates import TEMPLATE_MAP, TEMPLATE_SPECS
from athena.common.config import config
from athena.core.engine import AthenaEngine
from athena.common.models import GenerationConfig
from athena.core.hyperopt import HyperoptFinisher
from athena.services.telemetry import TelemetryCollector, generate_metrics, metrics_content_type

# ── single telemetry collector ───────────────────────────────────
_telemetry = TelemetryCollector()


# ── lifespan ──────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    if config.AUTO_SCHEDULER:
        from athena.live.scheduler import get_scheduler
        try:
            get_scheduler().start()
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(f"Auto-scheduler failed to start: {exc}")
    if config.FORWARD_SCHEDULER_AUTO:
        from athena.live.forward_scheduler import get_forward_scheduler
        try:
            get_forward_scheduler().start()
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(f"Forward scheduler failed to start: {exc}")
    yield


app = FastAPI(title="Athena", version="0.3.0", lifespan=lifespan)


# ── request / response schemas ────────────────────────────────────
class GenerateRequest(BaseModel):
    template: str = "trend_following"
    count: int = Field(default=1, ge=1, le=20)


class GenerateResponse(BaseModel):
    strategies: list


class EvolveRequest(BaseModel):
    template: str = "trend_following"
    population: int = Field(default=10, ge=3, le=100)
    generations: int = Field(default=5, ge=1, le=50)
    start_date: str = "2026-05-02"
    end_date: str = "2026-06-01"
    run_gates: bool = True


class EvolveResponse(BaseModel):
    best_strategy_id: Optional[str]
    best_score: float
    best_verdict: str
    strategies_evaluated: int
    promoted_count: int
    elapsed_seconds: float


class BacktestRequest(BaseModel):
    strategy_id: str
    start_date: str = "2026-05-02"
    end_date: str = "2026-06-01"


class BacktestResponse(BaseModel):
    strategy_id: str
    status: str
    metrics: Optional[dict] = None


class PromoteRequest(BaseModel):
    strategy_id: str


class PromoteResponse(BaseModel):
    strategy_id: str
    promoted: bool
    status: str
    gates_passed: bool
    score: float
    verdict: str


class DeployResponse(BaseModel):
    strategy_id: str
    path: str


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
def _get_engine(req: EvolveRequest) -> AthenaEngine:
    cfg = GenerationConfig(
        symbols=["BTC-USD"],
        timeframe="1h",
        start_date=req.start_date,
        end_date=req.end_date,
        population_size=req.population,
        generations=req.generations,
    )
    return AthenaEngine(cfg)


# ── endpoints ──────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "service": "athena", "version": "0.3.0", "framework": config.BACKTEST_FRAMEWORK}


@app.post("/strategies/generate", response_model=GenerateResponse)
async def generate_strategies(req: GenerateRequest):
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
        created.append({"id": sid, "name": record.name, "template": tpl.value, "dna": dna})
    session.commit()
    _telemetry.record_evaluation(req.count)
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
    row = session.query(StrategyModel).filter_by(id=strategy_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Strategy not found")
    return {
        "id": row.id, "name": row.name, "template": row.template,
        "dna": row.dna, "status": row.status, "generation": row.generation,
        "performance": {
            "total_return": row.total_return, "sharpe": row.sharpe,
            "sortino": row.sortino, "max_drawdown": row.max_drawdown,
            "total_trades": row.total_trades,
        },
        "score": {"raw_score": row.raw_score, "verdict": row.verdict},
        "metadata": row.metadata_json,
        "created_at": row.created_at.isoformat(),
    }


@app.post("/evolve", response_model=EvolveResponse)
async def evolve(req: EvolveRequest, background_tasks: BackgroundTasks):
    """Run full GA evolution + scoring + gates in a background task."""
    import time
    start = time.time()

    try:
        tpl = StrategyTemplate(req.template)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unknown template: {req.template}")

    engine = _get_engine(req)
    records = engine.evolve(tpl, run_gates=req.run_gates)
    elapsed = time.time() - start

    best = records[0] if records else None
    promoted = sum(1 for r in records if r.status == StrategyStatus.PROMOTED)

    return EvolveResponse(
        best_strategy_id=best.id if best else None,
        best_score=best.score.raw_score if best else 0.0,
        best_verdict=best.score.verdict if best else "demote",
        strategies_evaluated=len(records),
        promoted_count=promoted,
        elapsed_seconds=round(elapsed, 1),
    )


@app.post("/backtests/run", response_model=BacktestResponse)
async def run_backtest(req: BacktestRequest):
    """Run a single backtest for a stored strategy."""
    session = get_session()
    row = session.query(StrategyModel).filter_by(id=req.strategy_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Strategy not found")

    row.status = StrategyStatus.BACKTEST_RUNNING.value
    session.commit()

    engine = AthenaEngine()
    try:
        record = engine.evaluate(req.strategy_id, req.start_date, req.end_date)
        row.status = StrategyStatus.BACKTEST_DONE.value
        row.total_return = record.performance.total_return
        row.sharpe = record.performance.sharpe
        row.sortino = record.performance.sortino
        row.calmar = record.performance.calmar
        row.win_rate = record.performance.win_rate
        row.max_drawdown = record.performance.max_drawdown
        row.total_trades = record.performance.total_trades
        row.avg_trade = record.performance.avg_trade
        row.profit_factor = record.performance.profit_factor
        row.raw_score = record.score.raw_score
        row.verdict = record.score.verdict
        metrics = record.performance.model_dump()
        _telemetry.record_backtest()
    except Exception as exc:
        row.status = StrategyStatus.BACKTEST_FAILED.value
        metrics = {"error": str(exc)}

    row.updated_at = datetime.now(timezone.utc)
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
        .filter(StrategyModel.status.in_([StrategyStatus.BACKTEST_DONE.value, StrategyStatus.PROMOTED.value]))
        .order_by(StrategyModel.raw_score.desc())
        .limit(limit)
        .all()
    )
    return [
        BacktestListItem(
            strategy_id=r.id, name=r.name,
            total_return=r.total_return, sharpe=r.sharpe,
            max_drawdown=r.max_drawdown, total_trades=r.total_trades,
            win_rate=r.win_rate,
            created_at=r.created_at.isoformat(),
        )
        for r in rows
    ]


@app.get("/stats")
async def get_stats():
    session = get_session()
    total = session.query(StrategyModel).count()
    by_status = {s.value: session.query(StrategyModel).filter_by(status=s.value).count() for s in StrategyStatus}
    best = session.query(StrategyModel).order_by(StrategyModel.raw_score.desc()).first()
    return {
        "total_strategies": total,
        "by_status": by_status,
        "best_strategy": {
            "id": best.id if best else None,
            "name": best.name if best else None,
            "score": best.raw_score if best else 0.0,
            "verdict": best.verdict if best else None,
        },
    }


@app.post("/promote", response_model=PromoteResponse)
async def promote_strategy(req: PromoteRequest):
    """Run gates on a strategy and promote if it passes."""
    engine = AthenaEngine()
    try:
        record = engine.promote(req.strategy_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    return PromoteResponse(
        strategy_id=record.id,
        promoted=record.status == StrategyStatus.PROMOTED,
        status=record.status.value,
        gates_passed=record.metadata.get("robustness_passed", False) if record.metadata else False,
        score=record.score.raw_score,
        verdict=record.score.verdict,
    )


@app.post("/deploy")
async def deploy_strategy(strategy_id: str) -> DeployResponse:
    """Export a strategy to a deployable Python file."""
    engine = AthenaEngine()
    try:
        path = engine.deploy(strategy_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return DeployResponse(strategy_id=strategy_id, path=str(path))


# ── portfolio endpoints ────────────────────────────────────────────

from athena.portfolio.manager import PortfolioManager


class PortfolioAddRequest(BaseModel):
    strategy_id: str
    initial_weight: Optional[float] = None


class PortfolioRebalanceRequest(BaseModel):
    method: Optional[str] = None


@app.get("/portfolio/status")
async def portfolio_status():
    """Get current portfolio snapshot."""
    mgr = PortfolioManager()
    return mgr.snapshot().model_dump(mode="json")


@app.post("/portfolio/add")
async def portfolio_add(req: PortfolioAddRequest):
    """Add a PROMOTED strategy to the portfolio."""
    mgr = PortfolioManager()
    try:
        pos = mgr.add_strategy(req.strategy_id, req.initial_weight)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return pos.model_dump(mode="json")


@app.post("/portfolio/remove")
async def portfolio_remove(strategy_id: str, reason: str = "removed"):
    """Remove a strategy from the portfolio."""
    mgr = PortfolioManager()
    mgr.remove_strategy(strategy_id, reason)
    return {"strategy_id": strategy_id, "status": "removed"}


@app.post("/portfolio/pause")
async def portfolio_pause(strategy_id: str):
    """Pause a strategy in the portfolio."""
    mgr = PortfolioManager()
    mgr.pause_strategy(strategy_id)
    return {"strategy_id": strategy_id, "status": "paused"}


@app.post("/portfolio/resume")
async def portfolio_resume(strategy_id: str):
    """Resume a paused strategy."""
    mgr = PortfolioManager()
    mgr.resume_strategy(strategy_id)
    return {"strategy_id": strategy_id, "status": "resumed"}


@app.post("/portfolio/rebalance")
async def portfolio_rebalance(req: PortfolioRebalanceRequest):
    """Rebalance portfolio weights."""
    mgr = PortfolioManager()
    snap = mgr.rebalance(req.method)
    return snap.model_dump(mode="json")


@app.post("/portfolio/kill")
async def portfolio_kill(reason: str = "manual_kill"):
    """Emergency stop all portfolio positions."""
    mgr = PortfolioManager()
    mgr.kill_all(reason)
    return {"status": "killed", "reason": reason}


# ── drift monitoring endpoints ─────────────────────────────────────
from athena.live.feedback import FeedbackCollector  # noqa: E402
from athena.services.models import LiveSnapshot  # noqa: E402


@app.get("/drift/status")
async def drift_status(session_id: str):
    """Get latest drift classification for a live session."""
    collector = FeedbackCollector()
    snaps = collector.get_recent_snapshots(session_id, limit=1)
    if not snaps:
        raise HTTPException(status_code=404, detail="No snapshots for session")
    latest = snaps[0]
    return {
        "session_id": session_id,
        "strategy_id": latest.strategy_id,
        "timestamp": latest.timestamp.isoformat(),
        "is_degraded": latest.is_degraded,
        "live_sharpe": latest.sharpe_estimate,
        "live_drawdown": latest.max_drawdown,
        "backtest_sharpe": latest.backtest_sharpe,
        "backtest_drawdown": latest.backtest_max_drawdown,
        "sharpe_ratio": latest.sharpe_ratio,
        "drawdown_ratio": latest.drawdown_ratio,
    }


@app.get("/drift/history")
async def drift_history(session_id: str, limit: int = 50):
    """Get recent drift snapshots for a session."""
    collector = FeedbackCollector()
    snaps = collector.get_recent_snapshots(session_id, limit=limit)
    return [
        {
            "timestamp": s.timestamp.isoformat(),
            "is_degraded": s.is_degraded,
            "equity": s.equity,
            "sharpe_estimate": s.sharpe_estimate,
            "max_drawdown": s.max_drawdown,
            "sharpe_ratio": s.sharpe_ratio,
            "drawdown_ratio": s.drawdown_ratio,
        }
        for s in snaps
    ]


@app.post("/drift/demote")
async def drift_demote(strategy_id: str, reason: str = "manual"):
    """Manually demote a strategy (emergency stop + demote)."""
    from athena.live.feedback import AdaptiveLoop
    loop = AdaptiveLoop()
    await loop._demote_strategy(strategy_id, reason)
    return {"strategy_id": strategy_id, "status": "demoted", "reason": reason}





class HyperoptRequest(BaseModel):
    strategy_id: str
    epochs: int = 15
    loss_function: str = "SharpeHyperOptLoss"
    spaces: str = "buy sell"


class HyperoptResponse(BaseModel):
    strategy_id: str
    status: str
    best_loss: Optional[float] = None
    params: Optional[dict] = None


@app.post("/hyperopt", response_model=HyperoptResponse)
async def run_hyperopt(req: HyperoptRequest):
    """Run post-promotion hyperopt on a strategy to fine-tune parameters."""
    engine = AthenaEngine()
    record = engine.evaluate(req.strategy_id)
    finisher = HyperoptFinisher(
        epochs=req.epochs,
        loss_function=req.loss_function,
        spaces=req.spaces,
    )
    record = finisher.run(record)
    engine._persist(record)
    return HyperoptResponse(
        strategy_id=req.strategy_id,
        status="ok",
        best_loss=record.metadata.get("hyperopt", {}).get("best_loss"),
        params=record.metadata.get("hyperopt", {}).get("raw_params"),
    )

# ── scheduler control ──────────────────────────────────────────────
from athena.live.scheduler import get_scheduler  # noqa: E402


@app.post("/scheduler/start")
async def scheduler_start():
    get_scheduler().start()
    return {"running": True}


@app.post("/scheduler/stop")
async def scheduler_stop():
    get_scheduler().stop()
    return {"running": False}


@app.get("/scheduler/status")
async def scheduler_status():
    return get_scheduler().status()


# ── forward scheduler ──────────────────────────────────────────────
from athena.live.forward_scheduler import get_forward_scheduler  # noqa: E402


@app.post("/forward/scheduler/start")
async def forward_scheduler_start():
    get_forward_scheduler().start()
    return {"running": True}


@app.post("/forward/scheduler/stop")
async def forward_scheduler_stop():
    get_forward_scheduler().stop()
    return {"running": False}


@app.get("/forward/scheduler/status")
async def forward_scheduler_status():
    return get_forward_scheduler().status()


# ── live trading endpoints ─────────────────────────────────────────
from athena.live.runner import LiveRunner  # noqa: E402


class LiveStartRequest(BaseModel):
    strategy_id: str
    mode: str = "paper"
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


_runners: dict[str, LiveRunner] = {}


@app.post("/live/start", response_model=LiveStartResponse)
async def live_start(req: LiveStartRequest):
    session = get_session()
    strategy = session.query(StrategyModel).filter_by(id=req.strategy_id).first()
    if not strategy:
        raise HTTPException(status_code=404, detail="Strategy not found")
    session.close()

    if req.mode not in ("paper", "live"):
        raise HTTPException(status_code=400, detail="mode must be 'paper' or 'live'")

    if req.mode == "live" and (not req.exchange_key or not req.exchange_secret):
        raise HTTPException(status_code=400, detail="live mode requires exchange_key and exchange_secret")

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
    if runner.session_id:
        _runners[runner.session_id] = runner
    return LiveStartResponse(
        session_id=runner.session_id or "",
        strategy_id=req.strategy_id,
        status="running",
    )


@app.post("/live/stop")
async def live_stop(session_id: str):
    runner = _runners.pop(session_id, None)
    if not runner:
        raise HTTPException(status_code=404, detail="Session not found")
    await runner.stop("stopped_by_user")
    return {"session_id": session_id, "status": "stopped"}


@app.get("/live/status", response_model=LiveStatusResponse)
async def live_status(session_id: str):
    runner = _runners.get(session_id)
    if not runner:
        raise HTTPException(status_code=404, detail="Session not found")
    stats = runner.stats
    if not stats:
        raise HTTPException(status_code=404, detail="Session record missing")
    return LiveStatusResponse(**stats)


# ── telemetry ──────────────────────────────────────────────────────
from athena.services.forward_runner import ForwardRunner, run_forward


class ForwardRunRequest(BaseModel):
    strategy_id: str
    pairs: Optional[List[str]] = None
    timeframe: str = "1h"


class ForwardRunResponse(BaseModel):
    strategy_id: str
    bars: int
    total_signals: int
    open_positions: int
    total_closed: int
    total_pnl: float
    killed: bool


@app.post("/forward/run", response_model=ForwardRunResponse)
async def forward_run(req: ForwardRunRequest):
    """Run dry-run forward test for a stored strategy."""
    try:
        record, summary = run_forward(
            req.strategy_id,
            pairs=req.pairs,
            timeframe=req.timeframe,
            dry_run=True,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return ForwardRunResponse(
        strategy_id=req.strategy_id,
        bars=summary["bars"],
        total_signals=summary["total_signals"],
        open_positions=summary["open_positions"],
        total_closed=summary["total_closed"],
        total_pnl=summary["total_pnl"],
        killed=summary["killed"],
    )


# ── signals ─────────────────────────────────────────────────────────
from athena.services.models import Signal  # noqa: E402


class SignalListResponse(BaseModel):
    id: str
    strategy_id: str
    symbol: str
    direction: str
    confidence: float
    entry_price: float
    exit_price: Optional[float]
    pnl_pct: Optional[float]
    status: str
    created_at: str


@app.get("/signals")
async def list_signals(
    strategy_id: Optional[str] = None,
    symbol: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50,
):
    session = get_session()
    q = session.query(Signal)
    if strategy_id:
        q = q.filter(Signal.strategy_id == strategy_id)
    if symbol:
        q = q.filter(Signal.symbol == symbol)
    if status:
        q = q.filter(Signal.status == status)
    rows = q.order_by(Signal.created_at.desc()).limit(limit).all()
    return [
        {
            "id": r.id,
            "strategy_id": r.strategy_id,
            "symbol": r.symbol,
            "direction": r.direction,
            "confidence": r.confidence,
            "entry_price": r.entry_price,
            "exit_price": r.exit_price,
            "pnl_pct": r.pnl_pct,
            "status": r.status,
            "created_at": r.created_at.isoformat(),
        }
        for r in rows
    ]


# ── telemetry ──────────────────────────────────────────────────────
@app.get("/metrics")
async def metrics():
    return Response(content=generate_metrics(), media_type=metrics_content_type())
