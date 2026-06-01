"""Continuous feedback loop — live/paper PnL drives GA re-evolution & drift detection."""
import asyncio
import json
import logging
import math
import statistics
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from sqlalchemy import desc

from athena.services.models import get_session, LiveSnapshot, StrategyModel, LiveSessionModel
from athena.live.bot_manager import BotManager
from athena.common.models import (
    StrategyRecord, StrategyDNA, PerformanceMetrics, StrategyTemplate, StrategyStatus,
)
from athena.core.engine import AthenaEngine
from athena.common.config import config


logger = logging.getLogger(__name__)


class _StreamBuffer:
    """Rolling window of equity points for O(1) drawdown + Sharpe estimation."""

    def __init__(self, window_minutes: int = 480):
        self.points: List[Tuple[datetime, float]] = []  # (timestamp, equity)
        self.window_minutes = window_minutes
        self.peak = 0.0

    def append(self, ts: datetime, equity: float):
        self.points.append((ts, equity))
        cutoff = ts.timestamp() - self.window_minutes * 60
        while self.points and self.points[0][0].timestamp() < cutoff:
            self.points.pop(0)
        self.peak = max((p[1] for p in self.points), default=0.0)
        if equity > self.peak:
            self.peak = equity

    @property
    def drawdown(self) -> float:
        if not self.points:
            return 0.0
        return (self.peak - self.points[-1][1]) / self.peak if self.peak > 0 else 0.0

    @property
    def equity_returns(self) -> List[float]:
        """Simple period-to-period returns for rolling Sharpe estimate."""
        eqs = [p[1] for p in self.points]
        if len(eqs) < 2:
            return []
        returns = [(eqs[i] - eqs[i - 1]) / eqs[i - 1] for i in range(1, len(eqs)) if eqs[i - 1] > 0]
        return returns

    @property
    def estimated_sharpe(self) -> float:
        """Very crude annualized Sharpe from period returns."""
        rets = self.equity_returns
        if len(rets) < 2:
            return 0.0
        mean_r = statistics.mean(rets)
        std_r = statistics.stdev(rets)
        if std_r == 0:
            return 0.0
        n_periods = len(rets)
        # Assuming points are ~snapshot_interval apart; rough annualization factor
        annualization = math.sqrt(min(n_periods, 252))  # cap at ~1y
        return (mean_r / std_r) * annualization

    @property
    def estimated_win_rate(self) -> float:
        """Win rate from rolling returns ≥ 0."""
        rets = self.equity_returns
        if not rets:
            return 0.0
        wins = sum(1 for r in rets if r >= 0)
        return wins / len(rets)


# ─────────────────────────────────────────────────────────────────────
class FeedbackCollector:
    """Poll live/paper session, record snapshots, compute drift."""

    # Drift thresholds (live / backtest)
    SHARPE_DEGRADE_MILD = 0.60   # live_sharpe < backtest * 0.60
    SHARPE_DEGRADE_SEVERE = 0.30  # live_sharpe < backtest * 0.30
    DD_DEGRADE_MILD = 2.0         # live_dd > backtest_dd * 2.0
    DD_DEGRADE_SEVERE = 3.0       # live_dd > backtest_dd * 3.0

    def __init__(self, snapshot_interval_mins: int = 10, window_minutes: int = 480):
        self.interval = snapshot_interval_mins * 60  # seconds
        self.buffers: Dict[str, _StreamBuffer] = {}  # session_id -> buffer
        self.manager = BotManager()

    # ── polling ──────────────────────────────────────────────────────
    async def start_monitor(self, session_id: str):
        """Background task: run _poll loop for one session forever."""
        self.buffers[session_id] = _StreamBuffer(window_minutes=self.interval // 60 * 2)
        try:
            while True:
                await asyncio.sleep(self.interval)
                await self._poll(session_id)
        except asyncio.CancelledError:
            pass

    async def stop_monitor(self, session_id: str):
        self.buffers.pop(session_id, None)

    async def _poll(self, session_id: str):
        """Single snapshot capture."""
        stats = self.manager.status(session_id)
        if not stats:
            return

        equity = float(stats.get("equity", 0.0))
        total_trades = int(stats.get("total_trades", 0))
        p_closed = float(stats.get("profit_closed_pct", 0.0))
        p_all = float(stats.get("profit_all_pct", 0.0))
        ts = datetime.now(timezone.utc)

        # Rolling metrics
        buf = self.buffers.get(session_id)
        if buf:
            buf.append(ts, equity)
            live_sharpe = buf.estimated_sharpe
            live_dd = buf.drawdown
            live_wr = buf.estimated_win_rate
        else:
            live_sharpe = live_dd = live_wr = 0.0

        # Backtest baseline from strategies table
        strategy_id = self._resolve_strategy_id(session_id)
        bt_sharpe, bt_dd = self._get_backtest_baseline(strategy_id)

        # Drift classification
        is_degraded = self._classify(live_sharpe, live_dd, bt_sharpe, bt_dd)

        # Persist
        snap = LiveSnapshot(
            session_id=session_id,
            strategy_id=strategy_id,
            timestamp=ts,
            equity=equity,
            total_trades=total_trades,
            profit_closed_pct=p_closed,
            profit_all_pct=p_all,
            sharpe_estimate=live_sharpe,
            max_drawdown=live_dd,
            win_rate=live_wr,
            backtest_sharpe=bt_sharpe,
            backtest_max_drawdown=bt_dd,
            sharpe_ratio=(live_sharpe / bt_sharpe) if bt_sharpe > 0 else 0.0,
            drawdown_ratio=(live_dd / bt_dd) if bt_dd > 0 else 0.0,
            is_degraded=is_degraded,
        )
        sess = get_session()
        sess.add(snap)
        sess.commit()
        sess.close()

        return snap

    # ── helpers ──────────────────────────────────────────────────────
    def _resolve_strategy_id(self, session_id: str) -> str:
        sess = get_session()
        row = sess.query(LiveSessionModel).filter_by(id=session_id).first()
        sid = row.strategy_id if row else ""
        sess.close()
        return sid

    def _get_backtest_baseline(self, strategy_id: str) -> Tuple[float, float]:
        sess = get_session()
        row = sess.query(StrategyModel).filter_by(id=strategy_id).first()
        if row:
            sharpe = row.sharpe or 0.0
            dd = abs(row.max_drawdown or 0.0)
        else:
            sharpe = dd = 0.0
        sess.close()
        return sharpe, dd

    def _classify(
        self, live_sharpe: float, live_dd: float, bt_sharpe: float, bt_dd: float
    ) -> str:
        if bt_sharpe <= 0 and live_sharpe <= 0:
            return ""   # no meaningful comparison possible
        if bt_sharpe > 0:
            sharpe_ratio = live_sharpe / bt_sharpe
        else:
            sharpe_ratio = 1.0 if live_sharpe > 0 else 0.0

        if bt_dd > 0:
            dd_ratio = live_dd / bt_dd
        else:
            dd_ratio = 0.0

        severe = sharpe_ratio < self.SHARPE_DEGRADE_SEVERE or dd_ratio > self.DD_DEGRADE_SEVERE
        mild = sharpe_ratio < self.SHARPE_DEGRADE_MILD or dd_ratio > self.DD_DEGRADE_MILD

        if severe:
            return "severe"
        elif mild:
            return "mild"
        return ""

    def get_recent_snapshots(self, session_id: str, limit: int = 100) -> List[LiveSnapshot]:
        sess = get_session()
        rows = (
            sess.query(LiveSnapshot)
            .filter_by(session_id=session_id)
            .order_by(desc(LiveSnapshot.timestamp))
            .limit(limit)
            .all()
        )
        sess.close()
        return rows


# ─────────────────────────────────────────────────────────────────────
class AdaptiveLoop:
    """Triggers re-optimization when live/paper degradation is detected."""

    def __init__(
        self,
        collector: Optional[FeedbackCollector] = None,
        mini_pop: int = 10,
        mini_gens: int = 5,
        mutation_boost: float = 0.40,
    ):
        self.collector = collector or FeedbackCollector()
        self.mini_pop = mini_pop
        self.mini_gens = mini_gens
        self.mutation_boost = mutation_boost
        self._tasks: Dict[str, asyncio.Task] = {}

    # ── entry ──────────────────────────────────────────────────────
    async def watch_session(self, session_id: str):
        """Starts collector monitoring + adaptive re-optimization for one session."""
        collector_task = asyncio.create_task(
            self.collector.start_monitor(session_id)
        )
        adaptive_task = asyncio.create_task(
            self._adaptive_worker(session_id)
        )
        self._tasks[session_id] = asyncio.gather(collector_task, adaptive_task)

    async def stop_watch(self, session_id: str):
        """Cancel all bg tasks for session, stop collector."""
        task = self._tasks.pop(session_id, None)
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        await self.collector.stop_monitor(session_id)

    # ── adaptive worker ────────────────────────────────────────────
    async def _adaptive_worker(self, session_id: str):
        """Every poll checks drift. If severe → re-optimize → swap strategy."""
        while True:
            await asyncio.sleep(self.collector.interval)

            snaps = self.collector.get_recent_snapshots(session_id, limit=10)
            if not snaps:
                continue

            latest = snaps[0]

            # Action on severe degradation
            if latest.is_degraded == "severe":
                await self._handle_degradation(session_id, latest)

            # Mild degradation: just log / alert (no auto-stop, user can decide)
            elif latest.is_degraded == "mild" and len(snaps) >= 5:
                # Only act if mild persists for 5+ consecutive snapshots
                if all(s.is_degraded in ("mild", "severe") for s in snaps[:5]):
                    await self._handle_degradation(session_id, latest, aggressive=False)

    # ── re-optimization ─────────────────────────────────────────────
    async def _handle_degradation(
        self, session_id: str, latest: LiveSnapshot, aggressive: bool = True
    ):
        """Stop current bot, run mini-GA from current champion, start new winner."""
        strategy_id = latest.strategy_id

        # 1. Stop current bot
        self.collector.manager.stop(session_id, reason=f"degraded_{latest.is_degraded}")
        await asyncio.sleep(2)

        # 2. Load current champion DNA from DB
        sess = get_session()
        strategy_row = sess.query(StrategyModel).filter_by(id=strategy_id).first()
        sess.close()
        if not strategy_row:
            return

        dna_vector = strategy_row.dna.get("vector", {}) if isinstance(strategy_row.dna, dict) else {}
        template = StrategyTemplate(strategy_row.template)

        # 3. Mini-GA: seed from current champion + elevated mutation
        from athena.generator.ga_engine import GAEngine
        from athena.generator.dna import DNAEncoder

        encoder = DNAEncoder()
        ga = GAEngine(
            template=template,
            population_size=self.mini_pop,
            generations=self.mini_gens,
            mutation_rate=self.mutation_boost,
            crossover_rate=0.7,
            elitism_count=max(1, self.mini_pop // 5),
        )
        ga.initialize_population()
        # Inject champion
        ga.population[0].dna = dict(dna_vector)
        ga.population[0].generation = 0

        engine = AthenaEngine(self._mini_gen_config())

        def fitness_fn(ind):
            record = StrategyRecord(
                id=ind.id,
                name=f"{template.value}_{ind.id[-6:]}",
                template=template,
                dna=StrategyDNA(template=template, vector=ind.dna),
                generation=ind.generation,
            )
            record = engine.evaluate_record(record)
            return round(record.score.raw_score, 6)

        population = ga.evolve(fitness_fn)
        best = population[0]

        # 4. Save new strategy
        new_record = StrategyRecord(
            id=f"strat_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
            name=f"adaptive_{template.value}",
            template=template,
            dna=StrategyDNA(template=template, vector=best.dna),
            status=StrategyStatus.GENERATED,
        )
        engine._persist(new_record)

        # 5. If aggressive, auto-launch the new winner in paper mode
        if aggressive:
            await self._launch_new_best(new_record, session_id)

    async def _launch_new_best(self, record: StrategyRecord, old_session_id: str):
        """Start a new bot with the adapted strategy."""
        new_session = await self.collector.manager.start(
            strategy_id=record.id,
            mode="paper",
            risk={"max_drawdown": config.PROMOTE_THRESHOLD},
        )
        # Start monitoring the new session too
        await self.watch_session(new_session)

    # ── helpers ─────────────────────────────────────────────────────
    def _mini_gen_config(self):
        from athena.common.models import GenerationConfig
        return GenerationConfig(
            symbols=["BTC-USD"],
            timeframe="1h",
            population_size=self.mini_pop,
            generations=self.mini_gens,
            mutation_rate=self.mutation_boost,
            crossover_rate=0.7,
        )
