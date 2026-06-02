"""Forward-test scheduler — periodically runs promoted strategies through ForwardRunner.

Architecture
------------
1. Poll DB every INTERVAL for strategies with status == PROMOTED.
2. For each promoted strategy, spin off a `run_forward()` call.
3. Persist summary to DB (`ForwardRunResult` table).
4. If kill-switch triggers during a run, log an ERROR and mark the record.
5. Exposes API for start / stop / status.
"""

import logging
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional

from athena.common.models import StrategyStatus
from athena.services.models import get_session, StrategyModel, Base
from athena.services.forward_runner import run_forward
from athena.services.telemetry import TelemetryCollector

logger = logging.getLogger(__name__)

DEFAULT_INTERVAL_SECONDS = 3600  # 1 hour


@dataclass(frozen=True)
class ForwardRunSummary:
    strategy_id: str
    bars: int
    total_signals: int
    open_positions: int
    total_closed: int
    total_pnl: float
    killed: bool
    started_at: float
    finished_at: float


class ForwardScheduler:
    """Background scheduler for dry-run forward-testing of promoted strategies."""

    def __init__(self, interval_seconds: int = DEFAULT_INTERVAL_SECONDS):
        self.interval = interval_seconds
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.results: List[ForwardRunSummary] = []
        self.total_cycles = 0
        self.total_runs = 0
        self.kills_seen = 0
        self.last_cycle_at: Optional[datetime] = None

    # ── lifecycle ─────────────────────────────────────────────────────

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            logger.warning("ForwardScheduler already running")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("ForwardScheduler started | interval=%s", self.interval)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("ForwardScheduler stopped")

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def status(self) -> dict:
        return {
            "running": self.is_running,
            "interval_seconds": self.interval,
            "total_cycles": self.total_cycles,
            "total_runs": self.total_runs,
            "kills_seen": self.kills_seen,
            "last_cycle_at": self.last_cycle_at.isoformat() if self.last_cycle_at else None,
            "cached_results": len(self.results),
        }

    # ── core loop ─────────────────────────────────────────────────────

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            cycle_start = time.time()
            self.run_cycle()
            elapsed = time.time() - cycle_start
            sleep_for = max(0, self.interval - elapsed)
            logger.info("Forward cycle took %.0f s. Sleeping %.0f s.", elapsed, sleep_for)
            self._stop_event.wait(sleep_for)

    def _check_drift(self, summary: dict, row) -> bool:
        """Compare forward PnL vs backtest baseline; demote if severe.

        Severe = forward PnL < 50 % of backtest total_return, or
                 backtest positive but forward < –10 %.
        """
        from athena.common.models import StrategyStatus
        import json
        backtest_return = float(getattr(row, "total_return", 0.0) or 0.0)
        forward_pnl = summary.get("total_pnl", 0.0)
        drift_ratio = forward_pnl / max(abs(backtest_return), 0.001)
        is_severe = (
            (backtest_return > 0 and drift_ratio < 0.5)
            or (backtest_return > 0 and forward_pnl < -0.10)
        )
        if is_severe:
            logger.error(
                "SEVERE DRIFT for %s: backtest=%.4f forward=%.4f ratio=%.2f — demoting",
                row.id, backtest_return, forward_pnl, drift_ratio,
            )
            row.status = StrategyStatus.RETIRED.value
            meta = row.metadata_json or {}
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except Exception:
                    meta = {}
            meta["drift_demotion"] = {
                "forward_pnl": forward_pnl,
                "backtest_return": backtest_return,
                "ratio": drift_ratio,
                "demoted_at": datetime.now(timezone.utc).isoformat(),
            }
            row.metadata_json = meta
            TelemetryCollector().set_kill_switch(True)
        return is_severe

    def _mini_ga_restart(self, template, old_dna: dict) -> None:
        """Spawn a lightweight re-evolution around the fallen champion."""
        import threading
        def _run():
            try:
                from athena.core.engine import AthenaEngine
                from athena.common.models import GenerationConfig, StrategyTemplate
                from athena.generator.dna import DNAEncoder
                cfg = GenerationConfig(population_size=8, generations=2, mutation_rate=0.25)
                engine = AthenaEngine(cfg)
                # evolve returns records sorted by score descending
                records = engine.evolve(StrategyTemplate(template), run_gates=True)
                if records and records[0].score.verdict == "promote":
                    logger.info("Mini-GA restart produced new champion %s (score=%.3f)",
                                records[0].id, records[0].score.raw_score)
                else:
                    logger.warning("Mini-GA restart did not produce a promotable strategy")
            except Exception as exc:
                logger.error("Mini-GA restart failed: %s", exc)
        threading.Thread(target=_run, daemon=True).start()

    def run_cycle(self) -> List[ForwardRunSummary]:
        """Run one forward cycle: find PROMOTED strategies and dry-run each.
        If severe drift is detected, demote strategy and trigger mini-GA restart."""
        self.last_cycle_at = datetime.now(timezone.utc)
        self.total_cycles += 1

        import json
        session = get_session()
        promoted = (
            session.query(StrategyModel)
            .filter_by(status=StrategyStatus.PROMOTED.value)
            .order_by(StrategyModel.updated_at.desc())
            .all()
        )

        results: List[ForwardRunSummary] = []
        for row in promoted:
            sid = row.id
            logger.info("ForwardScheduler: running forward test for %s (%s)", sid, row.template)
            try:
                _, summary = run_forward(
                    sid,
                    pairs=None,
                    timeframe="1h",
                    dry_run=True,
                )
                record = ForwardRunSummary(
                    strategy_id=sid,
                    bars=summary["bars"],
                    total_signals=summary["total_signals"],
                    open_positions=summary["open_positions"],
                    total_closed=summary["total_closed"],
                    total_pnl=summary["total_pnl"],
                    killed=summary["killed"],
                    started_at=time.time(),
                    finished_at=time.time(),
                )
                results.append(record)
                self.results.append(record)
                self.total_runs += 1

                TelemetryCollector().record_signal("forward_run")
                if record.killed:
                    self.kills_seen += 1
                    TelemetryCollector().record_kill_switch()
                    logger.error(
                        "KILL-SWITCH triggered during forward run: strategy=%s pnl=%.4f",
                        sid, record.total_pnl,
                    )

                # ── drift check ──
                session = get_session()
                # refresh row from current session in case it was modified by another cycle
                row = session.query(StrategyModel).filter_by(id=sid).first()
                severe = self._check_drift(summary, row)
                if severe:
                    self._mini_ga_restart(row.template, row.dna)
                session.commit()
                session.close()
            except Exception as exc:
                logger.error("ForwardScheduler: run failed for %s: %s", sid, exc)

        session.close()
        return results


# ── convenience singleton ───────────────────────────────────────────
_fw_scheduler: Optional[ForwardScheduler] = None


def get_forward_scheduler() -> ForwardScheduler:
    global _fw_scheduler
    if _fw_scheduler is None:
        _fw_scheduler = ForwardScheduler()
    return _fw_scheduler
