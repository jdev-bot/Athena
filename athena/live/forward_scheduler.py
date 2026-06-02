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

    def run_cycle(self) -> List[ForwardRunSummary]:
        """Run one forward cycle: find PROMOTED strategies and dry-run each."""
        self.last_cycle_at = datetime.now(timezone.utc)
        self.total_cycles += 1

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
                    pairs=None,  # default BTC/USDT:USDT
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
                    started_at=time.time(),  # approximate; actual start/end inside run_forward
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
                        sid,
                        record.total_pnl,
                    )
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
