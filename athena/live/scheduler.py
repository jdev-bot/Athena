"""Autonomous scheduler — runs periodic strategy evolution without human intervention.

Design:
  1. Every INTERVAL_SECONDS, run a micro-evolution (small population, few generations).
  2. Best strategies auto-promoted if they pass gates.
  3. If no strategies promoted for PATIENCE_CYCLES, expand search (more templates / larger pop).
  4. Runs in a background thread; safe to start/stop via API or CLI.
"""
import time
import threading
import logging
from datetime import datetime, timezone
from typing import Optional, List, Callable

from athena.common.models import StrategyTemplate, GenerationConfig
from athena.common.config import config
from athena.core.engine import AthenaEngine

logger = logging.getLogger(__name__)


DEFAULT_INTERVAL_SECONDS = 3600       # 1 hour between evolutions
DEFAULT_MICRO_POP = 12                # Small population for quick iterations
DEFAULT_MICRO_GEN = 3                 # Few generations
PATIENCE_CYCLES = 3                   # If no promotions after 3 cycles, expand search
EXPANDED_TEMPLATES: List[StrategyTemplate] = list(StrategyTemplate)


class AutonomousScheduler:
    """Background scheduler for continuous strategy discovery."""

    def __init__(
        self,
        interval_seconds: int = DEFAULT_INTERVAL_SECONDS,
        micro_pop: int = DEFAULT_MICRO_POP,
        micro_gen: int = DEFAULT_MICRO_GEN,
        templates: Optional[List[StrategyTemplate]] = None,
        on_promotion: Optional[Callable[[str, float], None]] = None,
    ):
        self.interval = interval_seconds
        self.micro_pop = micro_pop
        self.micro_gen = micro_gen
        self.templates = templates or [StrategyTemplate.MEAN_REVERSION, StrategyTemplate.BREAKOUT]
        self.on_promotion = on_promotion
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.cycles_without_promotion = 0
        self.total_evaluated = 0
        self.total_promoted = 0
        self.last_cycle_at: Optional[datetime] = None

    # ── lifecycle ────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background evolution loop."""
        if self._thread and self._thread.is_alive():
            logger.warning("Scheduler already running")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info(f"AutonomousScheduler started | interval={self.interval}s pop={self.micro_pop} gen={self.micro_gen}")

    def stop(self) -> None:
        """Signal the loop to stop."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("AutonomousScheduler stopped")

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def status(self) -> dict:
        return {
            "running": self.is_running,
            "cycles_without_promotion": self.cycles_without_promotion,
            "total_evaluated": self.total_evaluated,
            "total_promoted": self.total_promoted,
            "last_cycle_at": self.last_cycle_at.isoformat() if self.last_cycle_at else None,
            "next_templates": [t.value for t in self.templates],
        }

    # ── core loop ────────────────────────────────────────────────────

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            cycle_start = time.time()
            self._run_cycle()
            elapsed = time.time() - cycle_start
            sleep_for = max(0, self.interval - elapsed)
            logger.info(f"Cycle took {elapsed:.0f}s. Sleeping {sleep_for:.0f}s.")
            self._stop_event.wait(sleep_for)

    def _run_cycle(self) -> None:
        """One evolution cycle across configured templates."""
        self.last_cycle_at = datetime.now(timezone.utc)
        promoted_this_cycle = 0
        evaluated_this_cycle = 0

        for tpl in self.templates:
            if self._stop_event.is_set():
                break
            gen_config = GenerationConfig(
                symbols=["BTC-USD"],
                timeframe="1h",
                population_size=self.micro_pop,
                generations=self.micro_gen,
            )
            engine = AthenaEngine(gen_config)
            try:
                records = engine.evolve(tpl, run_gates=True)
                evaluated_this_cycle += len(records)
                for r in records:
                    if r.status.value == "promoted":
                        promoted_this_cycle += 1
                        if self.on_promotion:
                            self.on_promotion(r.id, r.score.raw_score)
            except Exception as exc:
                logger.error(f"Cycle failed for {tpl.value}: {exc}")

        self.total_evaluated += evaluated_this_cycle
        self.total_promoted += promoted_this_cycle

        if promoted_this_cycle == 0:
            self.cycles_without_promotion += 1
        else:
            self.cycles_without_promotion = 0

        # telemetry
        from athena.services.telemetry import TelemetryCollector
        _tele = TelemetryCollector()
        _tele.record_scheduler_cycle()
        _tele.record_evaluation(evaluated_this_cycle)
        if promoted_this_cycle:
            _tele.record_scheduler_promotion()

        # ── Adaptive expansion ──
        if self.cycles_without_promotion >= PATIENCE_CYCLES:
            logger.warning(f"No promotions for {PATIENCE_CYCLES} cycles. Expanding search.")
            self.templates = EXPANDED_TEMPLATES
            self.micro_pop = min(30, self.micro_pop + 6)
            self.micro_gen = min(8, self.micro_gen + 2)
            self.cycles_without_promotion = 0

        logger.info(
            f"Cycle complete | evaluated={evaluated_this_cycle} promoted={promoted_this_cycle} "
            f"total_promoted={self.total_promoted}"
        )


# ── convenience singleton ───────────────────────────────────────────
_scheduler: Optional[AutonomousScheduler] = None


def get_scheduler() -> AutonomousScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = AutonomousScheduler()
    return _scheduler
