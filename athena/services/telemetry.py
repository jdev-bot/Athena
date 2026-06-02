"""Telemetry — Prometheus metrics for live session visibility.

All metrics are lazily-registered singletons so they can be imported
from anywhere without worrying about init order.  The scrape endpoint
is mounted inside the main FastAPI app via /metrics so there is only
one HTTP server to manage.
"""
import time
import logging
from typing import Callable

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

logger = logging.getLogger(__name__)

# ── metrics ────────────────────────────────────────────────────────
_STRATEGIES = Gauge("athena_strategies_total", "Number of strategies", ["status"])
_BACKTESTS = Counter("athena_backtests_total", "Backtests run")
_GENERATIONS = Counter("athena_generations_total", "GA generations completed")
_EVALUATIONS = Counter("athena_evaluations_total", "Strategies evaluated")
_SIGNALS = Counter("athena_signals_total", "Signals emitted", ["direction"])
_KILL_SWITCH = Gauge("athena_kill_switch_active", "1 if kill-switch is triggered")
_PNL = Gauge("athena_pnl", "Running PnL of dry-run session")
_SCHEDULER_CYCLES = Counter("athena_scheduler_cycles_total", "Scheduler cycles run")
_SCHEDULER_PROMOTIONS = Counter("athena_scheduler_promotions_total", "Strategies auto-promoted")
_API_REQUESTS = Counter("athena_api_requests_total", "API requests", ["endpoint", "status"])
_API_LATENCY = Histogram("athena_api_latency_seconds", "API latency", ["endpoint"])
_EVOLVE_LATENCY = Histogram("athena_evolve_latency_seconds", "Full evolve latency")


class TelemetryCollector:
    """Thin wrapper that exports Athena-specific events to Prometheus."""

    def __init__(self):
        self._kill = False
        self._pnl = 0.0

    # ── strategy lifecycle ──────────────────────────────────────────

    def record_strategy_count(self, status: str, count: int) -> None:
        _STRATEGIES.labels(status=status).set(count)

    def record_backtest(self) -> None:
        _BACKTESTS.inc()

    def record_generation(self) -> None:
        _GENERATIONS.inc()

    def record_evaluation(self, n: int = 1) -> None:
        _EVALUATIONS.inc(n)

    def record_signal(self, direction: str) -> None:
        _SIGNALS.labels(direction=direction).inc()

    # ── kill-switch / PnL ────────────────────────────────────────────

    def set_kill_switch(self, active: bool) -> None:
        self._kill = active
        _KILL_SWITCH.set(1 if active else 0)

    def set_pnl(self, value: float) -> None:
        self._pnl = value
        _PNL.set(value)

    # ── scheduler ────────────────────────────────────────────────────

    def record_scheduler_cycle(self) -> None:
        _SCHEDULER_CYCLES.inc()

    def record_scheduler_promotion(self) -> None:
        _SCHEDULER_PROMOTIONS.inc()

    # ── API middleware helpers ───────────────────────────────────────

    def wrap_endpoint(self, endpoint: str, fn: Callable):
        """Decorator that times an endpoint and records counts."""
        async def _wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = await fn(*args, **kwargs)
                _API_REQUESTS.labels(endpoint=endpoint, status="2xx").inc()
                return result
            except Exception:
                _API_REQUESTS.labels(endpoint=endpoint, status="5xx").inc()
                raise
            finally:
                _API_LATENCY.labels(endpoint=endpoint).observe(time.time() - start)
        return _wrapper


def generate_metrics() -> bytes:
    """Return the latest prometheus-formatted metrics."""
    return generate_latest()


def metrics_content_type() -> str:
    return CONTENT_TYPE_LATEST
