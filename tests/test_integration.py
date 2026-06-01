"""E2E integration tests for Athena unified engine + API.

Coverage:
  1. AthenaEngine — evolve() → backtest → score → gates → promote
  2. API (TestClient) — /evolve, /backtests/run, /promote, /deploy, /stats
  3. Orchestrator CLI — importable, thin wrapper
"""
import pytest
from fastapi.testclient import TestClient

from athena.services.api import app
from athena.core.engine import AthenaEngine
from athena.common.models import StrategyTemplate, GenerationConfig, StrategyStatus


client = TestClient(app)


# ── helpers ────────────────────────────────────────────────────────
def _get_or_create_strategy():
    """Generate a single strategy and return its ID."""
    resp = client.post("/strategies/generate", json={"template": "mean_reversion", "count": 1})
    assert resp.status_code == 200
    data = resp.json()
    return data["strategies"][0]["id"]


# ── 1. API health ──────────────────────────────────────────────────
def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["framework"] == "freqtrade"


# ── 2. Strategy generation ───────────────────────────────────────
def test_generate_strategies():
    resp = client.post("/strategies/generate", json={"template": "mean_reversion", "count": 3})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["strategies"]) == 3
    assert all("id" in s and "dna" in s for s in data["strategies"])


# ── 3. Engine evolve (small population for speed) ─────────────────
@pytest.mark.slow
@pytest.mark.timeout(300)
def test_engine_evolve_mean_reversion():
    cfg = GenerationConfig(
        symbols=["BTC-USD"], timeframe="1h",
        start_date="2026-05-02", end_date="2026-06-01",
        population_size=6, generations=2,
    )
    engine = AthenaEngine(cfg)
    records = engine.evolve(StrategyTemplate.MEAN_REVERSION, run_gates=True)
    assert len(records) == 6
    # Best record should have score computed
    best = records[0]
    assert best.score.raw_score >= 0.0
    assert best.performance.total_trades >= 0
    # Metadata should contain gate results
    assert "walk_forward" in (best.metadata or {})
    assert "monte_carlo" in (best.metadata or {})


# ── 4. API backtest run ──────────────────────────────────────────
@pytest.mark.slow
@pytest.mark.timeout(120)
def test_api_backtest_run():
    sid = _get_or_create_strategy()
    resp = client.post("/backtests/run", json={
        "strategy_id": sid,
        "start_date": "2026-05-02",
        "end_date": "2026-06-01",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["strategy_id"] == sid
    assert body["status"] in (StrategyStatus.BACKTEST_DONE.value, StrategyStatus.BACKTEST_FAILED.value)
    assert "metrics" in body
    if "error" not in body["metrics"]:
        assert "total_trades" in body["metrics"]


# ── 5. API stats ─────────────────────────────────────────────────
def test_api_stats():
    resp = client.get("/stats")
    assert resp.status_code == 200
    body = resp.json()
    assert "total_strategies" in body
    assert "by_status" in body
    assert "best_strategy" in body


# ── 6. API list strategies ───────────────────────────────────────
def test_api_list_strategies():
    resp = client.get("/strategies?limit=10")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)


# ── 7. API promote (best effort; may fail gates) ───────────────
@pytest.mark.slow
@pytest.mark.timeout(180)
def test_api_promote():
    sid = _get_or_create_strategy()
    # First run backtest so scores exist
    client.post("/backtests/run", json={
        "strategy_id": sid,
        "start_date": "2026-05-02",
        "end_date": "2026-06-01",
    })
    resp = client.post("/promote", json={"strategy_id": sid})
    assert resp.status_code == 200
    body = resp.json()
    assert body["strategy_id"] == sid
    assert isinstance(body["promoted"], bool)
    assert "gates_passed" in body


# ── 8. API deploy ────────────────────────────────────────────────
@pytest.mark.slow
@pytest.mark.timeout(60)
def test_api_deploy():
    sid = _get_or_create_strategy()
    resp = client.post("/deploy", params={"strategy_id": sid})
    assert resp.status_code == 200
    body = resp.json()
    assert body["strategy_id"] == sid
    assert body["path"].endswith(f"{sid}.py")
    from pathlib import Path
    assert Path(body["path"]).exists()


# ── 9. CLI orchestrator importable ───────────────────────────────
def test_orchestrator_importable():
    from athena import orchestrator
    assert hasattr(orchestrator, "main")


# ── 10. Engine evaluate existing strategy ────────────────────────
@pytest.mark.slow
@pytest.mark.timeout(120)
def test_engine_evaluate():
    sid = _get_or_create_strategy()
    cfg = GenerationConfig(
        symbols=["BTC-USD"], timeframe="1h",
        start_date="2026-05-02", end_date="2026-06-01",
    )
    engine = AthenaEngine(cfg)
    record = engine.evaluate(sid)
    assert record.id == sid
    assert record.score is not None
    assert record.performance.total_trades >= 0


# ── 11. Scheduler importable ───────────────────────────────────
def test_scheduler_importable():
    from athena.live.scheduler import AutonomousScheduler
    s = AutonomousScheduler(interval_seconds=9999)
    assert not s.is_running
    assert s.status()["running"] is False
