"""End-to-end tests for Athena."""
import pytest
from fastapi.testclient import TestClient

from athena.services.api import app, init_db
from athena.common.models import StrategyTemplate, StrategyStatus
from athena.generator.dna import DNAEncoder
from athena.generator.ga_engine import GAEngine
from athena.evaluator.scorer import Scorer
from athena.common.models import PerformanceMetrics, ScoreResult


# ── FastAPI fixtures ──────────────────────────────────────────────
@pytest.fixture
def client():
    """Yield a FastAPI TestClient with fresh DB."""
    init_db()
    with TestClient(app) as c:
        yield c


# ── API: health ───────────────────────────────────────────────────
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["service"] == "athena"


# ── API: generate ─────────────────────────────────────────────────
def test_generate_strategies(client):
    r = client.post("/strategies/generate", json={"template": "trend_following", "count": 3})
    assert r.status_code == 200
    body = r.json()
    assert len(body["strategies"]) == 3
    for s in body["strategies"]:
        assert s["id"].startswith("strat_")
        assert s["template"] == "trend_following"
        assert "dna" in s


def test_generate_unknown_template(client):
    r = client.post("/strategies/generate", json={"template": "nonexistent", "count": 1})
    assert r.status_code == 400


# ── API: list + get ───────────────────────────────────────────────
def test_list_and_get_strategy(client):
    # Generate
    r = client.post("/strategies/generate", json={"template": "mean_reversion", "count": 1})
    strat = r.json()["strategies"][0]
    sid = strat["id"]

    # List
    r2 = client.get("/strategies")
    assert r2.status_code == 200
    assert any(x["id"] == sid for x in r2.json())

    # Get by ID
    r3 = client.get(f"/strategies/{sid}")
    assert r3.status_code == 200
    assert r3.json()["id"] == sid

    # Filter by status
    r4 = client.get("/strategies?status=generated")
    assert r4.status_code == 200
    assert all(x["status"] == "generated" for x in r4.json())


def test_get_strategy_404(client):
    r = client.get("/strategies/strat_does_not_exist")
    assert r.status_code == 404


# ── API: backtest ───────────────────────────────────────────────────
def test_run_backtest(client):
    # Generate
    r = client.post("/strategies/generate", json={"template": "trend_following", "count": 1})
    strat = r.json()["strategies"][0]
    sid = strat["id"]

    # Run backtest
    r2 = client.post("/backtests/run", json={
        "strategy_id": sid,
        "start_date": "2024-01-01",
        "end_date": "2024-02-01",
    })
    assert r2.status_code == 200
    body = r2.json()
    assert body["strategy_id"] == sid
    # Should complete (even if zero trades on random candles)
    assert body["status"] in ("backtest_done", "backtest_failed")
    assert "metrics" in body
    m = body["metrics"]
    assert isinstance(m.get("total_return"), (int, float))
    assert isinstance(m.get("sharpe"), (int, float))
    assert isinstance(m.get("total_trades"), int)


def test_run_backtest_404(client):
    r = client.post("/backtests/run", json={
        "strategy_id": "strat_noexist",
        "start_date": "2024-01-01",
        "end_date": "2024-02-01",
    })
    assert r.status_code == 404


# ── API: backtests list ───────────────────────────────────────────
def test_list_backtests(client):
    # Generate + run backtest so at least one record has BACKTEST_DONE
    r = client.post("/strategies/generate", json={"template": "trend_following", "count": 1})
    strat = r.json()["strategies"][0]
    client.post("/backtests/run", json={
        "strategy_id": strat["id"],
        "start_date": "2024-01-01",
        "end_date": "2024-02-01",
    })

    r2 = client.get("/backtests")
    assert r2.status_code == 200
    # At least the one we just ran should appear
    assert len(r2.json()) >= 1


# ── API: stats ────────────────────────────────────────────────────
def test_stats_endpoint(client):
    # Generate a few
    client.post("/strategies/generate", json={"template": "breakout", "count": 2})
    r = client.get("/stats")
    assert r.status_code == 200
    body = r.json()
    assert body["total_strategies"] >= 2
    assert "by_status" in body
    assert "best_strategy" in body


# ── Generator: DNA ────────────────────────────────────────────────
def test_dna_encoder_random():
    enc = DNAEncoder()
    dna = enc.random_dna(StrategyTemplate.TREND_FOLLOWING)
    assert "fast_period" in dna
    assert "slow_period" in dna
    assert isinstance(dna["fast_period"], int)
    assert isinstance(dna["slow_period"], int)


def test_dna_mutate():
    enc = DNAEncoder()
    dna = enc.random_dna(StrategyTemplate.MEAN_REVERSION)
    mutated = enc.mutate(dna, StrategyTemplate.MEAN_REVERSION, mutation_rate=1.0)
    # With mutation_rate=1.0 every field should change (probabilistically)
    assert len(mutated) == len(dna)


def test_dna_crossover():
    enc = DNAEncoder()
    dna1 = enc.random_dna(StrategyTemplate.BREAKOUT)
    dna2 = enc.random_dna(StrategyTemplate.BREAKOUT)
    c1, c2 = enc.crossover(dna1, dna2, StrategyTemplate.BREAKOUT)
    assert len(c1) == len(dna1)
    assert len(c2) == len(dna2)


# ── Generator: GA Engine ──────────────────────────────────────────
def test_ga_initialize():
    ga = GAEngine(StrategyTemplate.MOMENTUM, population_size=10, generations=2)
    ga.initialize_population()
    assert len(ga.population) == 10
    assert ga.population[0].template == StrategyTemplate.MOMENTUM


def test_ga_evolve():
    ga = GAEngine(StrategyTemplate.VOLATILITY, population_size=8, generations=2)
    ga.initialize_population()

    def fitness_fn(ind):
        return ind.dna.get("atr_period", 14) / 100.0

    population = ga.evolve(fitness_fn)
    assert len(population) == 8
    # Best should have highest fitness
    assert population[0].fitness >= population[-1].fitness


# ── Evaluator: Scorer ───────────────────────────────────────────────
def test_scorer_low_trades():
    s = Scorer()
    metrics = PerformanceMetrics(total_trades=1, sharpe=1.5)
    score = s.score(metrics)
    assert score.verdict == "demote"
    assert score.raw_score == 0.0


def test_scorer_promote():
    s = Scorer()
    metrics = PerformanceMetrics(
        total_trades=50,
        sharpe=2.0,
        sortino=2.0,
        calmar=3.0,
        win_rate=0.6,
        max_drawdown=0.1,
    )
    score = s.score(metrics)
    assert score.raw_score > 0.5
    assert score.verdict == "promote"


def test_scorer_demote_high_drawdown():
    s = Scorer()
    metrics = PerformanceMetrics(
        total_trades=50,
        sharpe=2.0,
        sortino=2.0,
        calmar=3.0,
        win_rate=0.6,
        max_drawdown=0.35,
    )
    score = s.score(metrics)
    # High drawdown penalizes score
    assert score.raw_score < 1.0


# ── Integration: orchestrator round-trip ────────────────────────────
def test_orchestrator_roundtrip():
    from athena.orchestrator import AthenaOrchestrator
    from athena.common.models import GenerationConfig

    cfg = GenerationConfig(
        symbols=["BTC-USD"],
        timeframe="1h",
        population_size=4,
        generations=1,
    )
    orch = AthenaOrchestrator(cfg)
    records = orch.run_generation(StrategyTemplate.TREND_FOLLOWING, run_gates=False)
    assert len(records) > 0
    # All records should have been saved to DB
    best = records[0]
    assert best.score.raw_score >= 0.0


# ── FreqtradeWrapper: compile + backtest ───────────────────────────
def test_freqtrade_wrapper_compile():
    from athena.core.freqtrade_wrapper import FreqtradeWrapper
    from athena.services.models import StrategyModel

    record = StrategyModel(
        id="strat_test123",
        name="test",
        template="trend_following",
        dna={"fast_period": 10, "slow_period": 30, "trend_threshold": 0.01,
             "rsi_period": 14, "rsi_overbought": 70, "rsi_oversold": 30,
             "position_size_pct": 0.10, "min_stake_usd": 5.0, "max_stake_usd": 100.0,
             "max_open_trades": 1, "risk_capital_pct": 0.50, "reserve_capital_pct": 0.10},
    )
    code = FreqtradeWrapper.compile_strategy(record)
    assert "class AthenaStrategy(IStrategy)" in code
    assert "populate_entry_trend" in code


def test_freqtrade_wrapper_backtest():
    from athena.core.freqtrade_wrapper import FreqtradeWrapper
    from athena.generator.templates import TEMPLATE_MAP
    encoder = DNAEncoder()
    params = encoder.to_strategy_params(
        {"fast_period": 10, "slow_period": 30, "trend_threshold": 0.01,
         "rsi_period": 14, "rsi_overbought": 70, "rsi_oversold": 30,
         "position_size_pct": 0.10, "min_stake_usd": 5.0, "max_stake_usd": 100.0,
         "max_open_trades": 1, "risk_capital_pct": 0.50, "reserve_capital_pct": 0.10},
        StrategyTemplate.TREND_FOLLOWING,
    )
    params["class_name"] = "AthenaStrategy"
    params["template_name"] = "trend_following"
    params["timeframe"] = "1h"
    template = TEMPLATE_MAP.get(StrategyTemplate.TREND_FOLLOWING)
    code = template.format(**params)
    wrapper = FreqtradeWrapper()
    result = wrapper.run_backtest(code, start_date="2024-01-01", end_date="2024-01-15")
    assert "error" not in result or result["total_trades"] == 0
    assert isinstance(result["total_return"], (int, float))
    assert isinstance(result["total_trades"], int)


def test_live_start_stop(client):
    # Patch subprocess so we don't actually spawn a freqtrade bot in tests
    import subprocess as subprocess_mod
    from unittest.mock import MagicMock, patch

    mock_proc = MagicMock()
    mock_proc.pid = 12345
    mock_proc.poll.return_value = None
    mock_proc.send_signal = MagicMock()
    mock_proc.wait = MagicMock(return_value=0)

    with patch.object(subprocess_mod, "Popen", return_value=mock_proc):
        # Create strategy
        r = client.post("/strategies/generate", json={"template": "trend_following", "count": 1})
        strat = r.json()["strategies"][0]
        sid = strat["id"]

        # Start live session
        r2 = client.post("/live/start", json={
            "strategy_id": sid,
            "mode": "paper",
            "max_drawdown": 0.15,
            "daily_loss_limit": 0.10,
        })
        assert r2.status_code == 200
        body = r2.json()
        session_id = body["session_id"]
        assert body["status"] == "running"
        assert body["strategy_id"] == sid

        # Status (bot not fully booted in 3s, falls back to DB)
        r3 = client.get(f"/live/status?session_id={session_id}")
        assert r3.status_code == 200
        status = r3.json()
        assert status["session_id"] == session_id
        assert status["equity"] is not None

        # Stop
        r4 = client.post("/live/stop", params={"session_id": session_id})
        assert r4.status_code == 200
        assert r4.json()["status"] == "stopped"

        # Status after stop should 404 (runner removed from in-memory registry)
        r5 = client.get(f"/live/status?session_id={session_id}")
        assert r5.status_code == 404


def test_live_start_invalid_mode(client):
    r = client.post("/strategies/generate", json={"template": "trend_following", "count": 1})
    sid = r.json()["strategies"][0]["id"]
    r2 = client.post("/live/start", json={
        "strategy_id": sid,
        "mode": "invalid",
    })
    assert r2.status_code == 400
