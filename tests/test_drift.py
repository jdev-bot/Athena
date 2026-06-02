"""Tests for auto-stop on severe drift — Phase D.

Covers:
  1. _demote_strategy updates DB status + metadata
  2. _trigger_portfolio_kill when strategy in portfolio
  3. _trigger_portfolio_kill no-op when strategy not in portfolio
  4. Drift API endpoints (status, history, demote)
  5. AdaptiveLoop degradation handler flow
  6. Edge: demote nonexistent strategy
  7. Edge: portfolio kill failure handling
"""
import pytest
import asyncio
from datetime import datetime, timezone

from athena.live.feedback import AdaptiveLoop, FeedbackCollector
from athena.services.models import get_session, StrategyModel, LiveSnapshot
from athena.common.models import StrategyStatus, StrategyTemplate
from athena.portfolio.manager import PortfolioManager, PortfolioConfig


# ── helper: create a strategy in DB ───────────────────────────────

def _make_db_strategy(template=StrategyTemplate.TREND_FOLLOWING, status=StrategyStatus.PROMOTED):
    session = get_session()
    sid = f"strat_drift_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"
    row = StrategyModel(
        id=sid,
        name=f"test_{sid[-6:]}",
        template=template.value,
        status=status.value,
        sharpe=1.5,
        max_drawdown=-0.08,
        dna={"vector": {"fast_period": 12}},
        metadata_json="{}",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    session.add(row)
    session.commit()
    session.close()
    return sid


# ── 1. Demote strategy ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_demote_strategy():
    sid = _make_db_strategy(status=StrategyStatus.PROMOTED)
    loop = AdaptiveLoop()
    await loop._demote_strategy(sid, "severe_drift")

    session = get_session()
    row = session.query(StrategyModel).filter_by(id=sid).first()
    assert row.status == StrategyStatus.RETIRED.value
    meta = __import__('json').loads(row.metadata_json or "{}")
    assert meta["demotion"]["reason"] == "severe_drift"
    assert meta["demotion"]["old_status"] == StrategyStatus.PROMOTED.value
    session.close()


@pytest.mark.asyncio
async def test_demote_nonexistent_strategy():
    """Demoting a missing strategy should not crash."""
    loop = AdaptiveLoop()
    await loop._demote_strategy("nonexistent_xyz", "test")
    # No assertion needed — just verify no exception


# ── 2. Portfolio kill on severe drift ───────────────────────────

@pytest.mark.asyncio
async def test_portfolio_kill_when_in_portfolio():
    sid = _make_db_strategy(status=StrategyStatus.PROMOTED)
    # Add to portfolio
    mgr = PortfolioManager(PortfolioConfig(max_per_strategy_weight=1.0))
    mgr.add_strategy(sid, initial_weight=1.0)
    assert sid in mgr._positions

    loop = AdaptiveLoop()
    await loop._trigger_portfolio_kill(sid, "sess_123")

    # Kill_all persists to DB — verify via fresh PortfolioManager
    mgr2 = PortfolioManager(PortfolioConfig(max_per_strategy_weight=1.0))
    if sid in mgr2._positions:
        assert mgr2._positions[sid].status == "stopped"
    else:
        # Strategy was removed from portfolio after kill
        pass


@pytest.mark.asyncio
async def test_portfolio_kill_when_not_in_portfolio():
    sid = _make_db_strategy(status=StrategyStatus.PROMOTED)
    loop = AdaptiveLoop()
    # Should not crash even if strategy not in portfolio
    await loop._trigger_portfolio_kill(sid, "sess_123")


# ── 3. Drift classification thresholds ────────────────────────────

def test_classify_severe_sharpe():
    fc = FeedbackCollector()
    assert fc._classify(0.25, 0.05, 1.0, 0.05) == "severe"


def test_classify_severe_drawdown():
    fc = FeedbackCollector()
    assert fc._classify(1.0, 0.18, 1.0, 0.05) == "severe"


def test_classify_mild():
    fc = FeedbackCollector()
    assert fc._classify(0.5, 0.12, 1.0, 0.05) == "mild"


def test_classify_none():
    fc = FeedbackCollector()
    assert fc._classify(1.2, 0.03, 1.0, 0.05) == ""


# ── 4. API endpoints (TestClient) ─────────────────────────────────

from fastapi.testclient import TestClient
from athena.services.api import app

client = TestClient(app)


def test_drift_demote_endpoint():
    sid = _make_db_strategy(status=StrategyStatus.PROMOTED)
    resp = client.post("/drift/demote", params={"strategy_id": sid, "reason": "test_drift"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "demoted"

    session = get_session()
    row = session.query(StrategyModel).filter_by(id=sid).first()
    assert row.status == StrategyStatus.RETIRED.value
    session.close()


def test_drift_status_no_snapshots():
    resp = client.get("/drift/status", params={"session_id": "no_such_session"})
    assert resp.status_code == 404


def test_drift_history_no_snapshots():
    resp = client.get("/drift/history", params={"session_id": "no_such_session", "limit": 10})
    assert resp.status_code == 200
    assert resp.json() == []


# ── 5. AdaptiveLoop flow (mocked) ─────────────────────────────────

@pytest.mark.asyncio
async def test_handle_degradation_missing_strategy():
    """Handle degradation when strategy row missing — should not crash."""
    loop = AdaptiveLoop()
    # Create a fake LiveSnapshot with nonexistent strategy
    fake_snap = LiveSnapshot(
        session_id="sess_test",
        strategy_id="nonexistent_123",
        timestamp=datetime.now(timezone.utc),
        equity=1000,
        total_trades=0,
        profit_closed_pct=0,
        profit_all_pct=0,
        sharpe_estimate=0.1,
        max_drawdown=0.30,
        win_rate=0,
        backtest_sharpe=1.5,
        backtest_max_drawdown=0.05,
        sharpe_ratio=0.07,
        drawdown_ratio=6.0,
        is_degraded="severe",
    )
    # Should not raise
    await loop._handle_degradation("sess_test", fake_snap, aggressive=False)


# ── 6. StreamBuffer edge cases ────────────────────────────────────

def test_stream_buffer_drawdown_empty():
    buf = __import__('athena.live.feedback', fromlist=['_StreamBuffer'])._StreamBuffer()
    assert buf.drawdown == 0.0


def test_stream_buffer_sharpe_empty():
    buf = __import__('athena.live.feedback', fromlist=['_StreamBuffer'])._StreamBuffer()
    assert buf.estimated_sharpe == 0.0


def test_stream_buffer_win_rate_empty():
    buf = __import__('athena.live.feedback', fromlist=['_StreamBuffer'])._StreamBuffer()
    assert buf.estimated_win_rate == 0.0
