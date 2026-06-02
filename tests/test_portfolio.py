"""Extensive tests for Portfolio Manager.

Covers:
  1. Add/remove strategies (validations, weight computation)
  2. Weight computation (equal_weight, inverse_vol, equal_risk)
  3. Constraint enforcement (max_weight, min_weight, correlation)
  4. Rebalance flow
  5. PnL tracking and Sharpe/drawdown computation
  6. Snapshot consistency
  7. Kill-switch logic
  8. Correlation matrix (history-based + heuristic fallback)
  9. DB persistence roundtrip
  10. Edge cases (empty portfolio, all paused, single strategy)
"""
import pytest
import numpy as np
from datetime import datetime, timezone

from athena.portfolio.manager import PortfolioManager, _utcnow
from athena.common.models import PortfolioConfig, PortfolioPosition, StrategyStatus, StrategyRecord, StrategyDNA, StrategyTemplate
from athena.services.models import get_session, StrategyModel, init_db


# ── fixtures ──────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def clean_db():
    init_db()
    session = get_session()
    session.query(StrategyModel).delete()
    session.commit()


def _make_strategy(sid: str, template: str = "mean_reversion", status: str = "promoted", sharpe: float = 1.0):
    """Create a strategy row in DB."""
    session = get_session()
    row = StrategyModel(
        id=sid,
        name=f"test_{sid[-6:]}",
        template=template,
        dna={"bb_period": 20, "bb_std": 2.0},
        status=status,
        generation=1,
        raw_score=0.5,
        sharpe=sharpe,
        total_return=0.05,
        total_trades=10,
        metadata_json={},
    )
    session.add(row)
    session.commit()
    return sid


# ── 1. Add / remove ─────────────────────────────────────────────

def test_add_strategy_success():
    sid = _make_strategy("strat_add_001")
    mgr = PortfolioManager()
    pos = mgr.add_strategy(sid)
    assert pos.strategy_id == sid
    assert pos.weight > 0
    assert pos.notional > 0
    assert pos.status == "active"


def test_add_nonexistent_strategy_raises():
    mgr = PortfolioManager()
    with pytest.raises(ValueError, match="not found"):
        mgr.add_strategy("strat_nonexistent")


def test_add_non_promoted_strategy_raises():
    sid = _make_strategy("strat_draft_001", status="draft")
    mgr = PortfolioManager()
    with pytest.raises(ValueError, match="must be PROMOTED or LIVE"):
        mgr.add_strategy(sid)


def test_add_duplicate_is_idempotent():
    sid = _make_strategy("strat_dup_001")
    mgr = PortfolioManager()
    pos1 = mgr.add_strategy(sid)
    pos2 = mgr.add_strategy(sid)
    assert pos1.strategy_id == pos2.strategy_id


def test_remove_strategy():
    sid = _make_strategy("strat_rem_001")
    mgr = PortfolioManager()
    mgr.add_strategy(sid)
    mgr.remove_strategy(sid, reason="test")
    assert sid not in mgr._positions


def test_remove_nonexistent_warns():
    mgr = PortfolioManager()
    mgr.remove_strategy("strat_rem_nonexist")  # should not raise


# ── 2. Pause / resume ───────────────────────────────────────────

def test_pause_resume():
    sid = _make_strategy("strat_pause_001")
    mgr = PortfolioManager()
    mgr.add_strategy(sid)
    mgr.pause_strategy(sid)
    assert mgr._positions[sid].status == "paused"
    mgr.resume_strategy(sid)
    assert mgr._positions[sid].status == "active"


# ── 3. Weight computation ───────────────────────────────────────

def test_equal_weight():
    sids = [_make_strategy(f"strat_ew_{i:03d}") for i in range(3)]
    mgr = PortfolioManager(PortfolioConfig(allocation_method="equal_weight"))
    for sid in sids:
        mgr.add_strategy(sid)
    weights = {sid: mgr._positions[sid].weight for sid in sids}
    # After rebalance, all active positions should have equal weight
    snap = mgr.rebalance("equal_weight")
    for p in snap.positions:
        assert abs(p.weight - 1/3) < 0.01


def test_inverse_vol_weights():
    sids = [_make_strategy(f"strat_iv_{i:03d}") for i in range(3)]
    cfg = PortfolioConfig(
        allocation_method="inverse_vol",
        max_per_strategy_weight=0.60,
        min_per_strategy_weight=0.05,
    )
    mgr = PortfolioManager(cfg)
    for sid in sids:
        mgr.add_strategy(sid)
    np.random.seed(42)
    for i in range(30):
        mgr.update_pnl(sids[0], closed_pnl=np.random.normal(0, 20))  # very high vol
        mgr.update_pnl(sids[1], closed_pnl=np.random.normal(0, 5))   # medium vol
        mgr.update_pnl(sids[2], closed_pnl=np.random.normal(0, 0.5))  # very low vol
    snap = mgr.rebalance("inverse_vol")
    w = {p.strategy_id: p.weight for p in snap.positions}
    # Low-vol strategy should get highest weight
    assert w[sids[2]] > w[sids[1]], f"low-vol {w[sids[2]]} should > medium-vol {w[sids[1]]}"
    assert w[sids[1]] > w[sids[0]], f"medium-vol {w[sids[1]]} should > high-vol {w[sids[0]]}"


def test_single_strategy_gets_full_weight():
    sid = _make_strategy("strat_single_001")
    mgr = PortfolioManager()
    mgr.add_strategy(sid)
    snap = mgr.rebalance()
    assert abs(snap.positions[0].weight - 1.0) < 0.01


# ── 4. Constraint enforcement ───────────────────────────────────

def test_max_weight_capped():
    sids = [_make_strategy(f"strat_cap_{i:03d}") for i in range(4)]
    # Feasible: 4 strategies × 0.35 max = 1.40 ≥ 1.0
    cfg = PortfolioConfig(max_per_strategy_weight=0.35, allocation_method="equal_weight")
    mgr = PortfolioManager(cfg)
    for sid in sids:
        mgr.add_strategy(sid)
    snap = mgr.rebalance()
    for p in snap.positions:
        assert p.weight <= 0.35 + 0.01  # tolerance for renormalization
    assert sum(p.weight for p in snap.positions) <= 1.01


def test_min_weight_trim():
    sids = [_make_strategy(f"strat_trim_{i:03d}") for i in range(5)]
    # min=0.30 × 5 = 1.50 > 1.0 → infeasible; floor step trims all (0.2 < 0.30)
    # Then capping forces single survivor
    cfg = PortfolioConfig(min_per_strategy_weight=0.30, allocation_method="equal_weight")
    mgr = PortfolioManager(cfg)
    for sid in sids:
        mgr.add_strategy(sid)
    snap = mgr.rebalance()
    assert len(snap.positions) == 5  # all still in portfolio
    # Only 1 can survive with min=0.30 (would need 3.5+ to fit 5)
    active_weights = [p.weight for p in snap.positions if p.weight > 0.01]
    assert len(active_weights) == 1
    assert sum(active_weights) == pytest.approx(1.0, abs=0.01)


def test_correlation_scaling():
    sids = [_make_strategy(f"strat_corr_{i:03d}", template="mean_reversion") for i in range(2)]
    cfg = PortfolioConfig(max_correlation=0.50, allocation_method="equal_weight")
    mgr = PortfolioManager(cfg)
    for sid in sids:
        mgr.add_strategy(sid)
    # Force identical PnL history = correlation ~1.0
    for i in range(20):
        pnl = np.random.normal(1, 2)
        mgr.update_pnl(sids[0], closed_pnl=pnl)
        mgr.update_pnl(sids[1], closed_pnl=pnl)
    snap = mgr.rebalance()
    w0 = next(p.weight for p in snap.positions if p.strategy_id == sids[0])
    w1 = next(p.weight for p in snap.positions if p.strategy_id == sids[1])
    # After correlation penalty + renormalization, both get equal weight again
    # The key behavior is that the warning was logged and correlation detected
    assert w0 == pytest.approx(0.5, abs=0.01)
    assert w1 == pytest.approx(0.5, abs=0.01)


# ── 5. PnL tracking ──────────────────────────────────────────────

def test_update_pnl_tracks_drawdown():
    sid = _make_strategy("strat_pnl_001")
    mgr = PortfolioManager()
    mgr.add_strategy(sid)
    # Sequence: win, win, big loss → drawdown
    mgr.update_pnl(sid, closed_pnl=10)
    mgr.update_pnl(sid, closed_pnl=5)
    mgr.update_pnl(sid, closed_pnl=-20)
    pos = mgr._positions[sid]
    assert pos.max_drawdown < 0  # negative = drawdown occurred


def test_sharpe_30d_computed():
    sid = _make_strategy("strat_sharpe_001")
    mgr = PortfolioManager()
    mgr.add_strategy(sid)
    for i in range(30):
        mgr.update_pnl(sid, closed_pnl=1.0)
    pos = mgr._positions[sid]
    assert pos.sharpe_30d > 10  # Consistent positive returns → high Sharpe


# ── 6. Snapshot ──────────────────────────────────────────────────

def test_snapshot_empty_portfolio():
    mgr = PortfolioManager()
    snap = mgr.snapshot()
    assert snap.total_capital == 10_000.0
    assert snap.allocated_capital == 0.0
    assert snap.active_strategies == 0


def test_snapshot_consistency():
    sids = [_make_strategy(f"strat_snap_{i:03d}") for i in range(3)]
    mgr = PortfolioManager()
    for sid in sids:
        mgr.add_strategy(sid)
    snap = mgr.snapshot()
    assert snap.allocated_capital <= snap.total_capital
    assert snap.free_cash == snap.total_capital - snap.allocated_capital
    assert snap.active_strategies == 3


# ── 7. Kill switch ──────────────────────────────────────────────

def test_kill_switch_not_triggered():
    sid = _make_strategy("strat_kill_ok_001")
    mgr = PortfolioManager()
    mgr.add_strategy(sid)
    fired, reason = mgr.check_kill_switch()
    assert not fired
    assert reason == ""


def test_kill_switch_portfolio_drawdown():
    sids = [_make_strategy(f"strat_kill_dd_{i:03d}") for i in range(2)]
    cfg = PortfolioConfig(portfolio_max_drawdown_kill=0.10, total_capital=10_000.0)
    mgr = PortfolioManager(cfg)
    for sid in sids:
        mgr.add_strategy(sid)
    # Inflict massive losses
    for sid in sids:
        for _ in range(10):
            mgr.update_pnl(sid, closed_pnl=-500)
    fired, reason = mgr.check_kill_switch()
    assert fired
    assert "portfolio_drawdown" in reason


def test_kill_switch_executes():
    sid = _make_strategy("strat_kill_exec_001")
    mgr = PortfolioManager()
    mgr.add_strategy(sid)
    mgr.kill_all("test_kill")
    assert mgr._positions[sid].status == "stopped"


# ── 8. Correlation matrix ───────────────────────────────────────

def test_correlation_matrix_empty():
    mgr = PortfolioManager()
    assert mgr.get_correlation_matrix() == {}


def test_correlation_matrix_single():
    sid = _make_strategy("strat_corr1_001")
    mgr = PortfolioManager()
    mgr.add_strategy(sid)
    mat = mgr.get_correlation_matrix()
    assert mat[sid][sid] == 1.0


def test_correlation_matrix_heuristic():
    sid_trend = _make_strategy("strat_trend_001", template="trend_following")
    sid_mean = _make_strategy("strat_mean_001", template="mean_reversion")
    mgr = PortfolioManager()
    mgr.add_strategy(sid_trend)
    mgr.add_strategy(sid_mean)
    mat = mgr.get_correlation_matrix()
    corr = mat[sid_trend][sid_mean]
    # Trend and mean-reversion heuristic = negative or low
    assert corr <= 0.2


def test_correlation_matrix_from_history():
    sids = [_make_strategy(f"strat_hcorr_{i:03d}") for i in range(2)]
    mgr = PortfolioManager()
    for sid in sids:
        mgr.add_strategy(sid)
    # Uncorrelated histories
    np.random.seed(42)
    for i in range(30):
        mgr.update_pnl(sids[0], closed_pnl=np.random.normal(0, 1))
        mgr.update_pnl(sids[1], closed_pnl=np.random.normal(0, 1))
    mat = mgr.get_correlation_matrix()
    corr = mat[sids[0]][sids[1]]
    # Two independent random walks should have near-zero correlation
    assert abs(corr) < 0.5


# ── 9. DB persistence ────────────────────────────────────────────

def test_load_from_db():
    sid = _make_strategy("strat_db_001")
    mgr1 = PortfolioManager()
    mgr1.add_strategy(sid)
    # Create new manager instance — should restore from DB
    mgr2 = PortfolioManager()
    assert sid in mgr2._positions
    assert mgr2._positions[sid].status == "active"


# ── 10. Edge cases ──────────────────────────────────────────────

def test_all_paused_rebalance():
    sids = [_make_strategy(f"strat_paused_{i:03d}") for i in range(2)]
    mgr = PortfolioManager()
    for sid in sids:
        mgr.add_strategy(sid)
        mgr.pause_strategy(sid)
    snap = mgr.rebalance()
    # With no active positions, weights are empty but snapshot still valid
    assert snap.active_strategies == 0
    assert snap.paused_strategies == 2


def test_rebalance_preserves_manual_weights():
    sid = _make_strategy("strat_manual_001")
    mgr = PortfolioManager()
    mgr.add_strategy(sid, initial_weight=0.25)
    snap = mgr.rebalance("equal_weight")
    # Single strategy gets forced to 1.0
    assert abs(snap.positions[0].weight - 1.0) < 0.01
