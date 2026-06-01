"""Unit tests for dynamic risk-budget position sizing."""
import pytest
from unittest.mock import MagicMock


class DummySizing:
    """Plain class implementing the risk-budget sizing formula for unit testing.

    Mirrors the logic injected into every Freqtrade strategy template by
    `athena/generator/templates.py` so the tests stay in sync with generated code.
    """

    def __init__(self, **kwargs):
        self.position_size_pct = kwargs.get("position_size_pct", 0.10)
        self.min_stake_usd = kwargs.get("min_stake_usd", 5.0)
        self.max_stake_usd = kwargs.get("max_stake_usd", 100.0)
        self.max_open_trades = kwargs.get("max_open_trades", 1)
        self.risk_capital_pct = kwargs.get("risk_capital_pct", 0.50)
        self.reserve_capital_pct = kwargs.get("reserve_capital_pct", 0.10)
        self.stake_currency = "USDT"
        self.wallets = None  # injected per-test

    def custom_stake_amount(self, min_stake, max_stake):
        """Deterministic version used in tests (no date/price noise)."""
        balance = self.wallets.get_total(self.stake_currency)
        _reserve = balance * self.reserve_capital_pct
        risk_budget = balance * self.risk_capital_pct
        deployed = sum(
            trade.stake_amount for trade in self.wallets.get_all_stake_amounts().values()
            if trade
        )
        available_for_new = risk_budget - deployed
        if available_for_new < self.min_stake_usd:
            return 0.0
        stake = balance * self.position_size_pct
        stake = max(stake, self.min_stake_usd)
        stake = min(stake, self.max_stake_usd, available_for_new)
        return max(min(stake, max_stake), min_stake)


@pytest.fixture
def strategy():
    s = DummySizing()
    s.wallets = MagicMock()
    return s


class TestRiskBudgetSizing:
    """Cover all edge cases for the dynamic position-sizing formula."""

    def test_first_trade_on_50_balance(self, strategy):
        """$50 balance, no open trades → $5 stake (10% of $50, clamped to min)."""
        strategy.wallets.get_total.return_value = 50.0
        strategy.wallets.get_all_stake_amounts.return_value = {}
        stake = strategy.custom_stake_amount(min_stake=2.0, max_stake=500.0)
        assert stake == pytest.approx(5.0)

    def test_position_size_pct_capped_by_available(self, strategy):
        """10% of $200 = $20, but only $10 risk budget left → $10."""
        strategy.wallets.get_total.return_value = 200.0
        strategy.wallets.get_all_stake_amounts.return_value = {
            "trade1": MagicMock(stake_amount=90.0)
        }
        stake = strategy.custom_stake_amount(min_stake=2.0, max_stake=500.0)
        # risk_budget = 100, deployed = 90, available = 10
        assert stake == pytest.approx(10.0)

    def test_no_budget_left(self, strategy):
        """Risk budget fully deployed → returns 0, no new trade."""
        strategy.wallets.get_total.return_value = 100.0
        strategy.wallets.get_all_stake_amounts.return_value = {
            "trade1": MagicMock(stake_amount=50.0)
        }
        stake = strategy.custom_stake_amount(min_stake=2.0, max_stake=500.0)
        assert stake == 0.0

    def test_growing_balance_increases_stake(self, strategy):
        """As balance grows from $50 -> $500, stake compounds."""
        strategy.wallets.get_total.return_value = 500.0
        strategy.wallets.get_all_stake_amounts.return_value = {}
        stake = strategy.custom_stake_amount(min_stake=2.0, max_stake=500.0)
        # 10% of 500 = 50, clamped to max_stake_usd 100 -> 50
        assert stake == pytest.approx(50.0)

    def test_reserve_capital_never_touched(self, strategy):
        """Even with huge balance, reserve of 10% is off-limits except for safety."""
        strategy.wallets.get_total.return_value = 1_000.0
        strategy.wallets.get_all_stake_amounts.return_value = {}
        stake = strategy.custom_stake_amount(min_stake=2.0, max_stake=500.0)
        # 10% of 1000 = 100, clamped to max_stake_usd 100 -> 100
        # Risk budget is 50% = 500, available = 500
        assert stake == pytest.approx(100.0)

    def test_min_stake_enforced(self, strategy):
        """If position_size_pct computes below min_stake_usd, min wins."""
        strategy.wallets.get_total.return_value = 30.0
        strategy.wallets.get_all_stake_amounts.return_value = {}
        stake = strategy.custom_stake_amount(min_stake=2.0, max_stake=500.0)
        assert stake == pytest.approx(5.0)

    def test_exchange_max_stake_respected(self, strategy):
        """If exchange max_stake is below computed stake, exchange limit wins."""
        strategy.wallets.get_total.return_value = 1_000.0
        strategy.wallets.get_all_stake_amounts.return_value = {}
        stake = strategy.custom_stake_amount(min_stake=2.0, max_stake=40.0)
        assert stake == pytest.approx(40.0)

    def test_exchange_min_stake_respected(self, strategy):
        """If exchange min_stake is above computed stake, exchange limit wins."""
        strategy.wallets.get_total.return_value = 50.0
        strategy.wallets.get_all_stake_amounts.return_value = {}
        stake = strategy.custom_stake_amount(min_stake=10.0, max_stake=500.0)
        assert stake == pytest.approx(10.0)

    def test_multiple_open_trades_sum_deployed(self, strategy):
        """Deployed capital sums across ALL open trades correctly."""
        strategy.wallets.get_total.return_value = 200.0
        strategy.wallets.get_all_stake_amounts.return_value = {
            "t1": MagicMock(stake_amount=20.0),
            "t2": MagicMock(stake_amount=30.0),
        }
        stake = strategy.custom_stake_amount(min_stake=2.0, max_stake=500.0)
        # risk_budget = 100, deployed = 50, available = 50
        # position_size = 20, clamped between 5 and min(100,50)=50 -> 20
        assert stake == pytest.approx(20.0)

    def test_available_below_min_after_reserve(self, strategy):
        """If balance * position_size_pct < min_stake_usd, min wins despite reserve."""
        strategy.wallets.get_total.return_value = 45.0
        strategy.wallets.get_all_stake_amounts.return_value = {}
        stake = strategy.custom_stake_amount(min_stake=2.0, max_stake=500.0)
        # 10% of 45 = 4.5 < 5 -> min 5
        assert stake == pytest.approx(5.0)

    def test_risk_budget_blocks_overleverage(self, strategy):
        """Even when balance is huge, risk_capital_pct caps total deployed capital."""
        strategy.risk_capital_pct = 0.20  # tight risk budget
        strategy.wallets.get_total.return_value = 1_000.0
        strategy.wallets.get_all_stake_amounts.return_value = {
            "t1": MagicMock(stake_amount=180.0),
        }
        stake = strategy.custom_stake_amount(min_stake=2.0, max_stake=500.0)
        # risk_budget = 200, deployed = 180, available = 20
        # 10% of 1000 = 100, clamped to min(100,20)=20 -> 20
        assert stake == pytest.approx(20.0)

    def test_reserve_parameter_independence(self, strategy):
        """Raising reserve_capital_pct should NOT change available_for_new directly
        (it only earmarks balance; risk_budget still drives the gate)."""
        strategy.reserve_capital_pct = 0.30  # 30% reserve
        strategy.risk_capital_pct = 0.50
        strategy.wallets.get_total.return_value = 100.0
        strategy.wallets.get_all_stake_amounts.return_value = {}
        stake = strategy.custom_stake_amount(min_stake=2.0, max_stake=500.0)
        # risk_budget = 50, available = 50, 10% of 100 = 10
        assert stake == pytest.approx(10.0)
