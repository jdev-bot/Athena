"""Tests for the dry-run forward-testing bridge."""
import pytest
from datetime import datetime, timezone

from athena.portfolio.manager import PortfolioManager
from athena.common.models import PortfolioConfig
from athena.services.bridge import DryRunTrader, SignalEntry
from athena.services.forward_runner import ForwardRunner


@pytest.fixture
def portfolio():
    return PortfolioManager(PortfolioConfig(total_capital=10_000))


@pytest.fixture
def trader(portfolio):
    return DryRunTrader(portfolio=portfolio)


class TestDryRunTrader:
    def test_initial_state(self, trader):
        assert not trader.is_killed()
        assert trader.to_dict()["open_positions"] == 0

    def test_on_signal_opens_position(self, trader):
        trader.on_signal(
            strategy_id="s1", symbol="BTC/USDT", signal="long",
            close=100.0,
        )
        info = trader.to_dict()
        assert info["open_positions"] == 1
        assert info["total_signals"] == 1

    def test_opposite_signal_closes_position(self, trader):
        trader.on_signal(strategy_id="s1", symbol="BTC/USDT", signal="long", close=100.0)
        trader.on_signal(strategy_id="s1", symbol="BTC/USDT", signal="sell", close=110.0)
        info = trader.to_dict()
        assert info["open_positions"] == 0
        assert info["total_closed"] == 1

    def test_pnl_calculation(self, trader):
        trader.on_signal(strategy_id="s1", symbol="BTC/USDT", signal="long", close=100.0)
        trader.on_signal(strategy_id="s1", symbol="BTC/USDT", signal="short", close=110.0)
        info = trader.to_dict()
        assert info["total_pnl"] - 0.10 < 1e-6

    def test_no_action_when_killed(self, trader):
        trader.on_signal(strategy_id="s1", symbol="BTC/USDT", signal="long", close=100.0)
        trader.on_signal(strategy_id="s1", symbol="BTC/USDT", signal="short", close=80.0)
        # portfolio daily loss exceeds 10% after short?
        # Actually short at 80 after long at 100 is +0.20 for short, not loss.
        # Force kill manually to test that on_signal is ignored.
        trader._killed = True
        prev = trader.to_dict()["total_signals"]
        trader.on_signal(strategy_id="s1", symbol="BTC/USDT", signal="long", close=90.0)
        assert trader.to_dict()["total_signals"] == prev

    def test_reset(self, trader):
        trader.on_signal(strategy_id="s1", symbol="BTC/USDT", signal="long", close=100.0)
        trader.reset()
        assert not trader.is_killed()
        assert trader.to_dict()["open_positions"] == 0


class TestForwardRunnerUnit:
    def test_strategy_has_signal_methods_none(self):
        class Foo:
            pass
        entry, exit_ = ForwardRunner._strategy_has_signal_methods(Foo())
        assert not entry and not exit_

    def test_entry_signal_from_df(self):
        import pandas as pd
        df = pd.DataFrame({"close": [100, 101], "enter_long": [0, 1]})
        sig = ForwardRunner._entry_signal_from_df(None, df, "BTC/USDT")
        assert sig == "long"

    def test_exit_signal_from_df(self):
        import pandas as pd
        df = pd.DataFrame({"close": [100, 101], "exit_long": [0, 1]})
        assert ForwardRunner._exit_signal_from_df(None, df, "BTC/USDT") is True

    def test_run_no_data_raises(self, trader):
        from athena.common.models import StrategyRecord, StrategyTemplate, StrategyDNA
        from athena.generator.dna import DNAEncoder
        record = StrategyRecord(
            id="test",
            name="TestStrategy",
            template=StrategyTemplate.TREND_FOLLOWING,
            dna=StrategyDNA(template="trend_following", genes=[0.5, 0.5, 0.5, 0.5]),
            generation=0,
        )
        runner = ForwardRunner()
        # No cached data => raises
        with pytest.raises(RuntimeError, match="No candle data"):
            runner.run(record, trader, pair="FAKE/PAIR", timeframe="1h")

