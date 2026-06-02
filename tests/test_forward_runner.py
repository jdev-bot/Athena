"""Tests for multi-asset ForwardRunner."""
import pytest
from datetime import datetime, timezone

from athena.portfolio.manager import PortfolioManager
from athena.common.models import PortfolioConfig, StrategyRecord, StrategyTemplate, StrategyDNA
from athena.services.bridge import DryRunTrader
from athena.services.forward_runner import ForwardRunner


def test_multi_pair_no_data():
    """When no data exists for any pair the runner raises."""
    portfolio = PortfolioManager(PortfolioConfig(total_capital=10_000))
    trader = DryRunTrader(portfolio=portfolio)
    record = StrategyRecord(
        id="test",
        name="TestStrategy",
        template=StrategyTemplate.TREND_FOLLOWING,
        dna=StrategyDNA(template=StrategyTemplate.TREND_FOLLOWING, vector={"fast_period": 12, "slow_period": 26}),
        generation=0,
    )
    runner = ForwardRunner()
    with pytest.raises(RuntimeError, match="No candle data"):
        runner.run(record, trader, pairs=["FAKE/PAIR1", "FAKE/PAIR2"])


def test_run_single_pair_legacy_api():
    """Single pair (the default) also works."""
    portfolio = PortfolioManager(PortfolioConfig(total_capital=10_000))
    trader = DryRunTrader(portfolio=portfolio)
    record = StrategyRecord(
        id="test",
        name="TestStrategy",
        template=StrategyTemplate.TREND_FOLLOWING,
        dna=StrategyDNA(template=StrategyTemplate.TREND_FOLLOWING, vector={"fast_period": 12, "slow_period": 26}),
        generation=0,
    )
    runner = ForwardRunner()
    with pytest.raises(RuntimeError, match="No candle data"):
        runner.run(record, trader)
