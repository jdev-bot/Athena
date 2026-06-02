"""Tests for multi-asset per-symbol regime detection."""
import numpy as np
import pandas as pd
import pytest

from athena.market.regime import (
    Regime, detect_regime, detect_regimes, get_dominant_regime,
    get_suitable_templates_for_basket,
)


def _make_ohlcv(n: int = 200, trend: str = "up") -> pd.DataFrame:
    """Generate a synthetic OHLCV DataFrame for a single pair."""
    ts = pd.date_range("2026-01-01", periods=n, freq="h")
    if trend == "up":
        close = 100 + np.cumsum(np.random.randn(n) * 0.5 + 0.3)
    elif trend == "down":
        close = 100 - np.cumsum(np.random.randn(n) * 0.5 + 0.3)
    elif trend == "range":
        close = 100 + np.sin(np.linspace(0, 8 * np.pi, n)) * 3 + np.random.randn(n) * 0.2
    else:  # volatile
        close = 100 + np.cumsum(np.random.randn(n) * 2.0)
    close = np.maximum(close, 1.0)
    high = close + np.random.uniform(0.1, 1.0, n)
    low = close - np.random.uniform(0.1, 1.0, n)
    open_ = close + np.random.randn(n) * 0.1
    volume = np.random.uniform(10, 1000, n)
    return pd.DataFrame(
        {"date": ts, "open": open_, "high": high, "low": low, "close": close, "volume": volume}
    )


class TestDetectRegimes:

    def test_detect_regimes_multiple_pairs(self):
        dfs = {
            "BTC/USDT": _make_ohlcv(trend="up"),
            "ETH/USDT": _make_ohlcv(trend="down"),
        }
        regimes = detect_regimes(dfs)
        assert isinstance(regimes, dict)
        assert set(regimes.keys()) == {"BTC/USDT", "ETH/USDT"}
        # Values are Regime enum members
        assert all(isinstance(v, Regime) for v in regimes.values())

    def test_detect_regimes_empty(self):
        regimes = detect_regimes({})
        assert regimes == {}

    def test_detect_regimes_single_pair(self):
        dfs = {"BTC/USDT": _make_ohlcv(trend="up")}
        regimes = detect_regimes(dfs)
        assert list(regimes.keys()) == ["BTC/USDT"]

    def test_detect_regimes_with_ndarray(self):
        arr = np.column_stack([
            np.arange(200),
            np.ones(200) * 100,
            np.ones(200) * 101,
            np.ones(200) * 99,
            np.ones(200) * 100,
            np.ones(200) * 500,
        ])
        regimes = detect_regimes({"X": arr})
        assert isinstance(regimes, dict)


class TestGetDominantRegime:

    def test_dominant_up(self):
        regimes = {"BTC": Regime.TRENDING_UP, "ETH": Regime.TRENDING_UP, "SOL": Regime.RANGING}
        assert get_dominant_regime(regimes) == Regime.TRENDING_UP

    def test_dominant_down(self):
        regimes = {"BTC": Regime.TRENDING_DOWN, "ETH": Regime.TRENDING_DOWN, "SOL": Regime.VOLATILE}
        assert get_dominant_regime(regimes) == Regime.TRENDING_DOWN

    def test_dominant_tie(self):
        regimes = {"BTC": Regime.TRENDING_UP, "ETH": Regime.RANGING}
        result = get_dominant_regime(regimes)
        assert result in (Regime.TRENDING_UP, Regime.RANGING)

    def test_dominant_empty(self):
        assert get_dominant_regime({}) == Regime.UNKNOWN


class TestGetSuitableTemplatesForBasket:

    def test_returns_list(self):
        regimes = {"BTC": Regime.TRENDING_UP, "ETH": Regime.TRENDING_DOWN}
        tpls = get_suitable_templates_for_basket(regimes)
        assert isinstance(tpls, list)
        assert "trend_following" in tpls

    def test_empty_regimes(self):
        tpls = get_suitable_templates_for_basket({})
        assert tpls == []

    def test_volatile_basket(self):
        regimes = {"BTC": Regime.VOLATILE, "ETH": Regime.VOLATILE}
        tpls = get_suitable_templates_for_basket(regimes)
        assert "breakout" in tpls
        assert "volatility" in tpls
