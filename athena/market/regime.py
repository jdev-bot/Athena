"""Market regime detection — classify recent price action into regimes.

Regimes:
  • TRENDING_UP    — ADX > 25, price above midpoint, directional bias up
  • TRENDING_DOWN  — ADX > 25, price below midpoint, directional bias down
  • RANGING        — ADX < 20, low volatility, mean-reversion friendly
  • VOLATILE       — ATR percentile > 80, high volatility, breakout friendly
  • UNKNOWN        — insufficient data or conflicting signals

Templates suit regimes:
  • trend_following → TRENDING_UP / TRENDING_DOWN
  • mean_reversion  → RANGING
  • breakout        → VOLATILE
  • scalping        → all regimes (short timeframe)
  • swing           → all regimes (medium timeframe)
"""
import logging
from typing import List, Optional, Dict
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class Regime(Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"


# Template → suitable regimes
TEMPLATE_SUITABILITY: Dict[str, List[Regime]] = {
    "trend_following": [Regime.TRENDING_UP, Regime.TRENDING_DOWN],
    "mean_reversion": [Regime.RANGING],
    "breakout": [Regime.VOLATILE, Regime.TRENDING_UP, Regime.TRENDING_DOWN],
    "scalping": list(Regime),  # all
    "swing": list(Regime),      # all
}


def detect_regime(
    candles: np.ndarray,
    atr_period: int = 14,
    adx_period: int = 14,
    atr_lookback: int = 30,
    adx_trend_threshold: float = 25.0,
    adx_range_threshold: float = 20.0,
    atr_volatile_percentile: float = 80.0,
) -> Regime:
    """Detect market regime from OHLCV candles.

    candles: np.ndarray with shape (N, 6) — [timestamp, open, high, low, close, volume]
    """
    if len(candles) < max(atr_period, adx_period) * 2:
        return Regime.UNKNOWN

    closes = candles[:, 4].astype(float)
    highs = candles[:, 2].astype(float)
    lows = candles[:, 3].astype(float)

    # Compute ATR
    atr = _atr(highs, lows, closes, atr_period)
    if len(atr) == 0:
        return Regime.UNKNOWN

    # Compute ADX
    adx = _adx(highs, lows, closes, adx_period)
    if len(adx) == 0:
        return Regime.UNKNOWN

    latest_adx = adx[-1]
    latest_atr = atr[-1]
    latest_close = closes[-1]

    # Volatility check: ATR percentile over lookback
    atr_hist = atr[-atr_lookback:]
    atr_pct = np.percentile(atr_hist, atr_volatile_percentile)
    is_volatile = latest_atr >= atr_pct and latest_atr > np.mean(atr_hist) * 1.2

    # Range midpoint for trend direction
    recent_highs = highs[-atr_lookback:]
    recent_lows = lows[-atr_lookback:]
    mid = (np.max(recent_highs) + np.min(recent_lows)) / 2.0

    # Directional bias from recent slope
    slope = _linear_slope(closes[-10:])

    # Classification
    if is_volatile and latest_adx > adx_trend_threshold:
        # Volatile + trending = breakout-friendly, but classify by direction
        if slope > 0:
            return Regime.TRENDING_UP
        else:
            return Regime.TRENDING_DOWN

    if is_volatile:
        return Regime.VOLATILE

    if latest_adx >= adx_trend_threshold:
        if slope > 0 or latest_close > mid:
            return Regime.TRENDING_UP
        else:
            return Regime.TRENDING_DOWN

    if latest_adx <= adx_range_threshold:
        return Regime.RANGING

    # ADX between thresholds — ambiguous
    if slope > 0.5:
        return Regime.TRENDING_UP
    if slope < -0.5:
        return Regime.TRENDING_DOWN

    return Regime.UNKNOWN


def get_suitable_templates(regime: Regime) -> List[str]:
    """Return templates suitable for a given regime."""
    suitable = []
    for template, regimes in TEMPLATE_SUITABILITY.items():
        if regime in regimes:
            suitable.append(template)
    return suitable


def _true_range(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> np.ndarray:
    """Compute true range series."""
    prev_close = np.roll(closes, 1)
    prev_close[0] = closes[0]
    tr1 = highs - lows
    tr2 = np.abs(highs - prev_close)
    tr3 = np.abs(lows - prev_close)
    return np.maximum(np.maximum(tr1, tr2), tr3)


def _atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int) -> np.ndarray:
    """Average True Range."""
    tr = _true_range(highs, lows, closes)
    atr = np.zeros_like(tr)
    atr[:period] = np.mean(tr[:period])
    for i in range(period, len(tr)):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def _adx(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int) -> np.ndarray:
    """Average Directional Index."""
    tr = _true_range(highs, lows, closes)

    plus_dm = np.zeros_like(highs)
    minus_dm = np.zeros_like(highs)

    high_diff = np.diff(highs, prepend=highs[0])
    low_diff = np.diff(lows, prepend=lows[0])

    for i in range(1, len(highs)):
        up = highs[i] - highs[i - 1]
        down = lows[i - 1] - lows[i]
        if up > down and up > 0:
            plus_dm[i] = up
        if down > up and down > 0:
            minus_dm[i] = down

    atr = _atr(highs, lows, closes, period)
    # Avoid division by zero
    atr_safe = np.where(atr == 0, 1e-6, atr)

    plus_di = 100.0 * _wilder_smooth(plus_dm, period) / atr_safe
    minus_di = 100.0 * _wilder_smooth(minus_dm, period) / atr_safe

    dx = 100.0 * np.abs(plus_di - minus_di) / np.abs(plus_di + minus_di + 1e-6)
    adx = _wilder_smooth(dx, period)
    return adx


def _wilder_smooth(series: np.ndarray, period: int) -> np.ndarray:
    """Wilder's smoothing (RMA)."""
    result = np.zeros_like(series)
    result[:period] = np.mean(series[:period])
    for i in range(period, len(series)):
        result[i] = (result[i - 1] * (period - 1) + series[i]) / period
    return result


def _linear_slope(prices: np.ndarray) -> float:
    """Slope of linear regression over prices."""
    if len(prices) < 2:
        return 0.0
    x = np.arange(len(prices))
    return float(np.polyfit(x, prices, 1)[0])
