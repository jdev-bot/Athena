"""Robustness gates — Walk-forward validation + Monte Carlo stress test.

Only strategies passing BOTH gates are eligible for promotion to live/paper trading.
"""
import math
import random
import statistics
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple, Optional

import numpy as np

from athena.core.freqtrade_wrapper import FreqtradeWrapper
from athena.common.models import (
    PerformanceMetrics, StrategyRecord, StrategyDNA, StrategyTemplate, WalkForwardResult, MonteCarloResult,
)


class WalkForwardValidator:
    """Walk-forward analysis: train on first 70%, test on last 30%.

    Verdict passes if out-of-sample Sharpe ≥ 50% of in-sample Sharpe
    and out-of-sample MaxDrawdown ≤ 1.5× in-sample MaxDrawdown.
    """

    SHARPE_DEGRADATION_THRESHOLD = 0.50   # OOS sharpe ≥ 50% in-sample
    DRAWDOWN_DEGRADATION_THRESHOLD = 1.50  # OOS DD ≤ 1.5× in-sample

    def __init__(self, train_ratio: float = 0.70):
        self.train_ratio = train_ratio

    def run(self, record: StrategyRecord, start_date: str, end_date: str) -> WalkForwardResult:
        """Run in-sample + out-of-sample backtests and compare."""
        code = self._compile(record)
        train_start, train_end, test_start, test_end = self._split_dates(start_date, end_date)

        # In-sample
        in_metrics = self._backtest(code, train_start, train_end)
        # Out-of-sample
        out_metrics = self._backtest(code, test_start, test_end)

        # Degradation ratios
        sharpe_deg = (out_metrics["sharpe"] / in_metrics["sharpe"]) if in_metrics["sharpe"] > 0 else 0.0
        dd_deg = (out_metrics["max_drawdown"] / max(in_metrics["max_drawdown"], 1e-6)) if in_metrics["max_drawdown"] else 0.0

        degradation_ratio = (sharpe_deg + (1 / max(dd_deg, 1e-6))) / 2
        is_robust = (
            sharpe_deg >= self.SHARPE_DEGRADATION_THRESHOLD
            and dd_deg <= self.DRAWDOWN_DEGRADATION_THRESHOLD
            and out_metrics["total_trades"] >= 5
        )

        return WalkForwardResult(
            in_sample_metrics=PerformanceMetrics(
                total_return=in_metrics.get("total_return", 0.0),
                sharpe=in_metrics.get("sharpe", 0.0),
                max_drawdown=in_metrics.get("max_drawdown", 0.0),
                total_trades=in_metrics.get("total_trades", 0),
                win_rate=in_metrics.get("win_rate", 0.0),
            ),
            out_sample_metrics=PerformanceMetrics(
                total_return=out_metrics.get("total_return", 0.0),
                sharpe=out_metrics.get("sharpe", 0.0),
                max_drawdown=out_metrics.get("max_drawdown", 0.0),
                total_trades=out_metrics.get("total_trades", 0),
                win_rate=out_metrics.get("win_rate", 0.0),
            ),
            degradation_ratio=degradation_ratio,
            is_robust=is_robust,
        )

    # ── helpers ──────────────────────────────────────────────────────
    def _compile(self, record: StrategyRecord) -> str:
        from athena.generator.templates import TEMPLATE_MAP
        from athena.generator.dna import DNAEncoder
        encoder = DNAEncoder()
        params = encoder.to_strategy_params(record.dna.vector, record.dna.template)
        params["class_name"] = "AthenaStrategy"
        params["template_name"] = record.dna.template.value
        params["timeframe"] = getattr(record, "timeframe", "1h")
        template = TEMPLATE_MAP.get(record.dna.template)
        return template.format(**params)

    def _backtest(self, code: str, start: str, end: str) -> Dict[str, Any]:
        wrapper = FreqtradeWrapper()
        return wrapper.run_backtest(code, start_date=start, end_date=end, exchange="binance", symbol="BTC-USD")

    def _split_dates(self, start_date: str, end_date: str) -> Tuple[str, str, str, str]:
        s = datetime.strptime(start_date, "%Y-%m-%d")
        e = datetime.strptime(end_date, "%Y-%m-%d")
        total_days = (e - s).days
        train_days = int(total_days * self.train_ratio)
        train_cut = s + timedelta(days=train_days)
        # Ensure at least 7 days for test
        if (e - train_cut).days < 7:
            train_cut = e - timedelta(days=max(7, int(total_days * 0.15)))
        return (
            start_date,
            train_cut.strftime("%Y-%m-%d"),
            train_cut.strftime("%Y-%m-%d"),
            end_date,
        )


class MonteCarloStressTest:
    """Shuffle daily returns and check if original performance is statistically
    superior to random permutations.

    Verdict passes if p-value ≤ 0.05 (original Sharpe > 95% of shuffles).
    """

    P_VALUE_THRESHOLD = 0.05
    SHUFFLES = 200

    def run(self, record: StrategyRecord, start_date: str, end_date: str) -> MonteCarloResult:
        """Run backtest, extract per-trade PnL, shuffle order, compute significance."""
        code = self._compile(record)
        wrapper = FreqtradeWrapper()
        bt = wrapper.run_backtest(
            code, start_date=start_date, end_date=end_date,
            exchange="binance", symbol="BTC-USD", return_trades=True,
        )
        original_sharpe = bt.get("sharpe", 0.0)
        trades = bt.get("trades", [])

        if len(trades) < 5:
            return MonteCarloResult(
                original_sharpe=original_sharpe,
                shuffled_sharpe_mean=0.0,
                shuffled_sharpe_std=0.0,
                p_value=1.0,
                is_significant=False,
            )

        # Extract per-trade returns (use profit_abs)
        trade_returns = np.array([t["profit"] for t in trades if t.get("profit") is not None], dtype=float)
        if len(trade_returns) < 3:
            return MonteCarloResult(
                original_sharpe=original_sharpe,
                shuffled_sharpe_mean=0.0,
                shuffled_sharpe_std=0.0,
                p_value=1.0,
                is_significant=False,
            )

        # Original Sharpe from trade returns (annualized roughly assuming ~1 trade/day)
        original_trade_sharpe = self._sharpe_from_returns(trade_returns)

        shuffled_sharpes = []
        for _ in range(self.SHUFFLES):
            np.random.shuffle(trade_returns)
            shuffled_sharpes.append(self._sharpe_from_returns(trade_returns))

        shuffled_mean = float(np.mean(shuffled_sharpes))
        shuffled_std = float(np.std(shuffled_sharpes))

        # P-value: probability that shuffled Sharpe >= original
        if shuffled_std > 0:
            z = (shuffled_mean - original_trade_sharpe) / shuffled_std
            p_value = 1.0 - self._normal_cdf(z)
        else:
            p_value = 1.0 if original_trade_sharpe < shuffled_mean else 0.0

        is_significant = p_value <= self.P_VALUE_THRESHOLD

        return MonteCarloResult(
            original_sharpe=original_trade_sharpe,
            shuffled_sharpe_mean=shuffled_mean,
            shuffled_sharpe_std=shuffled_std,
            p_value=p_value,
            is_significant=is_significant,
        )

    # ── helpers ──────────────────────────────────────────────────────
    def _compile(self, record: StrategyRecord) -> str:
        from athena.generator.templates import TEMPLATE_MAP
        from athena.generator.dna import DNAEncoder
        encoder = DNAEncoder()
        params = encoder.to_strategy_params(record.dna.vector, record.dna.template)
        params["class_name"] = "AthenaStrategy"
        params["template_name"] = record.dna.template.value
        params["timeframe"] = getattr(record, "timeframe", "1h")
        template = TEMPLATE_MAP.get(record.dna.template)
        return template.format(**params)

    @staticmethod
    def _sharpe_from_returns(returns: np.ndarray) -> float:
        mean_r = np.mean(returns)
        std_r = np.std(returns)
        if std_r == 0 or math.isnan(mean_r) or math.isnan(std_r):
            return 0.0
        return float(mean_r / std_r * math.sqrt(len(returns)))

    @staticmethod
    def _normal_cdf(x: float) -> float:
        """Approximate CDF for standard normal."""
        if x == 0:
            return 0.5
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        p = 0.3275911
        sign = 1 if x >= 0 else -1
        x = abs(x) / math.sqrt(2)
        t = 1.0 / (1.0 + p * x)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)
        return 0.5 * (1.0 + sign * y)
