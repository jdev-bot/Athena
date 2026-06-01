"""Composite scoring for strategies."""
import numpy as np
from typing import Dict, Any
from athena.common.models import PerformanceMetrics, ScoreResult
from athena.common.config import config


class Scorer:
    """Calculate composite score from performance metrics."""

    def __init__(self):
        self.sharpe_weight = config.SHARPE_WEIGHT
        self.sortino_weight = config.SORTINO_WEIGHT
        self.calmar_weight = config.CALMAR_WEIGHT
        self.win_rate_weight = config.WIN_RATE_WEIGHT
        self.profit_factor_weight = getattr(config, "PROFIT_FACTOR_WEIGHT", 0.0)
        self.total_return_weight = getattr(config, "TOTAL_RETURN_WEIGHT", 0.0)

    def normalize_sharpe(self, sharpe: float) -> float:
        """Normalize Sharpe ratio to 0-1 scale."""
        return min(max(sharpe / 2.0, 0.0), 1.0)

    def normalize_sortino(self, sortino: float) -> float:
        return min(max(sortino / 2.0, 0.0), 1.0)

    def normalize_calmar(self, calmar: float) -> float:
        return min(max(calmar / 3.0, 0.0), 1.0)

    def normalize_win_rate(self, win_rate: float) -> float:
        return min(max(win_rate, 0.0), 1.0)

    def normalize_profit_factor(self, pf: float) -> float:
        # PF 2.0 is great, 1.0 is breakeven, 0 is terrible
        return min(max((pf - 0.5) / 2.0, 0.0), 1.0)

    def normalize_total_return(self, ret: float) -> float:
        # 20% return normalized to 1.0; scale non-linear for small windows
        return min(max(ret / 0.20, 0.0), 1.0)

    def score(self, metrics: PerformanceMetrics) -> ScoreResult:
        if metrics.total_trades < 2:
            return ScoreResult(
                raw_score=0.0,
                sharpe_contrib=0.0,
                sortino_contrib=0.0,
                calmar_contrib=0.0,
                win_rate_contrib=0.0,
                verdict="demote",
            )

        sharpe_norm = self.normalize_sharpe(metrics.sharpe)
        sortino_norm = self.normalize_sortino(metrics.sortino)
        calmar_norm = self.normalize_calmar(metrics.calmar)
        win_rate_norm = self.normalize_win_rate(metrics.win_rate)
        pf_norm = self.normalize_profit_factor(metrics.profit_factor)
        ret_norm = self.normalize_total_return(metrics.total_return)

        raw_score = (
            sharpe_norm * self.sharpe_weight +
            sortino_norm * self.sortino_weight +
            calmar_norm * self.calmar_weight +
            win_rate_norm * self.win_rate_weight +
            pf_norm * self.profit_factor_weight +
            ret_norm * self.total_return_weight
        )

        # Penalize high drawdown
        if metrics.max_drawdown > 0.3:
            raw_score *= 0.5
        elif metrics.max_drawdown > 0.2:
            raw_score *= 0.8

        if raw_score >= config.PROMOTE_THRESHOLD:
            verdict = "promote"
        elif raw_score < config.DEMOTE_THRESHOLD:
            verdict = "demote"
        else:
            verdict = "hold"

        return ScoreResult(
            raw_score=raw_score,
            sharpe_contrib=sharpe_norm * self.sharpe_weight,
            sortino_contrib=sortino_norm * self.sortino_weight,
            calmar_contrib=calmar_norm * self.calmar_weight,
            win_rate_contrib=win_rate_norm * self.win_rate_weight,
            verdict=verdict,
        )
