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
        
    def normalize_sharpe(self, sharpe: float) -> float:
        """Normalize Sharpe ratio to 0-1 scale."""
        # Sharpe 2.0 is excellent, 0 is poor, negative is bad
        return min(max(sharpe / 2.0, 0.0), 1.0)
    
    def normalize_sortino(self, sortino: float) -> float:
        """Normalize Sortino ratio to 0-1 scale."""
        return min(max(sortino / 2.0, 0.0), 1.0)
    
    def normalize_calmar(self, calmar: float) -> float:
        """Normalize Calmar ratio to 0-1 scale."""
        return min(max(calmar / 3.0, 0.0), 1.0)
    
    def normalize_win_rate(self, win_rate: float) -> float:
        """Normalize win rate to 0-1 scale."""
        return min(max(win_rate, 0.0), 1.0)
    
    def score(self, metrics: PerformanceMetrics) -> ScoreResult:
        """Calculate composite score."""
        if metrics.total_trades < 5:
            # Not enough trades for reliable scoring
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
        
        # Weighted composite
        raw_score = (
            sharpe_norm * self.sharpe_weight +
            sortino_norm * self.sortino_weight +
            calmar_norm * self.calmar_weight +
            win_rate_norm * self.win_rate_weight
        )
        
        # Penalize high drawdown
        if metrics.max_drawdown > 0.3:
            raw_score *= 0.5
        elif metrics.max_drawdown > 0.2:
            raw_score *= 0.8
        
        # Verdict
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
