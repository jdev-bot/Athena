"""Tests for feedback collector drift detection."""
import pytest
from athena.live.feedback import FeedbackCollector


class TestClassify:
    """Unit-test the drift classification logic in isolation."""

    collector = FeedbackCollector()

    def test_no_degradation(self):
        """Live metrics match backtest → no degradation."""
        assert self.collector._classify(1.5, 0.10, 1.5, 0.10) == ""

    def test_sharpe_mild(self):
        """Sharpe ratio 0.5 (50% of backtest) → mild."""
        assert self.collector._classify(0.5, 0.10, 1.0, 0.10) == "mild"

    def test_sharpe_severe(self):
        """Sharpe ratio 0.25 (25% of backtest) → severe."""
        assert self.collector._classify(0.25, 0.10, 1.0, 0.10) == "severe"

    def test_drawdown_mild(self):
        """Drawdown 2.5× backtest → mild."""
        assert self.collector._classify(1.0, 0.25, 1.0, 0.10) == "mild"

    def test_drawdown_severe(self):
        """Drawdown 4× backtest → severe."""
        assert self.collector._classify(1.0, 0.40, 1.0, 0.10) == "severe"

    def test_both_mild(self):
        """Both mild triggers fire → mild (not severe since neither crosses severe threshold)."""
        assert self.collector._classify(0.5, 0.25, 1.0, 0.10) == "mild"

    def test_both_severe(self):
        """Both severe triggers."""
        assert self.collector._classify(0.2, 0.50, 1.0, 0.10) == "severe"

    def test_zero_backtest_sharpe_positive_live(self):
        """Edge case: backtest Sharpe was 0 but live is positive."""
        assert self.collector._classify(0.5, 0.00, 0.0, 0.10) == ""

    def test_zero_backtest_sharpe_negative_live(self):
        """Edge case: backtest Sharpe was 0 and live is negative."""
        assert self.collector._classify(-0.1, 0.00, 0.0, 0.10) == ""

    def test_severe_drawdown_with_good_sharpe(self):
        """Good sharpe but DD severe → severe."""
        assert self.collector._classify(1.2, 0.35, 1.0, 0.10) == "severe"
