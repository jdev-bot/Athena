"""Portfolio Manager — multi-strategy capital allocation, risk budgeting, and rebalancing.

Manages a basket of PROMOTED strategies as a single portfolio with:
  • Inverse-volatility, equal-risk, or equal-weight allocation
  • Per-strategy and portfolio-level drawdown tracking
  • Kill-switch at portfolio max drawdown
  • Correlation-aware concentration limits
  • Automatic rebalancing
"""
import uuid
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np

from athena.common.models import (
    PortfolioConfig, PortfolioPosition, PortfolioSnapshot,
    StrategyStatus, StrategyRecord, PerformanceMetrics,
)
from athena.common.config import config
from athena.services.models import get_session, StrategyModel
from athena.core.engine import AthenaEngine

logger = logging.getLogger(__name__)


class PortfolioManager:
    """Manages capital allocation across multiple live/paper strategies."""

    def __init__(self, cfg: Optional[PortfolioConfig] = None):
        self.cfg = cfg or PortfolioConfig()
        self._positions: Dict[str, PortfolioPosition] = {}
        self._pnl_history: Dict[str, List[float]] = {}   # daily closed PnL per strategy
        self._equity_peak: float = self.cfg.total_capital
        self._last_rebalance: Optional[datetime] = None
        self._daily_pnl: List[float] = []
        self._live: bool = True
        self._load_from_db()

    # ── public lifecycle ───────────────────────────────────────────────

    def add_strategy(self, strategy_id: str, initial_weight: Optional[float] = None) -> PortfolioPosition:
        """Add a PROMOTED strategy to the portfolio, re-scaling existing positions."""
        session = get_session()
        row = session.query(StrategyModel).filter_by(id=strategy_id).first()
        if not row:
            raise ValueError(f"Strategy {strategy_id} not found")
        if row.status != StrategyStatus.PROMOTED.value and row.status != StrategyStatus.LIVE.value:
            raise ValueError(f"Strategy {strategy_id} status={row.status}; must be PROMOTED or LIVE")

        if strategy_id in self._positions:
            logger.warning(f"Strategy {strategy_id} already in portfolio; skipping")
            return self._positions[strategy_id]

        n_existing = len(self._positions)
        if initial_weight is None:
            weight = 1.0 / (n_existing + 1)
        else:
            weight = initial_weight

        # Re-scale existing positions so they still sum to (1 - new_weight)
        if n_existing > 0:
            scale = (1.0 - weight) / sum(p.weight for p in self._positions.values())
            for pos in self._positions.values():
                pos.weight *= scale
                pos.notional = pos.weight * self.cfg.total_capital
                self._persist_position(pos)

        weight = min(weight, self.cfg.max_per_strategy_weight)

        pos = PortfolioPosition(
            strategy_id=strategy_id,
            weight=weight,
            notional=weight * self.cfg.total_capital,
            status="active",
        )
        self._positions[strategy_id] = pos
        self._pnl_history[strategy_id] = []
        self._persist_position(pos)
        logger.info(f"Added {strategy_id} to portfolio weight={weight:.2%} notional={pos.notional:.2f}")
        return pos

    def remove_strategy(self, strategy_id: str, reason: str = "removed") -> None:
        """Remove a strategy from the portfolio, freeing its capital."""
        pos = self._positions.pop(strategy_id, None)
        if not pos:
            logger.warning(f"Strategy {strategy_id} not in portfolio")
            return
        self._pnl_history.pop(strategy_id, None)
        session = get_session()
        row = session.query(StrategyModel).filter_by(id=strategy_id).first()
        if row:
            meta_raw = row.metadata_json or "{}"
            if isinstance(meta_raw, str):
                try:
                    meta = json.loads(meta_raw)
                except json.JSONDecodeError:
                    meta = {}
            else:
                meta = meta_raw or {}
            meta["portfolio_removed_at"] = _utcnow().isoformat()
            meta["portfolio_remove_reason"] = reason
            row.metadata_json = json.dumps(meta)
            session.commit()
        logger.info(f"Removed {strategy_id} from portfolio ({reason})")

    def pause_strategy(self, strategy_id: str) -> None:
        """Pause a strategy — keep capital allocated but stop trading."""
        pos = self._positions.get(strategy_id)
        if pos:
            pos.status = "paused"
            self._persist_position(pos)
            logger.info(f"Paused {strategy_id}")

    def resume_strategy(self, strategy_id: str) -> None:
        """Resume a paused strategy."""
        pos = self._positions.get(strategy_id)
        if pos:
            pos.status = "active"
            self._persist_position(pos)
            logger.info(f"Resumed {strategy_id}")

    def update_pnl(self, strategy_id: str, closed_pnl: float, open_pnl: float = 0.0) -> None:
        """Update realized and unrealized PnL for a strategy."""
        pos = self._positions.get(strategy_id)
        if not pos:
            return
        pos.closed_pnl = closed_pnl
        pos.open_pnl = open_pnl
        # Track daily history for Sharpe / drawdown computation
        hist = self._pnl_history.setdefault(strategy_id, [])
        hist.append(closed_pnl)
        if len(hist) > 90:
            hist.pop(0)
        # Update position drawdown
        cumulative = np.cumsum(hist)
        if len(cumulative) > 0:
            peak = np.maximum.accumulate(cumulative)
            trough = cumulative - peak
            pos.max_drawdown = float(np.min(trough) / max(abs(peak[0]), 1e-6)) if len(peak) > 0 else 0.0
            # Rolling 30-day Sharpe
            if len(hist) >= 7:
                recent = np.array(hist[-30:])
                pos.sharpe_30d = float(np.mean(recent) / max(np.std(recent), 1e-6) * np.sqrt(30))
        self._persist_position(pos)

    def rebalance(self, method: Optional[str] = None) -> PortfolioSnapshot:
        """Recompute weights and reallocate capital."""
        method = method or self.cfg.allocation_method
        if not self._positions:
            return self.snapshot()

        # Compute raw weights
        raw_weights = self._compute_weights(method)

        # Apply constraints
        constrained = self._apply_constraints(raw_weights)

        # Normalize to sum=1
        total = sum(constrained.values())
        if total > 0:
            constrained = {sid: w / total for sid, w in constrained.items()}
        else:
            constrained = {sid: 1.0 / len(constrained) for sid in constrained}

        # Update positions: set weight=0 for anything not in constrained
        for sid in list(self._positions.keys()):
            if sid not in constrained:
                self._positions[sid].weight = 0.0
                self._positions[sid].notional = 0.0
                self._positions[sid].last_rebalanced_at = _utcnow()
                self._persist_position(self._positions[sid])

        # Update positions in constrained
        for sid, w in constrained.items():
            pos = self._positions[sid]
            pos.weight = w
            pos.notional = w * self.cfg.total_capital
            pos.last_rebalanced_at = _utcnow()
            self._persist_position(pos)

        self._last_rebalance = _utcnow()
        logger.info(f"Rebalanced portfolio method={method} positions={len(self._positions)} active_in_constraints={len(constrained)}")
        return self.snapshot()

    def snapshot(self) -> PortfolioSnapshot:
        """Current portfolio state."""
        total_closed = sum(p.closed_pnl for p in self._positions.values())
        total_open = sum(p.open_pnl for p in self._positions.values())
        allocated = sum(p.notional for p in self._positions.values() if p.weight > 0)
        portfolio_equity = self.cfg.total_capital + total_closed + total_open

        # Portfolio-level drawdown
        if portfolio_equity > self._equity_peak:
            self._equity_peak = portfolio_equity
        portfolio_dd = (portfolio_equity - self._equity_peak) / self._equity_peak if self._equity_peak > 0 else 0.0

        # Portfolio Sharpe (from combined daily PnLs)
        combined = self._combined_daily_pnl()
        portfolio_sharpe = 0.0
        if len(combined) >= 7:
            portfolio_sharpe = float(np.mean(combined) / max(np.std(combined), 1e-6) * np.sqrt(365))

        active = sum(1 for p in self._positions.values() if p.status == "active")
        paused = sum(1 for p in self._positions.values() if p.status == "paused")
        killed = sum(1 for p in self._positions.values() if p.status == "stopped")

        return PortfolioSnapshot(
            timestamp=_utcnow(),
            total_capital=self.cfg.total_capital,
            allocated_capital=allocated,
            free_cash=self.cfg.total_capital - allocated,
            total_closed_pnl=total_closed,
            total_open_pnl=total_open,
            portfolio_max_drawdown=portfolio_dd,
            portfolio_sharpe=portfolio_sharpe,
            positions=list(self._positions.values()),
            active_strategies=active,
            paused_strategies=paused,
            killed_strategies=killed,
        )

    def check_kill_switch(self) -> Tuple[bool, str]:
        """Check if portfolio-level kill switch should fire.
        
        Returns (fired, reason).
        """
        snap = self.snapshot()
        if abs(snap.portfolio_max_drawdown) >= self.cfg.portfolio_max_drawdown_kill:
            return True, f"portfolio_drawdown {snap.portfolio_max_drawdown:.2%} >= {self.cfg.portfolio_max_drawdown_kill:.2%}"
        # Also check if any single position exceeded its personal max
        for pos in self._positions.values():
            if abs(pos.max_drawdown) >= self.cfg.portfolio_max_drawdown_kill * 1.5:
                return True, f"position_drawdown {pos.strategy_id} {pos.max_drawdown:.2%}"
        return False, ""
    def kill_all(self, reason: str = "portfolio_kill_switch") -> None:
        """Set all positions to dead and persist."""
        self._live = False
        for sid, pos in self._positions.items():
            pos.status = "stopped"
            self._persist_position(pos)
            logger.critical(f"KILLED {sid}: {reason}")
        logger.critical(f"PORTFOLIO KILL SWITCH: {reason}")

    # ------------------------------------------------------------------
    # Dry-run bridge helpers
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset internal state for a fresh dry-run session."""
        self._daily_pnl = []
        self._live = True

    def on_trade(self, *, pair: str, pnl: float, timestamp: datetime, strategy_id: str) -> None:
        """Record a closed signal for trigger tracking."""
        self._daily_pnl.append(pnl)
        if len(self._daily_pnl) > 10_000:
            self._daily_pnl = self._daily_pnl[-5_000:]

    def is_kill_switch_triggered(self) -> bool:
        return not self._live or (self.max_drawdown() <= -self.cfg.max_drawdown)

    def daily_loss(self) -> float:
        if not self._daily_pnl:
            return 0.0
        return sum(self._daily_pnl[-500:])  # rough rolling

    def max_drawdown(self) -> float:
        if not self._daily_pnl:
            return 0.0
        cum = np.maximum.accumulate(np.cumsum(self._daily_pnl))
        if cum[-1] == 0:
            return 0.0
        trough = np.minimum.accumulate(np.cumsum(self._daily_pnl))
        dd = np.min((np.cumsum(self._daily_pnl) - cum) / (cum + 1e-9))
        return float(dd)

    def get_correlation_matrix(self) -> Dict[str, Dict[str, float]]:
        """Compute pairwise correlation matrix from backtest returns.

        Fallback: template-based heuristic when insufficient trade history.
        """
        sids = list(self._positions.keys())
        if len(sids) == 0:
            return {}
        if len(sids) == 1:
            return {sids[0]: {sids[0]: 1.0}}

        # Try actual returns history first
        matrix = {sid: {sid2: 0.0 for sid2 in sids} for sid in sids}
        for i, sid1 in enumerate(sids):
            matrix[sid1][sid1] = 1.0
            for j, sid2 in enumerate(sids):
                if i >= j:
                    continue
                h1 = self._pnl_history.get(sid1, [])
                h2 = self._pnl_history.get(sid2, [])
                if len(h1) >= 10 and len(h2) >= 10:
                    min_len = min(len(h1), len(h2))
                    corr = float(np.corrcoef(h1[-min_len:], h2[-min_len:])[0, 1])
                else:
                    corr = self._template_correlation_proxy(sid1, sid2)
                matrix[sid1][sid2] = corr
                matrix[sid2][sid1] = corr
        return matrix

    # ── weight computation ─────────────────────────────────────────────

    def _compute_weights(self, method: str) -> Dict[str, float]:
        """Compute raw allocation weights for all active positions."""
        active = {sid: p for sid, p in self._positions.items() if p.status == "active"}
        if not active:
            return {}

        if method == "equal_weight":
            n = len(active)
            return {sid: 1.0 / n for sid in active}

        if method == "inverse_vol":
            # Weight ∝ 1 / volatility from PnL history
            inv_vols = {}
            for sid, pos in active.items():
                hist = self._pnl_history.get(sid, [])
                if len(hist) >= 7:
                    vol = float(np.std(hist[-30:]))
                    inv_vols[sid] = 1.0 / max(vol, 1e-6)
                else:
                    # Fallback: use backtest Sharpe as vol proxy
                    inv_vols[sid] = max(pos.sharpe_30d, 0.1)
            total = sum(inv_vols.values())
            return {sid: w / total for sid, w in inv_vols.items()}

        if method == "equal_risk":
            # Weight to equalize risk contribution: w ∝ 1 / σ
            # Same formula as inverse_vol for uncorrelated assets;
            # can be extended with covariance matrix
            return self._compute_weights("inverse_vol")

        # Default
        n = len(active)
        return {sid: 1.0 / n for sid in active}

    def _apply_constraints(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Apply max/min weight constraints with iterative water-filling.
        
        Handles infeasible constraints (e.g. N * max_w < 1.0) by relaxing to equal-weight.
        """
        if not weights:
            return weights

        n = len(weights)
        # Check feasibility: can we satisfy both max and min simultaneously?
        # Only flag infeasibility if there are more strategies than can fit under the constraints
        # AND the raw weights would require redistribution
        max_allowed_by_min = int(1.0 / max(self.cfg.min_per_strategy_weight, 0.01))
        if n > max_allowed_by_min and self.cfg.min_per_strategy_weight > 1.0 / n:
            logger.warning(f"Infeasible min_weight constraint: min={self.cfg.min_per_strategy_weight} needs ≤{max_allowed_by_min} strategies; trimming to {max_allowed_by_min}")

        for iteration in range(50):
            # 1. Cap max weight
            capped = {}
            excess = 0.0
            uncapped_total = 0.0
            for sid, w in weights.items():
                if w > self.cfg.max_per_strategy_weight:
                    excess += w - self.cfg.max_per_strategy_weight
                    capped[sid] = self.cfg.max_per_strategy_weight
                else:
                    capped[sid] = w
                    uncapped_total += w

            # 2. Redistribute excess to uncapped positions
            if excess > 0 and uncapped_total > 0:
                for sid in capped:
                    if capped[sid] < self.cfg.max_per_strategy_weight:
                        capped[sid] += excess * (weights[sid] / uncapped_total)
            elif excess > 0:
                # All capped — infeasible, fallback
                active = [sid for sid, w in capped.items() if w > 0]
                return {sid: 1.0 / len(active) for sid in active}

            # 3. Check if any newly redistributed exceeded max
            still_over = any(w > self.cfg.max_per_strategy_weight + 1e-9 for w in capped.values())
            if still_over:
                weights = capped
                continue

            # 4. Floor min weight — trim strategies below threshold
            active_weights = {sid: w for sid, w in capped.items() if w >= self.cfg.min_per_strategy_weight}
            if not active_weights:
                best_sid, best_w = max(capped.items(), key=lambda item: item[1])
                active_weights = {best_sid: best_w}

            # 5. Renormalize to sum=1
            total = sum(active_weights.values())
            if total > 0:
                normalized = {sid: w / total for sid, w in active_weights.items()}
            else:
                normalized = active_weights

            # 6. Convergence check
            max_ok = all(w <= self.cfg.max_per_strategy_weight + 1e-9 for w in normalized.values())
            min_ok = all(w >= self.cfg.min_per_strategy_weight - 1e-9 for w in normalized.values())
            sum_ok = abs(sum(normalized.values()) - 1.0) < 1e-6

            if max_ok and min_ok and sum_ok:
                weights = normalized
                break

            weights = normalized

        # 7. Correlation penalty (applied once after weight constraints stable)
        corr_matrix = self.get_correlation_matrix()
        penalized = dict(weights)
        for sid1 in list(penalized.keys()):
            for sid2 in list(penalized.keys()):
                if sid1 >= sid2:
                    continue
                corr = corr_matrix.get(sid1, {}).get(sid2, 0.0)
                if abs(corr) >= self.cfg.max_correlation:
                    logger.warning(f"High correlation {sid1}-{sid2}: {corr:.2f}; scaling both by 0.8")
                    penalized[sid1] *= 0.8
                    penalized[sid2] *= 0.8

        # 8. Final renormalization after correlation penalty
        total = sum(penalized.values())
        if total > 0:
            penalized = {sid: w / total for sid, w in penalized.items()}
        return penalized

    # ── persistence ────────────────────────────────────────────────────

    def _persist_position(self, pos: PortfolioPosition) -> None:
        """Store position state in strategy metadata."""
        session = get_session()
        row = session.query(StrategyModel).filter_by(id=pos.strategy_id).first()
        if not row:
            return
        meta_raw = row.metadata_json or "{}"
        if isinstance(meta_raw, str):
            try:
                meta = json.loads(meta_raw)
            except json.JSONDecodeError:
                meta = {}
        else:
            meta = meta_raw or {}
        meta["portfolio"] = pos.model_dump(mode="json")
        row.metadata_json = json.dumps(meta)
        session.commit()

    def _load_from_db(self) -> None:
        """Restore portfolio state from DB on init."""
        session = get_session()
        promoted = (
            session.query(StrategyModel)
            .filter(StrategyModel.status.in_([StrategyStatus.PROMOTED.value, StrategyStatus.LIVE.value]))
            .all()
        )
        for row in promoted:
            meta_raw = row.metadata_json or "{}"
            if isinstance(meta_raw, str):
                try:
                    meta = json.loads(meta_raw)
                except json.JSONDecodeError:
                    meta = {}
            else:
                meta = meta_raw or {}
            pmeta = meta.get("portfolio")
            if pmeta:
                try:
                    # Parse ISO datetime strings back to datetime objects
                    for key in ("started_at", "last_rebalanced_at"):
                        if key in pmeta and isinstance(pmeta[key], str):
                            pmeta[key] = datetime.fromisoformat(pmeta[key].replace("Z", "+00:00"))
                    pos = PortfolioPosition(**pmeta)
                    self._positions[row.id] = pos
                    self._pnl_history[row.id] = []
                except Exception as exc:
                    logger.warning(f"Could not load portfolio position {row.id}: {exc}")
        if self._positions:
            logger.info(f"Restored {len(self._positions)} positions from DB")

    # ── helpers ──────────────────────────────────────────────────────

    def _compute_initial_weight(self, row: StrategyModel) -> float:
        """Compute starting weight for a newly added strategy."""
        # Start conservative: equal weight among all positions
        n = len(self._positions) + 1
        return 1.0 / n

    def _combined_daily_pnl(self) -> np.ndarray:
        """Sum daily PnL across all positions."""
        max_len = max((len(h) for h in self._pnl_history.values()), default=0)
        if max_len == 0:
            return np.array([])
        combined = np.zeros(max_len)
        for hist in self._pnl_history.values():
            if len(hist) == max_len:
                combined += np.array(hist)
            elif len(hist) > 0:
                # Pad with zeros if lengths differ
                padded = np.zeros(max_len)
                padded[-len(hist):] = hist
                combined += padded
        return combined

    def _template_correlation_proxy(self, sid1: str, sid2: str) -> float:
        """Heuristic correlation based on strategy templates."""
        session = get_session()
        r1 = session.query(StrategyModel).filter_by(id=sid1).first()
        r2 = session.query(StrategyModel).filter_by(id=sid2).first()
        if not r1 or not r2:
            return 0.0
        t1, t2 = r1.template, r2.template
        if t1 == t2:
            return 0.7  # Same template = likely correlated
        # Mean-reversion tends negatively correlated with trend/breakout
        if ("mean" in t1 and ("trend" in t2 or "breakout" in t2)) or \
           ("mean" in t2 and ("trend" in t1 or "breakout" in t1)):
            return -0.3
        return 0.2  # Default low correlation


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)
