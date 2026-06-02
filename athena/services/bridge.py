"""Forward-test (dry-run) bridge.
Intercepts strategy signals without submitting real exchange orders.
Persists signals to database and enforces kill-switch via portfolio manager."""

import datetime
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from freqtrade.strategy import IStrategy
from freqtrade.data.dataprovider import DataProvider

from athena.common.models import StrategyStatus
from athena.portfolio.manager import PortfolioManager
from athena.services.models import get_session, Signal, Base

logger = logging.getLogger(__name__)


@dataclass
class SignalEntry:
    strategy_id: str
    symbol: str
    direction: str  # "long" | "short" | "buy" | "sell"
    confidence: float  # 0.0..1.0
    entry_price: float
    timestamp: datetime.datetime = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc))
    exit_price: Optional[float] = None
    exit_time: Optional[datetime.datetime] = None
    pnl_abs: Optional[float] = None
    pnl_pct: Optional[float] = None
    triggered_kill_switch: bool = False


class DryRunTrader:
    """Paper-trade facade for Freqtrade strategy objects.

    Hooks into a strategy's next() via monkey-patching so we emit
    signals instead of real market orders.  No assets are held;
    PnL is marked-to-market when an opposite signal is received.
    """

    KILL_DAILY_LOSS_LIMIT: float = 0.10  # 10 % hard stop
    KILL_MAX_DRAWDOWN: float = 0.15      # 15 % hard stop

    def __init__(self, *, portfolio: PortfolioManager, db_session_factory=get_session) -> None:
        self._portfolio = portfolio
        self._signals: List[SignalEntry] = []
        self._open_positions: Dict[str, SignalEntry] = {}
        self._db = db_session_factory
        self._killed = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def is_killed(self) -> bool:
        return self._killed

    def on_signal(
        self,
        *,
        strategy_id: str,
        symbol: str,
        signal: str,
        close: float,
    ) -> None:
        """Called whenever would enter/exit on bar close."""
        if self._killed:
            return

        now = datetime.datetime.now(datetime.timezone.utc)

        # --- close existing position on opposite signal ---
        key = f"{strategy_id}:{symbol}"
        if key in self._open_positions and self._open_positions[key].direction != signal:
            self._close(signal_entry=self._open_positions[key], exit_price=close, exit_time=now)
            del self._open_positions[key]

        # --- open new ---
        if signal in ("long", "short") and key not in self._open_positions:
            entry = SignalEntry(
                strategy_id=strategy_id, symbol=symbol, direction=signal,
                confidence=1.0, entry_price=close,
            )
            self._signals.append(entry)
            self._open_positions[key] = entry
            self._persist(entry)
            logger.info("dry-run %s %s at %.2f  (strategy=%s)", signal, symbol, close, strategy_id)

            # telemetry
            from athena.services.telemetry import TelemetryCollector
            _tele = TelemetryCollector()
            _tele.record_signal(signal)
            _tele.set_kill_switch(self._killed)

        # --- equity / kill-switch guard ---
        self._check_kill_guards()

    def reset(self) -> None:
        self._open_positions.clear()
        self._signals.clear()
        self._killed = False
        self._portfolio.reset()

    def to_dict(self) -> Dict[str, Any]:
        closed = [s for s in self._signals if s.exit_price is not None]
        return {
            "killed": self._killed,
            "open_positions": len(self._open_positions),
            "total_signals": len(self._signals),
            "total_closed": len(closed),
            "total_pnl": sum(s.pnl_pct or 0 for s in closed),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _close(self, signal_entry: SignalEntry, exit_price: float, exit_time: datetime.datetime) -> None:
        signal_entry.exit_price = exit_price
        signal_entry.exit_time = exit_time
        raw = (exit_price - signal_entry.entry_price) / signal_entry.entry_price
        if signal_entry.direction == "short":
            raw = -raw
        signal_entry.pnl_pct = raw
        signal_entry.pnl_abs = raw * signal_entry.entry_price
        self._persist(signal_entry)

        # feed to portfolio manager
        self._portfolio.on_trade(
            pair=signal_entry.symbol,
            pnl=raw,
            timestamp=exit_time,
            strategy_id=signal_entry.strategy_id,
        )

    def _check_kill_guards(self) -> None:
        """Hard-stops based on portfolio manager waterline."""
        if self._portfolio.is_kill_switch_triggered():
            self._killed = True
            logger.error("Kill-switch triggered — portfolio drew down %.2f", self._portfolio.max_drawdown())
            return

        if self._portfolio.daily_loss() < -self.KILL_DAILY_LOSS_LIMIT:
            self._killed = True
            logger.error("Kill-switch triggered — daily loss %.2f", self._portfolio.daily_loss())
            return

    def _persist(self, sig: SignalEntry) -> None:
        import uuid
        # TODO wrap in an async / cron friendly model
        with self._db() as session:
            db_signal = Signal(
                id=str(uuid.uuid4()),
                strategy_id=sig.strategy_id,
                symbol=sig.symbol,
                direction=sig.direction,
                confidence=sig.confidence,
                entry_price=sig.entry_price,
                exit_price=sig.exit_price,
                timestamp=sig.timestamp,
                pnl_pct=sig.pnl_pct,
                pnl_abs=sig.pnl_abs,
                status="open" if sig.exit_price is None else "closed",
                triggered_kill_switch=sig.triggered_kill_switch,
            )
            # logger.info("Persisting signal for strategy %s on symbol %s", sig.strategy_id, sig.symbol)
            session.add(db_signal)
            session.commit()

