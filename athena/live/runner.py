"""Forward-test runner — evaluates a strategy on live candles (signal-only, no trades)."""
import asyncio
import uuid
import importlib
import sys
import types
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import pandas_ta as ta

from athena.live.feed import LiveFeed
from athena.core.freqtrade_wrapper import FreqtradeWrapper
from athena.services.models import get_session, StrategyModel, LiveSessionModel


# ── lightweight strategy executor ─────────────────────────────────
class ForwardRunner:
    """Evaluate a strategy's entry/exit logic on a live candle stream.

    This does NOT execute real trades and does NOT use the full Jesse
    backtest engine.  It recreates the indicator values from the
    strategy's DNA parameters and decides long/short/close based on
    the latest bar.  Signals are logged to DB.
    """

    def __init__(self, strategy_id: str, session_id: str, risk: dict = None):
        self.strategy_id = strategy_id
        self.session_id = session_id
        self.risk = risk or {}
        self._candles: list = []  # list of [ts, open, high, low, close, vol]
        self._position: str = "flat"  # flat, long, short
        self._equity: float = 10_000.0
        self._daily_start_equity: float = 10_000.0
        self._peak: float = 10_000.0
        self._trades = 0
        self._max_dd = 0.0
        self._last_day: int = datetime.utcnow().day

        # Load strategy from DB
        sess = get_session()
        self._record = sess.query(StrategyModel).filter_by(id=strategy_id).first()
        if not self._record:
            raise ValueError(f"Strategy {strategy_id} not found")

        self._dna = self._record.dna
        self._template = self._record.template
        sess.close()

    # ── indicator helpers ─────────────────────────────────────────
    def _ema(self, prices: np.ndarray, period: int) -> float:
        return ta.ema(prices, period, sequential=False) if len(prices) >= period else prices[-1]

    def _rsi(self, prices: np.ndarray, period: int) -> float:
        return ta.rsi(prices, period, sequential=False) if len(prices) >= period else 50.0

    def _atr(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int) -> float:
        if len(highs) < period:
            return closes[-1] * 0.01
        return ta.atr(highs, lows, closes, period, sequential=False)

    # ── decision ────────────────────────────────────────────────────
    def _should_long(self) -> bool:
        """Reproduce trend-following long entry logic from DNA."""
        if len(self._candles) < max(self._dna.get("slow_period", 30), self._dna.get("rsi_period", 14)):
            return False
        closes = np.array([c[4] for c in self._candles])
        fast = self._ema(closes, self._dna.get("fast_period", 10))
        slow = self._ema(closes, self._dna.get("slow_period", 30))
        rsi = self._rsi(closes, self._dna.get("rsi_period", 14))
        return fast > slow and rsi < self._dna.get("rsi_overbought", 70)

    def _should_short(self) -> bool:
        """Reproduce short entry logic."""
        if len(self._candles) < max(self._dna.get("slow_period", 30), self._dna.get("rsi_period", 14)):
            return False
        closes = np.array([c[4] for c in self._candles])
        fast = self._ema(closes, self._dna.get("fast_period", 10))
        slow = self._ema(closes, self._dna.get("slow_period", 30))
        rsi = self._rsi(closes, self._dna.get("rsi_period", 14))
        return fast < slow and rsi > self._dna.get("rsi_oversold", 30)

    def _should_cancel(self) -> bool:
        # Placeholder — exit on trend reversal
        if self._position == "long":
            return self._should_short() if self._template == "trend_following" else False
        if self._position == "short":
            return self._should_long() if self._template == "trend_following" else False
        return False

    # ── equity / risk ───────────────────────────────────────────────
    def _update_equity(self, candle: list):
        """Track unrealized PnL for open position."""
        close = candle[4]
        if self._position == "long" and hasattr(self, "_entry_price"):
            pnl = (close - self._entry_price) / self._entry_price * self._equity
            self._equity += pnl * self._dna.get("position_size", 0.1)
        elif self._position == "short" and hasattr(self, "_entry_price"):
            pnl = (self._entry_price - close) / self._entry_price * self._equity
            self._equity += pnl * self._dna.get("position_size", 0.1)
        if self._equity > self._peak:
            self._peak = self._equity
        dd = (self._peak - self._equity) / self._peak
        if dd > self._max_dd:
            self._max_dd = dd

    def _circuit_breaker(self) -> Optional[str]:
        """Return stop reason if tripped, else None."""
        max_dd = self.risk.get("max_drawdown", 0.15)
        if self._max_dd >= max_dd:
            return f"stopped_drawdown ({self._max_dd:.2%})"
        # Daily loss limit
        daily_limit = self.risk.get("daily_loss_limit", 0.10)
        daily_pnl = (self._equity - self._daily_start_equity) / self._daily_start_equity
        if daily_pnl <= -daily_limit:
            return f"stopped_daily_loss ({daily_pnl:.2%})"
        return None

    # ── public loop ───────────────────────────────────────────────
    def on_candle(self, candle: list) -> Optional[str]:
        """Process one live candle. Returns stop reason or None."""
        self._candles.append(candle)
        if len(self._candles) > 500:
            self._candles.pop(0)

        self._update_equity(candle)

        # Reset daily equity on new GMT day
        current_day = datetime.utcnow().day
        if current_day != self._last_day:
            self._last_day = current_day
            self._daily_start_equity = self._equity

        signal = None
        if self._position == "flat":
            if self._should_long():
                signal = {"side": "long", "price": candle[4], "ts": candle[0]}
                self._position = "long"
                self._entry_price = candle[4]
                self._trades += 1
            elif self._should_short():
                signal = {"side": "short", "price": candle[4], "ts": candle[0]}
                self._position = "short"
                self._entry_price = candle[4]
                self._trades += 1
        else:
            if self._should_cancel():
                signal = {"side": "close", "price": candle[4], "ts": candle[0]}
                self._position = "flat"

        # Persist state to DB
        self._persist_state(signal)

        return self._circuit_breaker()

    def _persist_state(self, signal: Optional[dict]):
        sess = get_session()
        row = sess.query(LiveSessionModel).filter_by(id=self.session_id).first()
        if row:
            row.equity = round(self._equity, 2)
            row.open_positions = 1 if self._position != "flat" else 0
            row.total_trades_taken = self._trades
            row.max_drawdown_seen = self._max_dd
            row.updated_at = datetime.utcnow()
            if signal:
                sigs = list(row.last_signals or [])
                sigs.append(signal)
                if len(sigs) > 50:
                    sigs = sigs[-50:]
                row.last_signals = sigs
            sess.commit()
        sess.close()


# ── daemon entrypoint ───────────────────────────────────────────
class LiveRunner:
    """Manages a single forward-test session with live candle feed."""

    def __init__(self, strategy_id: str, mode: str = "paper", risk: dict = None):
        self.strategy_id = strategy_id
        self.mode = mode
        self.risk = risk or {}
        self.session_id = f"live_{uuid.uuid4().hex[:12]}"
        self._runner: Optional[ForwardRunner] = None
        self._feed_task: Optional[asyncio.Task] = None
        self._stopped = False

    async def start(self):
        """Begin the forward-test loop."""
        # Persist session record
        sess = get_session()
        sess.add(LiveSessionModel(
            id=self.session_id,
            strategy_id=self.strategy_id,
            status="running",
            mode=self.mode,
        ))
        sess.commit()
        sess.close()

        self._runner = ForwardRunner(self.strategy_id, self.session_id, self.risk)
        self._feed = LiveFeed(
            symbol="BTC/USDT", timeframe="1m", exchange="binance",
            on_candle=self._on_candle,
        )
        self._feed_task = asyncio.create_task(self._feed.start())

    async def stop(self, reason: str = "stopped_by_user"):
        """Gracefully halt the runner."""
        self._stopped = True
        if self._feed:
            self._feed.stop()
        if self._feed_task:
            self._feed_task.cancel()
            try:
                await self._feed_task
            except asyncio.CancelledError:
                pass
        if self._feed and self._feed._exchange:
            await self._feed._exchange.close()

        sess = get_session()
        row = sess.query(LiveSessionModel).filter_by(id=self.session_id).first()
        if row:
            row.status = reason
            row.stopped_at = datetime.utcnow()
            sess.commit()
        sess.close()

    def _on_candle(self, candle: list):
        """Callback invoked by LiveFeed on each closed candle."""
        if self._stopped or not self._runner:
            return
        stop_reason = self._runner.on_candle(candle)
        if stop_reason:
            asyncio.create_task(self.stop(stop_reason))

    @property
    def stats(self) -> dict:
        sess = get_session()
        row = sess.query(LiveSessionModel).filter_by(id=self.session_id).first()
        if not row:
            sess.close()
            return {}
        stats = {
            "session_id": row.id,
            "strategy_id": row.strategy_id,
            "status": row.status,
            "mode": row.mode,
            "equity": row.equity,
            "open_positions": row.open_positions,
            "total_trades": row.total_trades_taken,
            "max_drawdown": row.max_drawdown_seen,
            "started_at": row.started_at.isoformat() if row.started_at else None,
            "stopped_at": row.stopped_at.isoformat() if row.stopped_at else None,
            "last_signals": row.last_signals,
        }
        sess.close()
        return stats
