"""Live trading runner — thin wrapper around Freqtrade BotManager.

The old ForwardRunner (signal-only, custom candle logic) is replaced by a
real Freqtrade bot subprocess.  This module preserves the same public API
so the FastAPI endpoints remain stable.
"""
import asyncio
from typing import Optional, Dict, Any

from athena.live.bot_manager import BotManager


class LiveRunner:
    """Manages a single live/paper trading session backed by a Freqtrade bot."""

    def __init__(
        self,
        strategy_id: str,
        mode: str = "paper",
        risk: dict = None,
        exchange_key: str = "",
        exchange_secret: str = "",
        sandbox: bool = False,
    ):
        self.strategy_id = strategy_id
        self.mode = mode
        self.risk = risk or {}
        self.exchange_key = exchange_key
        self.exchange_secret = exchange_secret
        self.sandbox = sandbox
        self.session_id: Optional[str] = None
        self._manager = BotManager()

    async def start(self):
        """Start the Freqtrade bot."""
        self.session_id = await self._manager.start(
            strategy_id=self.strategy_id,
            mode=self.mode,
            risk=self.risk,
            exchange_key=self.exchange_key,
            exchange_secret=self.exchange_secret,
            sandbox=self.sandbox,
        )

    async def stop(self, reason: str = "stopped_by_user"):
        """Stop the bot and clean up."""
        if self.session_id:
            await self._manager.stop(self.session_id, reason)

    @property
    def stats(self) -> Dict[str, Any]:
        """Current session stats (proxied from Freqtrade API)."""
        if not self.session_id:
            return {}
        return self._manager.status(self.session_id)
