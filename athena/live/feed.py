"""Live WebSocket data feed using ccxt Pro (async watch_ohlcv)."""
import asyncio
import time
from typing import Optional, Callable

import ccxt.pro as ccxtpro


class LiveFeed:
    """Async generator that yields real-time 1m OHLCV candles from Binance."""

    def __init__(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "1m",
        exchange: str = "binance",
        on_candle: Optional[Callable[[list], None]] = None,
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.exchange_name = exchange.lower()
        self.on_candle = on_candle
        self._running = False
        self._exchange = None
        self._last_ts = 0

    async def start(self):
        """Enter the watch loop. Call in an asyncio task."""
        self._running = True
        exchange_cls = getattr(ccxtpro, self.exchange_name, None)
        if exchange_cls is None:
            raise RuntimeError(f"ccxt.pro exchange '{self.exchange_name}' not available")
        self._exchange = exchange_cls({"enableRateLimit": True, "options": {"defaultType": "swap"}})

        while self._running:
            try:
                candle = await self._exchange.watch_ohlcv(self.symbol, self.timeframe)
                # watch_ohlcv returns the *latest* forming candle; we only yield when it closes
                # (timestamp changes)
                if candle and candle[-1][0] != self._last_ts:
                    self._last_ts = candle[-1][0]
                    if self.on_candle:
                        self.on_candle(candle[-1])
            except Exception as exc:
                await asyncio.sleep(1)

        await self._exchange.close()

    def stop(self):
        self._running = False
