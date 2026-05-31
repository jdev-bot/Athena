"""Market data provider using ccxt — fetches real OHLCV from live exchanges."""
import os
import csv
import time
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Tuple
import numpy as np

import ccxt


CANDLE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "candles"
CANDLE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_path(exchange: str, symbol: str, timeframe: str) -> Path:
    """Return local CSV path for cached candles."""
    sym = symbol.replace("/", "-")
    return CANDLE_DIR / f"{exchange}_{sym}_{timeframe}.csv"


class MarketDataProvider:
    """Fetch and cache real OHLCV candles from crypto exchanges via ccxt."""

    def __init__(self, exchange_name: str = "binance"):
        self.exchange = getattr(ccxt, exchange_name)({
            "enableRateLimit": True,
            "options": {"defaultType": "swap"},
        })

    # ── fetch ──────────────────────────────────────────────────────
    def fetch_ohlcv(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "1m",
        since: Optional[int] = None,
        until: Optional[int] = None,
        limit: int = 1000,
    ) -> List[List[float]]:
        """Fetch OHLCV candles from the exchange with automatic pagination."""
        all_candles: List[List[float]] = []
        while True:
            batch = self.exchange.fetch_ohlcv(
                symbol, timeframe=timeframe, since=since, limit=limit
            )
            if not batch:
                break
            # Remove duplicate last candle appended across pages
            if all_candles and batch and batch[0][0] == all_candles[-1][0]:
                batch = batch[1:]
            if not batch:
                break
            all_candles.extend(batch)
            since = batch[-1][0] + 1
            if until and since >= until:
                break
            time.sleep(self.exchange.rateLimit / 1000)
        return all_candles

    def fetch_and_cache(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "1m",
        start_date: str = "2025-01-01",
        end_date: str = "2025-03-01",
    ) -> Path:
        """Fetch full range, cache to CSV, return path."""
        since = self.exchange.parse8601(f"{start_date}T00:00:00Z")
        until = self.exchange.parse8601(f"{end_date}T00:00:00Z")
        candles = self.fetch_ohlcv(symbol, timeframe, since=since, until=until)
        path = _cache_path(self.exchange.id, symbol, timeframe)
        with open(path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
            for c in candles:
                writer.writerow(c)
        return path

    # ── load ───────────────────────────────────────────────────────
    def load_cached(
        self, symbol: str = "BTC/USDT", timeframe: str = "1m"
    ) -> np.ndarray:
        """Load cached candles from disk and return a Jesse-compatible numpy array."""
        path = _cache_path(self.exchange.id, symbol, timeframe)
        if not path.exists():
            raise FileNotFoundError(f"No cached candles at {path}. Run fetch_and_cache() first.")
        candles = []
        with open(path, "r") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                candles.append(
                    [
                        int(row["timestamp"]),
                        float(row["open"]),
                        float(row["high"]),
                        float(row["low"]),
                        float(row["close"]),
                        float(row["volume"]),
                    ]
                )
        return np.array(candles, dtype=np.float64)

    def candles_to_jesse_format(
        self, symbol: str = "BTC/USDT", timeframe: str = "1m",
        exchange_name: Optional[str] = None
    ) -> Tuple[dict, dict]:
        """Return (candles_dict, warmup_dict) for jesse.research.backtest()."""
        candles = self.load_cached(symbol, timeframe)
        key = f"{exchange_name or self.exchange.id}-{symbol.replace('/', '-')}"
        warmup = candles[:240] if len(candles) > 240 else candles
        candle_struct = {
            key: {
                "exchange": exchange_name or self.exchange.id,
                "symbol": symbol.replace("/", "-"),
                "candles": candles,
            }
        }
        warmup_struct = {
            key: {
                "exchange": exchange_name or self.exchange.id,
                "symbol": symbol.replace("/", "-"),
                "candles": warmup,
            }
        }
        return candle_struct, warmup_struct

    # ── real-time streaming (forward test) ─────────────────────────
    def watch_ohlcv(self, symbol: str = "BTC/USDT", timeframe: str = "1m"):
        """Blocking generator that yields new candles as they close."""
        last_ts = 0
        while True:
            try:
                candle = self.exchange.fetch_ohlcv(symbol, timeframe, limit=2)
                if candle and candle[-1][0] != last_ts:
                    last_ts = candle[-1][0]
                    yield candle[-1]
            except Exception as exc:
                print(f"[MarketData] watch error: {exc}")
            time.sleep(10)


# ── module-level convenience ────────────────────────────────────
def ensure_real_candles(
    symbol: str = "BTC/USDT",
    timeframe: str = "1m",
    start_date: str = "2025-01-01",
    end_date: str = "2025-03-01",
    exchange: str = "binance",
) -> np.ndarray:
    """Idempotent: fetch if missing, then return candles."""
    prov = MarketDataProvider(exchange)
    path = _cache_path(exchange, symbol, timeframe)
    if not path.exists():
        prov.fetch_and_cache(symbol, timeframe, start_date, end_date)
    return prov.load_cached(symbol, timeframe)
