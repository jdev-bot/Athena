"""Market data downloader — thin wrapper around Freqtrade's native download_data.

Replaces all synthetic / hand-crafted candle generation with real OHLCV
fetched from the exchange via ccxt (through freqtrade's infrastructure).
"""

import logging
import tempfile
import shutil
from pathlib import Path
from typing import Optional, List

logger = logging.getLogger(__name__)

# Shared data directory — persists across backtest / forward-test runs
DEFAULT_DATA_DIR = Path("/tmp/athena_shared_data/data")


class MarketDataDownloader:
    """Download historical candles using Freqtrade's native downloader.

    Usage
    -----
    dl = MarketDataDownloader()
    dl.ensure_data(pair="BTC/USDT", timeframe="1h", days=90)
    """

    def __init__(
        self,
        *,
        exchange: str = "binance",
        trading_mode: str = "spot",
        data_dir: Optional[Path] = None,
        data_format: str = "feather",
    ):
        self.exchange = exchange
        self.trading_mode = trading_mode
        self.data_dir = data_dir or DEFAULT_DATA_DIR
        self.data_format = data_format
        self._exchange_dir = self.data_dir / exchange
        self._exchange_dir.mkdir(parents=True, exist_ok=True)

    # ── public API ─────────────────────────────────────────────────

    def ensure_data(
        self, *,
        pair: str,
        timeframe: str = "1h",
        days: Optional[int] = None,
        timerange: Optional[str] = None,
        overwrite: bool = False,
    ) -> Path:
        """Download and return path to candle file.

        If the file already exists and overwrite=False, skip download.
        """
        file_path = self._file_path(pair, timeframe)
        if file_path.exists() and not overwrite:
            logger.info("Data already present: %s", file_path)
            return file_path

        self._download([pair], [timeframe], days=days, timerange=timerange)
        return file_path

    def ensure_multi(
        self,
        *,
        pairs: List[str],
        timeframes: List[str],
        days: int = 90,
        overwrite: bool = False,
    ) -> List[Path]:
        """Batch-download multiple pairs / timeframes."""
        missing = [
            (p, t)
            for p in pairs
            for t in timeframes
            if overwrite or not self._file_path(p, t).exists()
        ]
        if missing:
            all_pairs = list({p for p, _ in missing})
            all_tfs = list({t for _, t in missing})
            self._download(all_pairs, all_tfs, days=days)
        return [self._file_path(p, t) for p in pairs for t in timeframes]

    # ── internal ───────────────────────────────────────────────────

    def _file_path(self, pair: str, timeframe: str) -> Path:
        """Freqtrade feather naming:  data/&lt;exchange&gt;/&lt;pair&gt;_&lt;timeframe&gt;.feather"""
        # pair has slashes, freqtrade replaces with underscores
        safe_pair = pair.replace("/", "_").replace(":", "_")
        return self._exchange_dir / f"{safe_pair}-{timeframe}.{self.data_format}"

    def _download(
        self,
        pairs: List[str],
        timeframes: List[str],
        days: Optional[int] = None,
        timerange: Optional[str] = None,
    ) -> None:
        from freqtrade.commands.data_commands import start_download_data
        from freqtrade.configuration import Configuration

        config = {
            "exchange": {
                "name": self.exchange,
                "key": "",
                "secret": "",
            },
            "pairs": pairs,
            "timeframes": timeframes,
            "datadir": str(self.data_dir),
            "data_format_ohlcv": self.data_format,
            "trading_mode": self.trading_mode,
            "dry_run": True,
        }
        if days is not None:
            config["days"] = days
        if timerange is not None:
            config["timerange"] = timerange

        # Build minimal args dict for freqtrade CLI compat
        args = {
            "config": [],
            "exchange": self.exchange,
            "pairs": pairs,
            "timeframes": timeframes,
            "datadir": str(self.data_dir),
            "data_format_ohlcv": self.data_format,
            "trading_mode": self.trading_mode,
            "dry_run": True,
            "days": days,
            "timerange": timerange,
            "timeframe": timeframes[0] if timeframes else "1h",
        }

        logger.info("Downloading %s %s (days=%s timerange=%s) ...", pairs, timeframes, days, timerange)
        try:
            start_download_data(args)
            logger.info("Download complete.")
        except Exception as exc:
            logger.error("Download failed: %s", exc)
            raise

    def purge(self) -> None:
        """Delete all cached candle files."""
        if self._exchange_dir.exists():
            shutil.rmtree(self._exchange_dir)
            logger.info("Purged data dir: %s", self._exchange_dir)

