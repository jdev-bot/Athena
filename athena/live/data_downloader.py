"""Download historical OHLCV data for a pair before live/paper trading.

Spawns `freqtrade download-data --userdir <dir> ...` as a subprocess so the
exact same config validation and download pipeline used by the CLI is reused.
"""
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


def download_pair_data(
    deploy_dir: Path,
    pair: str,
    timeframe: str,
    days: int = 3,
    exchange_name: str = "binance",
    sandbox: bool = False,
) -> Path:
    """Download `days` of historical candles into `deploy_dir/data/`.

    Uses Freqtrade CLI so config validation and paths match exactly.
    Returns path to the data directory.
    """
    data_dir = deploy_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", "-m", "freqtrade", "download-data",
        "--userdir", str(deploy_dir),
        "--config", str(deploy_dir / "config.json"),
        "--timeframes", timeframe,
        "--pairs", pair,
        "--days", str(days),
        "--exchange", exchange_name,
        "--trading-mode", "futures",
    ]
    if sandbox:
        cmd.append("--dry-run-wallet")

    logger.info(f"Downloading {days}d {timeframe} data for {pair} ...")
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(deploy_dir),
    )
    if result.returncode != 0:
        logger.warning(f"download-data stderr: {result.stderr[:500]}")
        raise RuntimeError(f"download-data failed: {result.returncode}")

    logger.info(f"Downloaded data → {data_dir}")
    return data_dir
