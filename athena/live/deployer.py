"""Deploy a strategy + config into a Freqtrade user_data directory.

Reuses Freqtrade's own directory layout so the bot discovers strategies,
configs, and data files without custom search paths.
"""
import os
import shutil
from pathlib import Path
from typing import Any, Dict

from athena.core.freqtrade_wrapper import FreqtradeWrapper
from athena.services.models import get_session, StrategyModel
from athena.live.freqtrade_config import build_config, write_config
from athena.live.data_downloader import download_pair_data


class Deployer:
    """Write strategy code and Freqtrade config into a temporary user_data dir."""

    def __init__(self, base_dir: Path = None):
        self.base_dir = base_dir or Path("/tmp/athena_ft_deploys")
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def deploy(
        self,
        strategy_id: str,
        mode: str = "paper",
        risk: Dict[str, Any] = None,
        api_port: int = 0,
        db_url: str = None,
        exchange_key: str = "",
        exchange_secret: str = "",
        sandbox: bool = False,
    ) -> Path:
        """Create a user_data directory for a Freqtrade bot session.

        Returns the path to the directory (contains config.json,
        strategies/, data/, logs/).
        """
        sess = get_session()
        record = sess.query(StrategyModel).filter_by(id=strategy_id).first()
        if not record:
            raise ValueError(f"Strategy {strategy_id} not found")

        # Compile strategy source
        strategy_code = FreqtradeWrapper.compile_strategy(record)

        # Build deploy directory
        deploy_dir = self.base_dir / strategy_id
        if deploy_dir.exists():
            shutil.rmtree(deploy_dir, ignore_errors=True)
        deploy_dir.mkdir(parents=True, exist_ok=True)

        # Write strategy
        strat_dir = deploy_dir / "strategies"
        strat_dir.mkdir(parents=True, exist_ok=True)
        (strat_dir / "__init__.py").write_text("")
        (strat_dir / "AthenaStrategy.py").write_text(strategy_code)

        # Write config
        pair = Deployer._symbol_to_pair(record.dna.get("symbol", "BTC-USD"))
        timeframe = record.dna.get("timeframe", "1h")
        # Read sizing / risk params from DNA vector
        dna_vector = record.dna.get("vector", {}) if isinstance(record.dna, dict) else {}
        max_open_trades = int(dna_vector.get("max_open_trades", 1))

        cfg = build_config(
            strategy_name="AthenaStrategy",
            strategy_path=str(strat_dir),
            pair=pair,
            timeframe=timeframe,
            mode=mode,
            wallet_balance=50.0,
            max_open_trades=max_open_trades,
            dna=record.dna,
            risk=risk,
            db_url=db_url,
            api_port=api_port,
            exchange_key=exchange_key,
            exchange_secret=exchange_secret,
            sandbox=sandbox,
        )
        write_config(cfg, deploy_dir / "config.json")

        # Download historical candles so Freqtrade can warm up indicators
        try:
            download_pair_data(
                deploy_dir=deploy_dir,
                pair=pair,
                timeframe=timeframe,
                days=3,
                exchange_name="binance",
                sandbox=sandbox,
            )
        except Exception as exc:
            # Don't fail deploy if download errors — bot may still work if
            # exchange has enough warmup via ccxt streaming
            import logging
            logging.getLogger(__name__).warning(f"Data download failed: {exc}")

        # Data + logs dirs
        (deploy_dir / "data").mkdir(exist_ok=True)
        (deploy_dir / "logs").mkdir(exist_ok=True)

        sess.close()
        return deploy_dir

    def cleanup(self, strategy_id: str):
        """Remove a deploy directory."""
        deploy_dir = self.base_dir / strategy_id
        if deploy_dir.exists():
            shutil.rmtree(deploy_dir, ignore_errors=True)

    @staticmethod
    def _symbol_to_pair(symbol: str, futures: bool = True) -> str:
        """BTC-USD → BTC/USDT:USDT (futures) or BTC/USDT (spot)."""
        pair = symbol.replace("-", "/")
        if pair.endswith("USD") and not pair.endswith("USDT"):
            pair = pair + "T"
        if futures:
            pair = pair + ":USDT"
        return pair
