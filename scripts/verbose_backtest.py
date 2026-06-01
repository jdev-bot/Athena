"""Verbose backtest run with unconditional entry."""
import subprocess, json, tempfile, shutil
from pathlib import Path

STRATEGY = '''
import numpy as np, pandas as pd
from freqtrade.strategy.interface import IStrategy
import logging
logger = logging.getLogger(__name__)

class AthenaStrategy(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "1h"
    stoploss = -0.99
    can_short = False
    startup_candle_count = 0
    use_exit_signal = False

    def populate_indicators(self, df, metadata):
        return df

    def populate_entry_trend(self, df, metadata):
        df.loc[:, "enter_long"] = 1
        return df

    def populate_exit_trend(self, df, metadata):
        df.loc[:, "exit_long"] = 0
        return df
'''

tmp = Path(tempfile.mkdtemp(prefix="ft_v"))
strat_dir = tmp / "strategies"
strat_dir.mkdir(parents=True)
(strat_dir / "__init__.py").write_text("")
(strat_dir / "AthenaStrategy.py").write_text(STRATEGY)
shutil.copytree("/tmp/athena_shared_data/data", tmp / "data")

config = {
    "strategy": "AthenaStrategy",
    "strategy_path": str(strat_dir),
    "user_data_dir": str(tmp),
    "timeframe": "1h",
    "pairs": ["BTC/USDT:USDT"],
    "stake_currency": "USDT",
    "stake_amount": "unlimited",
    "tradable_balance_ratio": 1.0,
    "fiat_display_currency": "USD",
    "dry_run": True,
    "dry_run_wallet": 500.0,
    "max_open_trades": 1,
    "entry_pricing": {"price_side": "other", "use_order_book": True, "order_book_top": 1},
    "exit_pricing": {"price_side": "other", "use_order_book": True, "order_book_top": 1},
    "order_types": {"entry": "market", "exit": "market", "stoploss": "market", "stoploss_on_exchange": False},
    "exchange": {"name": "binance", "key": "", "secret": "", "ccxt_config": {"enableRateLimit": True}, "pair_whitelist": ["BTC/USDT:USDT"], "pair_blacklist": [], "sandbox": False},
    "pairlists": [{"method": "StaticPairList"}],
    "datadir": str(tmp / "data" / "binance"),
    "timerange": "20260501-20260515",
    "fee": 0.001,
    "trading_mode": "futures",
    "margin_mode": "cross",
    "dataformat_ohlcv": "feather",
}
(tmp / "config.json").write_text(json.dumps(config, indent=2))

cmd = [
    "python", "-m", "freqtrade", "backtesting",
    "-c", str(tmp / "config.json"),
    "--strategy", "AthenaStrategy",
    "--strategy-path", str(strat_dir),
    "--timerange", "20260501-20260515",
    "--timeframe", "1h",
    "--pairs", "BTC/USDT:USDT",
    "--dry-run-wallet", "500",
    "--max-open-trades", "1",
    "--fee", "0.001",
    "--verbose",
]
proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
print(proc.stdout[-5000:])
print("STDERR last 2000 chars:", proc.stderr[-2000:])
print("Return code:", proc.returncode)
shutil.rmtree(tmp, ignore_errors=True)
