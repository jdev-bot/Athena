"""Bypass wrapper, use bare Freqtrade CLI with a trivial unconditional-entry strategy."""
import subprocess, sys, json, tempfile, shutil
from pathlib import Path

STRATEGY = '''
import numpy as np, pandas as pd
from freqtrade.strategy.interface import IStrategy

class AthenaStrategy(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "1h"
    stoploss = -0.99
    can_short = False
    startup_candle_count = 0

    def populate_indicators(self, df, metadata):
        return df

    def populate_entry_trend(self, df, metadata):
        df.loc[:, "enter_long"] = 1
        return df

    def populate_exit_trend(self, df, metadata):
        df.loc[:, "exit_long"] = 0
        return df
'''

PAIR = "BTC/USDT:USDT"
TIMEFRAME = "1h"

# Build user_data tree
tmp = Path(tempfile.mkdtemp(prefix="ft_raw_"))
strat_dir = tmp / "strategies"
strat_dir.mkdir(parents=True)
(strat_dir / "__init__.py").write_text("")
(strat_dir / "AthenaStrategy.py").write_text(STRATEGY)

# Point data at shared cache
import shutil
src_data = Path("/tmp/athena_shared_data/data")
if src_data.exists():
    shutil.copytree(src_data, tmp / "data")

config = {
    "strategy": "AthenaStrategy",
    "strategy_path": str(strat_dir),
    "user_data_dir": str(tmp),
    "timeframe": TIMEFRAME,
    "pairs": [PAIR],
    "stake_currency": "USDT",
    "stake_amount": "unlimited",
    "tradable_balance_ratio": 0.99,
    "fiat_display_currency": "USD",
    "dry_run": True,
    "dry_run_wallet": 500.0,
    "max_open_trades": 1,
    "futures_leverage": 10,
    "futures_leverage_mode": "cross",
    "entry_pricing": {
        "price_side": "other",
        "use_order_book": True,
        "order_book_top": 1,
        "price_last_balance": 0.0,
        "check_depth_of_market": {"enabled": False, "bids_to_ask_delta": 1},
    },
    "exit_pricing": {
        "price_side": "other",
        "use_order_book": True,
        "order_book_top": 1,
    },
    "order_types": {
        "entry": "market",
        "exit": "market",
        "emergency_exit": "market",
        "force_exit": "market",
        "stoploss": "market",
        "take_profit": "limit",
        "stoploss_on_exchange": False,
        "stoploss_on_exchange_interval": 60,
    },
    "unfilledtimeout": {
        "entry": 10,
        "exit": 10,
        "unit": "minutes",
    },
    "exchange": {
        "name": "binance",
        "key": "",
        "secret": "",
        "ccxt_config": {"enableRateLimit": True},
        "ccxt_async_config": {"enableRateLimit": True},
        "pair_whitelist": [PAIR],
        "pair_blacklist": [],
        "sandbox": False,
    },
    "pairlists": [{"method": "StaticPairList"}],
    "datadir": str(tmp / "data" / "binance"),
    "timerange": "20260501-20260515",
    "fee": 0.001,
    "trading_mode": "futures",
    "margin_mode": "cross",
    "dataformat_ohlcv": "feather",
    "dataformat_trades": "feather",
}
(tmp / "config.json").write_text(json.dumps(config, indent=2))

# Run freqtrade backtesting via CLI
cmd = [
    sys.executable, "-m", "freqtrade", "backtesting",
    "-c", str(tmp / "config.json"),
    "--strategy", "AthenaStrategy",
    "--strategy-path", str(strat_dir),
    "--timerange", "20260501-20260515",
    "--timeframe", TIMEFRAME,
    "--pairs", PAIR,
    "--dry-run-wallet", "500",
    "--max-open-trades", "1",
    "--fee", "0.001",
]

print("Running:", " ".join(cmd))
proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
print(proc.stdout[-3000:])
print("STDERR:", proc.stderr[-1500:])
print("Return code:", proc.returncode)

shutil.rmtree(tmp, ignore_errors=True)
