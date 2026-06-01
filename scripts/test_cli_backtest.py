"""Run Freqtrade backtesting via CLI to isolate trade execution."""
import subprocess, sys, json, shutil
from pathlib import Path

USERDIR = Path("/tmp/athena_cli_test")
if USERDIR.exists():
    shutil.rmtree(USERDIR)
(USERDIR / "strategies").mkdir(parents=True, exist_ok=True)

STRATEGY = """
import numpy as np, pandas as pd, pandas_ta as ta
from freqtrade.strategy.interface import IStrategy

class AthenaStrategy(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = '1h'
    stoploss = -0.10
    can_short = False
    startup_candle_count = 12

    def populate_indicators(self, df, md):
        df['ema'] = ta.ema(df['close'], length=12)
        return df

    def populate_entry_trend(self, df, md):
        df.loc[df['close'] > df['ema'], 'enter_long'] = 1
        return df

    def populate_exit_trend(self, df, md):
        df.loc[df['close'] < df['ema'], 'exit_long'] = 1
        return df
"""
(USERDIR / "strategies" / "AthenaStrategy.py").write_text(STRATEGY)

CONFIG = {
    "strategy": "AthenaStrategy",
    "strategy_path": str(USERDIR / "strategies"),
    "timeframe": "1h",
    "pairs": ["BTC/USDT:USDT"],
    "stake_currency": "USDT",
    "stake_amount": "unlimited",
    "tradable_balance_ratio": 0.99,
    "fiat_display_currency": "USD",
    "dry_run": True,
    "dry_run_wallet": 50.0,
    "max_open_trades": 1,
    "cancel_open_orders_on_exit": True,
    "amend_last_stake_amount": False,
    "position_adjustment_enable": False,
    "max_entry_position_adjustment": 0,
    "use_exit_signal": True,
    "exit_profit_only": False,
    "ignore_roi_if_entry_signal": False,
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
        "password": "",
        "ccxt_config": {"enableRateLimit": True},
        "ccxt_async_config": {"enableRateLimit": True},
        "pair_whitelist": ["BTC/USDT:USDT"],
        "pair_blacklist": [],
        "sandbox": False,
    },
    "pairlists": [{"method": "StaticPairList"}],
    "datadir": str(USERDIR / "data"),
    "timerange": "20260501-20260601",
    "fee": 0.001,
    "trading_mode": "futures",
    "margin_mode": "cross",
    "dataformat_ohlcv": "feather",
    "dataformat_trades": "feather",
}
(USERDIR / "config.json").write_text(json.dumps(CONFIG, indent=2))

# Symlink data
BINANCE_DATA = Path("/tmp/athena_test_1h/data/binance")
if (BINANCE_DATA / "futures").exists():
    shutil.copytree(BINANCE_DATA / "futures", USERDIR / "data" / "binance" / "futures")

print("Running freqtrade backtesting...")
result = subprocess.run(
    ["python", "-m", "freqtrade", "backtesting", "--userdir", str(USERDIR), "--config", str(USERDIR / "config.json"), "--timerange", "20240101-20240201"],
    capture_output=True,
    text=True,
)
print(result.stdout[-4000:] if len(result.stdout) > 4000 else result.stdout)
if result.stderr:
    print("STDERR:", result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
print(f"Return code: {result.returncode}")
