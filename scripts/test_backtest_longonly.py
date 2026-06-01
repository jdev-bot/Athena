"""Test backtest — LONG ONLY to verify basic trade execution."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from athena.core.freqtrade_wrapper import FreqtradeWrapper

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
        # Enter long when price crosses above EMA
        df.loc[df['close'] > df['ema'], 'enter_long'] = 1
        return df

    def populate_exit_trend(self, df, md):
        # Exit long when price crosses below EMA
        df.loc[df['close'] < df['ema'], 'exit_long'] = 1
        return df
"""

wrapper = FreqtradeWrapper()
result = wrapper.run_backtest(
    strategy_code=STRATEGY,
    start_date="2024-01-01",
    end_date="2024-02-01",
    exchange="binance",
    symbol="BTC-USD",
    timeframe="1h",
)
print(json.dumps(result, indent=2))
