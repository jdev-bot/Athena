"""Test with a forced-entry strategy to confirm Freqtrade infrastructure works."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from athena.core.freqtrade_wrapper import FreqtradeWrapper

STRATEGY = """
import numpy as np, pandas as pd
from freqtrade.strategy.interface import IStrategy

class AthenaStrategy(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = '1h'
    stoploss = -0.50
    can_short = False
    startup_candle_count = 0

    def populate_indicators(self, df, md):
        return df

    def populate_entry_trend(self, df, md):
        # Enter long on EVERY candle
        df['enter_long'] = 1
        return df

    def populate_exit_trend(self, df, md):
        # Exit only on the very last candle
        df.loc[df.index[-1], 'exit_long'] = 1
        return df
"""

wrapper = FreqtradeWrapper()
result = wrapper.run_backtest(
    strategy_code=STRATEGY,
    start_date="2024-01-01",
    end_date="2024-01-15",  # shorter period
    exchange="binance",
    symbol="BTC-USD",
    timeframe="1h",
)
print(json.dumps(result, indent=2))
