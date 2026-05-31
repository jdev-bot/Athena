
from jesse.strategies import Strategy
import jesse.indicators as ta
import jesse.helpers as jh

class Strategy_strat_cda8eba63f90(Strategy):
    def __init__(self):
        super().__init__()
        self.fast_period = 7
        self.slow_period = 83
        self.trend_threshold = 0.005328426061356533
        self.rsi_period = 9
        self.rsi_overbought = 76
        self.rsi_oversold = 25
        self.position_size = 0.3547384453702068
        
    def should_long(self):
        fast_ema = ta.ema(self.candles, self.fast_period)
        slow_ema = ta.ema(self.candles, self.slow_period)
        rsi = ta.rsi(self.candles, self.rsi_period)
        
        if fast_ema is None or slow_ema is None or rsi is None:
            return False
            
        trend_strength = abs(fast_ema - slow_ema) / slow_ema
        return (fast_ema > slow_ema and 
                trend_strength > self.trend_threshold and
                rsi < self.rsi_overbought)
    
    def go_long(self):
        qty = jh.size_to_qty(self.position_size * self.balance, self.price)
        self.buy = qty, self.price
        
    def should_short(self):
        fast_ema = ta.ema(self.candles, self.fast_period)
        slow_ema = ta.ema(self.candles, self.slow_period)
        rsi = ta.rsi(self.candles, self.rsi_period)
        
        if fast_ema is None or slow_ema is None or rsi is None:
            return False
            
        trend_strength = abs(fast_ema - slow_ema) / slow_ema
        return (fast_ema < slow_ema and 
                trend_strength > self.trend_threshold and
                rsi > self.rsi_oversold)
    
    def go_short(self):
        qty = jh.size_to_qty(self.position_size * self.balance, self.price)
        self.sell = qty, self.price
    
    def should_cancel(self):
        return False
