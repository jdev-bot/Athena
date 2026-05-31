
from jesse.strategies import Strategy
import jesse.indicators as ta
import jesse.helpers as jh

class Strategy_strat_a091079905b3(Strategy):
    def __init__(self):
        super().__init__()
        self.fast_period = 35
        self.slow_period = 140
        self.trend_threshold = 0.019456909340190072
        self.rsi_period = 5
        self.rsi_overbought = 69
        self.rsi_oversold = 31
        self.position_size = 0.2878151947961988
        
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
