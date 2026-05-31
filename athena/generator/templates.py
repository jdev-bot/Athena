"""Generator templates for strategy generation."""
from typing import Dict, List, Any
from athena.common.models import DNASpec, StrategyTemplate


TREND_FOLLOWING_TEMPLATE = """
from jesse.strategies import Strategy
import jesse.indicators as ta
import jesse.helpers as jh

class {class_name}(Strategy):
    def __init__(self):
        super().__init__()
        self.fast_period = {fast_period}
        self.slow_period = {slow_period}
        self.trend_threshold = {trend_threshold}
        self.rsi_period = {rsi_period}
        self.rsi_overbought = {rsi_overbought}
        self.rsi_oversold = {rsi_oversold}
        self.position_size = {position_size}
        
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
"""

MEAN_REVERSION_TEMPLATE = """
from jesse.strategies import Strategy
import jesse.indicators as ta
import jesse.helpers as jh

class {class_name}(Strategy):
    def __init__(self):
        super().__init__()
        self.bb_period = {bb_period}
        self.bb_std = {bb_std}
        self.rsi_period = {rsi_period}
        self.rsi_overbought = {rsi_overbought}
        self.rsi_oversold = {rsi_oversold}
        self.mean_period = {mean_period}
        self.deviation_threshold = {deviation_threshold}
        self.position_size = {position_size}
        
    def should_long(self):
        sma = ta.sma(self.candles, self.mean_period)
        rsi = ta.rsi(self.candles, self.rsi_period)
        
        if sma is None or rsi is None:
            return False
            
        deviation = (sma - self.price) / sma
        
        return (self.price < sma and 
                rsi < self.rsi_oversold and
                deviation > self.deviation_threshold)
    
    def go_long(self):
        qty = jh.size_to_qty(self.position_size * self.balance, self.price)
        self.buy = qty, self.price
        self.take_profit = qty, self.price * 1.02
        self.stop_loss = qty, self.price * 0.98
    
    def should_short(self):
        sma = ta.sma(self.candles, self.mean_period)
        rsi = ta.rsi(self.candles, self.rsi_period)
        
        if sma is None or rsi is None:
            return False
            
        deviation = (self.price - sma) / sma
        
        return (self.price > sma and 
                rsi > self.rsi_overbought and
                deviation > self.deviation_threshold)
    
    def go_short(self):
        qty = jh.size_to_qty(self.position_size * self.balance, self.price)
        self.sell = qty, self.price
        self.take_profit = qty, self.price * 0.98
        self.stop_loss = qty, self.price * 1.02
    
    def should_cancel(self):
        return False
"""

BREAKOUT_TEMPLATE = """
from jesse.strategies import Strategy
import jesse.indicators as ta
import jesse.helpers as jh

class {class_name}(Strategy):
    def __init__(self):
        super().__init__()
        self.lookback = {lookback}
        self.atr_period = {atr_period}
        self.atr_multiplier = {atr_multiplier}
        self.volume_factor = {volume_factor}
        self.position_size = {position_size}
        
    def should_long(self):
        high = ta.ema(self.candles[:, 3], self.lookback)
        low = ta.ema(self.candles[:, 4], self.lookback)
        atr = ta.atr(self.candles, self.atr_period)
        
        if high is None or low is None or atr is None:
            return False
            
        breakout_level = high - atr * self.atr_multiplier
        
        return self.price > breakout_level and self.price > high * 0.999
    
    def go_long(self):
        qty = jh.size_to_qty(self.position_size * self.balance, self.price)
        self.buy = qty, self.price
        self.stop_loss = qty, self.price - atr * self.atr_multiplier * 2
    
    def should_short(self):
        high = ta.ema(self.candles[:, 3], self.lookback)
        low = ta.ema(self.candles[:, 4], self.lookback)
        atr = ta.atr(self.candles, self.atr_period)
        
        if high is None or low is None or atr is None:
            return False
            
        breakdown_level = low + atr * self.atr_multiplier
        
        return self.price < breakdown_level and self.price < low * 1.001
    
    def go_short(self):
        qty = jh.size_to_qty(self.position_size * self.balance, self.price)
        self.sell = qty, self.price
        self.stop_loss = qty, self.price + atr * self.atr_multiplier * 2
    
    def should_cancel(self):
        return False
"""

MOMENTUM_TEMPLATE = """
from jesse.strategies import Strategy
import jesse.indicators as ta
import jesse.helpers as jh

class {class_name}(Strategy):
    def __init__(self):
        super().__init__()
        self.momentum_period = {momentum_period}
        self.signal_period = {signal_period}
        self.momentum_threshold = {momentum_threshold}
        self.rsi_period = {rsi_period}
        self.position_size = {position_size}
        
    def should_long(self):
        macd_line, signal_line, _ = ta.macd(self.candles, fastperiod=12, slowperiod=26, signalperiod=9)
        rsi = ta.rsi(self.candles, self.rsi_period)
        
        if macd_line is None or signal_line is None or rsi is None:
            return False
            
        momentum = macd_line - signal_line
        return momentum > self.momentum_threshold and rsi < 70
    
    def go_long(self):
        qty = jh.size_to_qty(self.position_size * self.balance, self.price)
        self.buy = qty, self.price
    
    def should_short(self):
        macd_line, signal_line, _ = ta.macd(self.candles, fastperiod=12, slowperiod=26, signalperiod=9)
        rsi = ta.rsi(self.candles, self.rsi_period)
        
        if macd_line is None or signal_line is None or rsi is None:
            return False
            
        momentum = macd_line - signal_line
        return momentum < -self.momentum_threshold and rsi > 30
    
    def go_short(self):
        qty = jh.size_to_qty(self.position_size * self.balance, self.price)
        self.sell = qty, self.price
    
    def should_cancel(self):
        return False
"""

VOLATILITY_TEMPLATE = """
from jesse.strategies import Strategy
import jesse.indicators as ta
import jesse.helpers as jh

class {class_name}(Strategy):
    def __init__(self):
        super().__init__()
        self.atr_period = {atr_period}
        self.volatility_threshold = {volatility_threshold}
        self.position_size = {position_size}
        self.tp_multiplier = {tp_multiplier}
        self.sl_multiplier = {sl_multiplier}
        
    def should_long(self):
        atr = ta.atr(self.candles, self.atr_period)
        if atr is None:
            return False
            
        volatility = atr / self.price
        return volatility > self.volatility_threshold
    
    def go_long(self):
        qty = jh.size_to_qty(self.position_size * self.balance, self.price)
        self.buy = qty, self.price
        self.take_profit = qty, self.price * (1 + self.tp_multiplier * atr / self.price)
        self.stop_loss = qty, self.price * (1 - self.sl_multiplier * atr / self.price)
    
    def should_short(self):
        atr = ta.atr(self.candles, self.atr_period)
        if atr is None:
            return False
            
        volatility = atr / self.price
        return volatility > self.volatility_threshold
    
    def go_short(self):
        qty = jh.size_to_qty(self.position_size * self.balance, self.price)
        self.sell = qty, self.price
        self.take_profit = qty, self.price * (1 - self.tp_multiplier * atr / self.price)
        self.stop_loss = qty, self.price * (1 + self.sl_multiplier * atr / self.price)
    
    def should_cancel(self):
        return False
"""

TEMPLATE_MAP = {
    StrategyTemplate.TREND_FOLLOWING: TREND_FOLLOWING_TEMPLATE,
    StrategyTemplate.MEAN_REVERSION: MEAN_REVERSION_TEMPLATE,
    StrategyTemplate.BREAKOUT: BREAKOUT_TEMPLATE,
    StrategyTemplate.MOMENTUM: MOMENTUM_TEMPLATE,
    StrategyTemplate.VOLATILITY: VOLATILITY_TEMPLATE,
}

# DNA specifications for each template
TEMPLATE_SPECS: Dict[StrategyTemplate, List[DNASpec]] = {
    StrategyTemplate.TREND_FOLLOWING: [
        DNASpec(name="fast_period", type="int", min=5, max=50, default=8),
        DNASpec(name="slow_period", type="int", min=20, max=200, default=21),
        DNASpec(name="trend_threshold", type="float", min=0.001, max=0.05, default=0.01),
        DNASpec(name="rsi_period", type="int", min=5, max=30, default=14),
        DNASpec(name="rsi_overbought", type="int", min=60, max=90, default=70),
        DNASpec(name="rsi_oversold", type="int", min=10, max=40, default=30),
        DNASpec(name="position_size", type="float", min=0.01, max=0.5, default=0.05),
    ],
    StrategyTemplate.MEAN_REVERSION: [
        DNASpec(name="bb_period", type="int", min=10, max=50, default=20),
        DNASpec(name="bb_std", type="float", min=1.0, max=3.0, default=2.0),
        DNASpec(name="rsi_period", type="int", min=5, max=30, default=14),
        DNASpec(name="rsi_overbought", type="int", min=60, max=90, default=70),
        DNASpec(name="rsi_oversold", type="int", min=10, max=40, default=30),
        DNASpec(name="mean_period", type="int", min=10, max=100, default=50),
        DNASpec(name="deviation_threshold", type="float", min=0.001, max=0.05, default=0.01),
        DNASpec(name="position_size", type="float", min=0.01, max=0.5, default=0.05),
    ],
    StrategyTemplate.BREAKOUT: [
        DNASpec(name="lookback", type="int", min=10, max=100, default=20),
        DNASpec(name="atr_period", type="int", min=5, max=30, default=14),
        DNASpec(name="atr_multiplier", type="float", min=0.5, max=3.0, default=1.5),
        DNASpec(name="volume_factor", type="float", min=1.0, max=5.0, default=2.0),
        DNASpec(name="position_size", type="float", min=0.01, max=0.5, default=0.05),
    ],
    StrategyTemplate.MOMENTUM: [
        DNASpec(name="momentum_period", type="int", min=5, max=50, default=10),
        DNASpec(name="signal_period", type="int", min=5, max=30, default=9),
        DNASpec(name="momentum_threshold", type="float", min=0.1, max=5.0, default=1.0),
        DNASpec(name="rsi_period", type="int", min=5, max=30, default=14),
        DNASpec(name="position_size", type="float", min=0.01, max=0.5, default=0.05),
    ],
    StrategyTemplate.VOLATILITY: [
        DNASpec(name="atr_period", type="int", min=5, max=30, default=14),
        DNASpec(name="volatility_threshold", type="float", min=0.005, max=0.1, default=0.02),
        DNASpec(name="position_size", type="float", min=0.01, max=0.5, default=0.05),
        DNASpec(name="tp_multiplier", type="float", min=1.0, max=5.0, default=2.0),
        DNASpec(name="sl_multiplier", type="float", min=0.5, max=3.0, default=1.5),
    ],
}
