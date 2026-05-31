"""Core Jesse integration wrapper."""
import os
import sys
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional
import jesse
from jesse.config import config as jesse_config
from jesse.services import db
from athena.common.config import config


class JesseWrapper:
    """Wraps Jesse framework for programmatic use."""
    
    def __init__(self):
        self.initialized = False
        self.project_root = config.PROJECT_ROOT
        
    def init_jesse(self, exchange: str = "Sandbox", symbols: list = None, 
                   timeframe: str = "1h", start_date: str = "2024-01-01",
                   end_date: str = "2025-01-01") -> None:
        """Initialize Jesse with project config."""
        if self.initialized:
            return
            
        # Create temporary Jesse project structure
        self.temp_dir = tempfile.mkdtemp(prefix="athena_jesse_")
        self._setup_jesse_project(self.temp_dir, exchange, symbols, timeframe,
                                   start_date, end_date)
        
        # Change to project dir for Jesse
        os.chdir(self.temp_dir)
        
        # Initialize Jesse
        jesse_config.set_config(
            app_port=9001,
            debug_mode=False,
            logging=True,
            env='backtest',
            warm_up_candles=240,
            exchanges={
                exchange: {
                    'fee': 0.001,
                    'type': 'futures',
                    'balance': 10000,
                }
            },
            strategy_dir=str(Path(self.temp_dir) / 'strategies'),
        )
        
        # Initialize database
        db.open_connection()
        
        self.initialized = True
        self.exchange = exchange
        self.symbols = symbols or ["BTC-USD"]
        self.timeframe = timeframe
        
    def _setup_jesse_project(self, root: str, exchange: str, symbols: list,
                             timeframe: str, start_date: str, end_date: str) -> None:
        """Create minimal Jesse project structure."""
        root = Path(root)
        
        # Create directories
        (root / 'strategies').mkdir(parents=True, exist_ok=True)
        (root / 'storage').mkdir(parents=True, exist_ok=True)
        (root / 'logs').mkdir(parents=True, exist_ok=True)
        
        # Write config.py
        routes = []
        for symbol in symbols or ["BTC-USD"]:
            routes.append({
                'exchange': exchange,
                'symbol': symbol,
                'timeframe': timeframe,
                'strategy': 'AthenaStrategy',
            })
        
        config_py = f"""
from jesse.config import config

config['app']['trading_mode'] = 'backtest'
config['app']['debug_mode'] = False
config['env']['exchanges'] = {{
    '{exchange}': {{
        'fee': 0.001,
        'type': 'futures',
        'balance': 10000,
    }}
}}
config['env']['routes'] = {json.dumps(routes)}
config['env']['data']['start_date'] = '{start_date}'
config['env']['data']['end_date'] = '{end_date}'
"""
        (root / 'config.py').write_text(config_py)
        
        # Write requirements.txt
        (root / 'requirements.txt').write_text('jesse\n')
        
        # Create __init__.py for strategies package
        (root / 'strategies' / '__init__.py').write_text('')
        
    def run_backtest(self, strategy_code: str, strategy_name: str = "AthenaStrategy") -> Dict[str, Any]:
        """Run backtest for given strategy code."""
        if not self.initialized:
            raise RuntimeError("Jesse not initialized. Call init_jesse() first.")
        
        # Write strategy file
        strategy_path = Path(self.temp_dir) / 'strategies' / f'{strategy_name}.py'
        strategy_path.write_text(strategy_code)
        
        # Run backtest via Jesse
        from jesse.modes import backtest_mode
        
        try:
            result = backtest_mode.run(
                start_date=self.start_date if hasattr(self, 'start_date') else '2024-01-01',
                finish_date=self.end_date if hasattr(self, 'end_date') else '2025-01-01',
                chart=False,
                tradingview=False,
                csv=False,
                json=False,
            )
            
            # Extract metrics from result
            metrics = self._extract_metrics(result)
            return metrics
            
        except Exception as e:
            return {
                'error': str(e),
                'total_return': 0.0,
                'sharpe': 0.0,
                'max_drawdown': 0.0,
                'total_trades': 0,
            }
    
    def _extract_metrics(self, result: Any) -> Dict[str, Any]:
        """Extract performance metrics from Jesse result."""
        if result is None:
            return self._empty_metrics()
        
        try:
            # Jesse result is typically a dict or custom object
            if isinstance(result, dict):
                return {
                    'total_return': result.get('total_return', 0.0),
                    'sharpe': result.get('sharpe_ratio', 0.0),
                    'sortino': result.get('sortino_ratio', 0.0),
                    'calmar': result.get('calmar_ratio', 0.0),
                    'win_rate': result.get('winning_ratio', 0.0),
                    'max_drawdown': result.get('max_drawdown', 0.0),
                    'total_trades': result.get('total_trades', 0),
                    'avg_trade': result.get('average_trade', 0.0),
                    'profit_factor': result.get('profit_factor', 0.0),
                }
            else:
                # Handle Jesse's custom result object
                return {
                    'total_return': getattr(result, 'total_return', 0.0),
                    'sharpe': getattr(result, 'sharpe_ratio', 0.0),
                    'sortino': getattr(result, 'sortino_ratio', 0.0),
                    'calmar': getattr(result, 'calmar_ratio', 0.0),
                    'win_rate': getattr(result, 'winning_ratio', 0.0),
                    'max_drawdown': getattr(result, 'max_drawdown', 0.0),
                    'total_trades': getattr(result, 'total_trades', 0),
                    'avg_trade': getattr(result, 'average_trade', 0.0),
                    'profit_factor': getattr(result, 'profit_factor', 0.0),
                }
        except Exception:
            return self._empty_metrics()
    
    def _empty_metrics(self) -> Dict[str, Any]:
        return {
            'total_return': 0.0,
            'sharpe': 0.0,
            'sortino': 0.0,
            'calmar': 0.0,
            'win_rate': 0.0,
            'max_drawdown': 0.0,
            'total_trades': 0,
            'avg_trade': 0.0,
            'profit_factor': 0.0,
        }
    
    def cleanup(self) -> None:
        """Cleanup temporary files."""
        import shutil
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        self.initialized = False
