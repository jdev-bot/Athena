"""Strategy loader for dynamic strategy loading."""
import sys
import importlib.util
from pathlib import Path
from typing import Type, Optional


class StrategyLoader:
    """Load strategy classes from Python files."""
    
    def __init__(self, strategies_dir: Path):
        self.strategies_dir = strategies_dir
        self._cache = {}
        
    def load_strategy_class(self, name: str) -> Optional[Type]:
        """Load a strategy class by name from file."""
        if name in self._cache:
            return self._cache[name]
        
        file_path = self.strategies_dir / f"{name}.py"
        if not file_path.exists():
            return None
        
        spec = importlib.util.spec_from_file_location(name, file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        
        # Find the Strategy class
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (isinstance(attr, type) and 
                attr_name != 'Strategy' and 
                'Strategy' in attr_name):
                self._cache[name] = attr
                return attr
        
        return None
    
    def list_strategies(self) -> list:
        """List available strategy files."""
        return [f.stem for f in self.strategies_dir.glob('*.py') 
                if not f.name.startswith('_')]
