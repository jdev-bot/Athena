"""Core Jesse integration wrapper — uses jesse.research.backtest()."""
import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Any, Dict
import numpy as np

from jesse.research import backtest, fake_candle

from athena.common.config import config
from athena.generator.dna import DNAEncoder
from athena.generator.templates import TEMPLATE_MAP
from athena.common.models import StrategyTemplate


class JesseWrapper:
    """Wraps Jesse framework for programmatic backtesting."""

    def __init__(self):
        try:
            self._orig_dir = os.getcwd()
        except FileNotFoundError:
            self._orig_dir = os.path.dirname(os.path.abspath(__file__))

    def _temp_project(self, strategy_code: str) -> str:
        """Create a minimal temp Jesse project with the strategy."""
        tmp = tempfile.mkdtemp(prefix="athena_jesse_")
        tmp_path = Path(tmp)
        strategies_dir = tmp_path / "strategies"
        strategies_dir.mkdir(parents=True, exist_ok=True)
        (strategies_dir / "__init__.py").write_text("")
        strat_dir = strategies_dir / "AthenaStrategy"
        strat_dir.mkdir(parents=True, exist_ok=True)
        (strat_dir / "__init__.py").write_text(strategy_code)
        return tmp

    def _generate_1m_candles(self, n: int, exchange: str = "Sandbox",
                             symbol: str = "BTC-USD") -> tuple:
        """Generate n 1-minute fake candles in Jesse's expected format."""
        fake_candle({}, reset=True)
        candles = []
        for _ in range(n):
            candles.append(fake_candle({}))
        candles = np.array(candles, dtype=np.float64)
        warmup = candles[:240] if len(candles) > 240 else candles
        key = f"{exchange}-{symbol}"
        return {
            key: {"exchange": exchange, "symbol": symbol, "candles": candles}
        }, {
            key: {"exchange": exchange, "symbol": symbol, "candles": warmup}
        }

    def run_backtest(self, strategy_code: str, start_date: str = "2024-01-01",
                     end_date: str = "2024-02-01", exchange: str = "Sandbox",
                     symbol: str = "BTC-USD", timeframe: str = "1h") -> Dict[str, Any]:
        """Run an isolated Jesse backtest and return metrics."""
        tmp = self._temp_project(strategy_code)
        try:
            os.chdir(tmp)
            sys.path.insert(0, tmp)
            # Prevent Jesse from treating us as a unit test and looking
            # for strategies under the pytest package directory.
            old_pytest = os.environ.pop("PYTEST_CURRENT_TEST", None)
            for m in list(sys.modules.keys()):
                if m.startswith("strategies"):
                    del sys.modules[m]
            import importlib
            importlib.invalidate_caches()

            # Compute candle count (approx minutes in range)
            from datetime import datetime
            days = (datetime.strptime(end_date, "%Y-%m-%d") -
                    datetime.strptime(start_date, "%Y-%m-%d")).days
            n = max(days * 24 * 60, 240)

            candles, warmup = self._generate_1m_candles(n, exchange, symbol)

            result = backtest(
                config={
                    "starting_balance": 10_000,
                    "fee": 0.001,
                    "type": "futures",
                    "exchange": exchange,
                    "warm_up_candles": 240,
                    "futures_leverage": 1,
                    "futures_leverage_mode": "cross",
                },
                routes=[
                    {"exchange": exchange, "symbol": symbol,
                     "timeframe": timeframe, "strategy": "AthenaStrategy"}
                ],
                data_routes=[],
                candles=candles,
                warmup_candles=warmup,
            )

            if old_pytest is not None:
                os.environ["PYTEST_CURRENT_TEST"] = old_pytest

            metrics = result.get("metrics", {})
            return {
                "total_return": metrics.get("total", 0.0),
                "sharpe": metrics.get("sharpe_ratio", 0.0),
                "sortino": metrics.get("sortino_ratio", 0.0),
                "calmar": metrics.get("calmar_ratio", 0.0),
                "win_rate": metrics.get("winning_ratio", 0.0),
                "max_drawdown": metrics.get("max_drawdown", 0.0),
                "total_trades": metrics.get("total_trades", 0),
                "avg_trade": metrics.get("average_trade", 0.0),
                "profit_factor": metrics.get("profit_factor", 0.0),
            }
        except Exception as exc:
            return {
                "error": str(exc),
                "total_return": 0.0, "sharpe": 0.0, "sortino": 0.0,
                "calmar": 0.0, "win_rate": 0.0, "max_drawdown": 0.0,
                "total_trades": 0, "avg_trade": 0.0, "profit_factor": 0.0,
            }
        finally:
            os.chdir(self._orig_dir)
            if tmp in sys.path:
                sys.path.remove(tmp)
            shutil.rmtree(tmp, ignore_errors=True)

    @staticmethod
    def compile_strategy(record) -> str:
        """Render Jesse strategy source from a DB record / StrategyRecord."""
        encoder = DNAEncoder()
        params = encoder.to_strategy_params(record.dna, StrategyTemplate(record.template))
        params["class_name"] = "AthenaStrategy"
        template = TEMPLATE_MAP.get(StrategyTemplate(record.template))
        return template.format(**params)
