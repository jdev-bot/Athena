"""Core Freqtrade integration — backtesting via programmatic Freqtrade API with real market data."""
import os
import json
import tempfile
import shutil
from pathlib import Path
from typing import Any, Dict
from datetime import datetime

import pandas as pd
import numpy as np

from freqtrade.resolvers import StrategyResolver
from freqtrade.optimize.backtesting import Backtesting
from freqtrade.resolvers import ExchangeResolver
from freqtrade.configuration.configuration import Configuration
from freqtrade.configuration.config_validation import validate_config_consistency
from freqtrade.enums import RunMode, TradingMode, MarginMode
from freqtrade.data.history.history_utils import load_data
from freqtrade.configuration import TimeRange

from athena.common.config import config as athena_config
from athena.generator.dna import DNAEncoder
from athena.generator.templates import TEMPLATE_MAP, TEMPLATE_SPECS
from athena.common.models import StrategyTemplate
from athena.market.provider import MarketDataProvider


# ── helpers ─────────────────────────────────────────────────────
def _to_ft_pair(symbol: str) -> str:
    """BTC-USD → BTC/USDT for freqtrade."""
    pair = symbol.replace("-", "/")
    if pair.endswith("USD") and not pair.endswith("USDT"):
        pair = pair + "T"
    return pair


def _pair_to_key(pair: str) -> str:
    """BTC/USDT → BTC_USDT for filenames."""
    return pair.replace("/", "_")


class FreqtradeWrapper:
    """Wraps freqtrade for programmatic backtesting."""

    def __init__(self):
        self._tmpdir: Path | None = None

    # ── temp project scaffolding ──────────────────────────────────
    def _setup_project(self, strategy_code: str, pair: str, timeframe: str) -> Path:
        """Create a minimal freqtrade user_data directory with strategy + data."""
        tmpdir = Path(tempfile.mkdtemp(prefix="athena_ft_"))
        self._tmpdir = tmpdir

        # Strategy
        strat_dir = tmpdir / "strategies"
        strat_dir.mkdir(parents=True, exist_ok=True)
        (strat_dir / "__init__.py").write_text("")
        (strat_dir / "AthenaStrategy.py").write_text(strategy_code)

        # Data dir
        data_dir = tmpdir / "data" / "binance"
        data_dir.mkdir(parents=True, exist_ok=True)

        # Config
        cfg = {
            "strategy": "AthenaStrategy",
            "strategy_path": str(strat_dir),
            "timeframe": timeframe,
            "timeframe_detail": None,
            "pairs": [pair],
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
                "pair_whitelist": [pair],
                "pair_blacklist": [],
                "sandbox": False,
            },
            "pairlists": [{"method": "StaticPairList"}],
            "datadir": str(data_dir),
            "timerange": None,
            "fee": 0.001,
            "trading_mode": "futures",
            "margin_mode": "cross",
            "dataformat_ohlcv": "feather",
            "dataformat_trades": "feather",
            "enable_protections": False,
        }
        (tmpdir / "config.json").write_text(json.dumps(cfg, indent=2))
        return tmpdir

    # ── data preparation ────────────────────────────────────────────
    def _write_candle_data(
        self,
        data_dir: Path,
        pair: str,
        timeframe: str,
        start_date: str,
        end_date: str,
    ):
        """Fetch real candles via ccxt and write freqtrade feather files."""
        provider = MarketDataProvider(exchange_name="binance")
        ccxt_symbol = pair  # already BTC/USDT form

        provider.fetch_and_cache(
            symbol=ccxt_symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
        )
        candles = provider.load_cached(symbol=ccxt_symbol, timeframe=timeframe)

        # Convert to DataFrame with freqtrade column names
        df = pd.DataFrame(candles, columns=["date", "open", "high", "low", "close", "volume"])
        df["date"] = pd.to_datetime(df["date"], unit="ms", utc=True)
        df = df.set_index("date", drop=False)
        df = df.sort_index()

        # Write to freqtrade expected path: datadir/{timeframe}/{pair_key}.feather
        tf_dir = data_dir / timeframe
        tf_dir.mkdir(parents=True, exist_ok=True)
        pair_key = _pair_to_key(pair)
        feather_path = tf_dir / f"{pair_key}.feather"
        df.to_feather(feather_path)

    # ── backtest ──────────────────────────────────────────────────
    def run_backtest(
        self,
        strategy_code: str,
        start_date: str = "2024-01-01",
        end_date: str = "2024-02-01",
        exchange: str = "binance",
        symbol: str = "BTC-USD",
        timeframe: str = "1h",
    ) -> Dict[str, Any]:
        """Run a freqtrade backtest with real ccxt-sourced market data."""
        pair = _to_ft_pair(symbol)
        tmp = self._setup_project(strategy_code, pair, timeframe)

        try:
            self._write_candle_data(
                tmp / "data" / "binance",
                pair, timeframe, start_date, end_date,
            )

            # Load config
            config_path = tmp / "config.json"
            cfg = Configuration.from_files([str(config_path)])
            cfg["user_data_dir"] = str(tmp)

            # Ensure timerange matches data dates
            timerange = TimeRange.parse_timerange(
                f"{start_date.replace('-', '')}-{end_date.replace('-', '')}"
            )
            cfg["timerange"] = timerange

            # Run mode
            cfg["runmode"] = RunMode.BACKTEST
            cfg["dry_run"] = True

            # Load exchange
            exchange_instance = ExchangeResolver.load_exchange(cfg, load_leverage_tiers=True)

            # Backtest
            bt = Backtesting(cfg, exchange=exchange_instance)
            bt.start()

            # Extract metrics from bt.results
            results = bt.results
            if not results or "strategy" not in results:
                return self._empty_metrics()

            strategy_results = results["strategy"].get("AthenaStrategy", {})
            if not strategy_results:
                return self._empty_metrics()

            return {
                "total_return": strategy_results.get("profit_total_pct", 0.0),
                "sharpe": strategy_results.get("sharpe", 0.0),
                "sortino": strategy_results.get("sortino", 0.0),
                "calmar": strategy_results.get("calmar", 0.0),
                "win_rate": strategy_results.get("winrate", 0.0),
                "max_drawdown": abs(strategy_results.get("max_drawdown", 0.0) or 0.0),
                "total_trades": strategy_results.get("total_trades", 0),
                "avg_trade": strategy_results.get("avg_profit_pct", 0.0) or 0.0,
                "profit_factor": strategy_results.get("profit_factor", 0.0) or 0.0,
            }
        except Exception as exc:
            return {
                "error": str(exc),
                **self._empty_metrics(),
            }
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
            self._tmpdir = None

    @staticmethod
    def _empty_metrics() -> Dict[str, Any]:
        return {
            "total_return": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "calmar": 0.0,
            "win_rate": 0.0,
            "max_drawdown": 0.0,
            "total_trades": 0,
            "avg_trade": 0.0,
            "profit_factor": 0.0,
        }

    @staticmethod
    def compile_strategy(record) -> str:
        """Render Freqtrade strategy source from a DB record."""
        encoder = DNAEncoder()
        params = encoder.to_strategy_params(record.dna, StrategyTemplate(record.template))
        params["class_name"] = "AthenaStrategy"
        params["template_name"] = StrategyTemplate(record.template).value
        params["timeframe"] = getattr(record, "timeframe", "1h")
        template = TEMPLATE_MAP.get(StrategyTemplate(record.template))
        if not template:
            raise ValueError(f"Unknown template {record.template}")
        return template.format(**params)
