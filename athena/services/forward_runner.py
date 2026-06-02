"""ForwardRunner — iteratively paper-trade a strategy on one or many real market bar data.

Architecture
------------
1. Compile strategy code from DB record.
2. Load real candle DataFrames via FreqtradeWrapper.load_cached_candles().
3. Instantiate strategy with mock Freqtrade metadata.
4. Loop bar-by-bar, calling strategy.signal_entry()` and strategy.signal_exit()`.
   (If the strategy does not expose those helpers, fall back to evaluating
   `enter_long` / `exit_long` columns on the *current* bar.)
5. Emit calls to DryRunTrader.on_signal() — never hits a real exchange.
6. After each trade, check kill-switch / daily-loss via PortfolioManager.

No real orders are submitted. All signals are persisted for later analysis.
"""

import logging
import warnings
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd

from athena.common.models import StrategyRecord, StrategyStatus
from athena.core.freqtrade_wrapper import FreqtradeWrapper
from athena.services.bridge import DryRunTrader

logger = logging.getLogger(__name__)


class BasketRecord:
    """Lightweight per-pair state container."""

    def __init__(self, pair: str):
        self.pair = pair
        self.open_position: Optional[str] = None
        self.bars = 0
        self.regime: Optional[str] = None


class ForwardRunner:
    """Dry-run forward testing loop for a single strategy across multiple pairs."""

    # ── helpers ──────────────────────────────────────────────────

    @staticmethod
    def _strategy_has_signal_methods(strategy_obj) -> Tuple[bool, bool]:
        """Return (has_entry, has_exit) booleans."""
        return (
            hasattr(strategy_obj, "signal_entry") and callable(getattr(strategy_obj, "signal_entry")),
            hasattr(strategy_obj, "signal_exit") and callable(getattr(strategy_obj, "signal_exit")),
        )

    @staticmethod
    def _entry_signal_from_df(strategy, df: pd.DataFrame, pair: str) -> Optional[str]:
        """Evaluate the *last* row of a DataFrame produced by populate_*_trend.

        Returns 'long', 'short', or None.
        """
        if df.empty or "close" not in df.columns:
            return None
        row = df.iloc[-1]
        if "enter_long" in row and row["enter_long"] == 1:
            return "long"
        if "enter_short" in row and row["enter_short"] == 1:
            return "short"
        if "buy" in row and row["buy"] == 1:
            return "long"
        if "sell" in row and row["sell"] == 1:
            return "short"
        return None

    @staticmethod
    def _exit_signal_from_df(strategy, df: pd.DataFrame, pair: str) -> bool:
        """Evaluate the last row for exit signals."""
        if df.empty or "close" not in df.columns:
            return False
        row = df.iloc[-1]
        if "exit_long" in row and row["exit_long"] == 1:
            return True
        if "exit_short" in row and row["exit_short"] == 1:
            return True
        if "sell" in row and row["sell"] == 1:
            return True
        return False

    # ── public API ─────────────────────────────────────────────────

    def run(
        self,
        strategy_record: StrategyRecord,
        trader: DryRunTrader,
        *,
        pairs: List[str] = None,
        timeframe: str = "1h",
        warmup_bars: int = 200,
    ) -> Tuple[StrategyRecord, int]:
        """Execute forward test across multiple pairs.

        Returns
        -------
        (updated_record, total_bars_processed)
        """
        pairs = pairs or ["BTC/USDT:USDT"]
        if trader.is_killed():
            logger.warning("Trader already killed — skipping forward run")
            return strategy_record, 0

        # 1. Compile strategy code
        code = FreqtradeWrapper.compile_strategy(strategy_record)

        # 2. Load all pair dataframes
        wrapper = FreqtradeWrapper()
        dfs: Dict[str, pd.DataFrame] = {}
        for pair in pairs:
            df = wrapper.load_cached_candles(pair=pair, timeframe=timeframe)
            if df is None or df.empty:
                logger.warning("No candle data for %s %s", pair, timeframe)
                continue
            dfs[pair] = df
            logger.info("forward_runner: loaded %s bars for %s (%s)", len(df), pair, timeframe)

        if not dfs:
            raise RuntimeError(f"No candle data available for any pair")

        # 3. Resolve strategy class
        namespace: dict = {}
        exec(compile(code, "<GeneratedStrategy>", "exec"), namespace)
        StrategyClass = namespace["AthenaStrategy"]
        strategy = StrategyClass()
        strategy.dp = None

        has_entry, has_exit = self._strategy_has_signal_methods(strategy)
        logger.info("forward_runner: signal_entry=%s signal_exit=%s", has_entry, has_exit)

        # 4. Per-pair state
        baskets: Dict[str, BasketRecord] = {p: BasketRecord(p) for p in dfs}
        total_bars = 0

        # 5. Walk forward — simplest approach: iterate bars of the shortest series
        min_len = min(len(df) for df in dfs.values())
        for i in range(warmup_bars, min_len):
            if trader.is_killed():
                break

            for pair, df in dfs.items():
                basket = baskets[pair]
                window = df.iloc[i - warmup_bars : i + 1].copy()
                close = float(window.iloc[-1]["close"])

                basket.bars += 1
                total_bars += 1

                entry = None
                exiting = False

                if has_entry:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        raw = strategy.signal_entry(window, {"pair": pair})
                        entry = raw if raw in ("long", "short", None) else None
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        exiting = bool(strategy.signal_exit(window, {"pair": pair}))
                else:
                    try:
                        df_ind = strategy.populate_indicators(window.copy(), {"pair": pair})
                        df_ind = strategy.populate_entry_trend(df_ind, {"pair": pair})
                        df_ind = strategy.populate_exit_trend(df_ind, {"pair": pair})
                        entry = self._entry_signal_from_df(strategy, df_ind, pair)
                        exiting = self._exit_signal_from_df(strategy, df_ind, pair)
                    except Exception:
                        entry = None
                        exiting = False

                # --- manage position ---
                if basket.open_position is not None and exiting:
                    opposite = {"long": "short", "short": "long"}[basket.open_position]
                    trader.on_signal(
                        strategy_id=strategy_record.id,
                        symbol=pair,
                        signal=opposite,
                        close=close,
                    )
                    basket.open_position = None

                if basket.open_position is None and entry is not None:
                    trader.on_signal(
                        strategy_id=strategy_record.id,
                        symbol=pair,
                        signal=entry,
                        close=close,
                    )
                    basket.open_position = entry

        # 6. Close dangling positions at final close
        if not trader.is_killed():
            for pair, basket in baskets.items():
                if basket.open_position is not None:
                    opposite = {"long": "short", "short": "long"}[basket.open_position]
                    df = dfs[pair]
                    trader.on_signal(
                        strategy_id=strategy_record.id,
                        symbol=pair,
                        signal=opposite,
                        close=float(df.iloc[-1]["close"]),
                    )

        logger.info("forward_runner: finished %s bars, trader k=%s", total_bars, trader.is_killed())
        return strategy_record, total_bars


# ── convenience factory ──────────────────────────────────────────

def run_forward(
    strategy_id: str,
    *,
    pairs: Optional[List[str]] = None,
    timeframe: str = "1h",
    dry_run: bool = True,
) -> Tuple[StrategyRecord, dict]:
    """High-level factory — loads strategy from DB, instantiates DryRunTrader, runs."""
    if not dry_run:
        raise ValueError("run_forward only supports dry_run=True")
    if pairs is None:
        pairs = ["BTC/USDT:USDT"]

    from athena.services.models import get_session, StrategyModel
    session = get_session()
    row = session.query(StrategyModel).filter_by(id=strategy_id).first()
    if not row:
        raise ValueError(f"Strategy {strategy_id} not found")

    from athena.common.models import StrategyRecord, StrategyTemplate
    record = StrategyRecord(
        id=row.id,
        name=row.name,
        template=StrategyTemplate(row.template),
        dna=row.dna,
        generation=row.generation or 0,
        parent_id=row.parent_id,
    )

    from athena.portfolio.manager import PortfolioManager
    from athena.common.models import PortfolioConfig
    portfolio = PortfolioManager(PortfolioConfig(total_capital=10_000))
    trader = DryRunTrader(portfolio=portfolio)

    runner = ForwardRunner()
    _, bars = runner.run(record, trader, pairs=pairs, timeframe=timeframe)

    summary = trader.to_dict()
    summary["bars"] = bars
    summary["strategy_id"] = strategy_id
    summary["killed"] = trader.is_killed()
    return record, summary
