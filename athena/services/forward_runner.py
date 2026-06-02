"""ForwardRunner — iteratively paper-trade a single strategy on real market bar data.

Architecture
------------
1. Compile strategy code from DB record.
2. Load real candle DataFrame via FreqtradeWrapper.load_cached_candles().
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
from typing import Optional, Tuple

import pandas as pd

from athena.common.models import StrategyRecord, StrategyTemplate, StrategyStatus
from athena.core.freqtrade_wrapper import FreqtradeWrapper
from athena.services.bridge import DryRunTrader
from athena.services.models import get_session

logger = logging.getLogger(__name__)


class ForwardRunner:
    """Dry-run forward testing loop for a single strategy."""

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
        # Need at least one bar with indicator values.
        if df.empty or "close" not in df.columns:
            return None

        row = df.iloc[-1]

        # Standard Freqtrade column names
        if "enter_long" in row and row["enter_long"] == 1:
            return "long"
        if "enter_short" in row and row["enter_short"] == 1:
            return "short"

        # Fallback for strategies that use custom buy/sell columns
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
        pair: str = "BTC/USDT:USDT",
        timeframe: str = "1h",
        warmup_bars: int = 200,
    ) -> Tuple[StrategyRecord, int]:
        """Execute forward test.

        Returns
        -------
        (updated_record, bars_processed)
        """
        if trader.is_killed():
            logger.warning("Trader already killed — skipping forward run")
            return strategy_record, 0

        # 1. Compile strategy code
        code = FreqtradeWrapper.compile_strategy(strategy_record)

        # 2. Load real candle data
        wrapper = FreqtradeWrapper()
        df = wrapper.load_cached_candles(pair=pair, timeframe=timeframe)
        if df is None or df.empty:
            raise RuntimeError(f"No candle data available for {pair} {timeframe}")

        logger.info("forward_runner: loaded %s bars for %s (%s)", len(df), pair, timeframe)

        # 3. Resolve strategy class
        # We need an *instance* of the strategy.  StrategyResolver from freqtrade
        # expects a config dict with the class path.  Instead we simply exec the
        # code into a fresh namespace and pick up the AthenaStrategy class.
        namespace: dict = {}
        exec(compile(code, "<GeneratedStrategy>", "exec"), namespace)
        StrategyClass = namespace["AthenaStrategy"]
        strategy = StrategyClass()
        strategy.dp = None  # no dataprovider in dry-run

        has_entry, has_exit = self._strategy_has_signal_methods(strategy)
        logger.info("forward_runner: signal_entry=%s signal_exit=%s", has_entry, has_exit)

        # 4. Walk forward bar-by-bar
        bars = 0
        open_position: Optional[str] = None  # "long" | "short" | None

        for i in range(warmup_bars, len(df)):
            if trader.is_killed():
                break

            window = df.iloc[i - warmup_bars : i + 1].copy()
            close = float(window.iloc[-1]["close"])
            ts = window.index[-1]
            if hasattr(ts, "to_pydatetime"):
                ts = ts.to_pydatetime().replace(tzinfo=timezone.utc)
            else:
                ts = datetime.now(timezone.utc)

            bars += 1

            # -- evaluate entry / exit --
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
                # Populate indicators and trend columns on window slice so last row is current signal.
                # This is a *fallback* for regular populate_entry_trend / populate_exit_trend templates.
                try:
                    df_ind = strategy.populate_indicators(window.copy(), {"pair": pair})
                    df_ind = strategy.populate_entry_trend(df_ind, {"pair": pair})
                    df_ind = strategy.populate_exit_trend(df_ind, {"pair": pair})
                    entry = self._entry_signal_from_df(strategy, df_ind, pair)
                    exiting = self._exit_signal_from_df(strategy, df_ind, pair)
                except Exception:
                    entry = None
                    exiting = False

            # -- manage position --
            if open_position is not None and exiting:
                opposite = {"long": "short", "short": "long"}[open_position]
                trader.on_signal(
                    strategy_id=strategy_record.id,
                    symbol=pair,
                    signal=opposite,
                    close=close,
                )
                open_position = None

            if open_position is None and entry is not None:
                trader.on_signal(
                    strategy_id=strategy_record.id,
                    symbol=pair,
                    signal=entry,
                    close=close,
                )
                open_position = entry

        # 5. Close any dangling position at final close
        if open_position is not None and not trader.is_killed():
            opposite = {"long": "short", "short": "long"}[open_position]
            trader.on_signal(
                strategy_id=strategy_record.id,
                symbol=pair,
                signal=opposite,
                close=float(df.iloc[-1]["close"]),
            )

        logger.info("forward_runner: finished %s bars, trader k=%s", bars, trader.is_killed())
        return strategy_record, bars


# ── convenience factory ──────────────────────────────────────────

def run_forward(
    strategy_id: str,
    *,
    pair: str = "BTC/USDT:USDT",
    timeframe: str = "1h",
    dry_run: bool = True,
) -> Tuple[StrategyRecord, dict]:
    """High-level factory — loads strategy from DB, instantiates DryRunTrader, runs."""
    if not dry_run:
        raise ValueError("run_forward only supports dry_run=True")

    session = get_session()
    row = session.query(session.bind.engine.dialect.has_table("strategies"))
    # ^ dummy check, really just ensure tables exist

    from athena.services.models import StrategyModel
    row = session.query(StrategyModel).filter_by(id=strategy_id).first()
    if not row:
        raise ValueError(f"Strategy {strategy_id} not found")

    record = StrategyRecord(
        id=row.id,
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
    _, bars = runner.run(record, trader, pair=pair, timeframe=timeframe)

    summary = trader.to_dict()
    summary["bars"] = bars
    summary["strategy_id"] = strategy_id
    summary["killed"] = trader.is_killed()
    return record, summary

