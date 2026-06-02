"""Post-GA Hyperopt finisher — fine-tune generated strategy parameters via Freqtrade's native Hyperopt engine.

Flow:
1. Deploy a promoted strategy to a temporary user_data directory (config + data).
2. Run `freqtrade hyperopt` CLI programmatically.
3. Parse the `.fthypt` newline-delimited JSON results file to find the best epoch.
4. Merge optimized param values back into the strategy DNA.
5. Persist updated DNA and re-evaluate.
"""
import json
import logging
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from athena.core.freqtrade_wrapper import FreqtradeWrapper
from athena.common.models import StrategyRecord

logger = logging.getLogger(__name__)


class HyperoptFinisher:
    """Fine-tune IntParameter / RealParameter ranges via Freqtrade Hyperopt CLI."""

    DEFAULT_LOSSES = [
        "OnlyProfitHyperOptLoss",
        "SharpeHyperOptLoss",
        "SortinoHyperOptLoss",
    ]

    def __init__(
        self,
        epochs: int = 30,
        spaces: str = "buy sell",
        loss_function: Optional[str] = None,
        parallel_jobs: int = -1,
        timerange: Optional[str] = None,
    ):
        self.epochs = epochs
        self.spaces = spaces
        self.loss_function = loss_function or self.DEFAULT_LOSSES[0]
        self.parallel_jobs = parallel_jobs
        self.timerange = timerange  # e.g. "20260201-20260601"

    # ── main entry ─────────────────────────────────────────────────────

    def run(self, record: StrategyRecord, deploy_dir: Optional[Path] = None) -> StrategyRecord:
        """Run hyperopt on a deployed strategy and update DNA with optimized params."""
        tmp = deploy_dir or self._deploy(record)
        try:
            result = self._hyperopt(tmp, record.id)
            if result["status"] == "ok":
                record = self._merge_params(record, result["params"])
                record.metadata = record.metadata or {}
                record.metadata["hyperopt"] = {
                    "ran_at": datetime.now(timezone.utc).isoformat(),
                    "epochs": self.epochs,
                    "best_loss": result.get("loss"),
                    "loss_function": self.loss_function,
                    "raw_params": result["params"],
                }
                logger.info(
                    "Hyperopt complete for %s: best_loss=%s params=%s",
                    record.id,
                    result.get("loss"),
                    result["params"],
                )
            else:
                logger.warning(
                    "Hyperopt failed for %s: %s", record.id, result.get("error")
                )
        finally:
            if deploy_dir is None:
                shutil.rmtree(tmp, ignore_errors=True)
        return record

    # ── internal ─────────────────────────────────────────────────────

    def _deploy(self, record: StrategyRecord) -> Path:
        """Write strategy + minimal data dir for hyperopt inside a Freqtrade user_data layout."""
        tmp = Path(tempfile.mkdtemp(prefix="athena_ho_"))
        user_data = tmp / "user_data"
        user_data.mkdir(parents=True, exist_ok=True)

        # Strategy
        strat_dir = user_data / "strategies"
        strat_dir.mkdir(parents=True, exist_ok=True)
        (strat_dir / "__init__.py").write_text("")
        (strat_dir / "AthenaStrategy.py").write_text(
            FreqtradeWrapper.compile_strategy(record)
        )

        # Data dir + symlink shared cache so hyperopt has warm data
        data_dir = user_data / "data" / "binance" / "futures"
        data_dir.mkdir(parents=True, exist_ok=True)
        pair = _to_ft_pair(record.dna.vector.get("symbol", "BTC-USD"))
        timeframe = record.dna.vector.get("timeframe", "1h")
        key = f"{_pair_to_key(pair)}-{timeframe}-futures.feather"
        cache_feather = Path("/tmp/athena_shared_data/data/binance/futures") / key
        if cache_feather.exists():
            shutil.copy2(cache_feather, data_dir / key)

        # Minimal config
        cfg = {
            "strategy": "AthenaStrategy",
            "strategy_path": str(strat_dir),
            "timeframe": timeframe,
            "pairs": [pair],
            "stake_currency": "USDT",
            "stake_amount": "unlimited",
            "tradable_balance_ratio": 1.0,
            "fiat_display_currency": "USD",
            "dry_run": True,
            "dry_run_wallet": 500.0,
            "max_open_trades": 1,
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
            "exchange": {
                "name": "binance",
                "key": "",
                "secret": "",
                "password": "",
                "ccxt_config": {"enableRateLimit": True},
                "pair_whitelist": [pair],
                "pair_blacklist": [],
                "sandbox": False,
            },
            "pairlists": [{"method": "StaticPairList"}],
            "fee": 0.001,
            "trading_mode": "futures",
            "margin_mode": "cross",
            "dataformat_ohlcv": "feather",
            "datadir": str(user_data / "data" / "binance"),
        }
        (tmp / "config.json").write_text(json.dumps(cfg, indent=2))
        return tmp

    def _hyperopt(self, deploy_dir: Path, strategy_id: str) -> Dict[str, Any]:
        """Execute freqtrade hyperopt CLI and extract best parameters from .fthypt."""
        config_path = deploy_dir / "config.json"
        strat_dir = deploy_dir / "user_data" / "strategies"

        cmd = [
            "python", "-m", "freqtrade", "hyperopt",
            "--config", str(config_path),
            "--userdir", str(deploy_dir / "user_data"),
            "--strategy", "AthenaStrategy",
            "--strategy-path", str(strat_dir),
            "-e", str(self.epochs),
            "--spaces", *self.spaces.split(),
            "--hyperopt-loss", self.loss_function,
            "-j", str(self.parallel_jobs),
            "--disable-param-export",
            "--print-all",
        ]
        if self.timerange:
            cmd += ["--timerange", self.timerange]

        logger.info("Running hyperopt for %s: %s", strategy_id, " ".join(cmd))
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=900,
        )

        if proc.returncode != 0:
            return {
                "status": "error",
                "error": f"Exit {proc.returncode}: {proc.stderr[:1200]}",
            }

        # Find the generated .fthypt file
        results_dir = deploy_dir / "user_data" / "hyperopt_results"
        fthypt_files = sorted(
            results_dir.glob("*.fthypt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not fthypt_files:
            return {"status": "error", "error": "No .fthypt results file found"}

        best = self._parse_best_epoch(fthypt_files[0])
        if not best:
            return {"status": "error", "error": "Could not parse best epoch from .fthypt"}

        best["status"] = "ok"
        return best

    # ── parsing ───────────────────────────────────────────────────────

    @staticmethod
    def _parse_best_epoch(fthypt_path: Path) -> Optional[Dict[str, Any]]:
        """Read newline-delimited JSON and pick the epoch with minimum loss."""
        epochs: List[Dict[str, Any]] = []
        for line in fthypt_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                epochs.append(json.loads(line))
            except json.JSONDecodeError:
                continue

        if not epochs:
            return None

        # Filter out epochs with loss == 100000 (Freqtrade's "no trades" sentinel)
        valid = [
            e for e in epochs
            if e.get("loss") is not None and e.get("loss") != 100000
        ]
        if not valid:
            # Fall back to best of all epochs even if sentinel
            valid = epochs

        best = min(valid, key=lambda e: e.get("loss", float("inf")))
        params = best.get("params_dict", {})
        return {
            "params": params,
            "loss": best.get("loss"),
            "metrics": best.get("results_metrics", {}),
        }

    @staticmethod
    def _merge_params(record: StrategyRecord, params: Dict[str, Any]) -> StrategyRecord:
        """Merge optimized param dict back into record.dna.vector."""
        for key, val in params.items():
            if key in record.dna.vector:
                old = record.dna.vector[key]
                record.dna.vector[key] = val
                logger.info("Hyperopt param update: %s %s -> %s", key, old, val)
        return record

    @staticmethod
    def expand_ranges(record: StrategyRecord, factor: float = 0.20) -> StrategyRecord:
        """Widen DNA spec bounds around optimized values so future GA explores neighbors.

        factor: fraction of current value to add/subtract from min/max.
                e.g. 0.20 means new range = [val * 0.8, val * 1.2]
        """
        if not record.dna.spec:
            # Load spec from template if missing
            from athena.generator.templates import TEMPLATE_SPECS
            record.dna.spec = list(TEMPLATE_SPECS.get(record.dna.template, []))
        for s in record.dna.spec:
            if s.type not in ("int", "float"):
                continue
            val = record.dna.vector.get(s.name)
            if val is None:
                continue
            span = max(abs(val) * factor, 1 if s.type == "int" else 0.01)
            if s.min is not None:
                s.min = min(s.min, val - span)
            if s.max is not None:
                s.max = max(s.max, val + span)
            # Clamp to reasonable floors
            if s.type == "int":
                s.min = int(max(s.min, 2))
                s.max = int(max(s.max, s.min + 1))
            logger.info(
                "Expanded range for %s: %s -> [%s, %s]",
                s.name, val, s.min, s.max,
            )
        return record


# ── util ──────────────────────────────────────────

def _to_ft_pair(symbol: str) -> str:
    pair = symbol.replace("-", "/")
    if pair.endswith("USD") and not pair.endswith("USDT"):
        pair = pair + "T"
    if not pair.endswith(":USDT"):
        pair = pair + ":USDT"
    return pair


def _pair_to_key(pair: str) -> str:
    return pair.replace("/", "_").replace(":", "_")
