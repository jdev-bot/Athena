"""Diagnostic: capture Freqtrade bot stdout/stderr on startup."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from athena.orchestrator import AthenaOrchestrator
from athena.common.models import GenerationConfig, StrategyTemplate
from athena.live.deployer import Deployer
from athena.live.bot_manager import BotManager
from athena.services.models import init_db


async def main():
    init_db()
    print("[✓] DB init")

    # generate
    cfg = GenerationConfig(
        symbols=["BTC-USD"], timeframe="5m",
        population_size=4, generations=1, ml_boost=False,
    )
    orch = AthenaOrchestrator(cfg)
    records = orch.run_generation(StrategyTemplate.TREND_FOLLOWING, run_gates=False)
    best = records[0]
    print(f"[✓] Best {best.id}  verdict={best.score.verdict}")

    # deploy manually so we can inspect dir
    deployer = Deployer()
    deploy_dir = deployer.deploy(
        strategy_id=best.id,
        mode="paper",
        risk={"max_drawdown": 0.15},
        sandbox=True,
    )
    print(f"[✓] Deploy dir: {deploy_dir}")
    print(f"    data files: {list((deploy_dir / 'data').rglob('*'))}")
    print(f"    strategy: {(deploy_dir / 'strategies' / 'AthenaStrategy.py').read_text()[:300]}")

    # start bot with captured output
    manager = BotManager()
    # monkey-patch Popen to capture output to files
    import subprocess
    orig_popen = subprocess.Popen
    out_path = deploy_dir / "bot_stdout.log"
    err_path = deploy_dir / "bot_stderr.log"

    def _popen(*args, **kwargs):
        kwargs["stdout"] = open(out_path, "w")
        kwargs["stderr"] = open(err_path, "w")
        return orig_popen(*args, **kwargs)

    subprocess.Popen = _popen
    try:
        sid = await manager.start(
            strategy_id=best.id,
            mode="paper",
            risk={"max_drawdown": 0.15},
            sandbox=True,
        )
        print(f"[✓] Session {sid}")
        await asyncio.sleep(15)
        print("\n--- stdout (last 80 lines) ---")
        print(out_path.read_text()[-4000:] if out_path.exists() else "empty")
        print("\n--- stderr (last 80 lines) ---")
        print(err_path.read_text()[-4000:] if err_path.exists() else "empty")
        await manager.stop(sid)
        print("[✓] Stopped")
    finally:
        subprocess.Popen = orig_popen


if __name__ == "__main__":
    asyncio.run(main())
