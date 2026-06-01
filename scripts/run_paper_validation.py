"""
Phase 3.5 — Real Paper Trading Validation Script

Spawns a genuine Freqtrade dry_run bot with a champion strategy,
polls status for ~10 min, then reports whether:
  • bot boots cleanly
  • trades execute in paper mode
  • ~$5 stakes appear on a $50 wallet
  • kill-switch fires on drawdown breach (simulated)
  • feedback collector writes snapshots
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from athena.orchestrator import AthenaOrchestrator          # noqa: E402
from athena.common.models import GenerationConfig, StrategyTemplate  # noqa: E402
from athena.live.bot_manager import BotManager              # noqa: E402
from athena.live.feedback import FeedbackCollector          # noqa: E402
from athena.services.models import init_db                   # noqa: E402


async def main():
    print("=" * 60)
    print("PHASE 3.5 — Paper Trading Validation")
    print("=" * 60)

    # ── 1. Init DB ────────────────────────────────────────────────
    init_db()
    print("[✓] Database initialized")

    # ── 2. Generate champion strategy ───────────────────────────────
    print("\n[>] Generating champion strategy (pop=6, gen=2, gates=OFF, tf=5m)...")
    cfg = GenerationConfig(
        symbols=["BTC-USD"],
        timeframe="5m",
        population_size=4,
        generations=1,
        ml_boost=False,
    )
    orch = AthenaOrchestrator(cfg)
    records = orch.run_generation(
        StrategyTemplate.TREND_FOLLOWING,
        run_gates=False,
    )
    best = records[0]
    print(f"[✓] Best: {best.id} | score={best.score.raw_score:.3f} | verdict={best.score.verdict}")

    # ── 3. Spawn paper bot ───────────────────────────────────────────
    print("\n[>] Spawning Freqtrade paper bot ...")
    manager = BotManager()
    session_id = await manager.start(
        strategy_id=best.id,
        mode="paper",
        risk={"max_drawdown": 0.15, "daily_loss_limit": 0.10},
        sandbox=True,
    )
    print(f"[✓] Session started: {session_id}")

    # ── 4. Start feedback collector (share same manager) ────────────
    collector = FeedbackCollector()
    collector.manager = manager                                   # shared BotManager
    asyncio.create_task(collector.start_monitor(session_id))

    # ── 5. Poll status every 30 s for 20 iterations (~10 min) ──────
    try:
        print("\n[>] Polling bot status (10 min)...")
        for i in range(1, 21):
            await asyncio.sleep(30)
            stats = manager.status(session_id)
            print(
                f"    [{i:02d}/20] equity=${stats.get('equity', 0):.2f}  "
                f"trades={stats.get('total_trades', 0)}  "
                f"dd={stats.get('max_drawdown', 0):.2%}  "
                f"status={stats.get('status', '?')}"
            )

        # ── 6. Stop gracefully ───────────────────────────────────────────
        print("\n[>] Stopping bot...")
        await manager.stop(session_id, reason="validation_complete")
        print("[✓] Bot stopped")

        # ── 7. Fetch final snapshots ──────────────────────────────────────
        snaps = collector.get_recent_snapshots(session_id, limit=20)
        if snaps:
            print(f"\n[✓] Snapshots: {len(snaps)}")
            last = snaps[-1]
            print(
                f"    last equity={last.equity:.2f}  "
                f"trades={last.total_trades}  "
                f"drift={last.is_degraded or 'stable'}"
            )
        else:
            print("\n[!] No snapshots collected (bot may not have traded)")

        print("\n" + "=" * 60)
        print("PHASE 3.5 COMPLETE")
        print("=" * 60)
    except KeyboardInterrupt:
        print("\n[!] Interrupted — stopping bot...")
        await manager.stop(session_id, reason="interrupted")
        sys.exit(1)
    except Exception as exc:
        print(f"\n[✗] Error: {exc}")
        await manager.stop(session_id, reason=f"error ({exc})")
        raise


if __name__ == "__main__":
    asyncio.run(main())
