"""Quick smoke test — verify bot boots + API responds with existing strategy.
Reuses latest generated strategy and pre-downloads data."""
import asyncio, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from athena.live.bot_manager import BotManager
from athena.live.deployer import Deployer
from athena.services.models import init_db, get_session, StrategyModel

async def main():
    init_db()
    print("[✓] DB init")

    # Pick latest strategy
    sess = get_session()
    row = sess.query(StrategyModel).order_by(StrategyModel.created_at.desc()).first()
    if not row:
        print("[!] No strategies found")
        return
    sid = row.id
    print(f"[→] Strategy {sid}  tf={row.dna.get('timeframe')}  sym={row.dna.get('symbol')}")
    sess.close()

    # Pre-download shared data
    shared = Path("/tmp/athena_shared_data")
    shared.mkdir(parents=True, exist_ok=True)
    cfg = {
        "exchange": {"name": "binance", "key": "", "secret": "", "ccxt_config": {"enableRateLimit": True}, "sandbox": True},
        "timeframe": "5m",
        "pairs": ["BTC/USDT:USDT"],
        "dataformat_ohlcv": "feather",
        "trading_mode": "futures",
        "margin_mode": "cross",
        "dry_run": True,
    }
    (shared / "config.json").write_text(json.dumps(cfg))

    from athena.live.data_downloader import download_pair_data
    download_pair_data(shared, "BTC/USDT:USDT", "5m", days=3, sandbox=True)
    print(f"[✓] Shared data pre-downloaded ({len(list(shared.rglob('*.feather')))} files)")

    # Deploy with data symlink
    deployer = Deployer()
    deploy_dir = deployer.deploy(sid, mode="paper", risk={"max_drawdown":0.15}, sandbox=True)
    print(f"[✓] Deploy dir: {deploy_dir}")

    # Symlink shared data into deploy
    deploy_data = deploy_dir / "data" / "binance"
    deploy_data.mkdir(parents=True, exist_ok=True)
    shared_data = shared / "data" / "binance" / "futures"
    if shared_data.exists():
        for f in shared_data.iterdir():
            dest = deploy_data / "futures" / f.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                dest.symlink_to(f)
        print(f"[✓] Symlinked {len(list((deploy_data / 'futures').iterdir()))} data files")
    else:
        print("[!] No shared data to symlink")

    # Start bot
    mgr = BotManager()
    session_id = await mgr.start(sid, mode="paper", risk={"max_drawdown":0.15}, sandbox=True)
    print(f"[✓] Session {session_id}")

    # Poll 6 × 30s
    print("[→] Polling ...")
    for i in range(6):
        await asyncio.sleep(30)
        st = mgr.status(session_id)
        print(f"  [{i+1}/6] equity=${st.get('equity',0):.2f}  trades={st.get('total_trades',0)}  status={st.get('status','?')}")

    await mgr.stop(session_id)
    print("[✓] Stopped")

if __name__ == "__main__":
    asyncio.run(main())
