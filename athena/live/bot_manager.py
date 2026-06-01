"""Bot manager — spawn, monitor, and stop Freqtrade processes.

Uses Freqtrade's own CLI (`freqtrade trade --config ...`) so the bot runs
with all native functionality: ccxt data feed, order execution, position tracking,
profit logging, and the built-in REST API server.

Athena queries the bot's API for status and applies an external kill-switch
if drawdown exceeds the session threshold.
"""
import asyncio
import json
import os
import signal
import subprocess
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import requests

from athena.live.deployer import Deployer
from athena.services.models import get_session, LiveSessionModel


class BotManager:
    """Manages a single Freqtrade bot subprocess per session."""

    def __init__(self):
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._deployer = Deployer()

    # ── lifecycle ───────────────────────────────────────────────────
    async def start(
        self,
        strategy_id: str,
        mode: str = "paper",
        risk: Dict[str, Any] = None,
        exchange_key: str = "",
        exchange_secret: str = "",
        sandbox: bool = False,
    ) -> str:
        """Deploy config + strategy and spawn a Freqtrade bot.

        Returns the Athena session_id.
        """
        session_id = f"live_{uuid.uuid4().hex[:12]}"

        # Find a free port for the Freqtrade API server
        api_port = self._free_port()

        # Deploy user_data directory
        deploy_dir = self._deployer.deploy(
            strategy_id=strategy_id,
            mode=mode,
            risk=risk,
            api_port=api_port,
            exchange_key=exchange_key,
            exchange_secret=exchange_secret,
            sandbox=sandbox,
        )
        config_path = deploy_dir / "config.json"

        # Read API credentials from the generated config
        cfg = json.loads(config_path.read_text())
        api_creds = cfg["api_server"]

        # Spawn bot as subprocess (Freqtrade's own entry point)
        cmd = [
            "python", "-m", "freqtrade", "trade",
            "--config", str(config_path),
            "--userdir", str(deploy_dir),
        ]
        env = os.environ.copy()
        # Ensure the venv Python sees the same packages
        env["PYTHONPATH"] = str(deploy_dir / "strategies") + os.pathsep + env.get("PYTHONPATH", "")

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(deploy_dir),
            env=env,
        )

        # Persist session
        sess = get_session()
        sess.add(LiveSessionModel(
            id=session_id,
            strategy_id=strategy_id,
            status="running",
            mode=mode,
            equity=cfg.get("dry_run_wallet", 10_000.0),
        ))
        sess.commit()
        sess.close()

        self._sessions[session_id] = {
            "proc": proc,
            "deploy_dir": deploy_dir,
            "config": cfg,
            "api_port": api_port,
            "api_user": api_creds["username"],
            "api_pass": api_creds["password"],
            "started_at": datetime.utcnow(),
            "risk": risk or {},
            "strategy_id": strategy_id,
        }

        # Give Freqtrade a few seconds to boot its API server
        await asyncio.sleep(3)

        # Start kill-switch monitor
        asyncio.create_task(self._kill_switch_monitor(session_id))

        return session_id

    async def stop(self, session_id: str, reason: str = "stopped_by_user"):
        """Gracefully stop a bot session."""
        meta = self._sessions.pop(session_id, None)
        if not meta:
            sess = get_session()
            row = sess.query(LiveSessionModel).filter_by(id=session_id).first()
            if row:
                row.status = reason
                row.stopped_at = datetime.utcnow()
                sess.commit()
            sess.close()
            return

        proc = meta["proc"]
        # SIGTERM → Freqtrade's Worker catches it and exits cleanly
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

        # Cleanup deploy dir
        self._deployer.cleanup(meta["strategy_id"])

        # Update DB
        sess = get_session()
        row = sess.query(LiveSessionModel).filter_by(id=session_id).first()
        if row:
            row.status = reason
            row.stopped_at = datetime.utcnow()
            sess.commit()
        sess.close()

    # ── status ──────────────────────────────────────────────────────
    def status(self, session_id: str) -> Dict[str, Any]:
        """Query the Freqtrade bot's native REST API for live stats."""
        meta = self._sessions.get(session_id)
        if not meta:
            return self._db_status(session_id)

        # Try Freqtrade API first
        url = f"http://127.0.0.1:{meta['api_port']}"
        try:
            profit = self._api_get(url, "/api/v1/profit", meta)
            balance = self._api_get(url, "/api/v1/balance", meta)
            status_msg = self._api_get(url, "/api/v1/status", meta)
        except Exception:
            # Bot not ready yet — fall back to DB
            return self._db_status(session_id)

        # Compute drawdown from profit data
        starting = balance.get("starting_capital", 10_000.0)
        total_bot = balance.get("total_bot", starting)
        dd = (starting - total_bot) / starting if total_bot < starting else 0.0

        return {
            "session_id": session_id,
            "status": status_msg.get("status", "running"),
            "mode": meta["config"].get("dry_run", True),
            "equity": total_bot,
            "open_positions": profit.get("trade_count", 0) - profit.get("closed_trade_count", 0),
            "total_trades": profit.get("trade_count", 0),
            "max_drawdown": dd,
            "started_at": meta["started_at"].isoformat(),
            "stopped_at": None,
            "profit_closed_pct": profit.get("profit_closed_percent", 0.0),
            "profit_all_pct": profit.get("profit_all_percent", 0.0),
            "last_signals": [],  # Freqtrade handles signal logging internally
        }

    # ── kill-switch monitor ───────────────────────────────────────
    async def _kill_switch_monitor(self, session_id: str):
        """Poll Freqtrade API every 30s; kill bot if drawdown > threshold."""
        meta = self._sessions.get(session_id)
        if not meta:
            return

        max_dd = meta["risk"].get("max_drawdown", 0.15)
        daily_limit = meta["risk"].get("daily_loss_limit", 0.10)
        url = f"http://127.0.0.1:{meta['api_port']}"

        while session_id in self._sessions:
            await asyncio.sleep(30)
            meta = self._sessions.get(session_id)
            if not meta:
                break

            try:
                profit = self._api_get(url, "/api/v1/profit", meta)
                balance = self._api_get(url, "/api/v1/balance", meta)
            except Exception:
                continue

            starting = balance.get("starting_capital", 10_000.0)
            total_bot = balance.get("total_bot", starting)
            dd = (starting - total_bot) / starting if total_bot < starting else 0.0

            # Max drawdown check
            if dd >= max_dd:
                await self.stop(session_id, f"stopped_drawdown ({dd:.2%})")
                break

            # Daily loss check
            profit_closed_pct = profit.get("profit_closed_percent", 0.0)
            if profit_closed_pct <= -daily_limit * 100:
                await self.stop(session_id, f"stopped_daily_loss ({profit_closed_pct:.2f}%)")
                break

    # ── helpers ───────────────────────────────────────────────────
    def _api_get(self, base_url: str, endpoint: str, meta: Dict[str, Any]) -> Dict[str, Any]:
        """Authenticated GET against Freqtrade's API server."""
        resp = requests.get(
            f"{base_url}{endpoint}",
            auth=(meta["api_user"], meta["api_pass"]),
            timeout=5.0,
        )
        resp.raise_for_status()
        return resp.json()

    def _db_status(self, session_id: str) -> Dict[str, Any]:
        """Fallback when bot process is gone."""
        sess = get_session()
        row = sess.query(LiveSessionModel).filter_by(id=session_id).first()
        if not row:
            sess.close()
            return {}
        stats = {
            "session_id": row.id,
            "status": row.status,
            "mode": row.mode,
            "equity": row.equity,
            "open_positions": row.open_positions,
            "total_trades": row.total_trades_taken,
            "max_drawdown": row.max_drawdown_seen,
            "started_at": row.started_at.isoformat() if row.started_at else None,
            "stopped_at": row.stopped_at.isoformat() if row.stopped_at else None,
            "last_signals": row.last_signals,
        }
        sess.close()
        return stats

    @staticmethod
    def _free_port() -> int:
        """Find an available TCP port on localhost."""
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        return port
