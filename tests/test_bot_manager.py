"""Tests for the live trading BotManager."""
from unittest.mock import patch, MagicMock, AsyncMock
import pytest

from athena.live.bot_manager import BotManager


@pytest.fixture
def manager():
    return BotManager()


@pytest.mark.asyncio
async def test_start_paper_mode(manager, tmp_path, monkeypatch):
    """Test paper mode bot start with mocked subprocess."""
    # Mock deployer
    deploy_dir = tmp_path / "deploy"
    deploy_dir.mkdir()
    (deploy_dir / "config.json").write_text(
        """{"api_server": {"username": "u", "password": "p", "listen_port": 8080}}"""
    )
    manager._deployer.deploy = MagicMock(return_value=deploy_dir)
    manager._deployer.cleanup = MagicMock()

    # Mock subprocess.Popen
    mock_proc = MagicMock()
    mock_proc.poll.return_value = None
    mock_proc.communicate.return_value = (b"", b"")
    mock_proc.returncode = 0

    with patch("subprocess.Popen", return_value=mock_proc):
        session_id = await manager.start("strat_123", mode="paper")

    assert session_id.startswith("live_")
    assert session_id in manager._sessions
    assert manager._sessions[session_id]["config"].get("dry_run", True)
    mock_proc.send_signal.assert_not_called()

    # Cleanup
    await manager.stop(session_id)
    assert session_id not in manager._sessions


def test_free_port(manager):
    port = manager._free_port()
    assert 1024 < port < 65535


def test_api_get(manager, monkeypatch):
    import requests
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"status": "running", "profit": 0.1}
    mock_resp.raise_for_status = MagicMock()
    monkeypatch.setattr(requests, "get", lambda *a, **kw: mock_resp)

    meta = {"api_user": "u", "api_pass": "p"}
    result = manager._api_get("http://127.0.0.1:8080", "/api/v1/status", meta)
    assert result["status"] == "running"


@pytest.mark.asyncio
async def test_kill_switch_monitor_triggers(manager, monkeypatch):
    """Verify kill-switch monitor stops a bot when drawdown exceeds threshold."""
    manager._sessions["sess_1"] = {
        "proc": MagicMock(returncode=0),
        "config": {"dry_run": True},
        "api_port": 8080,
        "api_user": "u",
        "api_pass": "p",
        "started_at": None,
        "risk": {"max_drawdown": 0.05, "daily_loss_limit": 0.05},
        "strategy_id": "strat_1",
    }

    # Mock _api_get to return a drawdown of 10% (over threshold)
    monkeypatch.setattr(
        manager,
        "_api_get",
        lambda *a, **kw: {
            "starting_capital": 100.0,
            "total_bot": 90.0,
            "profit_closed_percent": -10.0,
        },
    )

    # Mock stop to avoid async subprocess cleanup in test
    monkeypatch.setattr(manager, "stop", AsyncMock())

    # Run one iteration of the monitor
    await manager._kill_switch_monitor("sess_1")

    # Since drawdown (10%) > threshold (5%), stop was called
    assert "sess_1" not in manager._sessions or manager.stop.called
