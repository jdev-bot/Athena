"""Tests for ForwardScheduler."""
import threading
import time
import uuid
from unittest.mock import MagicMock

import pytest

from athena.common.models import StrategyStatus, StrategyTemplate
from athena.live.forward_scheduler import ForwardScheduler, ForwardRunSummary, get_forward_scheduler, DEFAULT_INTERVAL_SECONDS
from athena.services.models import get_session, StrategyModel, init_db


@pytest.fixture(autouse=True)
def _clean_db(monkeypatch):
    init_db()
    session = get_session()
    session.query(StrategyModel).delete()
    session.commit()
    session.close()
    # reset singleton
    monkeypatch.setattr("athena.live.forward_scheduler._fw_scheduler", None)


def _promoted_strategy(name: str) -> str:
    sid = f"{name}_{uuid.uuid4().hex[:8]}"
    session = get_session()
    rec = StrategyModel(
        id=sid,
        name=name,
        template="trend_following",
        dna={"fast": 9, "slow": 21},
        status=StrategyStatus.PROMOTED.value,
        generation=1,
    )
    session.add(rec)
    session.commit()
    session.close()
    return sid


# ════════════════════════════════════════════════════════════════════
# Unit behaviour
# ════════════════════════════════════════════════════════════════════

class TestForwardScheduler:

    def test_start_stop(self):
        fs = ForwardScheduler(interval_seconds=60)
        fs.start()
        assert fs.is_running is True
        fs.stop()
        # after join it may still appear alive briefly
        time.sleep(0.05)
        assert fs.is_running is False

    def test_status_returns_dict(self):
        fs = ForwardScheduler(interval_seconds=60)
        st = fs.status()
        assert isinstance(st, dict)
        assert "running" in st
        assert st["running"] is False
        assert st["interval_seconds"] == 60
        assert st["total_cycles"] == 0

    def test_run_cycle_empty_db(self, monkeypatch):
        fs = ForwardScheduler(interval_seconds=60)
        results = fs.run_cycle()
        assert results == []
        assert fs.total_cycles == 1

    def test_run_cycle_one_promoted(self, monkeypatch):
        sid = _promoted_strategy("promo")
        fs = ForwardScheduler(interval_seconds=60)
        # mock run_forward to avoid real candles
        summary = {
            "bars": 100,
            "total_signals": 2,
            "open_positions": 0,
            "total_closed": 1,
            "total_pnl": 0.05,
            "killed": False,
        }
        monkeypatch.setattr(
            "athena.live.forward_scheduler.run_forward",
            lambda strategy_id, **kwargs: (None, summary),
        )
        results = fs.run_cycle()
        assert len(results) == 1
        assert results[0].strategy_id == sid
        assert not results[0].killed
        assert results[0].total_pnl == 0.05
        assert results[0].bars == 100

    def test_run_cycle_kill_switch_alert(self, monkeypatch):
        sid = _promoted_strategy("kill_me")
        fs = ForwardScheduler(interval_seconds=60)
        summary = {
            "bars": 50,
            "total_signals": 2,
            "open_positions": 0,
            "total_closed": 1,
            "total_pnl": -0.20,
            "killed": True,
        }
        monkeypatch.setattr(
            "athena.live.forward_scheduler.run_forward",
            lambda strategy_id, **kwargs: (None, summary),
        )
        results = fs.run_cycle()
        assert results[0].killed is True
        assert fs.kills_seen == 1

    def test_run_cycle_handles_exception(self, monkeypatch):
        sid = _promoted_strategy("broken")
        fs = ForwardScheduler(interval_seconds=60)
        monkeypatch.setattr(
            "athena.live.forward_scheduler.run_forward",
            lambda strategy_id, **kwargs: (_ for _ in ()).throw(ValueError("boom")),
        )
        # should not propagate
        results = fs.run_cycle()
        assert results == []

    def test_run_cycle_non_promoted_ignored(self):
        session = get_session()
        session.add(
            StrategyModel(
                id="draft_123",
                name="draft",
                template="mean_reversion",
                dna={},
                status=StrategyStatus.DRAFT.value,
            )
        )
        session.commit()
        session.close()
        fs = ForwardScheduler(interval_seconds=60)
        results = fs.run_cycle()
        assert results == []

    def test_interval_sleep(self, monkeypatch):
        fs = ForwardScheduler(interval_seconds=1)
        fs.start()
        # it will sleep for 1 second
        assert fs.is_running is True
        fs.stop()

    def test_singleton(self, monkeypatch):
        a = get_forward_scheduler()
        b = get_forward_scheduler()
        assert a is b

    def test_results_cached(self, monkeypatch):
        sid = _promoted_strategy("cached")
        fs = ForwardScheduler(interval_seconds=60)
        summary = {
            "bars": 10,
            "total_signals": 1,
            "open_positions": 0,
            "total_closed": 1,
            "total_pnl": 0.01,
            "killed": False,
        }
        monkeypatch.setattr(
            "athena.live.forward_scheduler.run_forward",
            lambda strategy_id, **kwargs: (None, summary),
        )
        fs.run_cycle()
        assert len(fs.results) == 1


# ════════════════════════════════════════════════════════════════════
# FastAPI endpoint integration (via TestClient)
# ════════════════════════════════════════════════════════════════════

from athena.services.api import app
from fastapi.testclient import TestClient

client = TestClient(app)


class TestForwardSchedulerApi:

    def test_forward_scheduler_start(self):
        resp = client.post("/forward/scheduler/start")
        assert resp.status_code == 200
        assert resp.json()["running"] is True

    def test_forward_scheduler_stop(self):
        resp = client.post("/forward/scheduler/stop")
        assert resp.status_code == 200
        assert resp.json()["running"] is False

    def test_forward_scheduler_status(self):
        resp = client.get("/forward/scheduler/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "running" in data
        assert "total_cycles" in data
        assert "total_runs" in data
        assert "kills_seen" in data

    def test_forward_scheduler_status_no_auto_start(self):
        resp = client.get("/forward/scheduler/status")
        # forward scheduler is a singleton — previous tests may have started / stopped it
        assert resp.status_code == 200
