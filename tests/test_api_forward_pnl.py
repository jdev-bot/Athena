"""Integration tests for forward PnL endpoint."""
import pytest
from fastapi.testclient import TestClient

from athena.services.api import app
from athena.services.models import init_db, get_session, Signal
import uuid
from datetime import datetime, timezone


@pytest.fixture
def client():
    init_db()
    session = get_session()
    session.query(Signal).delete()
    session.commit()
    session.close()
    return TestClient(app)


def _add_signals(session, strategy_id: str, pnls: list, symbol: str = "BTC/USDT"):
    for i, pnl in enumerate(pnls):
        s = Signal(
            id=str(uuid.uuid4()),
            strategy_id=strategy_id,
            symbol=symbol,
            direction="long",
            confidence=1.0,
            entry_price=100.0 + i,
            exit_price=100.0 + i + 1,
            pnl_pct=pnl,
            pnl_abs=pnl * 100,
            status="closed",
            created_at=datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
        )
        session.add(s)
    session.commit()


class TestForwardPnL:

    def test_empty(self, client):
        resp = client.get("/forward/pnl?strategy_id=missing")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_cumulative_pnl(self, client):
        sid = "strat_xyz"
        session = get_session()
        _add_signals(session, sid, [0.01, -0.005, 0.02])
        session.close()
        resp = client.get(f"/forward/pnl?strategy_id={sid}")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body) == 3
        assert body[-1]["cumulative_pnl"] == pytest.approx(0.025, abs=1e-4)

    def test_filter_by_symbol(self, client):
        sid = "strat_abc"
        session = get_session()
        _add_signals(session, sid, [0.01], symbol="BTC/USDT")
        _add_signals(session, sid, [0.02], symbol="ETH/USDT")
        session.close()
        resp = client.get(f"/forward/pnl?strategy_id={sid}&symbol=BTC/USDT")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body) == 1
        assert body[0]["cumulative_pnl"] == pytest.approx(0.01, abs=1e-4)
