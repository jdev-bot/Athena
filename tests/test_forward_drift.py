"""Tests for drift auto-demote inside ForwardScheduler."""
import uuid

import pytest

from athena.common.models import StrategyStatus, StrategyTemplate
from athena.live.forward_scheduler import ForwardScheduler, ForwardRunSummary
from athena.services.models import get_session, StrategyModel, init_db


@pytest.fixture(autouse=True)
def _clean_db(monkeypatch):
    init_db()
    session = get_session()
    session.query(StrategyModel).delete()
    session.commit()
    session.close()
    monkeypatch.setattr("athena.live.forward_scheduler._fw_scheduler", None)


def _promoted_strategy(total_return: float, metadata=None) -> str:
    sid = f"promo_{uuid.uuid4().hex[:8]}"
    session = get_session()
    rec = StrategyModel(
        id=sid,
        name="promo",
        template="trend_following",
        dna={"fast": 9, "slow": 21},
        status=StrategyStatus.PROMOTED.value,
        generation=1,
        total_return=total_return,
        metadata_json=metadata or {},
    )
    session.add(rec)
    session.commit()
    session.close()
    return sid


class TestDriftCheck:

    def test_no_drift(self):
        fs = ForwardScheduler(interval_seconds=60)
        sid = _promoted_strategy(total_return=0.20)
        session = get_session()
        row = session.query(StrategyModel).filter_by(id=sid).first()
        summary = {"total_pnl": 0.15}
        severe = fs._check_drift(summary, row)
        assert severe is False
        assert row.status == StrategyStatus.PROMOTED.value

    def test_severe_ratio(self):
        fs = ForwardScheduler(interval_seconds=60)
        sid = _promoted_strategy(total_return=0.20)
        session = get_session()
        row = session.query(StrategyModel).filter_by(id=sid).first()
        summary = {"total_pnl": 0.05}
        severe = fs._check_drift(summary, row)
        assert severe is True
        assert row.status == StrategyStatus.RETIRED.value

    def test_severe_loss(self):
        fs = ForwardScheduler(interval_seconds=60)
        sid = _promoted_strategy(total_return=0.20)
        session = get_session()
        row = session.query(StrategyModel).filter_by(id=sid).first()
        summary = {"total_pnl": -0.15}
        severe = fs._check_drift(summary, row)
        assert severe is True
        assert row.status == StrategyStatus.RETIRED.value

    def test_metadata_populated(self):
        fs = ForwardScheduler(interval_seconds=60)
        sid = _promoted_strategy(total_return=0.20, metadata={"foo": "bar"})
        session = get_session()
        row = session.query(StrategyModel).filter_by(id=sid).first()
        summary = {"total_pnl": -0.15}
        severe = fs._check_drift(summary, row)
        assert severe is True
        assert "drift_demotion" in row.metadata_json
        assert row.metadata_json["drift_demotion"]["forward_pnl"] == -0.15

    def test_no_drift_neutral_backtest(self):
        fs = ForwardScheduler(interval_seconds=60)
        sid = _promoted_strategy(total_return=0.0)
        session = get_session()
        row = session.query(StrategyModel).filter_by(id=sid).first()
        summary = {"total_pnl": 0.0}
        severe = fs._check_drift(summary, row)
        assert severe is False
