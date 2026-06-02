"""Tests for DNA versioning layer + API."""
import pytest
from fastapi.testclient import TestClient

from athena.services.api import app
from athena.services.models import init_db, get_session, StrategyModel, DnaSnapshot
from athena.core.dna_versioning import snapshot_dna, restore_dna, list_snapshots


@pytest.fixture
def client():
    init_db()
    session = get_session()
    session.query(StrategyModel).delete()
    session.query(DnaSnapshot).delete()
    session.commit()
    session.close()
    return TestClient(app)


def _seed_strategy(session, strategy_id: str):
    s = StrategyModel(
        id=strategy_id,
        name="test",
        template="trend_following",
        dna={"fast": 10, "slow": 30},
        status="promoted",
        generation=1,
    )
    session.add(s)
    session.commit()


class TestDnaVersioning:

    def test_snapshot_creates_version(self):
        init_db()
        sid = "dv_01"
        session = get_session()
        session.query(DnaSnapshot).delete()
        session.commit()
        v = snapshot_dna(sid, {"fast": 10, "slow": 30}, source="promotion")
        assert v == 1

    def test_snapshot_increments_version(self):
        init_db()
        sid = "dv_02"
        session = get_session()
        session.query(DnaSnapshot).delete()
        session.commit()
        v1 = snapshot_dna(sid, {"a": 1})
        v2 = snapshot_dna(sid, {"a": 2})
        assert v1 == 1
        assert v2 == 2

    def test_restore_latest(self):
        init_db()
        sid = "dv_03"
        session = get_session()
        session.query(DnaSnapshot).delete()
        session.commit()
        snapshot_dna(sid, {"x": 1})
        snapshot_dna(sid, {"x": 2})
        latest = restore_dna(sid)
        assert latest == {"x": 2}

    def test_restore_specific_version(self):
        init_db()
        sid = "dv_04"
        session = get_session()
        session.query(DnaSnapshot).delete()
        session.commit()
        snapshot_dna(sid, {"k": 10})
        snapshot_dna(sid, {"k": 20})
        vec = restore_dna(sid, version=1)
        assert vec == {"k": 10}

    def test_restore_missing_returns_none(self):
        assert restore_dna("missing_id", version=99) is None

    def test_list_snapshots(self):
        init_db()
        sid = "dv_05"
        session = get_session()
        session.query(DnaSnapshot).delete()
        session.commit()
        snapshot_dna(sid, {"m": 1}, source="manual")
        snapshot_dna(sid, {"m": 2}, source="hyperopt")
        versions = list_snapshots(sid)
        assert len(versions) == 2
        assert versions[0]["version"] == 1
        assert versions[1]["source"] == "hyperopt"


class TestDnaApi:

    def test_post_snapshot_404(self, client):
        resp = client.post("/dna/snapshot?strategy_id=noexist")
        assert resp.status_code == 404

    def test_post_snapshot_ok(self, client):
        session = get_session()
        _seed_strategy(session, "dv_api_1")
        session.close()
        resp = client.post("/dna/snapshot?strategy_id=dv_api_1&source=manual")
        assert resp.status_code == 200
        data = resp.json()
        assert data["version"] == 1
        assert data["source"] == "manual"

    def test_get_versions(self, client):
        session = get_session()
        _seed_strategy(session, "dv_api_2")
        session.query(DnaSnapshot).delete()
        session.commit()
        snapshot_dna("dv_api_2", {"q": 5}, "manual")
        session.close()
        resp = client.get("/dna/versions?strategy_id=dv_api_2")
        assert resp.status_code == 200
        assert len(resp.json()["versions"]) == 1

    def test_get_restore(self, client):
        session = get_session()
        _seed_strategy(session, "dv_api_3")
        session.query(DnaSnapshot).delete()
        session.commit()
        snapshot_dna("dv_api_3", {"r": 99}, "manual")
        session.close()
        resp = client.get("/dna/restore?strategy_id=dv_api_3&version=1")
        assert resp.status_code == 200
        assert resp.json()["dna"]["r"] == 99

    def test_get_restore_404(self, client):
        resp = client.get("/dna/restore?strategy_id=missing&version=1")
        assert resp.status_code == 404
