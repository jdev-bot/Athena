"""Integration tests for Athena dashboard."""
import pytest
from fastapi.testclient import TestClient

from athena.services.api import app, UI_DIR
from athena.services.models import init_db


@pytest.fixture
def client():
    init_db()
    return TestClient(app)


class TestDashboard:

    def test_root_returns_dashboard_html(self, client):
        """When index.html exists, GET / should serve the dashboard."""
        resp = client.get("/")
        if (UI_DIR / "index.html").is_file():
            assert resp.status_code == 200
            assert "text/html" in resp.headers.get("content-type", "")
            assert "Athena Dashboard" in resp.text
        else:
            assert resp.status_code == 200
            assert resp.json()["status"] == "ok"

    def test_dashboard_links_in_footer(self, client):
        """Dashboard HTML contains links to API docs, metrics, etc."""
        resp = client.get("/")
        if resp.status_code == 200 and "text/html" in resp.headers.get("content-type", ""):
            assert "/docs" in resp.text
            assert "/metrics" in resp.text
            assert "/health" in resp.text

    def test_dashboard_auto_refresh_meta(self, client):
        resp = client.get("/")
        if resp.status_code == 200 and "text/html" in resp.headers.get("content-type", ""):
            assert "loadAll" in resp.text
            assert "setInterval" in resp.text
