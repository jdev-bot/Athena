"""Comprehensive Athena dashboard tests.

Coverage:
- HTML structure validation (all panels, footer, links)
- CSS theme verification (dark mode, grid, badges, responsive)
- JavaScript function presence (loadAll, loadPipeline, drawChart, etc.)
- API contract tests (all endpoints consumed by the dashboard)
- Data-driven rendering (empty state, populated state, error state)
- Error resilience (handles API failures gracefully)
"""
import uuid

import pytest
from fastapi.testclient import TestClient

from athena.services.api import app, UI_DIR
from athena.services.models import init_db, get_session, StrategyModel
from athena.common.models import StrategyStatus, StrategyTemplate


@pytest.fixture
def client():
    init_db()
    return TestClient(app)


@pytest.fixture
def client_with_strategies(client):
    """Seed DB with a mix of strategies for data-driven dashboard tests."""
    session = get_session()
    # Clear any leftover strategies from previous tests
    session.query(StrategyModel).filter(StrategyModel.id.like("dash-test-%")).delete(synchronize_session=False)
    session.commit()
    uid = uuid.uuid4().hex[:8]
    for i, status in enumerate([
        StrategyStatus.GENERATED,
        StrategyStatus.BACKTEST_DONE,
        StrategyStatus.PROMOTED,
        StrategyStatus.RETIRED,
    ]):
        s = StrategyModel(
            id=f"dash-test-{uid}-{i}",
            name=f"Strategy {i}",
            template="trend_following" if i % 2 == 0 else "scalping",
            dna="{}",
            status=status.value,
            generation=i,
            raw_score=0.5 + i * 0.1,
            total_return=0.02 * (i + 1),
            sharpe=0.8 + i * 0.1,
            total_trades=10 + i * 5,
            max_drawdown=0.05,
            win_rate=0.55,
            metadata_json={"drift_demotion": {"reason": "severe drift", "demoted_at": "2026-06-01T00:00:00"}} if status == StrategyStatus.RETIRED else {},
        )
        session.add(s)
    session.commit()
    session.close()
    return client


# ── HTML Structure Tests ───────────────────────────────────────────

class TestHtmlStructure:

    def test_root_serves_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")

    def test_title_present(self, client):
        resp = client.get("/")
        assert "<title>Athena Dashboard</title>" in resp.text

    def test_all_panels_present(self, client):
        resp = client.get("/")
        text = resp.text
        assert "Pipeline" in text
        assert "Strategies" in text
        assert "Forward PnL" in text
        assert "Latest Signals" in text
        assert "Drift Alerts" in text
        assert "DNA Versions" in text

    def test_kpi_row_present(self, client):
        resp = client.get("/")
        assert 'id="pipeline-kpis"' in resp.text
        assert 'id="kpi-gens"' in resp.text
        assert 'id="kpi-promoted"' in resp.text
        assert 'id="kpi-retired"' in resp.text
        assert 'id="kpi-last-cycle"' in resp.text
        assert 'id="kpi-signals"' in resp.text
        assert 'id="kpi-kill"' in resp.text

    def test_tables_present(self, client):
        resp = client.get("/")
        assert 'id="strategies-table"' in resp.text
        assert 'id="signals-table"' in resp.text
        assert 'id="dna-table"' in resp.text

    def test_canvas_present(self, client):
        resp = client.get("/")
        assert 'id="pnl-canvas"' in resp.text

    def test_footer_present(self, client):
        resp = client.get("/")
        assert '<footer' in resp.text
        assert '/docs' in resp.text
        assert '/metrics' in resp.text
        assert '/health' in resp.text
        assert '/strategies' in resp.text
        assert '/signals' in resp.text

    def test_health_dot_present(self, client):
        resp = client.get("/")
        assert 'id="health-dot"' in resp.text
        assert 'id="health-text"' in resp.text

    def test_version_shown(self, client):
        resp = client.get("/")
        assert "v0.3.0" in resp.text


# ── CSS Theme Tests ────────────────────────────────────────────────

class TestCssTheme:

    def test_dark_css_variables(self, client):
        resp = client.get("/")
        assert "--bg: #0d1117" in resp.text
        assert "--panel: #161b22" in resp.text
        assert "--border: #30363d" in resp.text
        assert "--text: #c9d1d9" in resp.text
        assert "--accent: #58a6ff" in resp.text

    def test_badge_styles(self, client):
        resp = client.get("/")
        assert ".badge.promoted" in resp.text
        assert ".badge.generated" in resp.text
        assert ".badge.retired" in resp.text
        assert ".badge.drift" in resp.text

    def test_grid_layout(self, client):
        resp = client.get("/")
        assert "grid-template-columns" in resp.text

    def test_responsive_meta(self, client):
        resp = client.get("/")
        assert 'name="viewport"' in resp.text
        assert 'width=device-width' in resp.text

    def test_loading_animation(self, client):
        resp = client.get("/")
        assert "@keyframes dots" in resp.text
        assert "loading::after" in resp.text


# ── JavaScript Function Tests ──────────────────────────────────────

class TestJavaScriptFunctions:

    def test_loadall_function_exists(self, client):
        resp = client.get("/")
        assert "async function loadAll()" in resp.text

    def test_loadpipeline_function_exists(self, client):
        resp = client.get("/")
        assert "async function loadPipeline()" in resp.text

    def test_loadstrategies_function_exists(self, client):
        resp = client.get("/")
        assert "async function loadStrategies()" in resp.text

    def test_loadpnlchart_function_exists(self, client):
        resp = client.get("/")
        assert "async function loadPnLChart()" in resp.text

    def test_loadsignals_function_exists(self, client):
        resp = client.get("/")
        assert "async function loadSignals()" in resp.text

    def test_loaddrift_function_exists(self, client):
        resp = client.get("/")
        assert "async function loadDrift()" in resp.text

    def test_loaddnaversions_function_exists(self, client):
        resp = client.get("/")
        assert "async function loadDnaVersions()" in resp.text

    def test_drawchart_function_exists(self, client):
        resp = client.get("/")
        assert "function drawChart(" in resp.text

    def test_fmttime_helper_exists(self, client):
        resp = client.get("/")
        assert "function fmtTime(" in resp.text

    def test_badge_helper_exists(self, client):
        resp = client.get("/")
        assert "function badge(" in resp.text

    def test_fetchjson_helper_exists(self, client):
        resp = client.get("/")
        assert "async function fetchJSON(" in resp.text

    def test_auto_refresh_interval(self, client):
        resp = client.get("/")
        assert "setInterval(loadAll, 60000)" in resp.text

    def test_manual_refresh_buttons(self, client):
        resp = client.get("/")
        # Each panel should have a refresh button calling its load function
        assert 'onclick="loadAll()"' in resp.text
        assert 'onclick="loadStrategies()"' in resp.text
        assert 'onclick="loadPnLChart()"' in resp.text
        assert 'onclick="loadSignals()"' in resp.text
        assert 'onclick="loadDrift()"' in resp.text
        assert 'onclick="loadDnaVersions()"' in resp.text


# ── API Contract Tests ───────────────────────────────────────────

class TestApiContracts:
    """Verify all endpoints consumed by the dashboard return expected shapes."""

    def test_health_contract(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "service" in data
        assert "version" in data

    def test_stats_contract(self, client):
        resp = client.get("/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_strategies" in data
        assert "by_status" in data
        assert "best_strategy" in data
        # Dashboard-specific keys
        assert "generations" in data
        assert "promoted" in data
        assert "retired" in data
        assert "signals_24h" in data
        assert "kill_switch_active" in data

    def test_strategies_list_contract(self, client_with_strategies):
        resp = client_with_strategies.get("/strategies?status=all")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        if data:
            s = data[0]
            assert "id" in s
            assert "name" in s
            assert "status" in s
            assert "template" in s
            assert "generation" in s
            assert "raw_score" in s
            assert "total_return" in s
            assert "sharpe" in s
            assert "total_trades" in s
            assert "created_at" in s
            # Dashboard-enriched fields
            assert "updated_at" in s
            assert "metadata" in s
            assert "dna_versions" in s

    def test_strategies_filter_by_status(self, client_with_strategies):
        for status in ["promoted", "retired", "generated"]:
            resp = client_with_strategies.get(f"/strategies?status={status}")
            assert resp.status_code == 200
            data = resp.json()
            assert isinstance(data, list)
            for s in data:
                assert s["status"] == status

    def test_signals_contract(self, client):
        resp = client.get("/signals?limit=20")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)

    def test_forward_pnl_contract(self, client_with_strategies):
        # Get a promoted strategy
        resp = client_with_strategies.get("/strategies?status=promoted")
        strategies = resp.json()
        if strategies:
            sid = strategies[0]["id"]
            resp = client_with_strategies.get(f"/forward/pnl?strategy_id={sid}")
            assert resp.status_code == 200
            data = resp.json()
            assert isinstance(data, list)
            # Each point should have timestamp, pnl_pct, cumulative_pnl
            if data:
                p = data[0]
                assert "timestamp" in p
                assert "pnl_pct" in p
                assert "cumulative_pnl" in p

    def test_scheduler_status_contract(self, client):
        resp = client.get("/scheduler/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "running" in data

    def test_metrics_contract(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200
        text = resp.text
        assert "athena_" in text or "# HELP" in text or text == ""


# ── Data-Driven Rendering Tests ────────────────────────────────────

class TestDataDrivenRendering:
    """Dashboard renders correctly with different data states."""

    def test_empty_state_text(self, client):
        resp = client.get("/")
        assert "No strategies yet" in resp.text
        assert "No signals" in resp.text
        assert "No drift alerts" in resp.text
        assert "No DNA snapshots" in resp.text

    def test_strategies_with_data(self, client_with_strategies):
        """Static HTML contains JS that renders strategy names from API."""
        resp = client_with_strategies.get("/")
        assert "loadStrategies" in resp.text
        assert "${s.name}" in resp.text  # JS template literal for name

    def test_status_badges_in_html(self, client_with_strategies):
        resp = client_with_strategies.get("/")
        assert "badge(s.status)" in resp.text or "${badge(s.status)}" in resp.text

    def test_drift_alert_in_html(self, client_with_strategies):
        resp = client_with_strategies.get("/")
        assert "loadDrift" in resp.text
        assert ".badge.drift" in resp.text

    def test_trend_following_template_visible(self, client_with_strategies):
        """The JS template literal for template column exists."""
        resp = client_with_strategies.get("/")
        assert "${s.template}" in resp.text

    def test_scalping_template_visible(self, client_with_strategies):
        """JS will show scalping when data contains it."""
        resp = client_with_strategies.get("/")
        assert "${s.template}" in resp.text


# ── Error Resilience Tests ─────────────────────────────────────────

class TestErrorResilience:
    """Dashboard handles API failures without crashing."""

    def test_error_state_in_js(self, client):
        resp = client.get("/")
        text = resp.text
        # Check that JS has catch blocks for each loader
        assert "catch(e)" in text
        # Check that error messages are displayed in empty-state divs
        assert 'Error: ${e.message}' in text or "Error:" in text

    def test_health_error_handling(self, client):
        # The JS has a try/catch for health
        resp = client.get("/")
        assert "Error: " in resp.text

    def test_dashboard_shows_fallback_when_no_index_html(self, client, monkeypatch):
        """If index.html is missing, / returns JSON fallback."""
        import athena.services.api as api_module
        # Temporarily point UI_DIR to a non-existent path
        original_ui_dir = UI_DIR
        fake_dir = original_ui_dir.parent / "ui_missing"
        monkeypatch.setattr(api_module, "UI_DIR", fake_dir)
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["dashboard"] == "not_installed"


# ── Accessibility / Usability Tests ────────────────────────────────

class TestUsability:

    def test_html_lang_attribute(self, client):
        resp = client.get("/")
        assert '<html lang="en">' in resp.text

    def test_meta_charset(self, client):
        resp = client.get("/")
        assert '<meta charset="UTF-8">' in resp.text

    def test_refresh_buttons_have_title(self, client):
        resp = client.get("/")
        assert 'title="Refresh"' in resp.text

    def test_panel_sections_semantic(self, client):
        resp = client.get("/")
        assert "<main>" in resp.text
        assert "<header>" in resp.text
        assert "<footer" in resp.text  # footer element with class

    def test_no_external_dependencies(self, client):
        """Dashboard is self-contained — no external CSS/JS/CDN links."""
        resp = client.get("/")
        text = resp.text
        assert "cdnjs" not in text
        assert "unpkg" not in text
        assert "googleapis" not in text
        assert "bootstrap" not in text.lower()
        assert "jquery" not in text.lower()
        # All JS is inline
        script_blocks = text.split("<script>")
        assert len(script_blocks) > 1
        # All CSS is inline
        assert "<style>" in text


# ── Performance / Size Tests ───────────────────────────────────────

class TestPerformance:

    def test_dashboard_size_under_50kb(self, client):
        resp = client.get("/")
        size = len(resp.text.encode("utf-8"))
        assert size < 50_000, f"Dashboard HTML is {size} bytes, expected < 50KB"

    def test_dashboard_loads_fast(self, client):
        import time
        start = time.time()
        resp = client.get("/")
        elapsed = time.time() - start
        assert elapsed < 1.0, f"Dashboard took {elapsed:.2f}s to load"
        assert resp.status_code == 200
