"""Tests for the Hyperopt finisher."""
import json
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from athena.core.hyperopt import HyperoptFinisher, _to_ft_pair, _pair_to_key
from athena.common.models import StrategyRecord, StrategyDNA, StrategyTemplate


@pytest.fixture
def dummy_record(tmp_path):
    return StrategyRecord(
        id="test_strat_001",
        name="TestStrategy",
        template=StrategyTemplate.TREND_FOLLOWING,
        dna=StrategyDNA(
            template=StrategyTemplate.TREND_FOLLOWING,
            vector={"fast_period": 10, "slow_period": 40, "symbol": "BTC-USD", "timeframe": "1h"},
        ),
        generation=1,
    )


def test_pair_conversion():
    assert _to_ft_pair("BTC-USD") == "BTC/USDT:USDT"
    assert _to_ft_pair("ETH-USD") == "ETH/USDT:USDT"
    assert _pair_to_key("BTC/USDT:USDT") == "BTC_USDT_USDT"


def test_hyperopt_deploy(dummy_record, tmp_path):
    finisher = HyperoptFinisher(epochs=1, spaces="buy")
    deploy_dir = finisher._deploy(dummy_record)
    assert deploy_dir.exists()
    config_path = deploy_dir / "config.json"
    assert config_path.exists()
    config = json.loads(config_path.read_text())
    assert config["strategy"] == "AthenaStrategy"
    assert config["pairs"] == ["BTC/USDT:USDT"]
    # Cleanup
    shutil.rmtree(deploy_dir, ignore_errors=True)


def test_parse_best_epoch(tmp_path):
    finisher = HyperoptFinisher()
    fthypt = tmp_path / "test.fthypt"
    epoch1 = json.dumps({"loss": 0.5, "params_dict": {"fast_period": 20}})
    epoch2 = json.dumps({"loss": 0.1, "params_dict": {"fast_period": 30}})
    epoch3 = json.dumps({"loss": 100000, "params_dict": {"fast_period": 40}})
    fthypt.write_text("\n".join([epoch1, epoch2, epoch3]) + "\n")
    best = finisher._parse_best_epoch(fthypt)
    assert best["loss"] == 0.1
    assert best["params"]["fast_period"] == 30


def test_merge_params(dummy_record):
    finisher = HyperoptFinisher()
    updated = finisher._merge_params(dummy_record, {"fast_period": 99, "unknown": 42})
    assert updated.dna.vector["fast_period"] == 99
    assert "unknown" not in updated.dna.vector


@pytest.fixture
def mock_hyperopt_results(tmp_path):
    """Create a fake .fthypt file and mock subprocess to simulate hyperopt."""
    def _make(record):
        finisher = HyperoptFinisher(epochs=1, spaces="buy")
        deploy_dir = finisher._deploy(record)
        results_dir = deploy_dir / "user_data" / "hyperopt_results"
        results_dir.mkdir(parents=True, exist_ok=True)
        fthypt = results_dir / f"strategy_AthenaStrategy_test.fthypt"
        best_epoch = {
            "loss": 0.05,
            "params_dict": {"fast_period": 22},
            "results_metrics": {"total_trades": 10},
        }
        fthypt.write_text(json.dumps(best_epoch) + "\n")
        return deploy_dir
    return _make


def test_run_end_to_end(dummy_record, mock_hyperopt_results):
    finisher = HyperoptFinisher(epochs=1, spaces="buy")
    deploy_dir = mock_hyperopt_results(dummy_record)
    with patch.object(finisher, "_deploy", return_value=deploy_dir):
        result = finisher.run(dummy_record, deploy_dir=deploy_dir)
    assert result.dna.vector["fast_period"] == 22
    assert result.metadata["hyperopt"]["best_loss"] == 0.05
    shutil.rmtree(deploy_dir, ignore_errors=True)
