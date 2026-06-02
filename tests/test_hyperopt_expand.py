"""Tests for HyperoptFinisher.expand_ranges."""
import pytest

from athena.common.models import DNASpec, StrategyDNA, StrategyRecord, StrategyTemplate, PerformanceMetrics
from athena.core.hyperopt import HyperoptFinisher


@pytest.fixture
def record_with_specs() -> StrategyRecord:
    """Build a record with DNA spec containing int + float params."""
    spec = [
        DNASpec(name="fast", type="int", min=5, max=20, default=10),
        DNASpec(name="slow", type="int", min=20, max=50, default=30),
        DNASpec(name="atr_mult", type="float", min=1.0, max=5.0, default=2.0),
        DNASpec(name="use_filter", type="bool", default=False),
    ]
    dna = StrategyDNA(
        template=StrategyTemplate.TREND_FOLLOWING,
        vector={"fast": 12, "slow": 35, "atr_mult": 3.0, "use_filter": True},
        spec=spec,
    )
    return StrategyRecord(
        id="test_123",
        name="test",
        template=StrategyTemplate.TREND_FOLLOWING,
        dna=dna,
        performance=PerformanceMetrics(),
    )


class TestExpandRanges:

    def test_int_range_widened(self, record_with_specs):
        rec = HyperoptFinisher.expand_ranges(record_with_specs, factor=0.25)
        fast_spec = next(s for s in rec.dna.spec if s.name == "fast")
        # val=12, span=12*0.25=3  => min=min(5, 12-3)=5, max=max(20, 12+3)=20
        assert fast_spec.min <= 12
        assert fast_spec.max >= 12
        assert fast_spec.min >= 2
        assert fast_spec.max > fast_spec.min

    def test_float_range_widened(self, record_with_specs):
        rec = HyperoptFinisher.expand_ranges(record_with_specs, factor=0.30)
        atr_spec = next(s for s in rec.dna.spec if s.name == "atr_mult")
        assert atr_spec.min <= 3.0
        assert atr_spec.max >= 3.0
        assert atr_spec.max > atr_spec.min

    def test_bool_unchanged(self, record_with_specs):
        rec = HyperoptFinisher.expand_ranges(record_with_specs, factor=0.20)
        bool_spec = next(s for s in rec.dna.spec if s.name == "use_filter")
        assert bool_spec.min is None
        assert bool_spec.max is None

    def test_no_spec_loads_from_template(self):
        """Spec missing -> fall back to TEMPLATE_SPECS."""
        dna = StrategyDNA(
            template=StrategyTemplate.MEAN_REVERSION,
            vector={"fast": 8, "slow": 25, "rsi_period": 14},
            spec=[],
        )
        rec = StrategyRecord(
            id="x", name="x",
            template=StrategyTemplate.MEAN_REVERSION,
            dna=dna,
            performance=PerformanceMetrics(),
        )
        rec2 = HyperoptFinisher.expand_ranges(rec, factor=0.20)
        assert rec2.dna.spec  # should now have loaded template specs

    def test_factor_zero_means_no_change(self, record_with_specs):
        rec1 = HyperoptFinisher.expand_ranges(record_with_specs, factor=0.0)
        # span = max(val*0, ...) = epsilon; min should stay same if already smaller than val
        fast = next(s for s in rec1.dna.spec if s.name == "fast")
        assert fast.min == 5
        assert fast.max == 20
