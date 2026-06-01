"""Quick roundtrip test: generate strategy code and backtest."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from athena.generator.dna import DNAEncoder
from athena.generator.templates import TEMPLATE_MAP, TEMPLATE_SPECS
from athena.common.models import StrategyTemplate
from athena.core.freqtrade_wrapper import FreqtradeWrapper

# Build a simple DNA for trend_following
template = StrategyTemplate.TREND_FOLLOWING
specs = TEMPLATE_SPECS[template]
encoder = DNAEncoder()
dna = encoder.random_dna(template)
params = encoder.to_strategy_params(dna, template)
params["class_name"] = "AthenaStrategy"
params["template_name"] = template.value
params["timeframe"] = "1h"

code = TEMPLATE_MAP[template].format(**params)
print("--- strategy code snippet ---")
print(code[:500])

wrapper = FreqtradeWrapper()
result = wrapper.run_backtest(
    strategy_code=code,
    start_date="2026-05-01",
    end_date="2026-05-15",
    exchange="binance",
    symbol="BTC-USD",
    timeframe="1h",
)
print("\n--- Result ---")
print(json.dumps(result, indent=2))
