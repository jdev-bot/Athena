from athena.orchestrator import AthenaOrchestrator
from athena.common.models import GenerationConfig, StrategyTemplate
import json, sys

cfg = GenerationConfig(
    symbols=['BTC-USD'],
    timeframe='1h',
    start_date='2026-05-02',
    end_date='2026-06-01',
    population_size=10,
    generations=3,
    mutation_rate=0.25,
    crossover_rate=0.8,
    elitism_count=2,
    ml_boost=False,
)
orch = AthenaOrchestrator(cfg)
all_results = {}
for tpl in [StrategyTemplate.TREND_FOLLOWING, StrategyTemplate.MEAN_REVERSION]:
    print(f'\n=== {tpl.value} ===')
    records = orch.run_generation(tpl, run_gates=False)
    best = max(records, key=lambda r: r.score.raw_score)
    worst = min(records, key=lambda r: r.score.raw_score)
    all_results[tpl.value] = {
        'best': {'id': best.id, 'score': best.score.raw_score, 'verdict': best.score.verdict, 'metrics': best.performance.model_dump()},
        'worst': {'id': worst.id, 'score': worst.score.raw_score, 'verdict': worst.score.verdict, 'metrics': worst.performance.model_dump()},
        'n_trades': [r.performance.total_trades for r in records],
    }
    print(f"Best  score={best.score.raw_score:.4f} verdict={best.score.verdict} trades={best.performance.total_trades}")
    print(f"Worst score={worst.score.raw_score:.4f} verdict={worst.score.verdict} trades={worst.performance.total_trades}")

print('\n=== SUMMARY ===')
for k,v in all_results.items():
    print(f"{k:20s} best={v['best']['score']:.4f}/{v['best']['verdict']} worst={v['worst']['score']:.4f}/{v['worst']['verdict']} trades={v['n_trades']}")

# Save results
with open('/tmp/athena_search_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)
print('\nSaved to /tmp/athena_search_results.json')
