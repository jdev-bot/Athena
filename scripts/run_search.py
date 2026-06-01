from athena.orchestrator import AthenaOrchestrator
from athena.common.models import GenerationConfig, StrategyTemplate

cfg = GenerationConfig(
    symbols=['BTC-USD'],
    timeframe='1h',
    start_date='2026-05-02',
    end_date='2026-06-01',
    population_size=20,
    generations=5,
    mutation_rate=0.25,
    crossover_rate=0.8,
    elitism_count=3,
    ml_boost=False,
)
orch = AthenaOrchestrator(cfg)
results = {}
for tpl in [StrategyTemplate.TREND_FOLLOWING, StrategyTemplate.MEAN_REVERSION]:
    print(f'\n=== Template: {tpl.value} ===')
    records = orch.run_generation(tpl, run_gates=False)
    best = max(records, key=lambda r: r.score.raw_score)
    print(f'Best score: {best.score.raw_score:.4f}  Verdict: {best.score.verdict}')
    print(f'Metrics: {best.performance.model_dump()}')
    results[tpl.value] = {
        'best_id': best.id,
        'score': best.score.raw_score,
        'verdict': best.score.verdict,
        'metrics': best.performance.model_dump(),
    }

print('\n=== SUMMARY ===')
for k,v in results.items():
    print(f"{k:20s} score={v['score']:.4f} verdict={v['verdict']}")
