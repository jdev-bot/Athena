"""CLI entry-point for Athena — thin wrapper around AthenaEngine."""
import argparse
import json
import logging
from pathlib import Path

from athena.common.models import StrategyTemplate, GenerationConfig
from athena.common.config import config
from athena.core.engine import AthenaEngine

logging.basicConfig(level=getattr(logging, config.LOG_LEVEL), format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Athena — AI Strategy Generator")
    parser.add_argument("--mode", choices=["evolve", "search", "evaluate", "promote", "deploy", "list", "export"],
                        default="evolve")
    parser.add_argument("--template", choices=[t.value for t in StrategyTemplate], default="trend_following")
    parser.add_argument("--symbols", default="BTC-USD")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--start-date", default="2026-05-02")
    parser.add_argument("--end-date", default="2026-06-01")
    parser.add_argument("--population", type=int, default=30)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--strategy-id")
    parser.add_argument("--output")
    parser.add_argument("--gates", type=lambda x: x.lower() in ("true", "1", "yes"), default=True,
                        help="Run robustness gates after evolution")
    args = parser.parse_args()

    gen_config = GenerationConfig(
        symbols=args.symbols.split(","),
        timeframe=args.timeframe,
        start_date=args.start_date,
        end_date=args.end_date,
        population_size=args.population,
        generations=args.generations,
    )
    engine = AthenaEngine(gen_config)

    if args.mode == "evolve":
        template = StrategyTemplate(args.template)
        records = engine.evolve(template, run_gates=args.gates)
        for r in records[:5]:
            logger.info(f"  {r.id}: score={r.score.raw_score:.3f} verdict={r.score.verdict} gates={r.metadata.get('robustness_passed', False)}")
        if args.output:
            Path(args.output).write_text(json.dumps([r.model_dump() for r in records], indent=2, default=str))
            logger.info(f"Saved to {args.output}")

    elif args.mode == "search":
        results = engine.run_search()
        for tpl, recs in results.items():
            if recs:
                logger.info(f"{tpl}: best={recs[0].id} score={recs[0].score.raw_score:.3f}")
        if args.output:
            out = {k: [r.model_dump() for r in v] for k, v in results.items()}
            Path(args.output).write_text(json.dumps(out, indent=2, default=str))

    elif args.mode == "evaluate":
        if not args.strategy_id:
            logger.error("--strategy-id required")
            return
        record = engine.evaluate(args.strategy_id)
        logger.info(f"Score={record.score.raw_score:.3f} | {record.score.verdict} | trades={record.performance.total_trades}")

    elif args.mode == "promote":
        if not args.strategy_id:
            logger.error("--strategy-id required")
            return
        record = engine.promote(args.strategy_id)
        logger.info(f"Status={record.status.value} | gates={record.metadata.get('robustness_passed')} | verdict={record.score.verdict}")

    elif args.mode == "deploy":
        if not args.strategy_id:
            logger.error("--strategy-id required")
            return
        path = engine.deploy(args.strategy_id)
        logger.info(f"Deployed to {path}")

    elif args.mode == "list":
        records = engine.run_search([StrategyTemplate(args.template)])
        for r in records.get(args.template, [])[:20]:
            logger.info(f"{r.id}: score={r.score.raw_score:.3f} status={r.status.value}")

    elif args.mode == "export":
        records = engine.run_search([StrategyTemplate(args.template)])
        for r in records.get(args.template, [])[:10]:
            path = engine.deploy(r.id)
            logger.info(f"Exported {r.id} → {path}")


if __name__ == "__main__":
    main()
