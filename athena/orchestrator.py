"""Main orchestrator for Athena generator + evaluator."""
import os
import sys
import json
import asyncio
import argparse
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from athena.common.models import (
    StrategyTemplate, StrategyRecord, StrategyDNA, StrategyStatus,
    PerformanceMetrics, ScoreResult, GenerationConfig
)
from athena.common.config import config
from athena.generator.templates import TEMPLATE_MAP
from athena.generator.dna import DNAEncoder
from athena.generator.ga_engine import GAEngine, Individual
from athena.generator.ml_predictor import MLPredictor
from athena.evaluator.scorer import Scorer
from athena.services.models import init_db, get_session, StrategyModel


class AthenaOrchestrator:
    """Orchestrates strategy generation, evaluation, and lifecycle."""
    
    def __init__(self, gen_config: GenerationConfig):
        self.config = gen_config
        self.encoder = DNAEncoder()
        self.scorer = Scorer()
        self.ml_predictor: Dict[StrategyTemplate, MLPredictor] = {}
        init_db()
        
    def generate_strategy_code(self, record: StrategyRecord) -> str:
        """Generate Jesse-compatible strategy Python code."""
        template = TEMPLATE_MAP.get(record.dna.template)
        if not template:
            raise ValueError(f"Unknown template: {record.dna.template}")
        
        params = self.encoder.to_strategy_params(record.dna.vector, record.dna.template)
        params['class_name'] = f"Strategy_{record.id.replace('-', '_')}"
        
        return template.format(**params)
    
    def evaluate_strategy(self, record: StrategyRecord) -> PerformanceMetrics:
        """Run backtest and return metrics."""
        # Write strategy code to file
        code = self.generate_strategy_code(record)
        strategy_file = config.GENERATED_DIR / f"{record.id}.py"
        strategy_file.parent.mkdir(parents=True, exist_ok=True)
        strategy_file.write_text(code)
        
        # TODO: Actually run Jesse backtest here
        # For now, simulate with random metrics for testing
        import random
        return PerformanceMetrics(
            total_return=random.uniform(-0.2, 0.5),
            sharpe=random.uniform(-0.5, 2.0),
            sortino=random.uniform(-0.5, 2.0),
            calmar=random.uniform(-0.5, 3.0),
            win_rate=random.uniform(0.3, 0.7),
            max_drawdown=random.uniform(0.05, 0.4),
            total_trades=random.randint(10, 200),
            avg_trade=random.uniform(-0.01, 0.05),
            profit_factor=random.uniform(0.5, 3.0),
        )
    
    def run_generation(self, template: StrategyTemplate) -> List[StrategyRecord]:
        """Run full generation + evaluation cycle."""
        print(f"\n{'='*60}")
        print(f"Starting generation for template: {template.value}")
        print(f"Population: {self.config.population_size}, Generations: {self.config.generations}")
        print(f"{'='*60}\n")
        
        # Initialize GA engine
        ga = GAEngine(
            template=template,
            population_size=self.config.population_size,
            generations=self.config.generations,
            mutation_rate=self.config.mutation_rate,
            crossover_rate=self.config.crossover_rate,
            elitism_count=self.config.elitism_count,
        )
        
        # Initialize population
        ga.initialize_population()
        
        # ML predictor
        ml = MLPredictor(template)
        
        # Fitness function
        def fitness_fn(individual: Individual) -> float:
            # Create record
            record = StrategyRecord(
                id=individual.id,
                name=f"{template.value}_{individual.id[-6:]}",
                template=template,
                dna=StrategyDNA(template=template, vector=individual.dna),
                generation=individual.generation,
            )
            
            # Evaluate
            metrics = self.evaluate_strategy(record)
            record.performance = metrics
            
            # Score
            score = self.scorer.score(metrics)
            record.score = score
            
            # Save to DB
            self._save_strategy(record)
            
            return score.raw_score
        
        # Evolve
        population = ga.evolve(fitness_fn)
        
        # Train ML predictor on final population
        if self.config.ml_boost:
            ml.train(population)
            self.ml_predictor[template] = ml
        
        # Convert to records
        records = ga.to_strategy_records()
        
        # Update records with actual scores
        for record in records:
            record.performance = self.evaluate_strategy(record)
            record.score = self.scorer.score(record.performance)
            self._save_strategy(record)
        
        # Sort by score
        records.sort(key=lambda x: x.score.raw_score, reverse=True)
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"Generation complete!")
        print(f"Best score: {records[0].score.raw_score:.3f}")
        print(f"Best strategy: {records[0].id}")
        print(f"Verdict: {records[0].score.verdict}")
        print(f"{'='*60}\n")
        
        return records
    
    def _save_strategy(self, record: StrategyRecord) -> None:
        """Save strategy to database."""
        session = get_session()
        
        existing = session.query(StrategyModel).filter(
            StrategyModel.id == record.id
        ).first()
        
        if existing:
            # Update
            existing.status = record.status.value
            existing.generation = record.generation
            existing.total_return = record.performance.total_return
            existing.sharpe = record.performance.sharpe
            existing.sortino = record.performance.sortino
            existing.calmar = record.performance.calmar
            existing.win_rate = record.performance.win_rate
            existing.max_drawdown = record.performance.max_drawdown
            existing.total_trades = record.performance.total_trades
            existing.avg_trade = record.performance.avg_trade
            existing.profit_factor = record.performance.profit_factor
            existing.raw_score = record.score.raw_score
            existing.verdict = record.score.verdict
            existing.updated_at = datetime.utcnow()
        else:
            # Create
            strat = StrategyModel(
                id=record.id,
                name=record.name,
                template=record.dna.template.value,
                dna=record.dna.vector,
                objective=record.objective,
                status=record.status.value,
                generation=record.generation,
                parent_id=record.parent_id,
                total_return=record.performance.total_return,
                sharpe=record.performance.sharpe,
                sortino=record.performance.sortino,
                calmar=record.performance.calmar,
                win_rate=record.performance.win_rate,
                max_drawdown=record.performance.max_drawdown,
                total_trades=record.performance.total_trades,
                avg_trade=record.performance.avg_trade,
                profit_factor=record.performance.profit_factor,
                raw_score=record.score.raw_score,
                verdict=record.score.verdict,
            )
            session.add(strat)
        
        session.commit()
    
    def get_best_strategies(self, template: StrategyTemplate = None,
                           limit: int = 10) -> List[StrategyRecord]:
        """Get best strategies from database."""
        session = get_session()
        query = session.query(StrategyModel).order_by(StrategyModel.raw_score.desc())
        if template:
            query = query.filter(StrategyModel.template == template.value)
        rows = query.limit(limit).all()
        
        records = []
        for row in rows:
            records.append(self._row_to_record(row))
        return records
    
    def _row_to_record(self, row: StrategyModel) -> StrategyRecord:
        """Convert DB row to StrategyRecord."""
        return StrategyRecord(
            id=row.id,
            name=row.name,
            template=StrategyTemplate(row.template),
            dna=StrategyDNA(template=StrategyTemplate(row.template), vector=row.dna),
            objective=row.objective,
            status=StrategyStatus(row.status),
            generation=row.generation,
            parent_id=row.parent_id,
            performance=PerformanceMetrics(
                total_return=row.total_return,
                sharpe=row.sharpe,
                sortino=row.sortino,
                calmar=row.calmar,
                win_rate=row.win_rate,
                max_drawdown=row.max_drawdown,
                total_trades=row.total_trades,
                avg_trade=row.avg_trade,
                profit_factor=row.profit_factor,
            ),
            score=ScoreResult(raw_score=row.raw_score, verdict=row.verdict),
            created_at=row.created_at,
            updated_at=row.updated_at,
        )


def main():
    parser = argparse.ArgumentParser(description="Athena Strategy Generator")
    parser.add_argument("--mode", choices=["evolve", "evaluate", "list", "export"],
                        default="evolve")
    parser.add_argument("--template", choices=[t.value for t in StrategyTemplate],
                        default="trend_following")
    parser.add_argument("--symbols", default="BTC-USD")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--population", type=int, default=30)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--output", default=None)
    
    args = parser.parse_args()
    
    gen_config = GenerationConfig(
        symbols=args.symbols.split(","),
        timeframe=args.timeframe,
        population_size=args.population,
        generations=args.generations,
    )
    
    orch = AthenaOrchestrator(gen_config)
    
    if args.mode == "evolve":
        template = StrategyTemplate(args.template)
        records = orch.run_generation(template)
        
        if args.output:
            with open(args.output, 'w') as f:
                data = [r.model_dump() for r in records]
                json.dump(data, f, indent=2, default=str)
            print(f"Results saved to {args.output}")
    
    elif args.mode == "list":
        records = orch.get_best_strategies(limit=20)
        for r in records:
            print(f"{r.id}: {r.name} | Score: {r.score.raw_score:.3f} | Status: {r.status.value}")
    
    elif args.mode == "evaluate":
        print("Evaluate mode: specify --strategy-id")
    
    elif args.mode == "export":
        records = orch.get_best_strategies(limit=10)
        for r in records:
            code = orch.generate_strategy_code(r)
            path = config.GENERATED_DIR / f"{r.id}.py"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(code)
            print(f"Exported {r.id} to {path}")


if __name__ == "__main__":
    main()
