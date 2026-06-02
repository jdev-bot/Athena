"""Genetic Algorithm engine for strategy evolution."""
import random
import numpy as np
from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass
from athena.common.models import StrategyTemplate, StrategyDNA, StrategyRecord, PerformanceMetrics
from athena.common.config import config
from athena.generator.dna import DNAEncoder


@dataclass
class Individual:
    """A candidate strategy in the population."""
    id: str
    template: StrategyTemplate
    dna: Dict[str, Any]
    fitness: float = 0.0
    generation: int = 0
    parent_ids: List[str] = None

    def __post_init__(self):
        if self.parent_ids is None:
            self.parent_ids = []


class GAEngine:
    """Genetic Algorithm engine for strategy evolution."""

    def __init__(
        self,
        template: StrategyTemplate,
        population_size: int = 30,
        generations: int = 10,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        elitism_count: int = 2,
        allowed_templates: Optional[List[StrategyTemplate]] = None,
    ):
        self.template = template
        self.allowed_templates = allowed_templates or [template]
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_count = elitism_count
        self.encoder = DNAEncoder()
        self.population: List[Individual] = []
        self.generation = 0
        self.best_fitness_history: List[float] = []
        self.avg_fitness_history: List[float] = []
        self._rng = np.random.default_rng()

    def _random_template(self) -> StrategyTemplate:
        idx = self._rng.integers(0, len(self.allowed_templates))
        return self.allowed_templates[int(idx)]

    def initialize_population(self, seed_strategies: List[StrategyRecord] = None) -> None:
        """Create initial population."""
        self.population = []

        # Add seeded strategies if provided
        if seed_strategies:
            for s in seed_strategies[:self.population_size]:
                ind = Individual(
                    id=s.id,
                    template=s.dna.template,
                    dna=s.dna.vector,
                    generation=0,
                )
                self.population.append(ind)

        # Fill remaining with random from allowed templates
        while len(self.population) < self.population_size:
            tmpl = self._random_template()
            dna = self.encoder.random_dna(tmpl)
            ind = Individual(
                id=self._generate_id(),
                template=tmpl,
                dna=dna,
                generation=0,
            )
            self.population.append(ind)
    
    def evolve(self, fitness_fn: Callable[[Individual], float], parallel_workers: int = 1) -> List[Individual]:
        """Run GA evolution for configured generations.

        If parallel_workers > 1, fitness evaluation is dispatched to a
        ThreadPoolExecutor.  The caller's fitness_fn must be thread-safe.
        """
        for gen in range(self.generations):
            self.generation = gen
            
            # Evaluate fitness (parallel or sequential)
            if parallel_workers > 1:
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=parallel_workers) as pool:
                    results = list(pool.map(fitness_fn, self.population))
                for ind, fit in zip(self.population, results):
                    ind.fitness = fit
            else:
                for ind in self.population:
                    ind.fitness = fitness_fn(ind)
            
            # Sort by fitness
            self.population.sort(key=lambda x: x.fitness, reverse=True)
            
            # Record history
            self.best_fitness_history.append(self.population[0].fitness)
            avg = np.mean([ind.fitness for ind in self.population])
            self.avg_fitness_history.append(avg)
            
            # Elitism
            new_population = self.population[:self.elitism_count]
            
            # Generate offspring
            while len(new_population) < self.population_size:
                parent1 = self._tournament_select()
                parent2 = self._tournament_select()

                # Determine child template
                if parent1.template == parent2.template:
                    child_template = parent1.template
                else:
                    # Different templates — pick one at random, skip crossover
                    child_template = self._rng.choice([parent1.template, parent2.template])

                if random.random() < self.crossover_rate and parent1.template == parent2.template:
                    child1_dna, child2_dna = self.encoder.crossover(
                        parent1.dna, parent2.dna, child_template
                    )
                else:
                    child1_dna = dict(parent1.dna)
                    child2_dna = dict(parent2.dna)

                child1_dna = self.encoder.mutate(
                    child1_dna, child_template, self.mutation_rate
                )
                child2_dna = self.encoder.mutate(
                    child2_dna, child_template, self.mutation_rate
                )

                new_population.append(Individual(
                    id=self._generate_id(),
                    template=child_template,
                    dna=child1_dna,
                    generation=gen + 1,
                    parent_ids=[parent1.id, parent2.id],
                ))

                if len(new_population) < self.population_size:
                    new_population.append(Individual(
                        id=self._generate_id(),
                        template=child_template,
                        dna=child2_dna,
                        generation=gen + 1,
                        parent_ids=[parent1.id, parent2.id],
                    ))
            
            self.population = new_population
        
        # Final evaluation
        if parallel_workers > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=parallel_workers) as pool:
                results = list(pool.map(fitness_fn, self.population))
            for ind, fit in zip(self.population, results):
                ind.fitness = fit
        else:
            for ind in self.population:
                ind.fitness = fitness_fn(ind)
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        
        return self.population
    
    def _tournament_select(self, tournament_size: int = 3) -> Individual:
        """Tournament selection."""
        contestants = random.sample(self.population, 
                                   min(tournament_size, len(self.population)))
        return max(contestants, key=lambda x: x.fitness)
    
    def _generate_id(self) -> str:
        """Generate unique strategy ID."""
        import uuid
        return f"strat_{uuid.uuid4().hex[:12]}"
    
    def to_strategy_records(self) -> List[StrategyRecord]:
        """Convert population to strategy records."""
        records = []
        for ind in self.population:
            records.append(StrategyRecord(
                id=ind.id,
                name=f"{ind.template.value}_{ind.id[-6:]}",
                template=ind.template,
                dna=StrategyDNA(template=ind.template, vector=ind.dna),
                generation=ind.generation,
                performance=PerformanceMetrics(),
            ))
        return records
