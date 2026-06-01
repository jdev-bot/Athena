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
    """Genetic Algorithm engine for evolving strategies."""
    
    def __init__(self, template: StrategyTemplate, population_size: int = None,
                 generations: int = None, mutation_rate: float = None,
                 crossover_rate: float = None, elitism_count: int = None):
        self.template = template
        self.population_size = population_size or config.DEFAULT_POPULATION_SIZE
        self.generations = generations or config.DEFAULT_GENERATIONS
        self.mutation_rate = mutation_rate or config.MUTATION_RATE
        self.crossover_rate = crossover_rate or config.CROSSOVER_RATE
        self.elitism_count = elitism_count or config.ELITISM_COUNT
        
        self.encoder = DNAEncoder()
        self.population: List[Individual] = []
        self.generation = 0
        self.best_fitness_history = []
        self.avg_fitness_history = []
        
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
        
        # Fill remaining with random
        while len(self.population) < self.population_size:
            dna = self.encoder.random_dna(self.template)
            ind = Individual(
                id=self._generate_id(),
                template=self.template,
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
                
                if random.random() < self.crossover_rate:
                    child1_dna, child2_dna = self.encoder.crossover(
                        parent1.dna, parent2.dna, self.template
                    )
                else:
                    child1_dna = dict(parent1.dna)
                    child2_dna = dict(parent2.dna)
                
                child1_dna = self.encoder.mutate(
                    child1_dna, self.template, self.mutation_rate
                )
                child2_dna = self.encoder.mutate(
                    child2_dna, self.template, self.mutation_rate
                )
                
                new_population.append(Individual(
                    id=self._generate_id(),
                    template=self.template,
                    dna=child1_dna,
                    generation=gen + 1,
                    parent_ids=[parent1.id, parent2.id],
                ))
                
                if len(new_population) < self.population_size:
                    new_population.append(Individual(
                        id=self._generate_id(),
                        template=self.template,
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
