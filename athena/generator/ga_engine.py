"""Genetic Algorithm engine for strategy evolution."""
import logging
import random
import numpy as np
from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass
from athena.common.models import StrategyTemplate, StrategyDNA, StrategyRecord, PerformanceMetrics
from athena.common.config import config
from athena.generator.dna import DNAEncoder

logger = logging.getLogger(__name__)


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
        ml_seed_ratio: float = 0.0,
    ):
        self.template = template
        self.allowed_templates = allowed_templates or [template]
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_count = elitism_count
        self.ml_seed_ratio = max(0.0, min(1.0, ml_seed_ratio))
        self.encoder = DNAEncoder()
        self.population: List[Individual] = []
        self.generation = 0
        self.best_fitness_history: List[float] = []
        self.avg_fitness_history: List[float] = []
        self._rng = np.random.default_rng()
        self._predictors: Dict[str, Any] = {}

    # ── predictor management ──────────────────────────────────────────

    @property
    def predictors(self) -> Dict[str, Any]:
        return self._predictors

    @predictors.setter
    def predictors(self, val: Dict[str, Any]):
        self._predictors = val

    def _get_predictor(self, template: StrategyTemplate):
        if not self._predictors:
            return None
        return self._predictors.get(template)

    def _train_predictor(self, population: List[Individual], template: StrategyTemplate) -> None:
        """Train a predictor for the given template on the evaluated population."""
        # Lazy import to avoid heavy sklearn load on startup
        try:
            from athena.generator.ml_predictor import MLPredictor
        except ImportError:
            logger.warning("MLPredictor not available (sklearn missing)")
            return

        pred = self._predictors.get(template)
        if pred is None:
            pred = MLPredictor(template)
            self._predictors[template] = pred

        # Filter to individuals matching this template
        subset = [ind for ind in population if ind.template == template]
        if len(subset) < 10:
            return
        pred.train(subset)
        logger.info(f"Predictor trained for {template.value} on {len(subset)} samples, R² check done")

    # ── initialization ────────────────────────────────────────────────

    def _random_template(self) -> StrategyTemplate:
        idx = self._rng.integers(0, len(self.allowed_templates))
        return self.allowed_templates[int(idx)]

    def initialize_population(self, seed_strategies: Optional[List[StrategyRecord]] = None) -> None:
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

    # ── evolution ─────────────────────────────────────────────────────

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

            # ── Elitism ──
            new_population = self.population[:self.elitism_count]

            # ── ML-guided seeding ──
            pred = self._get_predictor(self.template)
            if pred is not None and pred.is_trained and self.ml_seed_ratio > 0:
                ml_count = int(self.ml_seed_ratio * self.population_size)
                ml_candidates = pred.generate_promising_candidates(
                    self.template, n_candidates=ml_count
                )
                for dna in ml_candidates[:ml_count]:
                    if len(new_population) >= self.population_size:
                        break
                    tmpl_idx = self._rng.integers(0, len(self.allowed_templates))
                    tmpl = self.allowed_templates[int(tmpl_idx)]
                    new_population.append(Individual(
                        id=self._generate_id(),
                        template=tmpl,
                        dna=dna,
                        generation=gen + 1,
                        parent_ids=["ml_predictor"],
                    ))

            # ── Generate offspring ──
            while len(new_population) < self.population_size:
                parent1 = self._tournament_select()
                parent2 = self._tournament_select()

                # Determine child template
                if parent1.template == parent2.template:
                    child_template = parent1.template
                else:
                    # Different templates — pick one at random, skip DNA crossover
                    idx = self._rng.integers(0, 2)
                    child_template = [parent1.template, parent2.template][int(idx)]

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

            # ── Train predictors after generation 2 ──
            if gen >= 2 and self.ml_seed_ratio > 0:
                self._train_predictor(self.population, self.template)
                for tmpl in self.allowed_templates:
                    if tmpl != self.template:
                        self._train_predictor(self.population, tmpl)

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

    # ── helpers ───────────────────────────────────────────────────────

    def _tournament_select(self) -> Individual:
        """Tournament selection."""
        contestants = self._rng.choice(
            self.population, size=min(3, len(self.population)), replace=False
        )
        return max(contestants, key=lambda x: x.fitness)

    def _generate_id(self) -> str:
        return f"ga_{self.generation}_{self._rng.integers(0, 1_000_000)}"

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
