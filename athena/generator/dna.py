"""DNA encoding/decoding for strategies."""
import random
from typing import Dict, List, Any
from athena.common.models import DNASpec, StrategyTemplate
from athena.generator.templates import TEMPLATE_SPECS


class DNAEncoder:
    """Encode/decode strategy DNA vectors."""
    
    def get_spec(self, template: StrategyTemplate) -> List[DNASpec]:
        """Get DNA specification for template."""
        return TEMPLATE_SPECS.get(template, [])
    
    def random_dna(self, template: StrategyTemplate) -> Dict[str, Any]:
        """Generate random DNA vector for template."""
        spec = self.get_spec(template)
        dna = {}
        for s in spec:
            dna[s.name] = self._random_value(s)
        return dna
    
    def _random_value(self, spec: DNASpec) -> Any:
        """Generate random value for a DNA parameter."""
        if spec.type == "int":
            return random.randint(int(spec.min), int(spec.max))
        elif spec.type == "float":
            return random.uniform(spec.min, spec.max)
        elif spec.type == "bool":
            return random.choice([True, False])
        elif spec.type == "choice":
            return random.choice(spec.choices)
        return spec.default
    
    def mutate(self, dna: Dict[str, Any], template: StrategyTemplate,
               mutation_rate: float = 0.2) -> Dict[str, Any]:
        """Mutate a DNA vector."""
        spec = self.get_spec(template)
        result = dict(dna)
        
        for s in spec:
            if random.random() < mutation_rate:
                result[s.name] = self._random_value(s)
        
        return result
    
    def crossover(self, dna1: Dict[str, Any], dna2: Dict[str, Any],
                  template: StrategyTemplate) -> tuple:
        """Perform two-point crossover between two DNA vectors."""
        spec = self.get_spec(template)
        names = [s.name for s in spec]
        
        if len(names) < 2:
            return dict(dna1), dict(dna2)
        
        point1 = random.randint(0, len(names) - 1)
        point2 = random.randint(point1, len(names) - 1)
        
        child1 = dict(dna1)
        child2 = dict(dna2)
        
        for i in range(point1, point2 + 1):
            name = names[i]
            child1[name] = dna2.get(name, dna1[name])
            child2[name] = dna1.get(name, dna2[name])
        
        return child1, child2
    
    def to_strategy_params(self, dna: Dict[str, Any], template: StrategyTemplate) -> Dict[str, Any]:
        """Convert DNA to strategy constructor parameters.

        Merges with spec defaults so cross-template individuals don't
        trigger KeyError during template formatting.
        """
        spec = self.get_spec(template)
        params = {s.name: s.default for s in spec}
        params.update(dna)
        return params
