from TSGP.core.engine import BaseEvolutionaryAlgorithm
from typing import Dict, Any, List
import random


class StandardGP(BaseEvolutionaryAlgorithm):
    """Standard Genetic Programming with crossover and mutation operators."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.crossover_prob = config['variation']['crossover']['application_prob']
        self.mutation_prob = config['variation']['mutation']['application_prob']

    def _variation(self, population: List) -> List:
        """Apply crossover and mutation operators to the population."""
        offspring = [self.toolbox.clone(ind) for ind in population]
        random.shuffle(offspring)

        # Apply crossover
        for i in range(1, len(offspring), 2):
            if random.random() < self.crossover_prob:
                offspring[i - 1], offspring[i] = self.toolbox.mate(offspring[i - 1], offspring[i])
                del offspring[i - 1].fitness.values
                del offspring[i].fitness.values
                
        # Apply mutation
        for i in range(len(offspring)):
            if random.random() < self.mutation_prob:
                try:
                    offspring[i], = self.toolbox.mutate(offspring[i])
                    del offspring[i].fitness.values
                except Exception:
                    offspring[i] = self.toolbox.createIndividual()
                    del offspring[i].fitness.values
        return offspring


