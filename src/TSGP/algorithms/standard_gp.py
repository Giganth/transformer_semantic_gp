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
        """Clone, shuffle to randomize mating, then apply crossover and mutation.

        The shuffle breaks the positional link to ``population``, so each clone
        carries ``parent_index`` for measures that need its origin.
        """
        offspring = [self.toolbox.clone(ind) for ind in population]
        for index, individual in enumerate(offspring):
            individual.parent_index = index
        random.shuffle(offspring)

        for i in range(1, len(offspring), 2):
            if random.random() < self.crossover_prob:
                offspring[i - 1], offspring[i] = self.toolbox.mate(offspring[i - 1], offspring[i])
                del offspring[i - 1].fitness.values
                del offspring[i].fitness.values

        for i in range(len(offspring)):
            if random.random() < self.mutation_prob:
                parent_index = offspring[i].parent_index
                try:
                    offspring[i], = self.toolbox.mutate(offspring[i])
                    del offspring[i].fitness.values
                except Exception:
                    offspring[i] = self.toolbox.createIndividual()
                    del offspring[i].fitness.values
                # A failed mutation replaces the object, so re-tag it.
                offspring[i].parent_index = parent_index
        return offspring


