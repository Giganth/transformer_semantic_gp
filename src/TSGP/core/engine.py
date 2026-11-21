from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List
import os
import random
import time

import numpy as np

from TSGP.symbolic_regression.evaluation import create_fitness_functions
from TSGP.core.pset import create_pset
from TSGP.core.toolbox import create_toolbox
from TSGP.symbolic_regression.dataset import prepare_dataset
from .monitoring import Monitor
from deap import tools


class BaseEvolutionaryAlgorithm(ABC):
    """Base class for evolutionary search. Subclasses implement _variation()."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the algorithm with configuration."""
        self.config = config
        self.generation = 0
        self.best_individual = None
        self.start_time = None
        self.termination_conditions = self._setup_termination_conditions()

        self._set_random_seed()

        # The data generator supplies its own dataset.
        if hasattr(self, '_custom_dataset'):
            X_train, Y_train, X_test, Y_test, X_semantic, variable_names = self._custom_dataset()
            print("Using custom dataset function for data preparation.")
        else:
            X_train, Y_train, X_test, Y_test, X_semantic, variable_names = prepare_dataset(config)

        self.semantic_testset_loaded = X_semantic is not None

        config['problem']['variables'] = variable_names
        self.pset = create_pset(config)

        evaluation_function, evaluation_function_test, evaluation_function_semantic = create_fitness_functions(config, self.pset, X_train, Y_train, X_test, Y_test, X_semantic)

        self.toolbox = create_toolbox(config, self.pset, evaluation_function, evaluation_function_test, evaluation_function_semantic)
        self.monitor = Monitor(config, console_out=config['console_out'])

    def _set_random_seed(self):
        """Set random seed for reproducibility across all components."""
        seed = self.config['random_seed']
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

    @abstractmethod
    def _variation(self, population: List) -> List:
        """Apply variation operators to the population."""

    def _selection(self, population: List, num_selected: int) -> List:
        """Select parents for reproduction."""
        return self.toolbox.select(population, num_selected)
    
    def _evaluation(self, population: List, evaluate_train=True, evaluate_test=True, evaluate_semantics=True) -> None:
        """Evaluate fitness of individuals in the population.

        A skipped objective keeps the value the individual already holds.
        """
        if evaluate_train:
            for ind, (fit, out_vec) in zip(
                population, self.toolbox.map(self.toolbox.evaluate, population)
            ):
                ind.fitness.values = (fit,)
                ind.output_vector = out_vec

        if evaluate_test:
            for ind, (fit_test, _out_vec_test) in zip(
                population, self.toolbox.map(self.toolbox.evaluate_test, population)
            ):
                ind.fitness_test.values = (fit_test,)

        if evaluate_semantics and self.semantic_testset_loaded:
            for ind, (fit_semantic, out_vec_semantic) in zip(
                population, self.toolbox.map(self.toolbox.evaluate_semantic, population)
            ):
                ind.fitness_semantic.values = (fit_semantic,)
                ind.output_vector_semantic = out_vec_semantic
            
    def _initialize(self) -> None:
        """Create the initial population."""
        print(f"Initializing population using {self.config['initialization']['mode']} method...")
        self.population = list()
        while len(self.population) < self.config['popsize']:
            self.population.append(self.toolbox.createIndividual())

    def run(self) -> Any:
        """Execute the evolutionary search."""
        self.start_time = time.time()

        self._initialize()
        self._evaluation(self.population)

        self.best_individual = tools.selBest(self.population, 1)[0]

        self.monitor.compile(self)

        while not self._should_terminate():
            self.selected = self._selection(self.population, len(self.population))
            self.offspring = self._variation(self.selected)
            self._evaluation(self.offspring)
            self._semantic_similarity(self.selected, self.offspring)

            self.population = self.offspring
            self._update_best()

            self.monitor.compile(self)
            self.generation += 1

        self._finalize()
        return self.best_individual

    def _update_best(self):
        """Update the best individual found so far."""
        best_ind_curr = tools.selBest(self.population, 1)[0]
        self.best_individual = self.best_individual if self.best_individual.fitness.values[0] <= best_ind_curr.fitness.values[0] else best_ind_curr

    def _finalize(self):
        """Save results and cleanup resources."""
        if 'monitor_folder' in self.config:
            self.monitor.save_results(self.config)

        # Clean up multiprocessing pool
        if hasattr(self.toolbox, 'pool') and self.toolbox.pool is not None:
            self.toolbox.pool.close() 
            self.toolbox.pool.join()


    def _setup_termination_conditions(self) -> List[Callable]:
        """Configure termination conditions based on config."""
        conditions = []
        if self.config['termination']['max_generations'] < float('inf'):
            conditions.append(self._check_max_generations)
        return conditions
    
    def _check_max_generations(self) -> bool:
        """Check if maximum generations limit has been reached."""
        return self.generation >= self.config['termination']['max_generations']

    def _should_terminate(self) -> bool:
        """Check if any termination condition is met."""
        return any(condition() for condition in self.termination_conditions)
    
    @staticmethod
    def _pair_distance(parent, child, vector_attr: str, normalize: bool) -> float:
        """Euclidean distance between the output vectors of parent and child; with
        ``normalize``, divided by sqrt(n) (RMS difference per case). nan if an output is
        not finite."""
        parent_vec = np.asarray(getattr(parent, vector_attr, np.nan), dtype=np.float64)
        child_vec = np.asarray(getattr(child, vector_attr, np.nan), dtype=np.float64)
        if not (np.all(np.isfinite(parent_vec)) and np.all(np.isfinite(child_vec))):
            return np.nan
        diff = parent_vec - child_vec
        return float(np.sqrt(np.mean(diff ** 2))) if normalize else float(np.linalg.norm(diff))

    def _semantic_similarity(self, selected_population: List, offspring: List):
        """Semantic distance of each child to its parent (``child.parent_index``):
        ``semantic_distance`` on the semantic testset, ``semantic_distance_data`` on the
        training cases. nan for children identical to their parent."""
        moved = 0

        for position, child in enumerate(offspring):
            parent_index = getattr(child, 'parent_index', position)
            if not 0 <= parent_index < len(selected_population):
                child.semantic_distance = np.nan
                child.semantic_distance_data = np.nan
                continue
            parent = selected_population[parent_index]

            if str(child) == str(parent):
                child.semantic_distance = np.nan
                child.semantic_distance_data = np.nan
                continue
            moved += 1

            child.semantic_distance = (
                self._pair_distance(parent, child, 'output_vector_semantic', normalize=False)
                if self.semantic_testset_loaded else np.nan)
            child.semantic_distance_data = self._pair_distance(
                parent, child, 'output_vector', normalize=True)

        self.variation_move_rate = moved / len(offspring) if offspring else np.nan
