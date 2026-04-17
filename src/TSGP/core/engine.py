from abc import ABC, abstractmethod
from typing import List, Dict, Any, Callable
import time
import random
import numpy as np

from TSGP.symbolic_regression.evaluation import create_fitness_functions
from TSGP.core.pset import create_pset
from TSGP.core.toolbox import create_toolbox
from TSGP.symbolic_regression.dataset import prepare_dataset
from .monitoring import Monitor
from deap import tools
import os

class BaseEvolutionaryAlgorithm(ABC):
    """Base class for evolutionary search.

    This class provides the core framework for implementing evolutionary search.
    Subclasses should implement abstract methods for variation operators.

    Public API:
        __init__(config): Initialize the algorithm with configuration
        run(): Execute the evolutionary search and return the best solution

    Protected methods (override in subclasses):
        _variation(): Apply variation operators (crossover, mutation, transformer semantic variation, etc.) - ABSTRACT
        
    Protected methods (internal framework):
        _set_random_seed(): Set random seed for reproducibility
        _initialize(): Create and evaluate initial population
        _evaluation(): Evaluate fitness of individuals in population
        _selection(): Select parents for reproduction
        _setup_termination_conditions(): Configure stopping criteria
        _check_max_generations(): Check if max generations limit reached
        _should_terminate(): Check if any termination condition is met
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the algorithm with configuration."""
        self.config = config
        self.generation = 0
        self.best_individual = None
        self.start_time = None
        self.termination_conditions = self._setup_termination_conditions()
         
        # set random seed
        self._set_random_seed()

        # Prepare Dataset (returns number of variables)
        # Allow subclasses to provide custom dataset function
        if hasattr(self, '_custom_dataset'):
            X_train, Y_train, X_test, Y_test, X_semantic, variable_names, underlying_function = self._custom_dataset()
            print("Using custom dataset function for data preparation.")
            print(f"Underlying function: {underlying_function}")
        else:
            X_train, Y_train, X_test, Y_test, X_semantic, variable_names = prepare_dataset(config)
    
        # Dynamically adjust variables in config based on dataset dimensionality
        config['problem']['variables'] = variable_names
        
        # Create toolbox
        self.pset = create_pset(config)
        
        # Create Fitness Function
        evaluation_function, evaluation_function_test, evaluation_function_semantic = create_fitness_functions(config, self.pset, X_train, Y_train, X_test, Y_test, X_semantic)

        # Create Toolbox
        self.toolbox = create_toolbox(config, self.pset, evaluation_function, evaluation_function_test, evaluation_function_semantic)

        # Create monitor
        self.monitor = Monitor(config, console_out=config['console_out'])

    def _set_random_seed(self):
        """Set random seed for reproducibility across all components."""
        seed = self.config['random_seed'] 
        
        # Set Python's random seed
        random.seed(seed)
        
        # Set NumPy's random seed
        np.random.seed(seed)
        
        # Set Python hash seed (for consistent dictionary ordering)
        os.environ['PYTHONHASHSEED'] = str(seed)

    @abstractmethod
    def _variation(self, population: List) -> List:
        """Apply variation operators to the population."""
        # defined in the subclasses
        pass

    def _selection(self, population: List, num_selected: int) -> List:
        """Select parents for reproduction."""
        return self.toolbox.select(population, num_selected)
    
    def _evaluation(self, population: List, evaluate_train=True, evaluate_test=True, evaluate_semantics=True) -> None:
        """Evaluate fitness of individuals in the population."""
        nan = float('nan')

        # Evaluate or create placeholder results
        results_train = (
            self.toolbox.map(self.toolbox.evaluate, population) if evaluate_train
            else [(nan, nan) for _ in population]
        )
        results_test = (
            self.toolbox.map(self.toolbox.evaluate_test, population) if evaluate_test
            else [(nan, nan) for _ in population]
        )
        results_semantic = (
            self.toolbox.map(self.toolbox.evaluate_semantic, population) if evaluate_semantics
            else [(nan, nan) for _ in population]
        )
        
        # Assign results to individuals
        for ind, (fit, out_vec), (fit_test, out_vec_test), (fit_semantic, out_vec_semantic) in zip(
            population, results_train, results_test, results_semantic
        ):
            ind.fitness.values = (fit,)
            ind.output_vector = out_vec
            ind.fitness_test.values = (fit_test,)
            ind.output_vector_test = out_vec_test
            ind.fitness_semantic.values = (fit_semantic,)
            ind.output_vector_semantic = out_vec_semantic
            
    def _initialize(self) -> None:
        """Create the initial population."""
        self.population = list()
        while len(self.population) < self.config['popsize']:
            ind = self.toolbox.createIndividual()
            self.population.append(ind)

    def run(self) -> Any:
        """Execute the evolutionary search."""
        self.start_time = time.time()

        # Initialize search
        self._initialize()
        self._evaluation(self.population)
        self.best_individual = tools.selBest(self.population, 1)[0]
        self.monitor.compile(self)

        while not self._should_terminate():

            # Selection
            self.selected = self._selection(self.population, len(self.population))
            
            # Variation
            self.offspring = self._variation(self.selected)

            # Evaluation
            self._evaluation(self.offspring)

            # Calculate Semantic Similarity
            self._semantic_similarity(self.selected, self.offspring)

            # Replacement
            self.population = self.offspring

            # Update best individual
            self._update_best()

            # Log generation
            self.monitor.compile(self)

            # Increment generation counter
            self.generation += 1

        # Save results and cleanup
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
    
    def _semantic_similarity(self, selected_population: List, offspring: List):
        """Compute semantic similarity between selected population and offspring."""
        for parent, child in zip(selected_population, offspring):
            # Compute Euclidean distance between output vectors
            if hasattr(parent, 'output_vector_semantic') and hasattr(child, 'output_vector_semantic'):
                parent_vec = parent.output_vector_semantic
                child_vec = child.output_vector_semantic
                
                # Compute semantic distance (Euclidean distance)
                semantic_distance = np.linalg.norm(parent_vec - child_vec)
                
                # Store semantic distance as attribute
                child.semantic_distance = semantic_distance
            else:
                child.semantic_distance = None
            