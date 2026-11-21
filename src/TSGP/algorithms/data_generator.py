from sklearn.datasets import make_regression
from TSGP.algorithms.standard_gp import StandardGP
from TSGP.core.simplify import SymbolicSimplifier
from TSGP.symbolic_regression.synthetic_problem_generator import SyntheticProblemGenerator
from typing import Dict, Any, List, Tuple
import time
import numpy as np
from deap import gp, tools, creator

from TSGP.symbolic_regression.dataset import DATASETS, load_dataset, train_test_split

class DataGenerator(StandardGP):
    def __init__(self, config: Dict[str, Any]):
        """Initialize the DataGenerator; its dataset is drawn by _prepare_custom_dataset."""
        # Set before the parent __init__, which loads the dataset through it.
        self._custom_dataset = self._prepare_custom_dataset
        self._prepared_dataset = None

        super().__init__(config)
        
        self.simplifier = SymbolicSimplifier(self.pset)

    def _prepare_custom_dataset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Draw a target function and its dataset.

        dataset_generation selects how:
        - 'random_function_target': a random function from SyntheticProblemGenerator
        - 'make_regression': a random linear function from sklearn's make_regression

        Returns:
            tuple: (X_train, Y_train, X_test, Y_test, X_semantic, variable_names)
        """

        if self._prepared_dataset is not None:
            return self._prepared_dataset

        num_features = self.config['data_generator']['num_features']

        self.config['problem']['variables'] = [f'x{i}' for i in range(num_features)]

        dataset_mode = self.config['data_generator']['dataset_generation']
        noise_level = self.config['data_generator']['noise_level']

        if dataset_mode == 'random_function_target':
            self.problem_generator = SyntheticProblemGenerator(self.config)
            X, Y, function, _, _ = self.problem_generator.generate_dataset()

            # Add Gaussian noise to outputs (noise_level is fraction of Y's std)
            if noise_level > 0.0:
                rng = np.random.default_rng(self.config['random_seed'])
                noise = rng.normal(0, noise_level * np.std(Y), size=Y.shape)
                Y = Y + noise

            self.generated_function = str(function)

        elif dataset_mode == 'make_regression':
            X, Y, coef = make_regression(
                n_samples=self.config['data_generator']['num_samples_dataset'],
                n_features=num_features,
                n_informative=num_features,
                effective_rank=num_features,
                noise=noise_level,
                coef=True,
                random_state=self.config['random_seed']
            )
            self.generated_function = ' + '.join([f'{c:.6f}*x{i}' for i, c in enumerate(coef)])

        else:
            raise ValueError(f"Unknown dataset generation mode: {dataset_mode}")

        print(f"Underlying Function: {self.generated_function}")
        X_train, Y_train, X_test, Y_test = train_test_split(
            X, Y,
            test_size=self.config['problem']['test_split_ratio'],
            standardization=self.config['problem']['standardization'],
            random_state=self.config['random_seed']
        )

        # Required: the archive stores each individual's output on this basis.
        X_semantic, _ = load_dataset(str(DATASETS / "Semantic_Testset.csv"))
        if num_features > X_semantic.shape[1]:
            raise ValueError(
                f"Semantic testset has only {X_semantic.shape[1]} usable features but the "
                f"generator needs {num_features}. Regenerate Semantic_Testset.csv with more columns."
            )
        X_semantic = X_semantic[:, :num_features]  # Match number of variables
        
        variable_names = [f'x{i}' for i in range(num_features)]

        self._prepared_dataset = (
            X_train, Y_train, X_test, Y_test, X_semantic, variable_names
        )
        return self._prepared_dataset


    def run(self):
        """Execute the evolutionary search."""
        self.start_time = time.time()
        self.archive = {}

        self._initialize()
        self._evaluation(self.population, evaluate_semantics=False)
        self.best_individual = tools.selBest(self.population, 1)[0]
        
        self._update_archive(self.population)
        
        self.monitor.compile(self)

        while not self._should_terminate():
            self.selected = self._selection(self.population, len(self.population))
            
            self.offspring = self._variation(self.selected)
            
            self._evaluation(self.offspring, evaluate_semantics=False)
            self._update_archive(self.offspring)

            self.population = self.offspring
            
            self._update_best()
            self.monitor.compile(self)
            self.generation += 1

        self._finalize()
        return self.best_individual
    
    def _update_archive(self, population: List):
        """Add population to archive after optional simplification."""
        pop_to_archive = population
        if self.config['data_generator']['simplification']:
            pop_to_archive = [self._simplify(ind) for ind in population]
            self._evaluation(pop_to_archive, evaluate_train=False, evaluate_test=False, evaluate_semantics=True)
        
        for ind in pop_to_archive:
            expr_str = str(ind)
            if expr_str not in self.archive:
                self.archive[expr_str] = np.array(ind.output_vector_semantic)
    
    def _finalize(self):
        """Save results and cleanup resources."""
        if 'monitor_folder' in self.config:
            self.monitor.save_results(self.config)
        
        self._save_archive()
        self._save_generated_function()
        
        if hasattr(self.toolbox, 'pool') and self.toolbox.pool is not None:
            self.toolbox.pool.close()
            self.toolbox.pool.join()
    
    def _simplify(self, individual) -> Any:
        """
        Simplify the given individual using SymPy.
        
        Args:
            individual: DEAP individual to simplify
            
        Returns:
            Simplified individual (or original if simplification fails)
        """
        simplified_tree = self.simplifier.simplify(
            individual,
            evaluate_expression=True,
            evaluate_constants=False
        )
        
        simplified_ind = creator.Individual(simplified_tree)
        return simplified_ind
    
    def _save_archive(self) -> None:
        """Save the archive as pickle file."""
        import pickle
        import os

        archive_folder = self.config['data_generator']['archive_folder']
        archive_path = self.config['data_generator'].get(
            'archive_path', 
            os.path.join(archive_folder, "behavior_archive.pkl")
        )
        
        print(f"Saving {len(self.archive)} expressions to {archive_path}")
        
        os.makedirs(os.path.dirname(archive_path), exist_ok=True)
        with open(archive_path, 'wb') as f:
            pickle.dump(list(self.archive.items()), f)
    
    def _save_generated_function(self) -> None:
        """Save the generated function expression to a file."""
        import os
        
        archive_folder = self.config['data_generator']['archive_folder']
        functions_path = self.config['data_generator'].get(
            'function_path', 
            os.path.join(archive_folder, "generated_function.txt")
        )
        print(f"Saving generated function to {functions_path}")
        os.makedirs(os.path.dirname(functions_path), exist_ok=True)
        with open(functions_path, 'w') as f:
            f.write(self.generated_function)
    
    def _should_terminate(self) -> bool:
        """Check termination conditions."""
        target_size = self.config['data_generator']['num_samples_archive']
        max_generations = self.config['termination']['max_generations']
        return len(self.archive) >= target_size or self.generation >= max_generations