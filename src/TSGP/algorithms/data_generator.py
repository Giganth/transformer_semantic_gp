from sklearn.datasets import make_regression
from TSGP.algorithms.standard_gp import StandardGP
from TSGP.core.simplify import SymbolicSimplifier
from typing import Dict, Any, List, Tuple
import time
import numpy as np
from deap import tools, creator

from TSGP.symbolic_regression.dataset import load_dataset, train_test_split


class DataGenerator(StandardGP):
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize DataGenerator with optional custom dataset function.
        
        Args:
            config: Configuration dictionary
            custom_dataset_fn: Optional function that returns (X_train, Y_train, X_test, Y_test, X_semantic, variable_names)
                              If provided, this will be used instead of _prepare_custom_dataset().
        """
        # Store custom dataset function before calling parent __init__
        self._custom_dataset = self._prepare_custom_dataset
        
        super().__init__(config)
        
        # Initialize symbolic simplifier
        self.simplifier = SymbolicSimplifier(self.pset)
    
    def _prepare_custom_dataset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Generate a synthetic regression dataset for data generation.

        Supports two dataset generation modes controlled by
        ``config['data_generator']['dataset_generation']``:

        * ``'make_regression'`` *(default)* – generates a random linear
          regression dataset via :func:`sklearn.datasets.make_regression`.
        * ``'target_function'`` – evaluates a user-supplied symbolic
          expression given in ``config['data_generator']['target_function']``
          (e.g. ``"sin(x0) + x1**2"``) on uniformly-sampled inputs.
          The expression is evaluated using NumPy, so standard NumPy
          functions (``sin``, ``cos``, ``exp``, ``log``, ``sqrt``, …) are
          available.  The number of variables is inferred from
          ``config['data_generator']['num_features']``.

        Returns:
            tuple: (X_train, Y_train, X_test, Y_test, X_semantic,
                    variable_names, function_string)
        """
        num_samples_dataset = self.config['data_generator']['num_samples_dataset']
        num_features = self.config['data_generator']['num_features']
        noise_level = self.config['data_generator']['noise_level']
        random_seed = self.config['random_seed']
        dataset_generation = self.config['data_generator'].get('dataset_generation', 'make_regression')

        if dataset_generation == 'target_function':
            X, Y, function_string = self._generate_from_target_function(
                num_samples_dataset, num_features, noise_level, random_seed
            )
        else:
            # Default: make_regression
            X, Y, coef = make_regression(
                n_samples=num_samples_dataset,
                n_features=num_features,
                n_informative=num_features,
                effective_rank=num_features,
                noise=noise_level,
                coef=True,
                random_state=random_seed,
            )
            function_string = " + ".join([f"{c:.3f}*x{i}" for i, c in enumerate(coef)])

        # Get number of variables from dataset
        num_variables = X.shape[1]
        
        X_train, Y_train, X_test, Y_test = train_test_split(
            X, Y, 
            test_size=self.config['problem']['test_split_ratio'], 
            standardization=self.config['problem']['standardization'], 
            random_state=self.config['random_seed']
        )

        # Get Randomly Generated Dataset for Testing Semantics
        X_semantic, Y = load_dataset("src/TSGP/symbolic_regression/datasets/Semantic_Testset.csv", self.config['problem']['subsample'])
        
        # Remove columns so that number of variables matches
        X_semantic = X_semantic[:, :num_variables]

        variable_names = [f'x{i}' for i in range(num_variables)]

        return X_train, Y_train, X_test, Y_test, X_semantic, variable_names, function_string

    def _generate_from_target_function(
        self,
        num_samples: int,
        num_features: int,
        noise_level: float,
        random_seed: int,
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Generate a dataset by evaluating a user-defined target function.

        The target function string is read from
        ``config['data_generator']['target_function']``.  Variables must be
        named ``x0``, ``x1``, … matching ``num_features``.  Input values are
        drawn uniformly from the range specified by
        ``config['data_generator'].get('input_range', [-1, 1])``.

        Args:
            num_samples: Number of data points to generate.
            num_features: Number of input variables (``x0`` … ``x{n-1}``).
            noise_level: Standard deviation of Gaussian noise added to ``Y``.
            random_seed: Random seed for reproducibility.

        Returns:
            tuple: (X, Y, function_string) where X has shape
            ``(num_samples, num_features)`` and Y has shape ``(num_samples,)``.

        Raises:
            ValueError: If ``target_function`` is missing from the config or
                evaluation fails.
        """
        target_function = self.config['data_generator'].get('target_function')
        if not target_function:
            raise ValueError(
                "dataset_generation is set to 'target_function' but "
                "'target_function' is not specified in the data_generator config."
            )

        input_range = self.config['data_generator'].get('input_range', [-1.0, 1.0])
        low, high = float(input_range[0]), float(input_range[1])

        rng = np.random.default_rng(random_seed)
        X = rng.uniform(low, high, size=(num_samples, num_features))

        # Build a namespace where x0, x1, … resolve to column vectors so that
        # the expression can be evaluated with standard NumPy broadcasting.
        namespace = {f'x{i}': X[:, i] for i in range(num_features)}
        namespace.update({
            name: getattr(np, name)
            for name in dir(np)
            if not name.startswith('_') and callable(getattr(np, name))
        })

        try:
            Y = eval(target_function, {"__builtins__": {}}, namespace)  # noqa: S307
            Y = np.asarray(Y, dtype=float).flatten()
        except Exception as exc:
            raise ValueError(
                f"Failed to evaluate target_function '{target_function}': {exc}"
            ) from exc

        if Y.shape[0] != num_samples:
            raise ValueError(
                f"Evaluated target function produced {Y.shape[0]} values, "
                f"expected {num_samples}."
            )

        if noise_level > 0:
            Y = Y + rng.normal(0.0, noise_level, size=Y.shape)

        return X, Y, target_function

    def run(self):
        """Execute the evolutionary search."""
        self.start_time = time.time()
        self.archive = {}

        # Initialize and evaluate population
        self._initialize()
        self._evaluation(self.population, evaluate_semantics=False)
        self.best_individual = tools.selBest(self.population, 1)[0]

        # Archive initial population
        self._update_archive(self.population)
        
        self.monitor.compile(self)

        while not self._should_terminate():
            # Selection and Variation
            self.selected = self._selection(self.population, len(self.population))
            self.offspring = self._variation(self.selected)

            # Evaluate and archive offspring
            self._evaluation(self.offspring, evaluate_semantics=False)
            self._update_archive(self.offspring)

            # Replacement
            self.population = self.offspring
            self._update_best()
            self.monitor.compile(self)
            self.generation += 1

        # Save results and cleanup
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
        
        # Clean up multiprocessing pool
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
        
        # Wrap simplified tree in Individual (required for fitness attribute)
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
    
    def _should_terminate(self) -> bool:
        """Check termination conditions."""
        target_size = self.config['data_generator']['num_samples_archive']
        return len(self.archive) >= target_size