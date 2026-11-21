from typing import Tuple, Any, Dict

import numpy as np
from deap import gp
from functools import partial

def create_fitness_functions(config, pset, X_train, Y_train, X_test, Y_test, X_semantic) -> Tuple[Any, Any, Any]:
    """ Create fitness evaluation functions for training, testing and semantic distance.

    ``X_semantic`` may be None when semantic evaluation is disabled; its evaluator is then None.
    """
    train_eval = partial(evaluate_SR, X=X_train, Y=Y_train, pset=pset, config=config)
    test_eval  = partial(evaluate_SR, X=X_test,  Y=Y_test,  pset=pset, config=config)
    semantic_eval = None if X_semantic is None else partial(evaluate_Semantics, X=X_semantic, pset=pset, config=config)
    return train_eval, test_eval, semantic_eval

def evaluate_SR(
    individual, 
    X: np.ndarray,
    Y: np.ndarray,
    pset,
    config: dict,
    ) -> tuple[float, np.ndarray]:
    """
    Evaluate the symbolic regression individual on the provided dataset.

    Args:
        individual: The GP individual to evaluate
        X: Input features (array-like)
        Y: Target values (array-like)
        pset: Primitive set for GP
        config: Configuration dictionary

    Returns:
        tuple: (fitness value, output vector) for DEAP compatibility
    """
    try:
        func = gp.compile(expr=individual, pset=pset)  # Transform the tree expression into a callable function
    except Exception as e:
        print(f"Expression compilation failed: {e}")
        print(str(individual))
        return (float('inf'), np.array([]))

    output_vector = [np.longdouble(func(*x)) for x in X]

    if config['problem']['error_metric'] == 'RMSE':
        errors = [np.longdouble((pred - y)) ** 2 for pred, y in zip(output_vector, Y)]
        fitness = np.sqrt(sum(errors) / len(X))
    else:
        raise ValueError(f"Unknown error metric: {config['problem']['error_metric']}")

    if np.isnan(fitness):
        return (float('inf'), np.array(output_vector))

    return (fitness, np.array(output_vector))

def evaluate_Semantics(
    individual, 
    X: np.ndarray, 
    pset, 
    config: dict
    ) -> tuple[float, np.ndarray]:    
    """
    Evaluate the individual on a fixed, randomly generated, standardized dataset.
    
    Args:
        individual: The GP individual to evaluate
        X: Input features (array-like)
        pset: Primitive set for GP
            
    Returns:
        tuple: (fitness value, output vector) for DEAP compatibility
    """
    try:
        func = gp.compile(expr=individual, pset=pset)  # Transform the tree expression into a callable function
    except Exception as e:
        print(f"Expression compilation failed: {e}")
        print(str(individual))
        return (float('inf'), np.array([]))

    output_vector = [np.longdouble(func(*x)) for x in X]

    return (float('inf'), np.array(output_vector))
