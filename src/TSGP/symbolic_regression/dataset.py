"""
Dataset loading and preprocessing for symbolic regression.
"""

from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Any, Dict

# The benchmark CSVs live next to this module, so the package finds them from any
# working directory.
DATASETS = Path(__file__).resolve().parent / "datasets"

def load_dataset(file_path: str, subsample = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset from CSV file.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        tuple: (X, Y) where X is features and Y is target
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is malformed or empty
    """
    print(f"Loading dataset from: {file_path}")
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    except Exception as e:
        raise ValueError(f"Error loading dataset from {file_path}: {str(e)}")

    if subsample and len(data) > 10000:
        print(f"Subsampling dataset from {len(data)} rows to 10000")
        data = data.sample(n=10000, random_state=42).reset_index(drop=True)
    
    if data.empty:
        raise ValueError("Dataset is empty")
    if data.isnull().values.any():
        raise ValueError("Dataset contains NaN values")
    if data.shape[1] < 2:
        raise ValueError("Dataset must have at least 2 columns (features + target)")
    
    X = data.iloc[:, :-1].values
    Y = data.iloc[:, -1].values
    return X, Y

def train_test_split(X: np.ndarray, Y: np.ndarray, test_size: float = 0.25,
                    standardization: bool = True, random_state: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split dataset into train and test sets.

    Args:
        X: Feature matrix
        Y: Target vector
        test_size: Proportion of test set
        standardization: Whether to standardize features and target
        random_state: Random seed for reproducibility

    Returns:
        tuple: (X_train, Y_train, X_test, Y_test)
    """
    if random_state is not None:
        np.random.seed(random_state)

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    split_idx = int(X.shape[0] * (1 - test_size))
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    X_train, X_test = X[train_indices], X[test_indices]
    Y_train, Y_test = Y[train_indices], Y[test_indices]

    if standardization:
        XScaler = StandardScaler()
        yScaler = StandardScaler()
        X_train = XScaler.fit_transform(X_train)
        X_test = XScaler.transform(X_test)
        Y_train = yScaler.fit_transform(Y_train.reshape(-1, 1)).flatten()
        Y_test = yScaler.transform(Y_test.reshape(-1, 1)).flatten()

    return X_train, Y_train, X_test, Y_test

def prepare_dataset(config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Load and prepare dataset for symbolic regression.

    Args:
        config: Configuration dictionary containing problem settings

    Returns:
        tuple: (X_train, Y_train, X_test, Y_test, X_semantic, variable_names)
    """
    dataset_name = config['problem']['dataset']
    file_path = DATASETS / f"{dataset_name}.csv"
    X, Y = load_dataset(str(file_path), config['problem']['subsample'])

    num_variables = X.shape[1]

    X_train, Y_train, X_test, Y_test = train_test_split(
        X, Y,
        test_size=config['problem']['test_split_ratio'],
        standardization=config['problem']['standardization'],
        random_state=config['random_seed']
    )

    if config['problem'].get('load_semantic_testset', True):
        semantic_path = DATASETS / "Semantic_Testset.csv"
        print(f"Loading semantic testset from: {semantic_path}")
        semantic_data = pd.read_csv(semantic_path)
        X_semantic = semantic_data.values  # All columns are features

        if num_variables > X_semantic.shape[1]:
            raise ValueError(
                f"Semantic testset has only {X_semantic.shape[1]} features but dataset "
                f"'{config['problem']['dataset']}' has {num_variables}. Set "
                f"problem.load_semantic_testset: False, or regenerate Semantic_Testset.csv "
                f"with at least {num_variables} columns."
            )
        X_semantic = X_semantic[:, :num_variables]
    else:
        X_semantic = None

    variable_names = [f'x{i}' for i in range(num_variables)]

    return X_train, Y_train, X_test, Y_test, X_semantic, variable_names