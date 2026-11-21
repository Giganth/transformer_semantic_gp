"""
This module implements a probabilistic tree-building approach for generating random
symbolic mathematical expressions. It is based on the data generation framework from:
**Neural Symbolic Regression That Scales**
Paper: https://arxiv.org/pdf/2106.06427.pdf
GitHub: https://github.com/SymposiumOrganization/NeuralSymbolicRegressionThatScales/tree/main/scripts/data_creation

The generator creates expressions by:
1. Sampling operators (unary/binary) according to probability distributions
2. Building expression trees with controlled complexity
3. Ensuring all variables appear in the generated expression
4. Adding multiplicative and additive constants to variables
5. Evaluating expressions to generate synthetic datasets
Key Features:
- Configurable operator weights and probabilities
- Ensures minimum variable coverage
- Handles invalid expressions (NaN, Inf, extreme values)
- Uses prefix notation internally, converts to infix for SymPy processing

Classes:
    SyntheticProblemGenerator: Main generator class for creating random functions
"""

import numpy as np
import sympy as sp
from typing import List, Set, Tuple, Dict, Optional


class SyntheticProblemGenerator:

    """
    Generator for random mathematical expressions using a probabilistic tree-building approach.
    
    This class generates random symbolic expressions by sampling operators and operands
    according to specified probability distributions.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the synthetic problem generator.
        
        Args:
            config: Configuration dictionary containing:
                - data_generator['nl']: Node level factor for position sampling
                - data_generator['functions']: List of operator names
                - data_generator['operator_weights']: Dict mapping operator names to weights
                - data_generator['ephemerals'], data_generator['ephemeral_stepsize']:
                  range and step of the constants
                - problem['variables']: List of variable names
                - random_seed: Random seed for reproducibility
        """
        self.config = config
        
        data_gen_config = config['data_generator']
        problem_config = config['problem']
        
        self.variables = problem_config['variables']
        
        ephemerals = data_gen_config['ephemerals']
        stepsize = data_gen_config['ephemeral_stepsize']
        self.coefficients = np.round(
            np.arange(ephemerals[0], ephemerals[1] + stepsize, stepsize), 1
        ).tolist()

        self.min_ops = len(self.variables) - 1 # Minimum operators = number of variables - 1
        self.max_ops = len(self.variables) + 1 # Maximum operators = variables + 1
        self.nl = data_gen_config['nl']  # node level factor
        
        seed = config['random_seed']
        self.rng = np.random.default_rng(seed)
        
        self._extract_operators_from_config()
        
        self._precompute_distributions()
        
        print(f"Synthetic Problem Generator initialized:")
        print(f"  Variables: {self.variables}")
        print(f"  Coefficients: {len(self.coefficients)} values")
        print(f"  Operators range: {self.min_ops}-{self.max_ops}")
        print(f"  Unary/Binary probs: {self.p1:.2f}/{self.p2:.2f}")
        print(f"  Unary ops: {self.una_ops}")
        print(f"  Binary ops: {self.bin_ops}")
        print(f"  Random seed: {seed}")
    
    def _extract_operators_from_config(self):
        """Extract and categorize operators from config with weights."""
        self.una_ops = []
        self.bin_ops = []
        una_weights = []
        bin_weights = []
        
        operator_weights = self.config['data_generator']['operator_weights']
        
        unary_ops = {'neg', 'abs', 'sin', 'cos', 'tan', 'exp', 'ln', 'sqrt'}
        binary_ops = {'add', 'sub', 'mul', 'div', 'pow'}
        
        for func in self.config['data_generator']['functions']:
            weight = operator_weights.get(func, 1.0)
            if weight > 0:
                if func in unary_ops:
                    self.una_ops.append(func)
                    una_weights.append(weight)
                elif func in binary_ops:
                    self.bin_ops.append(func)
                    bin_weights.append(weight)
        
        una_weights = np.array(una_weights, dtype=np.float64)
        bin_weights = np.array(bin_weights, dtype=np.float64)

        # Unary vs binary split is the total weight mass of each arity class.
        total_weight = una_weights.sum() + bin_weights.sum()
        if total_weight == 0:
            raise ValueError("No operators with positive weight found in config")
        self.p1 = float(una_weights.sum() / total_weight)  # unary probability
        self.p2 = float(bin_weights.sum() / total_weight)  # binary probability

        if self.una_ops:
            self.una_ops_probs = una_weights / una_weights.sum()
        else:
            self.una_ops_probs = np.array([])

        if self.bin_ops:
            self.bin_ops_probs = bin_weights / bin_weights.sum()
        else:
            self.bin_ops_probs = np.array([])

        self.all_ops = self.una_ops + self.bin_ops
    
    def _precompute_distributions(self):
        """
        Precompute probability distributions for operator placement.
        """
        # D[e][n] represents number of different trees with n operators
        # that can be generated from e empty nodes
        # D(0, n) = 0
        # D(e, 0) = nl ** e
        # D(e, n) = nl * D(e - 1, n) + p1 * D(e, n - 1) + p2 * D(e + 1, n - 1)
        
        D = []
        D.append([0] + [self.nl ** i for i in range(1, 2 * self.max_ops + 1)])
        
        for n in range(1, 2 * self.max_ops + 1):  # number of operators
            s = [0]
            for e in range(1, 2 * self.max_ops - n + 1):  # number of empty nodes
                s.append(
                    self.nl * s[e - 1]
                    + self.p1 * D[n - 1][e]
                    + self.p2 * D[n - 1][e + 1]
                )
            D.append(s)
        
        assert all(len(D[i]) >= len(D[i + 1]) for i in range(len(D) - 1))
        
        D = [
            [D[j][i] for j in range(len(D)) if i < len(D[j])]
            for i in range(max(len(x) for x in D))
        ]
        
        self.ubi_dist = D

    def generate_random_function(
        self,
        min_ops: Optional[int] = None,
        max_ops: Optional[int] = None,
        require_variables: Optional[Set[str]] = None,
        max_attempts: int = 200
    ) -> Tuple[sp.Basic, str, Set[str]]:
        """
        Generate a random mathematical function.

        Args:
            min_ops: Minimum number of operators (defaults to self.min_ops)
            max_ops: Maximum number of operators (defaults to self.max_ops)
            require_variables: Variables that must appear. Defaults to all problem
                variables, guaranteeing full variable coverage.
            max_attempts: Max resampling attempts to satisfy coverage.

        Returns:
            (sympy_expr, prefix_string, variables_used)

        Raises:
            ValueError: If no valid, fully-covering expression is found.
        """
        if min_ops is None:
            min_ops = self.min_ops
        if max_ops is None:
            max_ops = self.max_ops
        if require_variables is None:
            require_variables = set(self.variables)

        for _ in range(max_attempts):
            nb_ops = self.rng.integers(min_ops, max_ops + 1)
            f_expr = self._generate_expr(nb_ops)
            f_expr = self._add_constants_to_prefix(f_expr)
            infix = self.prefix_to_infix(f_expr)

            try:
                f = self.process_equation(infix)
            except ValueError:
                continue

            if f == 0 or isinstance(f, (int, float, str)):
                continue

            variables_used = set(map(str, f.free_symbols))
            if require_variables and not require_variables.issubset(variables_used):
                continue

            return f, f_expr, variables_used

        raise ValueError(
            f"Could not generate an expression covering {require_variables} "
            f"after {max_attempts} attempts"
        )
    
    def _generate_expr(
        self,
        nb_total_ops: int
    ) -> List[str]:
        """
        Create a tree with exactly `nb_total_ops` operators.
        
        Args:
            nb_total_ops: Number of operators to include
            
        Returns:
            List of tokens in prefix notation
        """
        stack = [None]
        nb_empty = 1  # number of empty nodes
        l_leaves = 0  # left leaves - None states reserved for leaves
        t_leaves = 1  # total number of leaves (just used for sanity check)
        
        for nb_ops in range(nb_total_ops, 0, -1):
            skipped, arity = self.sample_next_pos_ubi(nb_empty, nb_ops)

            if arity == 1:
                op = str(self.rng.choice(self.una_ops, p=self.una_ops_probs))
            else:
                op = str(self.rng.choice(self.bin_ops, p=self.bin_ops_probs))
            
            nb_empty += arity - 1 - skipped  # created empty nodes - skipped future leaves
            t_leaves += arity - 1  # update number of total leaves
            l_leaves += skipped  # update number of left leaves
            
            pos = [i for i, v in enumerate(stack) if v is None][l_leaves]
            stack = (
                stack[:pos]
                + [op]
                + [None for _ in range(arity)]
                + stack[pos + 1 :]
            )
        
        assert len([1 for v in stack if v in self.all_ops]) == nb_total_ops
        assert len([1 for v in stack if v is None]) == t_leaves
        
        # Generate leaves - ensure all variables appear at least once
        leaves = []
        remaining_vars = self.variables.copy()
        
        for i in range(t_leaves):
            if remaining_vars:
                leaf = str(self.rng.choice(remaining_vars))
                remaining_vars.remove(leaf)
            else:
                leaf = str(self.rng.choice(self.variables))
            leaves.append([leaf])
        
        for pos in range(len(stack) - 1, -1, -1):
            if stack[pos] is None:
                stack = stack[:pos] + leaves.pop() + stack[pos + 1 :]
        
        assert len(leaves) == 0
        return stack
    
    def sample_next_pos_ubi(
        self,
        nb_empty: int,
        nb_ops: int
    ) -> Tuple[int, int]:
        """
        Sample the position of the next node (unary-binary case).
        Sample a position in {0, ..., `nb_empty` - 1}, along with an arity.
        
        Args:
            nb_empty: Number of empty slots available
            nb_ops: Number of operators remaining to place
            
        Returns:
            Tuple of (position_skipped, arity)
        """
        assert nb_empty > 0
        assert nb_ops > 0
        
        # Single-arity configs: forced choice.
        if self.p1 == 0:
            return self.rng.integers(0, nb_empty), 2
        if self.p2 == 0:
            return self.rng.integers(0, nb_empty), 1

        # Otherwise sample arity from the tree-count distribution (allows a
        # unary root); variable coverage is enforced in generate_random_function.
        probs = []
        
        for i in range(nb_empty):
            probs.append(
                (self.nl ** i) * self.p1 * self.ubi_dist[nb_empty - i][nb_ops - 1]
            )
        
        for i in range(nb_empty):
            probs.append(
                (self.nl ** i) * self.p2 * self.ubi_dist[nb_empty - i + 1][nb_ops - 1]
            )
        
        total = self.ubi_dist[nb_empty][nb_ops]
        if total == 0:
            e = self.rng.integers(0, 2 * nb_empty)
        else:
            probs = [p / total for p in probs]
            probs = np.array(probs, dtype=np.float64)
            e = self.rng.choice(2 * nb_empty, p=probs)
        
        arity = 1 if e < nb_empty else 2
        e = e % nb_empty
        
        return e, arity
    
    def prefix_to_infix(
        self,
        expr: List[str]
    ) -> str:
        """
        Convert prefix notation to infix notation.
        
        Args:
            expr: Expression in prefix notation (list of tokens)
            
        Returns:
            Expression in infix notation
        """
        if not expr:
            raise ValueError("Empty expression")
        
        op_map = {
            'add': '+',
            'sub': '-',
            'mul': '*',
            'div': '/',
            'pow': '**',
            'neg': '-',
            'abs': 'abs',
            'sin': 'sin',
            'cos': 'cos',
            'tan': 'tan',
            'exp': 'exp',
            'ln': 'log',
            'sqrt': 'sqrt'
        }
        
        unary_ops = {'neg', 'abs', 'sin', 'cos', 'tan', 'exp', 'ln', 'sqrt'}
        binary_ops = {'add', 'sub', 'mul', 'div', 'pow'}
        all_ops = unary_ops | binary_ops
        
        def convert(tokens, index=[0]):
            if index[0] >= len(tokens):
                raise ValueError("Invalid prefix expression: unexpected end")
            
            token = tokens[index[0]]
            index[0] += 1
            
            if token in all_ops:
                arity = 2  # Default to binary
                if token in unary_ops:
                    arity = 1
                
                if arity == 1:
                    arg = convert(tokens, index)
                    op_symbol = op_map.get(token, token)
                    return f"{op_symbol}({arg})"
                elif arity == 2:
                    left = convert(tokens, index)
                    right = convert(tokens, index)
                    op_symbol = op_map.get(token, token)
                    return f"({left} {op_symbol} {right})"
            else:
                return token
        
        try:
            result = convert(expr[:])  # Make a copy to avoid modifying original
            return result
        except Exception as e:
            raise ValueError(f"Failed to convert prefix to infix: {e}")
    
    def process_equation(self, infix: str) -> sp.Basic:
        """
        Process and simplify equation using SymPy.
        
        Args:
            infix: Expression in infix notation
            
        Returns:
            Processed SymPy expression
        """
        try:
            expr = sp.parse_expr(infix, evaluate=True)
            
            symbols = set([str(x) for x in expr.free_symbols])
            if not symbols:
                raise ValueError("No variables in the expression")

            expr = sp.cancel(expr)

            return expr
        except Exception as e:
            raise ValueError(f"Failed to process equation: {e}")
    
    def _add_constants_to_prefix(self, prefix_expr: List[str]) -> List[str]:
        """
        Add multiplicative and additive constants to variables in prefix notation.
        
        This function wraps each variable with: add(mul(coeff, var), coeff)
        
        Args:
            prefix_expr: Expression in prefix notation
            
        Returns:
            Modified prefix expression with constants added
        """
        result = []
        
        for token in prefix_expr:
            if token in self.variables:
                # Wrap variable with: add(mul(coeff, var), coeff)
                nonzero_coeffs = [c for c in self.coefficients if abs(c) != 0]
                mult_coeff = str(float(self.rng.choice(nonzero_coeffs)))
                add_coeff = str(float(self.rng.choice(self.coefficients)))
                
                result.extend(['add', 'mul', mult_coeff, token, add_coeff])
            else:
                result.append(token)
        
        return result
    
    def generate_dataset(
        self,
        num_samples: Optional[int] = None,
        max_retries: int = 10
    ) -> Tuple[np.ndarray, np.ndarray, sp.Basic, str, Set[str]]:
        """
        Generate a synthetic dataset using a random mathematical function.

        This method generates a random function, evaluates it on random input data,
        and retries on invalid values. Noise is added by the caller.
        
        Args:
            num_samples: Number of samples to generate (uses config if None)
            max_retries: Maximum attempts to generate valid function
            
        Returns:
            Tuple of (X, Y, sympy_expr, string_repr, variables_used) where:
                - X: Input data array of shape (num_samples, num_variables)
                - Y: Output data array of shape (num_samples,), without noise (not standardized)
                - sympy_expr: SymPy expression
                - string_repr: String representation of the expression
                - variables_used: Set of variable names in the expression
                
        Raises:
            ValueError: If unable to generate valid function after max_retries
        """
        from sympy import lambdify
        import pandas as pd
        
        if num_samples is None:
            num_samples = self.config['data_generator']['num_samples_dataset']
        
        num_variables = len(self.variables)
        
        for attempt in range(max_retries):
            try:
                sympy_expr, string_repr, vars_used = self.generate_random_function()
                
                func = lambdify(self.variables, sympy_expr, modules='numpy')
                
                # Inputs drawn uniformly from [1, 5]
                X = self.rng.uniform(1,5, size=(num_samples, num_variables))
                try:
                    Y = func(*X.T)  # Transpose to pass each variable as a separate array
                except (ZeroDivisionError, OverflowError, ValueError, RuntimeWarning) as e:
                    print(f"Attempt {attempt + 1}: Exception during evaluation - {e}, retrying...")
                    continue
                
                if not np.all(np.isfinite(Y)):
                    invalid_count = np.sum(~np.isfinite(Y))
                    print(f"Attempt {attempt + 1}: {invalid_count} invalid values encountered, retrying...")
                    continue
                
                # Check for extreme values (very positive/negative)
                Y_max = np.max(np.abs(Y))
                if Y_max > 1e4:  # Threshold for extreme values
                    print(f"Attempt {attempt + 1}: Extreme values detected (max={Y_max:.2e}), retrying...")
                    continue

                print(f"Dataset shape: X={X.shape}, Y={Y.shape}")
                print(f"Y statistics: mean={np.mean(Y):.3f}, std={np.std(Y):.3f}, "
                      f"min={np.min(Y):.3f}, max={np.max(Y):.3f}")

                return X, Y, sympy_expr, string_repr, vars_used
                
            except ValueError as e:
                print(f"Attempt {attempt + 1}: Invalid expression - {e}")
                continue
            except Exception as e:
                print(f"Attempt {attempt + 1}: Error - {e}")
                continue
        
        raise ValueError(f"Failed to generate valid dataset after {max_retries} attempts")