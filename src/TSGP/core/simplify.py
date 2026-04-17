"""
Symbolic simplification for genetic programming individuals using SymPy.

This module provides functionality to simplify DEAP individuals by converting them
to SymPy expressions, applying symbolic simplification, and converting back to DEAP format.
"""

import re
from typing import List, Dict, Tuple, Any
import sympy as sp
from deap import gp

class SymbolicSimplifier:
    """
    Simplifies genetic programming individuals using symbolic computation.
    
    Converts DEAP expression trees to SymPy expressions, applies simplification,
    and converts back to DEAP format while preserving numerical precision.
    """
    
    # Mapping from SymPy operators to Deap function names
    SYMPY_TO_PREFIX = {
        sp.Add: "add",
        sp.Mul: "mul",
        sp.Pow: "pow",
        sp.exp: "exp",
        sp.log: "ln",
        sp.Abs: "abs",
        sp.sin: "sin",
        sp.cos: "cos",
        sp.tan: "tan",
        sp.asin: "asin",
        sp.acos: "acos",
        sp.atan: "atan",
        sp.sinh: "sinh",
        sp.cosh: "cosh",
        sp.tanh: "tanh",
    }
    
    # Mapping from Deap function names to SymPy operators
    PREFIX_TO_SYMPY = {
        "add": "+",
        "mul": "*",
        "sub": "-",
        "pow": "**",
        "div": "/",
        "protected_division": "/",
        "protected_log": "log",
        "protected_exp": "exp",
        "neg": "neg",
        "ln": "log",
        "abs": "abs",
        "sin": "sin",
        "cos": "cos",
        "tan": "tan",
        "asin": "asin",
        "acos": "acos",
        "atan": "atan",
        "sinh": "sinh",
        "cosh": "cosh",
        "tanh": "tanh",
    }
    
    def __init__(self, pset: gp.PrimitiveSet):
        """
        Initialize the simplifier.
        
        Args:
            pset: DEAP primitive set defining the function and terminal set
        """
        self.pset = pset
    
    def simplify(
        self, 
        individual: gp.PrimitiveTree, 
        evaluate_expression: bool = True,
        evaluate_constants: bool = False
    ) -> gp.PrimitiveTree:
        """
        Simplify a DEAP individual using SymPy symbolic computation.
        
        Args:
            individual: DEAP individual to simplify
            evaluate_expression: Whether to evaluate the expression during simplification
            evaluate_constants: Whether to evaluate constant subexpressions (e.g., 2*3 -> 6)
        
        Returns:
            Simplified DEAP individual, or original if simplification fails
            
        Note:
            SymPy handles division by zero differently than DEAP. If the primitive set
            contains division, simplification may not be exact.
        """
        try:
            nodes, edges, labels = gp.graph(individual)

            # Extract expressions from labels
            expressions = list()
            for node in range(len(labels)):
                expressions.append(labels[node])

            # Handle constant preservation
            if not evaluate_constants:
                expressions, placeholder_dict = self._preserve_constants(expressions)
            else:
                placeholder_dict = None
            
            # Build SymPy-compatible function string
            function_string = self._build_function_string(nodes, edges, expressions, 0)
            
            # Parse and simplify with SymPy
            parsed_expr = sp.parse_expr(
                function_string, 
                evaluate=evaluate_expression
            )
            
            # Restore original constants if needed
            if placeholder_dict:
                simplified_str = str(parsed_expr)
                simplified_str = self._restore_constants(simplified_str, placeholder_dict)
                parsed_expr = sp.parse_expr(simplified_str, evaluate=False)
            
            # Check for invalid expressions
            if parsed_expr.has(sp.zoo) or parsed_expr.has(sp.I) or parsed_expr.has(sp.nan):
                return individual
            
            # Convert back to DEAP format
            prefix_list = self._sympy_to_prefix(parsed_expr)
            prefix_str = self._format_prefix_list(prefix_list)
            
            simplified_individual = gp.PrimitiveTree.from_string(prefix_str, self.pset)
        
            return simplified_individual
            
        except (ValueError, AttributeError, SyntaxError, ZeroDivisionError, TypeError):
            # Return original if simplification fails
            return individual
    
    def _preserve_constants(
        self, 
        expressions: List[Any]
    ) -> Tuple[List[Any], Dict[str, float]]:
        """
        Replace float constants with placeholders (except -1, 0, 1).
        
        Args:
            expressions: List of expression elements
            
        Returns:
            Tuple of (modified expressions, placeholder dictionary)
        """
        constants_to_optimize = {-1.0, -0.0, 0.0, 1.0}
        
        # Extract float values to preserve
        float_values = [
            item for item in expressions 
            if isinstance(item, float) and item not in constants_to_optimize
        ]
        
        # Create placeholder mapping
        placeholder_dict = {
            f'c_{i+1}': float_values[i] 
            for i in range(len(float_values))
        }
        
        # Replace floats with placeholders
        modified_expressions = []
        for item in expressions:
            if isinstance(item, float) and item not in constants_to_optimize:
                placeholder = next(
                    key for key, value in placeholder_dict.items() 
                    if value == item
                )
                modified_expressions.append(placeholder)
            else:
                modified_expressions.append(item)
        
        return modified_expressions, placeholder_dict
    
    def _restore_constants(self, expr_string: str, placeholder_dict: Dict[str, float]) -> str:
        """
        Restore original constants from placeholders.
        
        Args:
            expr_string: Expression string with placeholders
            placeholder_dict: Mapping from placeholders to original values
            
        Returns:
            Expression string with restored constants
        """
        pattern = r'\b(c_\d+)\b'
        
        def replacement(match):
            placeholder = match.group(0)
            return str(placeholder_dict.get(placeholder, 0))
        
        return re.sub(pattern, replacement, expr_string)
    
    def _build_function_string(
        self, 
        nodes: List[int], 
        edges: List[Tuple[int, int]], 
        expressions: List[Any], 
        current_node: int
    ) -> str:
        """
        Recursively build SymPy-compatible function string from DEAP tree.
        
        Args:
            nodes: List of node indices
            edges: List of (parent, child) edge tuples
            expressions: List of expression elements
            current_node: Current node index being processed
            
        Returns:
            String representation of the expression tree
        """
        # Find children of current node
        next_nodes = [
            edge[1] for edge in edges 
            if edge[0] == current_node
        ]
        
        # Terminal node (leaf)
        if not next_nodes:
            return str(expressions[current_node])
        
        # Unary operator
        if len(next_nodes) == 1:
            op_symbol = self._get_operation_symbol(expressions[current_node])
            child_expr = self._build_function_string(nodes, edges, expressions, next_nodes[0])
            return f"({op_symbol}({child_expr}))"
        
        # Binary operator
        left_expr = self._build_function_string(nodes, edges, expressions, next_nodes[0])
        right_expr = self._build_function_string(nodes, edges, expressions, next_nodes[1])
        op_symbol = self._get_operation_symbol(expressions[current_node])
        
        return f"({left_expr}{op_symbol}{right_expr})"
    
    def _get_operation_symbol(self, op: str) -> str:
        """
        Get SymPy operator symbol for a given function name.
        
        Args:
            op: Function name (e.g., 'add', 'mul')
            
        Returns:
            SymPy operator symbol (e.g., '+', '*')
            
        Raises:
            ValueError: If operation is not supported
        """
        if op not in self.PREFIX_TO_SYMPY:
            raise ValueError(f"Unsupported operation: '{op}'")
        return self.PREFIX_TO_SYMPY[op]
    
    def _sympy_to_prefix(self, expr: sp.Basic) -> List[str]:
        """
        Convert SymPy expression to prefix notation.
        
        Args:
            expr: SymPy expression
            
        Returns:
            List of tokens in prefix notation
        """
        # Symbols and constants
        if isinstance(expr, sp.Symbol):
            return [str(expr)]
        elif isinstance(expr, sp.Integer):
            return [str(expr)]
        elif isinstance(expr, sp.Float):
            return [f"{expr:.1f}"]
        elif isinstance(expr, sp.Rational):
            return ["div", str(expr.p), str(expr.q)]
        elif expr == sp.E:
            return ["E"]
        elif expr == sp.pi:
            return ["pi"]
        
        # Check for square root (special case of pow)
        if (isinstance(expr, sp.Pow) and 
            isinstance(expr.args[1], sp.Rational) and
            expr.args[1].p == 1 and expr.args[1].q == 2):
            return ["sqrt"] + self._sympy_to_prefix(expr.args[0])
        
        # SymPy operators
        for op_type, op_name in self.SYMPY_TO_PREFIX.items():
            if isinstance(expr, op_type):
                return self._parse_sympy_operator(op_name, expr)
        
        # Unknown operator
        raise ValueError(f"Unknown SymPy operator: {expr}")
    
    def _parse_sympy_operator(self, op_name: str, expr: sp.Basic) -> List[str]:
        """
        Parse a SymPy expression with a given operator.
        
        Args:
            op_name: Operator name (e.g., 'add', 'mul')
            expr: SymPy expression
            
        Returns:
            List of tokens in prefix notation
        """
        n_args = len(expr.args)
        
        # Validate argument count
        if op_name in ["add", "mul"]:
            if n_args < 2:
                raise ValueError(f"{op_name} requires at least 2 arguments, got {n_args}")
        else:
            if not (1 <= n_args <= 2):
                raise ValueError(f"{op_name} requires 1-2 arguments, got {n_args}")
        
        # Build prefix list
        parse_list = []
        for i in range(n_args):
            if i == 0 or i < n_args - 1:
                parse_list.append(op_name)
            parse_list.extend(self._sympy_to_prefix(expr.args[i]))
        
        return parse_list
    
    def _format_prefix_list(self, prefix_list: List[Any]) -> str:
        """
        Format prefix list to DEAP-compatible string.
        
        Args:
            prefix_list: List of prefix tokens
            
        Returns:
            Space-separated string representation
        """
        formatted = []
        for token in prefix_list:
            if isinstance(token, (int, float)):
                formatted.append(f"{float(token):.1f}")
            elif isinstance(token, str) and token.lstrip('-').replace('.', '', 1).isdigit():
                formatted.append(f"{float(token):.1f}")
            else:
                formatted.append(str(token))
        
        # Handle negative zero
        formatted = ["0.0" if x == "-0.0" else x for x in formatted]
        
        return " ".join(formatted)
    