"""Tree validation functionality for Transformer sampling."""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

from TSGP.config.defaults import MAX_TREE_DEPTH, PADDING_INDEX, UNKNOWN_INDEX


@dataclass
class TreeNode:
    """A node in the sampled tree during Transformer decoding."""
    token_index: Optional[int]
    depth: int
    arity: Optional[int]
    children_count: Optional[int]
    is_open: Optional[bool]

class TreeValidator:
    """Validates tree structure during Transformer sampling."""
    
    def __init__(self, transformer_model):
        """
        Initialize the validator.
        
        Args:
            transformer_model: Instance of TransformerModel
        """
        self.vocab_arity_dict = transformer_model.vocab_arity_dict
        self.terminal_keys = transformer_model.terminal_keys
        self.variables = transformer_model.variables
        self.end_token_index = len(self.vocab_arity_dict) - 1
        self._special_indices = {PADDING_INDEX, UNKNOWN_INDEX, self.end_token_index - 1, self.end_token_index}
    
    def validate_next_token(self, output_tokens: np.ndarray, sampled_token_index: Optional[int], sampled_tree: List, dimensionality: int, remaining_tokens: Optional[int] = None) -> Tuple[np.ndarray, List]:
        """
        Validates the next token based on tree structure constraints.
        
        Args:
            output_tokens: Array of probabilities for each token
            sampled_token_index: Index of previously sampled token
            sampled_tree: List of sampled nodes with their properties
            dimensionality: Number of problem variables
            remaining_tokens: Number of tokens left in the sequence budget
            
        Returns:
            tuple: (updated_probabilities, updated_sampled_tree)
        """
        if sampled_token_index is None:
            arity = None
            is_open_function = None
            current_depth = 0
            no_children_nodes = None
        else:
            no_children_nodes = 0
            arity = self.vocab_arity_dict[sampled_token_index]
            is_open_function = (arity > 0)
            
            if sampled_tree[-1].arity is None:  # First node
                current_depth = 0
            elif sampled_tree[-1].arity > 0:  # Previous node was a function
                current_depth = sampled_tree[-1].depth + 1
                sampled_tree[-1].children_count = int(sampled_tree[-1].children_count) + 1
            else:  # Previous node was a terminal
                for node in range(2, len(sampled_tree)):
                    if sampled_tree[-node].is_open is True:
                        current_depth = sampled_tree[-node].depth + 1
                        sampled_tree[-node].children_count += 1
                        break
            
            if len(sampled_tree) > 1:
                for node in range(1, len(sampled_tree)):
                    if sampled_tree[node].arity == sampled_tree[node].children_count:
                        sampled_tree[node].is_open = False
        
        sampled_tree.append(TreeNode(sampled_token_index, current_depth, arity, no_children_nodes, is_open_function))
        
        valid_nodes = self._get_valid_next_nodes(sampled_tree, sampled_token_index, remaining_tokens)

        valid_nodes = self._adjust_for_dimensionality(valid_nodes, dimensionality)

        return self._calculate_probability_table(output_tokens, valid_nodes), sampled_tree
    
    def _get_valid_next_nodes(self, sampled_tree: List, sampled_token_index: Optional[int], remaining_tokens: Optional[int] = None) -> List[int]:
        """Determine valid next nodes based on tree structure."""
        # Check if all functions are closed
        if len(sampled_tree) > 1:
            all_functions_closed = True
            for node in range(1, len(sampled_tree)):
                if sampled_tree[node].is_open is True:
                    all_functions_closed = False
                    break
            if all_functions_closed:
                return [len(self.vocab_arity_dict) - 1]  # Assuming end token is last
        
        # Count outstanding open slots (children still needed to complete the tree)
        # Each open function needs (arity - children_so_far) more children
        outstanding = 0
        if len(sampled_tree) > 1:
            for node in range(1, len(sampled_tree)):
                if sampled_tree[node].is_open is True:
                    outstanding += sampled_tree[node].arity - sampled_tree[node].children_count
        
        # Only terminals once the remaining tokens are just enough to close the tree.
        if remaining_tokens is not None and remaining_tokens <= outstanding:
            return self._get_terminal_keys()
        
        if sampled_token_index in self._get_function_keys():
            depth_next_node = sampled_tree[-1].depth + 1
        elif sampled_token_index in self._get_terminal_keys():
            for node in range(2, len(sampled_tree)):
                if sampled_tree[-node].is_open is True:
                    depth_next_node = sampled_tree[-node].depth + 1
                    break
        else:  # First node or None
            depth_next_node = 0
        
        # Apply depth restrictions (max depth per Koza)
        if depth_next_node < MAX_TREE_DEPTH:
            return self._get_function_keys() + self._get_terminal_keys()
        elif depth_next_node == MAX_TREE_DEPTH:
            return self._get_terminal_keys()
        else:
            return []
    
    def _get_function_keys(self) -> List[int]:
        """Get keys for function tokens (excludes special tokens)."""
        return [k for k, v in self.vocab_arity_dict.items() if v > 0]
    
    def _get_terminal_keys(self) -> List[int]:
        """Get keys for terminal tokens (excludes special tokens like [end], [start], padding, [UNK])."""
        return [k for k, v in self.vocab_arity_dict.items() if v == 0 and k not in self._special_indices]
    
    def _calculate_probability_table(self, output_tokens: np.ndarray, valid_nodes: List[int]) -> np.ndarray:
        """Calculate probability table for valid nodes."""
        new_prob_table = np.zeros_like(output_tokens)
        sum_probs = sum(output_tokens[node] for node in valid_nodes)
        
        if sum_probs > 0:
            for node in valid_nodes:
                new_prob_table[node] = output_tokens[node] / sum_probs
        
        return new_prob_table
    
    def _adjust_for_dimensionality(self, valid_nodes: List[int], dimensionality: int) -> List[int]:
        """Adjust valid nodes based on dimensionality constraints."""
        adjusted_valid_nodes = []
                
        variable_keys = self.terminal_keys[:len(self.variables)]
        invalid_variables = variable_keys[dimensionality:]
        for node in valid_nodes:
            if node not in invalid_variables:
                adjusted_valid_nodes.append(node)
        return adjusted_valid_nodes