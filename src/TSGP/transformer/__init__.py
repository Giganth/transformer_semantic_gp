# src/TSGP/transformer/__init__.py
"""Transformer module for TSGP project."""

from .model import TransformerModel
from .sampling import TransformerSampler
from .validation import TreeValidator
from .pretraining import TransformerPretrainer
from .utils import (
    create_vectorization,
    generate_vocab,
    generate_vocab_arity_dict,
    get_terminal_keys,
    get_function_keys,
    prepare_transformer_config
)

# Export main classes for easy import
__all__ = [
    'TransformerModel',
    'TransformerSampler',
    'TreeValidator',
    'TransformerPretrainer',
    'create_vectorization',
    'generate_vocab',
    'generate_vocab_arity_dict',
    'get_terminal_keys',
    'get_function_keys',
    'prepare_transformer_config'
]