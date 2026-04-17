# src/TSGP/transformer/sampling.py
"""Sampling functionality for Transformer model."""

import numpy as np
import tensorflow as tf
from .validation import TreeValidator

class TransformerSampler:
    """Sampling functionality for the Transformer model in TSGP."""
    
    def __init__(self, transformer_model):
        """
        Initialize the sampler.
        
        Args:
            transformer_model: Instance of TransformerModel
        """
        self.model = transformer_model
        self.validator = TreeValidator(transformer_model)

    def sample_population(self, input_sequences, encoder_vectorization, decoder_vectorization, dimensionality, target_semantic_distance):
        """
        Sample a population of individuals from the Transformer model.
        
        Args:
            input_sequences: List of input sequences.
            encoder_vectorization: Encoder vectorization function.
            decoder_vectorization: Decoder vectorization function.
            dimensionality: Controls the max. amount of variables to be sampled.
            target_semantic_distance: Target semantic distance for sampling. Controls the semantic similarity of the sampled individuals.

        Returns:
            List of sampled sequences
        """
        max_decoded_sequence_length = self.model.max_length
        batch_size = len(input_sequences)
        
        # Vectorize encoder inputs once at the start
        tokenized_input_sequences = encoder_vectorization(input_sequences)

        # Get the start token index
        start_token_index = decoder_vectorization(["[start]"]).numpy()[0, 0]
        end_token_index = decoder_vectorization(["[end]"]).numpy()[0, 0]
        
        # Initialize decoder input as token indices
        # Shape: (batch_size, max_length)
        decoder_input_indices = np.zeros((batch_size, max_decoded_sequence_length), dtype=np.int64)
        decoder_input_indices[:, 0] = start_token_index
        
        # Track which sequences are still active
        active_mask = np.ones(batch_size, dtype=bool)
        decoded_sequence_list = [""] * batch_size
        sampled_trees = [[] for _ in range(batch_size)]
        prev_sampled_token_indices = [None] * batch_size
        
        # Prepare float inputs once
        floats = []
        if self.model.include_distance:
            floats.append(target_semantic_distance)
        if self.model.include_amt_variables:
            floats.append(dimensionality)
        floats_batch = tf.constant([floats] * batch_size, dtype=tf.float32)
        
        for i in range(max_decoded_sequence_length - 1):
            if not active_mask.any():
                break
            
            current_decoder_input = tf.constant(decoder_input_indices, dtype=tf.int64)
            
            # Get predictions for all sequences
            predictions = self.model.model([
                tokenized_input_sequences,
                current_decoder_input,
                floats_batch
            ], training=False)
            
            # Process each sequence
            for idx in range(batch_size):

                # Skip if sequence is already finished
                if not active_mask[idx]:
                    continue
                    
                probabilities = predictions[idx, i, :].numpy()
                
                # Apply tree validation
                probabilities, sampled_trees[idx] = self.validator.validate_next_token(
                    probabilities,
                    prev_sampled_token_indices[idx],
                    sampled_trees[idx],
                    dimensionality,
                    remaining_tokens=max_decoded_sequence_length - 1 - i
                )
                
                # Normalize probabilities
                probabilities = probabilities / np.sum(probabilities)
                
                # Sample token
                sampled_token_index = np.random.choice(len(probabilities), p=probabilities)
                
                # Store the token index directly (no string manipulation!)
                decoder_input_indices[idx, i + 1] = sampled_token_index
                prev_sampled_token_indices[idx] = sampled_token_index
                
                # Check if sequence is finished
                if sampled_token_index == end_token_index:
                    active_mask[idx] = False
            
        # Convert token indices back to strings only at the end
        for idx in range(batch_size):
            
            # Find where [end] token appears (or use full sequence)
            token_indices = decoder_input_indices[idx]
            end_pos = np.where(token_indices == end_token_index)[0]
            if len(end_pos) > 0:
                token_indices = token_indices[1:end_pos[0]]  # Skip [start], stop at [end]
            else:
                # Find first zero (padding)
                non_zero = np.where(token_indices != 0)[0]
                if len(non_zero) > 0:
                    token_indices = token_indices[1:non_zero[-1]+1]  # Skip [start]
                else:
                    token_indices = []
            
            # Convert indices to tokens
            tokens = [self.model.dec_index_lookup[int(tok_idx)] for tok_idx in token_indices]
            decoded_sequence_list[idx] = " ".join(tokens)
        
        return decoded_sequence_list