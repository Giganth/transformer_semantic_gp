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

        # Encoder and decoder are separate sub-models: the encoder runs once per population.
        self.encoder = transformer_model.model.get_layer("encoder")
        decoder = transformer_model.model.get_layer("decoder")
        amt_floats = transformer_model.amt_floats
        # Dynamic dimensions so the shrinking batch and growing prefix trace once.
        self._next_token_probabilities = tf.function(
            lambda enc, dec, floats: decoder([dec, enc, floats], training=False)[:, -1, :],
            input_signature=[
                tf.TensorSpec([None, None, transformer_model.embed_dim + amt_floats], tf.float32),
                tf.TensorSpec([None, None], tf.int64),
                tf.TensorSpec([None, amt_floats], tf.float32),
            ]
        )


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
        
        tokenized_input_sequences = encoder_vectorization(input_sequences)

        start_token_index = decoder_vectorization(["[start]"]).numpy()[0, 0]
        end_token_index = decoder_vectorization(["[end]"]).numpy()[0, 0]
        
        decoder_input_indices = np.zeros((batch_size, max_decoded_sequence_length), dtype=np.int64)
        decoder_input_indices[:, 0] = start_token_index
        
        active_mask = np.ones(batch_size, dtype=bool)
        decoded_sequence_list = [""] * batch_size
        sampled_trees = [[] for _ in range(batch_size)]
        prev_sampled_token_indices = [None] * batch_size
        
        floats = []
        if self.model.include_distance:
            floats.append(target_semantic_distance)
        if self.model.include_amt_variables:
            floats.append(dimensionality)
        floats_batch = tf.constant([floats] * batch_size, dtype=tf.float32)

        # The encoder input never changes while decoding, so encode once.
        encoder_outputs = self.encoder([tokenized_input_sequences, floats_batch], training=False)

        for i in range(max_decoded_sequence_length - 1):
            active_indices = np.flatnonzero(active_mask)
            if active_indices.size == 0:
                break

            # Only unfinished sequences, and only the prefix up to position i (causal mask).
            gather_indices = tf.constant(active_indices)
            preds_np = self._next_token_probabilities(
                tf.gather(encoder_outputs, gather_indices),
                tf.constant(decoder_input_indices[active_indices, :i + 1]),
                tf.gather(floats_batch, gather_indices)
            ).numpy()

            for row, idx in enumerate(active_indices):

                probabilities = preds_np[row]
                
                # Apply tree validation
                probabilities, sampled_trees[idx] = self.validator.validate_next_token(
                    probabilities,
                    prev_sampled_token_indices[idx],
                    sampled_trees[idx],
                    dimensionality,
                    remaining_tokens=max_decoded_sequence_length - 1 - i
                )
                
                probabilities = probabilities / np.sum(probabilities)
                
                sampled_token_index = np.random.choice(len(probabilities), p=probabilities)
                
                decoder_input_indices[idx, i + 1] = sampled_token_index
                prev_sampled_token_indices[idx] = sampled_token_index
                
                if sampled_token_index == end_token_index:
                    active_mask[idx] = False
            
        for idx in range(batch_size):
            
            token_indices = decoder_input_indices[idx]
            end_pos = np.where(token_indices == end_token_index)[0]
            if len(end_pos) > 0:
                token_indices = token_indices[1:end_pos[0]]  # Skip [start], stop at [end]
            else:
                non_zero = np.where(token_indices != 0)[0]
                if len(non_zero) > 0:
                    token_indices = token_indices[1:non_zero[-1]+1]  # Skip [start]
                else:
                    token_indices = []
            
            tokens = [self.model.dec_index_lookup[int(idx)] for idx in token_indices]
            decoded_sequence_list[idx] = " ".join(tokens)
        
        return decoded_sequence_list
