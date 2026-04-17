# src/TSGP/transformer/model.py
"""Transformer model definition for TSGP."""

import tensorflow as tf
from keras import layers, optimizers
from .layers import TransformerEncoder, TransformerDecoder, PositionalEmbedding

import warnings

warnings.filterwarnings(
    "ignore",
    message="Skipping variable loading for optimizer"
)

class TransformerModel:
    """Transformer model which acts as semantic variation operator in GP for symbolic regression."""
    
    def __init__(self, config):
        """
        Initialize the Transformer model.
        
        Args:
            config: Configuration dictionary containing:
                - max_length: Maximum sequence length
                - latent_dim: Latent dimension
                - num_heads: Number of attention heads
                - num_stacks: Number of transformer stacks
                - dropout_rate: Dropout rate
                - learning_rate: Learning rate
                - include_distance: Whether to include distance information
                - include_amt_variables: Whether to include number of variables
                - vocab_arity_dict: Dictionary mapping tokens to arities
                - terminal_keys: List of terminal token indices
                - function_keys: List of function token indices
                - dec_vocab: Decoder vocabulary
                - embed_dim: Embedding dimension
                - variables: List of variable tokens
                - legacy_mode: If True, use flat architecture for old weight compatibility (default: False)
        """
        self.config = config
        self.max_length = config['max_length']
        self.latent_dim = config['latent_dim']
        self.num_heads = config['num_heads']
        self.num_stacks = config['num_stacks']
        self.dropout_rate = config['dropout_rate']
        self.learning_rate = config['learning_rate']
        self.include_distance = config['include_semantic_distance']
        self.include_amt_variables = config['include_dimensionality']
        self.vocab_arity_dict = config.get('vocab_arity_dict', None)
        self.terminal_keys = config.get('terminal_keys', None)
        self.function_keys = config.get('function_keys', None)
        self.dec_vocab = config.get('dec_vocab', None)
        self.embed_dim = config.get('embed_dim', None)
        self.variables = config.get('variables', None)
        self.legacy_mode = self.config['legacy_mode']
        
        # Create index lookup for decoder
        self.dec_index_lookup = dict(zip(range(len(self.dec_vocab)), self.dec_vocab)) if self.dec_vocab else None
        self.vocab_keys = list(range(len(self.dec_vocab))) if self.dec_vocab else None
        
        # Calculate number of float inputs
        self.amt_floats = sum([
            1 if self.include_distance else 0,
            1 if self.include_amt_variables else 0
        ])
        
        # Build model
        self.model = self._build_model()
    
    def _build_model(self):
        """Build the Transformer model architecture."""
        # Encoder
        encoder_inputs = tf.keras.Input(shape=(None,), dtype="int64", name="encoder_inputs")
        float_inputs = tf.keras.Input(shape=(self.amt_floats,), dtype="float32", name="float_inputs")
        
        x = PositionalEmbedding(self.max_length, self.config['vocab_size'], self.embed_dim)(encoder_inputs)
        for stack in range(self.num_stacks):
            x = TransformerEncoder(
                self.embed_dim + self.amt_floats, 
                self.latent_dim, 
                self.num_heads, 
                self.dropout_rate, 
                stack_index=stack
            )(x, float_inputs)
        encoder_outputs = x
        encoder = tf.keras.Model([encoder_inputs, float_inputs], encoder_outputs, name="encoder")
        
        # Explicitly enable masking support for the encoder model
        encoder.supports_masking = True
        
        # Decoder
        decoder_inputs = tf.keras.Input(shape=(None,), dtype="int64", name="decoder_inputs")
        encoded_seq_inputs = tf.keras.Input(shape=(None, self.embed_dim + self.amt_floats), name="decoder_state_inputs")
        
        x = PositionalEmbedding(self.max_length, self.config['vocab_size'], self.embed_dim)(decoder_inputs)
        for stack in range(self.num_stacks):
            x = TransformerDecoder(
                self.embed_dim + self.amt_floats, 
                self.latent_dim, 
                self.num_heads, 
                self.dropout_rate, 
                stack_index=stack
            )(x, encoded_seq_inputs, float_inputs)
        
        x = layers.Dropout(self.dropout_rate)(x)
        decoder_outputs = layers.Dense(self.config['vocab_size'], activation="softmax")(x)
        decoder = tf.keras.Model([decoder_inputs, encoded_seq_inputs, float_inputs], decoder_outputs, name="decoder")
        
        # Explicitly enable masking support for the decoder model
        decoder.supports_masking = True
        
        # Connect encoder and decoder
        if self.legacy_mode:
            # Legacy mode: Use flat architecture (encoder/decoder outputs directly)
            # This maintains compatibility with old saved weights
            decoder_outputs = decoder([decoder_inputs, encoder_outputs, float_inputs])
        else:
            # New mode: Use nested model architecture (encoder/decoder as functional layers)
            # This creates cleaner model structure with encoder and decoder as single layers
            encoder_outputs = encoder([encoder_inputs, float_inputs])
            decoder_outputs = decoder([decoder_inputs, encoder_outputs, float_inputs])
        
        # Learning rate schedule
        lr_schedule = optimizers.schedules.CosineDecay(
            initial_learning_rate=0.0,
            decay_steps=self.config['decay_steps'],
            warmup_target=self.learning_rate,
            warmup_steps=self.config['decay_steps']/ 100
        )
        
        # Create final model
        model = tf.keras.Model([encoder_inputs, decoder_inputs, float_inputs], decoder_outputs, name="transformer")
        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=optimizers.AdamW(learning_rate=lr_schedule),
            metrics=["accuracy"],
            jit_compile=True
        )
        
        return model
    
    def get_config(self):
        """Return model configuration."""
        return {
            'max_length': self.max_length,
            'latent_dim': self.latent_dim,
            'num_heads': self.num_heads,
            'num_stacks': self.num_stacks,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'include_distance': self.include_distance,
            'include_amt_variables': self.include_amt_variables,
            'vocab_size': self.config['vocab_size'],
            'vocab_arity_dict': self.vocab_arity_dict,
            'terminal_keys': self.terminal_keys,
            'function_keys': self.function_keys,
            'dec_vocab': self.dec_vocab,
            'embed_dim': self.embed_dim,
            'legacy_mode': self.legacy_mode
        }
    
    def save_config(self, filepath):
        """Save model configuration to file."""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.get_config(), f, indent=2)