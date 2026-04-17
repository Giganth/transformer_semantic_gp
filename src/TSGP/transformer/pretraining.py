"""Transformer pretraining module for TSGP."""

import os
import pickle
import random
import glob
import re
from typing import Dict, Any, List, Tuple
import tensorflow as tf
import numpy as np
from keras.callbacks import ModelCheckpoint
import tensorflow.data as tf_data

from .model import TransformerModel
from .utils import create_vectorization, generate_vocab


class TransformerPretrainer:
    """Handles pretraining of Transformer models for TSGP."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the pretrainer.
        
        Args:
            config: Configuration dictionary containing pretraining settings
        """
        self.config = config
        self.pretraining_config = config["model"]
        
        # Setup random seeds
        self._setup_random_seeds()
        
        # Setup GPU configuration
        self._setup_gpu()
        
        # Extract pretraining parameters
        self.model_directory = self.pretraining_config['model_directory']
        self.max_epochs = self.pretraining_config["max_epochs"]
        self.batch_size = self.pretraining_config["batch_size"]
        self.learning_rate = self.pretraining_config["learning_rate"]
        self.val_split_ratio = self.pretraining_config["val_split_ratio"]
        self.similarity_threshold = self.pretraining_config["similarity_threshold"]
        self.max_samples_per_file = self.pretraining_config["max_samples_per_file"]
        self.max_length = self.pretraining_config["max_length"]
        
        # Feature flags
        self.include_distance = self.pretraining_config["include_semantic_distance"]
        self.include_amt_variables = self.pretraining_config["include_dimensionality"]
        
        # Data paths
        self.archive_directories = self.pretraining_config['archive_directories']
        
        print("=" * 50)
        print("Transformer Pretraining Initialized")
        print("=" * 50)
        print(f"Model directory: {self.model_directory}")
        print(f"Max epochs: {self.max_epochs}")
        print(f"Batch size: {self.batch_size}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Validation split: {self.val_split_ratio}")
        print(f"GPU devices: {tf.config.list_physical_devices('GPU')}")
        print("=" * 50)
    
    def _setup_random_seeds(self):
        """Set random seeds for reproducibility."""
        if 'random_seed' in self.config:
            seed = self.config['random_seed']
            random.seed(seed)
            np.random.seed(seed)
            tf.random.set_seed(seed)
            os.environ['PYTHONHASHSEED'] = str(seed)
    
    def _setup_gpu(self):
        """Configure GPU settings."""
        use_gpu = self.config['use_gpu']
        
        if not use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        else:
            gpus = tf.config.list_physical_devices('GPU')
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        
        # Suppress TensorFlow warnings
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    def load_training_data(self) -> List[Tuple[str, str, List[float]]]:
        """
        Load training data from archive directories.
        
        Returns:
            List of (encoder_string, decoder_string, additional_inputs) tuples
        """
        text_pairs = []

        # Get archive directories from config
        archive_dirs = self.config['model']['archive_directories']
        
        for i, archive_dir in enumerate(archive_dirs):
            print(f"\n Processing directory: {archive_dir}")
            
            # Find all pickle files in directory
            input_paths = glob.glob(os.path.join(archive_dir, "**/*similar_pairs.pkl"), recursive=True)
            print(f"Found {len(input_paths)} archive file(s)")
            
            for file_path in input_paths:
                print(f"Loading: {file_path}")
                
                try:
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)
                    
                    print(f"  Loaded {len(data)} samples")
                    random.shuffle(data)
                    
                    # Extract dimensionality from path if needed
                    amt_variables = None
                    if self.include_amt_variables:
                        match = re.search(r'Archive_(\d+)', file_path)
                        if match:
                            amt_variables = int(match.group(1))
                            print(f"Detected functions valid for {amt_variables}-dimensional problems")
                    
                    # Process samples
                    samples_added = 0
                    for ind in data:
                        if samples_added >= self.max_samples_per_file:
                            break
                        
                        string_encoder = ind[0]
                        string_decoder = ind[1]
                        distance = ind[2]
                        
                        # Skip if distance exceeds threshold
                        if distance >= self.similarity_threshold:
                            continue
                        
                        # Prepare strings for tokenization
                        adjusted_encoder = str(string_encoder).replace("(", " ").replace(")", "").replace(",", "").replace("-0.0", "0.0")
                        adjusted_decoder = str(string_decoder).replace("(", " ").replace(")", "").replace(",", "").replace("-0.0", "0.0")
                        
                        # Build additional inputs
                        additional_inputs = []
                        if self.include_distance:
                            additional_inputs.append(distance)
                        if self.include_amt_variables and amt_variables is not None:
                            additional_inputs.append(amt_variables)

                        # Add flags to decoder
                        adjusted_decoder = "[start] " + adjusted_decoder + " [end]"
                        
                        text_pairs.append((adjusted_encoder, adjusted_decoder, additional_inputs))
                        samples_added += 1
                    
                    print(f"  Added {samples_added} samples (total: {len(text_pairs)})")
                    
                except Exception as e:
                    print(f"  Error loading {file_path}: {e}")
                    continue
        return text_pairs
    
    def prepare_datasets(self, text_pairs: List[Tuple[str, str, List[float]]]):
        """
        Prepare training and validation datasets.
        
        Args:
            text_pairs: List of (encoder_string, decoder_string, additional_inputs) tuples
            
        Returns:
            Tuple of (train_dataset, val_dataset, encoder_vectorization, decoder_vectorization, vocab_size)
        """
        # Shuffle data
        random.shuffle(text_pairs)
        
        # Train-val split
        num_val_samples = int(self.val_split_ratio * len(text_pairs))
        num_train_samples = len(text_pairs) - num_val_samples
        
        train_pairs = text_pairs[:num_train_samples]
        val_pairs = text_pairs[num_train_samples:]
        
        print(f"\n{len(text_pairs)} total pairs")
        print(f"{len(train_pairs)} training pairs")
        print(f"{len(val_pairs)} validation pairs")
        
        # Create vectorization layers
        # Extract vocabulary components from config
        functions = self.pretraining_config['functions']
        variables = self.pretraining_config['variables']
        ephemerals = self.pretraining_config['ephemerals']
        
        vocab = generate_vocab(functions, variables, ephemerals)
        
        encoder_vectorization = create_vectorization(
            vocab_data=vocab, 
            flags=False, 
            max_length=self.max_length
        )
        decoder_vectorization = create_vectorization(
            vocab_data=vocab, 
            flags=True, 
            max_length=self.max_length
        )
        
        # Get vocabulary
        dec_vocab = decoder_vectorization.get_vocabulary()
        vocab_size = len(dec_vocab)
        
        print(f"\nVocabulary size: {vocab_size}")
        print(f"Encoder vocab: {encoder_vectorization.get_vocabulary()[:10]}...")
        print(f"Decoder vocab: {dec_vocab[:10]}...")
        
        # Create datasets
        def make_dataset(pairs):
            enc_texts, dec_texts, additional_inputs = zip(*pairs)
            enc_texts = list(enc_texts)
            dec_texts = list(dec_texts)
            additional_inputs = list(additional_inputs)
            
            dataset = tf_data.Dataset.from_tensor_slices((enc_texts, dec_texts, additional_inputs))
            dataset = dataset.shuffle(buffer_size=min(len(enc_texts), 100000), reshuffle_each_iteration=True)
            dataset = dataset.batch(self.batch_size)
            dataset = dataset.map(lambda enc, dec, floats: format_dataset(
                enc, dec, floats, encoder_vectorization, decoder_vectorization
            ))
            return dataset.prefetch(tf.data.AUTOTUNE)
        
        def format_dataset(enc, dec, additional_inputs, enc_vec, dec_vec):
            enc = enc_vec(enc)
            dec = dec_vec(dec)
            additional_inputs = tf.convert_to_tensor(additional_inputs, dtype=tf.float32)
            return (
                {
                    "encoder_inputs": enc,
                    "decoder_inputs": dec[:, :-1],
                    "float_inputs": additional_inputs
                },
                dec[:, 1:],
            )
        
        train_ds = make_dataset(train_pairs)
        val_ds = make_dataset(val_pairs)
        
        return train_ds, val_ds, vocab_size
    
    def train(self, train_ds, val_ds, vocab_size):
        """
        Train the transformer model.
        
        Args:
            train_ds: Training dataset
            val_ds: Validation dataset
            vocab_size: Size of vocabulary
            
        Returns:
            Training history
        """
        # Build transformer configuration
        transformer_config = {
            'max_length': self.max_length,
            'vocab_size': vocab_size,
            'embed_dim': self.pretraining_config['embed_dim'],
            'latent_dim': self.pretraining_config['latent_dim'],
            'num_heads': self.pretraining_config['num_heads'],
            'num_stacks': self.pretraining_config['num_stacks'],
            'dropout_rate': self.pretraining_config['dropout_rate'],
            'learning_rate': self.learning_rate,
            'include_semantic_distance': self.include_distance,
            'include_dimensionality': self.include_amt_variables,
            'legacy_mode': self.pretraining_config['legacy_mode'],
            'decay_steps': self.pretraining_config['decay_steps'],
        }
        
        # Create model
        transformer_model = TransformerModel(transformer_config)
        
        # Setup callbacks
        callbacks = []
          
        mc = ModelCheckpoint(
            filepath=self.model_directory,
            monitor='val_loss',
            mode='min',
            verbose=1,
            save_best_only=True,
            save_weights_only=True
        )
        callbacks.append(mc)
        
        # Try loading existing weights
        try:
            transformer_model.model.load_weights(self.model_directory)
            print(f"\nLoaded existing model from {self.model_directory}")
            print("Continuing training (fine-tuning)...")
        except (OSError, ValueError) as e:
            print(f"\nNo existing model found at {self.model_directory} ({e})")
            print("Training new model from scratch...")
        
        
        # Train
        print(f"\nStarting training for up to {self.max_epochs} epochs...")
        history = transformer_model.model.fit(
            train_ds,
            epochs=self.max_epochs,
            validation_data=val_ds,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def run(self):
        """Execute the complete pretraining pipeline."""
        print("\n" + "=" * 50)
        print("STARTING TRANSFORMER PRETRAINING")
        print("=" * 50)
        
        # Load data
        print("\nStep 1: Loading training data...")
        text_pairs = self.load_training_data()
        
        print(f"\nTotal training samples loaded: {len(text_pairs)}")
        if len(text_pairs) == 0:
            raise ValueError("No training data loaded! Check archive directories.")
        
        # Prepare datasets
        print("\nStep 2: Preparing datasets...")
        train_ds, val_ds, vocab_size = self.prepare_datasets(text_pairs)
        
        # Show sample
        print("\nSample batch:")
        for inputs, targets in train_ds.take(5):
            print(f"  Encoder inputs shape: {inputs['encoder_inputs'].shape}")
            print(f"  Decoder inputs shape: {inputs['decoder_inputs'].shape}")
            print(f"  Float inputs shape: {inputs['float_inputs'].shape}")
            print(f"  Targets shape: {targets.shape}")
            print(f"  First encoder input: {inputs['encoder_inputs'][0][:20]}...")
            print(f"  First float input: {inputs['float_inputs'][0]}")
        
        # Train
        print("\nStep 3: Training model...")
        history = self.train(train_ds, val_ds, vocab_size)
        
        # Save history
        history_path = os.path.join(
            os.path.dirname(self.model_directory),
            'training_history.csv'
        )
        os.makedirs(os.path.dirname(history_path), exist_ok=True)
        
        import pandas as pd
        history_df = pd.DataFrame(history.history)
        history_df.to_csv(history_path, index=False)
        print(f"\nTraining history saved to {history_path}")
        
        print("\n" + "=" * 50)
        print("PRETRAINING COMPLETE")
        print("=" * 50)
        
        return history
