import tensorflow as tf
import os
from TSGP.core.engine import BaseEvolutionaryAlgorithm
from TSGP.transformer import TransformerModel, TransformerSampler
from TSGP.transformer.utils import create_vectorization, prepare_transformer_config, generate_vocab, transform_deap_population_to_keras_strings
from typing import List

class TransformerGP(BaseEvolutionaryAlgorithm):
    """Transformer Semantic GP implementation."""
    
    def __init__(self, config: dict):
        # Configure GPU usage
        use_gpu = config.get('use_gpu')
        
        if not use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        else:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)

        # set random seed for TensorFlow operations
        tf.random.set_seed(config['random_seed'])
        
        # Prepare transformer configuration
        model_config = config['model']
        model_config = prepare_transformer_config(model_config)
               
        config['model'] = model_config
        
        # Generate vocabulary for vectorization using MODEL's vocabulary (not problem's)
        functions = model_config['functions']
        variables = model_config['variables']
        ephemerals = model_config['ephemerals']
        vocab = generate_vocab(functions, variables, ephemerals)

        # Create encoder layers (length None for variable input length)
        self.encoder_vectorization = create_vectorization(
            vocab_data=vocab, 
            flags=False, 
            max_length=None
        )
        
        # Decoder layers (fixed max length for output sequences)
        self.decoder_vectorization = create_vectorization(
            vocab_data=vocab, 
            flags=True, 
            max_length=model_config['max_length']
        )

        # Initialize Transformer model and sampler
        self.transformer_model = TransformerModel(model_config)
        self.sampler = TransformerSampler(self.transformer_model)
        
        # Load pretrained model weight
        model_directory = config['variation']['transformer']['model_directory']
        self.transformer_model.model.load_weights(model_directory)

        # Call parent init (which sets random seed and initializes population)
        # This ensures TensorFlow operations don't consume random numbers before population init
        super().__init__(config)
        
        self.false_expressions_counter = 0

    def _variation(self, population: List) -> List:
        """Apply tsgp mutation to population using Transformer sampling."""
        from deap import gp, creator
        
        input_sequences = transform_deap_population_to_keras_strings(population)

        # Get configuration parameters
        dimensionality = len(self.config['problem']['variables'])
        target_semantic_distance = self.config['variation']['transformer']['target_semantic_distance']

        # Sample new population using transformer
        offspring_sequences = self.sampler.sample_population(
            input_sequences=input_sequences,
            encoder_vectorization=self.encoder_vectorization,
            decoder_vectorization=self.decoder_vectorization,
            dimensionality=dimensionality,
            target_semantic_distance=target_semantic_distance
        )

        offspring = []
        self.false_expressions_counter = 0
        
        for i, expr_str in enumerate(offspring_sequences):
            try:
                deap_expression = gp.PrimitiveTree.from_string(str(expr_str), self.pset)
                deap_expression_ind = creator.Individual(deap_expression)

                # Check validity of the expression
                offspring.append(deap_expression_ind)
                
            except Exception as e:
                # If the sampled individual is not syntactically correct or invalid,
                # use the input individual instead
                print(f"Invalid expression sampled: {expr_str}")
                self.false_expressions_counter += 1
                deap_expression = gp.PrimitiveTree.from_string(str(population[i]), self.pset)
                deap_expression_ind = creator.Individual(deap_expression)
                offspring.append(deap_expression_ind)
        return offspring
