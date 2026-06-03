# Maximum tree depth as recommended by DEAP developers based on Koza (1992)
MAX_TREE_DEPTH = 17

# Vocabulary special token indices (set by TextVectorization ordering)
PADDING_INDEX = 0
UNKNOWN_INDEX = 1

STANDARD_GP_DEFAULTS = {
    'algorithm': 'standard_gp',
    'popsize': 100,
    'initialization': {
        'mode': 'ramped-half-and-half',
        'min_tree_depth': 2,
        'max_tree_depth': 5,
    },
    'selection': {
        'type': 'tournament', # 'tournament' or 'double_tournament'
        'tournament_size': 5,
        'parsimony_size': 1,    # Only used for double tournament
        'fitness_first': True    # Only used for double tournament
    },
    'variation': {
        'crossover': {
            'application_prob': 0.9,
            'node_bias_terminals': 0.1
        },
        'mutation': {
            'application_prob': 0.1,
            'mode': 'ramped-half-and-half',
            'min_tree_depth': 0,
            'max_tree_depth': 2
        }
    },
    'termination': {
        'max_generations': 100,
    },
    'random_seed': 1000,
    'console_out': True,
    'monitor_folder': 'experiments_data/logs/',
    'num_cores': 1,
    'use_gpu': True  # Whether to use GPU for TensorFlow operations

}

TS_GP_DEFAULTS = {
    **STANDARD_GP_DEFAULTS,  # Inherit most settings
    'algorithm': 'transformer_gp',
    
    'variation': {
        'transformer': {
            'model_directory': 'src/TSGP/transformer/models/TSGP.weights.h5',  # Path to pretrained model
            'target_semantic_distance': 1,
        }
    },
    'model': {
        # Model Settings
        'num_stacks'                       : 2,  # Amount of encoders and decoder layers - in "Attention is all you need" -> stacks: 6
        'num_heads'                        : 8,  # Amount of heads within the self-attention layer - "Attention is all you need" -> heads: 8
        'dropout_rate'                     : 0.1, 
        'latent_dim'                       : 512, 
        'embed_dim'                        : 128, 

        # Settings for similarity search
        'max_length'                       : 100,    # max length of expressions in tokens
        'similarity_threshold'             : 100,    # 100 is default value, lower -> more similar (sqrt(similarity_threshold) == Euclidean Distance)
        'nlist'                            : 5000,   # amt. of clusters -> higher nlist -> faster search but lower accuracy and indexing takes longer
        'nprobe'                           : 10,     # amt. of clusters to be searched -> higher nprobe -> higher accuracy, but increased search duration
        'n_neighbors'                      : 3,     # amt. of neighbors to retrieve from similarity search, 3 default
                
        # Settings for pretraining
        'max_samples_per_file'             : 5000000,  # Maximum samples to load per archive file during pretraining
        'batch_size'                       : 256, # 256 is default value
        'learning_rate'                    : 0.001, # 0.001 is default value
        'decay_steps'                      : 550000, # 1000000 is default value
        'max_epochs'                       : 8, # 8 is default value
        'patience'                         : 3, # for early stopping and convergence, no. of epochs that we accept no improvement
        'val_split_ratio'                  : 0.1,   # percentage how much of the data is used for the validation set during training
        
        
        # Feature flags - what additional inputs to include
        'include_semantic_distance'        : True,  # if True the semantic distance is included in training
        'include_dimensionality'           : True,  # if True the dimensionality is included in training
        
        # Architecture mode
        'legacy_mode'                      : False,  # if True uses flat architecture for old weight compatibility, if False uses nested encoder/decoder architecture

        # Settings for vocabulary. Defines the token of the transformer and thus which functions can be created
        'functions'                        : ['add', 'sub', 'mul', 'div','pow'], 
        'variables'                        : ['x0', 'x1', 'x2', 'x3','x4'], 
        'ephemerals'                       : [-5.0,-4.0,-3.0,-2.0,-1.0,-0.5,-0.4,-0.3,-0.2,-0.1, 0.0, 0.1,0.2 ,0.3, 0.4,0.5,1.0,2.0,3.0,4.0,5.0],
    
        # Paths for saving/loading models and logs
        'model_directory'                 : 'src/TSGP/transformer/models',
        
        # Data paths - directories containing training archives
        'archive_directories': [
        'trainingdata/Archive_2',  # 2D problems
        'trainingdata/Archive_3',  # 3D problems
        'trainingdata/Archive_4',  # 4D problems
        'trainingdata/Archive_5',  # 5D problems
    ]
    },

}

SYMBOLIC_REGRESSION_DEFAULTS = {
    'problem': {
        'dataset': 'feynman_I_43_43',   # Default dataset name
        'test_split_ratio': 0.25,
        'subsample': True, # Whether to subsample data on large datasets to max 10000 samples
        'error_metric': 'RMSE',
        'functions': ['add', 'sub', 'mul', 'div', 'pow'],
        'ephemerals': [-0.5, 0.5],  # Range for ephemeral constants
        'standardization': True
    }
}

DATA_GENERATOR_DEFAULTS = {
    **STANDARD_GP_DEFAULTS,  # Inherit most settings
    'algorithm': 'data_generator',
    'data_generator': {
        # Dataset generation mode:
        #   'make_regression'   – random linear dataset (sklearn)
        #   'target_function'   – evaluate a user-supplied symbolic expression
        'dataset_generation': 'make_regression',
        # Symbolic expression used when dataset_generation == 'target_function'.
        # Variables must be named x0, x1, … matching num_features.
        # Standard NumPy functions (sin, cos, exp, log, sqrt, …) are available.
        # Example: "sin(x0) + x1**2"
        'target_function': None,
        # Optional pool of target functions for 'target_function' mode.
        # When set to a non-empty list, one function is randomly selected per
        # run using the run's random_seed, giving each GP job a different
        # non-linear landscape and maximising archive diversity across many runs.
        # Example: ["sin(x0) + x1**2", "x0**3 - x1", "exp(-x0) * cos(x1)"]
        'target_function_pool': None,
        # Input range [low, high] for uniform sampling when using target_function.
        'input_range': [-1.0, 1.0],
        'num_samples_dataset': 1000,
        'num_features': 5,
        'noise_level': 1.0,
        'num_samples_archive': 150000,
        'archive_folder': 'trainingdata/',
        'simplification': True
    }
}