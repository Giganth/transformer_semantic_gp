from copy import deepcopy

# Maximum tree depth as recommended by DEAP developers based on Koza (1992)
MAX_TREE_DEPTH = 17

# Vocabulary special token indices (set by TextVectorization ordering)
PADDING_INDEX = 0
UNKNOWN_INDEX = 1

STANDARD_GP_DEFAULTS = {
    'algorithm': 'standard_gp',
    'popsize': 100,
    'initialization': {
        'mode': 'ramped-half-and-half',  # the only mode
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
    'num_cores': 10,
    'use_gpu': True,  # Whether to use GPU for TensorFlow operations
}

TS_GP_DEFAULTS = {
    **STANDARD_GP_DEFAULTS,  # Inherit most settings
    'algorithm': 'transformer_gp',
    
    'variation': {
        'transformer': {
            'model_directory': 'src/TSGP/transformer/models/TSGP-2.weights.h5',  # Path to pretrained model
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
        'similarity_threshold'             : 100,    # largest Euclidean distance between the semantics of a pair
        'n_neighbors'                      : 3,     # amt. of neighbors to retrieve from similarity search, 3 default
                
        # Settings for pretraining
        'max_samples_per_archive'          : 2000000,  # Maximum samples to load per archive file
        'batch_size'                       : 256,
        'learning_rate'                    : 0.001,
        'decay_steps'                      : 550000,
        'max_epochs'                       : 8,
        'val_split_ratio'                  : 0.1,   # percentage how much of the data is used for the validation set during training
        
        
        # Feature flags - what additional inputs to include
        'include_semantic_distance'        : True,  # if True the semantic distance is included in training
        'include_dimensionality'           : True,  # if True the dimensionality is included in training

        # Settings for vocabulary. Defines the token of the transformer and thus which functions can be created
        'functions'                        : ['add', 'sub', 'mul', 'div','pow'], 
        'variables'                        : ['x0', 'x1', 'x2', 'x3','x4'], 
        'ephemerals'                       : [-5.0,-4.0,-3.0,-2.0,-1.0,-0.5,-0.4,-0.3,-0.2,-0.1, 0.0, 0.1,0.2 ,0.3, 0.4,0.5,1.0,2.0,3.0,4.0,5.0],
    
        # The pairs the model is trained with are stored here, plus a test file
        # of further pairs taken from the same archives.
        'train_data_path'                  : 'trainingdata/train_pairs.pkl',
        'test_data_path'                   : 'trainingdata/test_pairs.pkl',
        'test_split_ratio'                 : 0.1,

        # Paths for saving/loading models and logs
        'model_directory'                 : 'src/TSGP/transformer/models',
        
        # Data paths - directories containing training archives
        'archive_directories': [
        'trainingdata/Archive_2',  # 2D problems
        'trainingdata/Archive_3',  # 3D problems
        'trainingdata/Archive_4',  # 4D problems
        'trainingdata/Archive_5',  # 5D problems
        ],  
    },

}

SYMBOLIC_REGRESSION_DEFAULTS = {
    'problem': {
        'dataset': 'feynman_II_13_17',  # Default dataset name
        'test_split_ratio': 0.25,
        'subsample': True, # Whether to subsample data on large datasets to max 10000 samples
        'error_metric': 'RMSE',  # the only metric
        'functions': ['add', 'sub', 'mul', 'div', 'pow'],
        'ephemerals': [-0.5, 0.5],  # Range for ephemeral constants
        'standardization': True,  # Whether to standardize inputs and target
        'load_semantic_testset': False,  # If False, semantics are measured on the training cases.
    }
}

DATA_GENERATOR_DEFAULTS = {
    **STANDARD_GP_DEFAULTS,  # Inherit most settings
    **TS_GP_DEFAULTS,
    **SYMBOLIC_REGRESSION_DEFAULTS,
    'algorithm': 'data_generator',
    'variation': deepcopy(STANDARD_GP_DEFAULTS['variation']),
    'data_generator': {
        'dataset_generation': 'random_function_target',  # 'random_function_target' or 'make_regression'
        'num_samples_dataset': 1000,
        'num_features': 2,
        'functions': ['add', 'sub', 'mul', 'div', 'pow','sin','cos'],
        'ephemerals': [-0.5, 0.5],  # Range for ephemeral constants
        'ephemeral_stepsize': 0.1,  # Step between the constants sampled from that range
        'noise_level': 0.1,  # Standard deviation of Gaussian noise added to outputs
        'num_samples_archive': 100000,
        'archive_folder': 'trainingdata/',
        'simplification': True,

        # Options of the synthetic problem generator
        'nl': 1.0,                     # Node level factor for position sampling (1.0 = uniform, >1.0 = deeper trees, <1.0 = shallower trees)

        'operator_weights': {
            'add': 1.0,
            'sub': 1.0,
            'mul': 1.0,
            'div': 1.0,    # Lower weight = less likely to be selected
            'sin': 1.0,
            'sqrt': 1.0,
        }
    }
}