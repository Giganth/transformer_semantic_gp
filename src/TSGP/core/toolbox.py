"""
Factory for creating DEAP toolboxes for GP algorithms.
"""

from deap import base, gp, tools, creator
from typing import Any, Dict, Callable
import operator
import multiprocessing

from TSGP.config.defaults import MAX_TREE_DEPTH

def create_toolbox(
    config: Dict[str, Any], pset: gp.PrimitiveSet, evaluation_function: Callable, evaluation_function_test: Callable, evaluation_function_semantic: Callable
    ) -> base.Toolbox:

    """
    Create a DEAP toolbox configured for GP.
    
    Args:
        config: Configuration dictionary
        pset: Primitive set
        evaluation_function: Function to evaluate individuals
        evaluation_function_test: Function to evaluate individuals on test set
        evaluation_function_semantic: Function to evaluate individuals for semantic distance
    
    Returns:
        Configured DEAP toolbox
    """
    toolbox = base.Toolbox()
    toolbox.register("evaluate", evaluation_function)
    toolbox.register("evaluate_test", evaluation_function_test)
    toolbox.register("evaluate_semantic", evaluation_function_semantic)
    
    # Enable multiprocessing only if num_cores > 1
    if config['num_cores'] > 1:
        pool = multiprocessing.Pool(processes=config['num_cores'])
        toolbox.register("map", pool.map)
        # Store pool reference for proper cleanup
        toolbox.pool = pool
    else:
        toolbox.pool = None

    # Individual and Population
    creator.create("Fitness", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.Fitness,fitness_test = creator.Fitness, fitness_semantic = creator.Fitness)
    
    # Initialization Algorithm
    if config['initialization']['mode'] == 'ramped-half-and-half':
        toolbox.register("initialization_alg", gp.genHalfAndHalf, pset=pset,
                            min_=config['initialization']['min_tree_depth'],
                            max_=config['initialization']['max_tree_depth'])
    elif config['initialization']['mode'] == 'full':
        toolbox.register("initialization_alg", gp.genFull, pset=pset,
                            min_=config['initialization']['min_tree_depth'],
                            max_=config['initialization']['max_tree_depth'])
    elif config['initialization']['mode'] == 'grow':
        toolbox.register("initialization_alg", gp.genGrow, pset=pset,
                            min_=config['initialization']['min_tree_depth'],
                            max_=config['initialization']['max_tree_depth'])

    toolbox.register("createIndividual", tools.initIterate, creator.Individual, toolbox.initialization_alg)

    # Selection Configuration
    if config['selection']['type'] == 'tournament':
        toolbox.register('select', tools.selTournament, tournsize=config['selection']['tournament_size'])
    elif config['selection']['type'] == 'random':
        toolbox.register('select', tools.selRandom)
    elif config['selection']['type'] == 'fitness_proportionate':
        toolbox.register('select', tools.selRoulette)
    elif config['selection']['type'] == 'aut_epsilon_lexicase':
        toolbox.register('select', tools.selAutomaticEpsilonLexicase)
    elif config['selection']['type'] == 'double_tournament':
        toolbox.register('select', tools.selDoubleTournament, fitness_size=config['selection']['tournament_size'], parsimony_size=config['selection']['parsimony_size'], fitness_first=config['selection']['fitness_first'])
    elif config['selection']['type'] == 'none':
        print("Selection is set to none!")

    # Variation Configuration
    if config['algorithm'] == "standard_gp" or config['algorithm'] == "data_generator":
        variation = config['variation']
        
        # Register crossover
        if 'crossover' in variation:
            crossover_config = variation['crossover']
            toolbox.register("mate", gp.cxOnePointLeafBiased, termpb=crossover_config['node_bias_terminals'])
            # recommended by DEAP developers based on Koza:
            toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=MAX_TREE_DEPTH))
        
        # Register mutation
        if 'mutation' in variation:
            mutation_config = variation['mutation']
            if mutation_config['mode'] == 'ramped-half-and-half':
                toolbox.register("expr_mut", gp.genHalfAndHalf, min_=mutation_config['min_tree_depth'],
                                    max_=mutation_config['max_tree_depth'])
            elif mutation_config['mode'] == 'full':
                toolbox.register("expr_mut", gp.genFull, min_=mutation_config['min_tree_depth'],
                                    max_=mutation_config['max_tree_depth'])
            elif mutation_config['mode'] == 'grow':
                toolbox.register("expr_mut", gp.genGrow, min_=mutation_config['min_tree_depth'],
                                    max_=mutation_config['max_tree_depth'])
            toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
            # recommended by DEAP developers based on Koza:
            toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=MAX_TREE_DEPTH))
    elif config['algorithm'] == "transformer_gp":
        # Not added to the toolbox as variation is handled in the TransformerGP class
        pass
    else:
        raise ValueError(f"Unknown algorithm type: {config['algorithm']}")
            
    return toolbox