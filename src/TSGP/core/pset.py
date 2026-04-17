import random
from deap import gp
import operator
from functools import partial

import numpy as np

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


def ephemeral_generator(values):
    """Generate a random ephemeral constant from the given values."""
    return random.choice(values)

def div(x: float, y: float) -> float: 
    """ Protected Division as in Koza 1992, S. 82f."""
    return 1. if y == 0 else float(x) / float(y)

def pow(base: float, exp: float) -> float:
    """ Protected Power function to avoid overflow and complex numbers."""
    try:
        # Handle cases where base is negative and exponent is fractional
        if base < 0 and not float(exp).is_integer():
            return 1.0  # Avoid complex numbers
        result = base ** exp
        
        # Check for infinity or NaN
        if not np.isfinite(result):
            return 1.0
            
        return result
    except (OverflowError, ZeroDivisionError, ValueError):
        return 1.0 

def create_pset(config: dict) -> gp.PrimitiveSet:
    """
    Creates a DEAP PrimitiveSet based on the configuration.

    Args:
        config (dict): Configuration dictionary containing function set and terminal set.

    Returns:
        gp.PrimitiveSet: Configured DEAP PrimitiveSet.
    """
    pset = gp.PrimitiveSet("MAIN", len(config['problem']['variables']))

    # Add functions to the primitive set
    for func in config['problem']['functions']:
        if func == 'add':
            pset.addPrimitive(operator.add, 2)
        elif func == 'sub':
            pset.addPrimitive(operator.sub, 2)
        elif func == 'mul':
            pset.addPrimitive(operator.mul, 2)
        elif func == 'div':
            pset.addPrimitive(div, 2)
        elif func == 'pow':
            pset.addPrimitive(pow, 2)

    rename_dict = {f'ARG{i}': var for i, var in enumerate(config['problem']['variables'])}
    pset.renameArguments(**rename_dict)
    
    # Generate ephemeral values
    ephemeral_values = [
        float(round(i, 2)) for i in
        np.arange(float(config['problem']['ephemerals'][0]),
                float(config['problem']['ephemerals'][1]) + 0.1,
                0.1)
    ]
    
    eph_callable = partial(ephemeral_generator, ephemeral_values)
    pset.addEphemeralConstant('eph', eph_callable)

    return pset
