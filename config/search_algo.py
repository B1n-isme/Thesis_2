from ray import tune
from ray.tune.search.basic_variant import BasicVariantGenerator
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search.bohb import TuneBOHB
from ray.tune.search.hebo import HEBOSearch
from ray.tune.search.optuna import OptunaSearch

from typing import Any


# For initial baseline establishment (as recommended in algo.txt)
BASELINE = ['random', 'hyperopt']

# For efficient optimization (BOHB and Optuna with TPE)
EFFICIENT = ['bohb', 'optuna']

# For noisy time series data
NOISY_DATA = ['hebo', 'random']

def get_search_algorithm_class(algorithm: str) -> Any:
    if algorithm == 'random':
        return BasicVariantGenerator()
        
    elif algorithm == 'hyperopt':
        return HyperOptSearch()
        
    elif algorithm == 'bohb':
        return TuneBOHB()
        
    elif algorithm == 'hebo':
        return HEBOSearch()
        
    elif algorithm == 'optuna':
        return OptunaSearch()
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")