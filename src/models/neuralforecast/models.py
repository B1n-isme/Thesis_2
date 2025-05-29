# Standard library imports
from typing import List, Optional, Any, Dict

# Third-party imports
import ray
from ray import tune
import torch
from neuralforecast.auto import (
    AutoNHITS,
    AutoNBEATS, 
    AutoTFT,
    AutoLSTM,
    AutoGRU
)
from neuralforecast.losses.pytorch import (
    MAE,
    MSE,
    RMSE,
    MAPE,
    SMAPE,
    DistributionLoss
)

# Local imports
from config.base import SCALER_TYPE, DEFAULT_SEARCH_ALGORITHM
from config.search_algo import get_search_algorithm_class
from src.models.neuralforecast.auto_cfg import (
    neural_auto_model_cfg,
    neural_auto_model_cfg_legacy
)

def get_neural_models(
    horizon: int, num_samples: int = 10, hist_exog_list: Optional[List[str]] = None
) -> List[Any]:
    """
    Get auto models for direct use in the pipeline.

    Args:
        horizon: Forecast horizon
        num_samples: Number of hyperparameter samples per model
        hist_exog_list: List of historical exogenous features

    Returns:
        List of auto model instances for direct use
    """

    configs = neural_auto_model_cfg(horizon)
    search_alg_name = DEFAULT_SEARCH_ALGORITHM
    search_alg = get_search_algorithm_class(search_alg_name)

    init_config = {
        "h": horizon,
        # "loss": MAE(),
        "loss": DistributionLoss("Normal", level=[90]),
        "search_alg": search_alg,
        "num_samples": num_samples,
        "verbose": False,
        "refit_with_val": False,
    }

    base_auto_config = {
        "input_size": tune.choice([horizon * 2, horizon * 3, horizon * 4, horizon * 6]),
        "learning_rate": tune.choice([1e-4, 1e-3, 5e-3]),
        "scaler_type": tune.choice(SCALER_TYPE),
        "max_steps": tune.choice([50, 100]),
        "batch_size": tune.choice([16, 32, 64]),
        "windows_batch_size": tune.choice([128, 256, 512]),
        "val_check_steps": 50, 
        "random_seed": tune.randint(1, 20),
        "early_stop_patience_steps": 10,
        # trainer_kwargs values
        'accelerator': 'gpu',
        'logger': False,
    }

    if hist_exog_list:
        base_auto_config["hist_exog_list"] = hist_exog_list

    for model_name, model_config in configs.items():
        configs[model_name] = {**base_auto_config, **model_config}

    models = [
        # Primary auto models for direct use
        AutoNHITS(**init_config, config=configs["nhits"]),
        # AutoNBEATS(**init_config, config=configs["nbeats"]),
        # AutoLSTM(**init_config, config=configs["lstm"]),
        # AutoTFT(**init_config, config=configs["tft"]),
    ]

    return models

