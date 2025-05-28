from typing import List, Optional, Any, Dict
import ray
from ray import tune
from ray.tune.search.hyperopt import HyperOptSearch
import torch
from neuralforecast.auto import AutoNHITS, AutoNBEATS, AutoTFT, AutoLSTM, AutoGRU
from neuralforecast.losses.pytorch import MAE, MSE, RMSE, MAPE, SMAPE
from ray.tune import CLIReporter
from ray.tune.logger import LoggerCallback

from config.base import SCALER_TYPE
from config.search_algo import get_search_algorithm_class
from src.models.neuralforecast.auto_cfg import (
    neural_auto_model_cfg,
    neural_auto_model_cfg_legacy,
)

def get_auto_models(
    h: int, num_samples: int = 10, hist_exog_list: Optional[List[str]] = None
) -> List[Any]:
    """
    Get auto models for hyperparameter optimization.

    Args:
        h: Forecast horizon
        loss_fn: Loss function (default MAE)
        num_samples: Number of hyperparameter samples per model
        hist_exog_list: List of historical exogenous features

    Returns:
        List of auto model instances for HPO
    """

    configs = neural_auto_model_cfg(h)
    search_alg = get_search_algorithm_class("hyperopt")

    init_config = {
        "h": h,
        "loss": MAE(),
        "search_alg": search_alg,
        "num_samples": num_samples,
        "verbose": False,
        "early_stop_patience_steps": 10
    }

    # base_auto_config = {
    #     "input_size": tune.choice([h * 2, h * 3, h * 4, h * 6]),
    #     "learning_rate": tune.choice([1e-4, 1e-3, 5e-3]),
    #     "scaler_type": tune.choice(SCALER_TYPE),
    #     "max_steps": tune.choice([500, 1000, 1500]),
    #     "batch_size": tune.choice([16, 32, 64]),
    #     "windows_batch_size": tune.choice([128, 256, 512]),
    #     "val_check_steps": 50,
    #     "random_seed": tune.randint(1, 20),
    # }
    base_auto_config = {
        "input_size": tune.choice([h * 2, h * 3, h * 4, h * 6]),
        "learning_rate": tune.choice([1e-4, 1e-3, 5e-3]),
        "scaler_type": tune.choice(SCALER_TYPE),
        "max_steps": tune.choice([50, 100]),
        "batch_size": tune.choice([16, 32, 64]),
        "windows_batch_size": tune.choice([128, 256, 512]),
        "val_check_steps": 50,
        "random_seed": tune.randint(1, 20),
    }

    for model_name, model_config in configs.items():
        configs[model_name] = {**base_auto_config, **model_config}

    if hist_exog_list:
        base_auto_config["hist_exog_list"] = hist_exog_list

    models = [
        # Primary auto models for HPO
        AutoNHITS(**init_config, config=configs["nhits"]),
        # AutoNBEATS(**init_config, config=configs["nbeats"]),
        # AutoLSTM(**init_config, config=configs["lstm"]),
        # AutoTFT(**init_config, config=configs["tft"]),
    ]

    return models

