# Standard library imports
from typing import List, Optional, Any, Dict

# Third-party imports
import ray
from ray import tune
import torch
from neuralforecast.auto import (
    AutoDLinear,
    AutoTFT,
    AutoiTransformer,
    AutoNBEATSx, 
    AutoTSMixerx,
    AutoPatchTST,
    AutoTimesNet,
    AutoFEDformer,
    AutoAutoformer
)
from neuralforecast.losses.pytorch import (
    MAE,
    # MSE,
    # RMSE,
    # MAPE,
    # SMAPE,
    # DistributionLoss
)

# Local imports
from config.base import get_search_algorithm_class
from src.models.neuralforecast.auto_cfg import neural_auto_cfg

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

    configs = neural_auto_cfg(horizon)
    search_alg = get_search_algorithm_class('hyperopt')

    init_config = {
        "h": horizon,
        "loss": MAE(),
        # "loss": DistributionLoss("StudentT", level=[90]),
        "search_alg": search_alg,
        "num_samples": num_samples,
        "verbose": False,
        "refit_with_val": False,
    }

    base_auto_config = {
        "input_size": tune.choice([horizon * 2, horizon * 3, horizon * 4, horizon * 6]),
        "learning_rate": tune.choice([1e-4, 5e-4, 1e-3, 2e-3]),
        "scaler_type": 'robust',
        "max_steps": tune.choice([50, 100, 150, 200]),
        "batch_size": tune.choice([16, 32, 64]),
        "windows_batch_size": tune.choice([256, 512, 1024]),
        "val_check_steps": 50, 
        "random_seed": tune.randint(1, 20),
        "early_stop_patience_steps": 10,
        # trainer_kwargs values
        'accelerator': 'gpu',
        'logger': False,
    }

    # base_auto_config = {
    #     "input_size": tune.choice([horizon * 2]),
    #     "learning_rate": tune.choice([5e-3]),
    #     "scaler_type": 'robust',
    #     "max_steps": tune.choice([30]),
    #     "batch_size": tune.choice([32]),
    #     # "windows_batch_size": tune.choice([128]),
    #     "val_check_steps": 20, 
    #     "random_seed": tune.randint(1, 20),
    #     "early_stop_patience_steps": 2,
    #     # trainer_kwargs values
    #     'accelerator': 'gpu',
    #     'logger': False,
    # }

    transformer_config = configs["transformer"].copy()
    transformer_config = {**base_auto_config, **transformer_config}

    if hist_exog_list:
        base_auto_config["hist_exog_list"] = hist_exog_list

    for model_name, model_config in configs.items():
        configs[model_name] = {**base_auto_config, **model_config}

    models = [
        # Primary auto models for direct use
        # AutoNHITS(**init_config, config=configs["nhits"]),
        AutoTFT(**init_config, config=configs["tft"]),
        AutoiTransformer(**init_config, n_series=1, config=transformer_config),
        AutoNBEATSx(**init_config, config=configs["nbeats"]),
        AutoTSMixerx(**init_config, n_series=1, config=configs["tsmixer"]),
        # AutoLSTM(**init_config, config=configs["lstm"]),
    ]

    return models

