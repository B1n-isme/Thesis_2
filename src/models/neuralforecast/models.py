# Standard library imports
from typing import List, Optional, Any, Dict
from pathlib import Path

# Third-party imports
import ray
from ray import tune
import torch
from neuralforecast.auto import (
    AutoTFT,
    AutoNBEATSx, 
    AutoTSMixerx,
    AutoiTransformer,
    AutoBiTCN,
    AutoKAN,
)
from neuralforecast.models import (
    TFT,
    NBEATSx,
    TSMixerx,
    iTransformer,
    BiTCN,
    KAN,
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
from src.utils.utils import load_yaml_to_dict

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
        # "early_stop_patience_steps": 10,
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
        AutoTFT(**init_config, config=configs["tft"]),
        AutoiTransformer(**init_config, n_series=1, config=transformer_config),
        AutoNBEATSx(**init_config, config=configs["nbeats"]),
        AutoTSMixerx(**init_config, n_series=1, config=configs["tsmixer"]),
        AutoBiTCN(**init_config, config=configs["bitcn"]),
        AutoKAN(**init_config, config=configs["kan"]),
    ]

    return models


def get_normal_neural_models(
    horizon: int,
    config_path: Path,
    hist_exog_list: Optional[List[str]] = None,
) -> List[Any]:
    """
    Get normal model instances with best hyperparameters from a config file.

    Args:
        horizon: Forecast horizon.
        config_path: Path to the YAML configuration file with best hyperparameters.
        hist_exog_list: List of historical exogenous features.

    Returns:
        List of normal model instances.
    """
    configs = load_yaml_to_dict(config_path)

    # Map of known neural models. Models not in this map will be skipped.
    neural_model_map = {
        "TFT": TFT,
        "NBEATSx": NBEATSx,
        "TSMixerx": TSMixerx,
        "iTransformer": iTransformer,
        "BiTCN": BiTCN,
        "KAN": KAN,
    }

    # Add "Auto" prefixed versions to map to the same models for flexibility
    auto_model_map = {f"Auto{k}": v for k, v in neural_model_map.items()}
    neural_model_map.update(auto_model_map)

    models = []
    for model_name, model_config in configs.items():
        if model_name in neural_model_map:
            ModelClass = neural_model_map[model_name]
            
            # Basic model config
            init_config = {
                "h": horizon,
                "loss": MAE(),
            }
            if hist_exog_list and ModelClass != iTransformer:
                init_config['hist_exog_list'] = hist_exog_list
            
            # For models that require n_series, check against the actual class
            if ModelClass in [TSMixerx, iTransformer]:
                init_config['n_series'] = 1

            # Parameters to skip from CV config as they are not needed for final fitting
            params_to_skip = [
                'early_stop_patience_steps', 
                'accelerator',
                'logger'
            ]
            for param in params_to_skip:
                model_config.pop(param, None)

            # Combine all configs
            final_config = {**init_config, **model_config}

            # Print the final config
            print(f"  Final config for {ModelClass.__name__}: {final_config}")

            try:
                model_instance = ModelClass(**final_config)
                models.append(model_instance)
                print(f"  ✓ Instantiated {ModelClass.__name__} with loaded config.")
            except Exception as e:
                print(f"  ✗ Error instantiating {ModelClass.__name__}: {e}")
                print(f"    Config used: {final_config}")

        else:
            # This is expected for ML models like AutoXGBoost, etc.
            print(f"  - Skipping non-neural model from config: '{model_name}'")
            
    return models

