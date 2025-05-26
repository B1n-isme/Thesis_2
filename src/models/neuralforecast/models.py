"""
Enhanced model configurations optimized for Bitcoin forecasting.
"""
from typing import List, Optional, Any, Dict
from neuralforecast.models import NHITS, NBEATS, TFT, LSTM, GRU, RNN, TCN, DeepAR
from neuralforecast.losses.pytorch import MAE
from config.base import LOCAL_SCALER_TYPE, TUNING_RESULTS_DIR
from src.utils.hyperparam import load_best_hyperparameters_from_json
import numpy as np
import pandas as pd
import os
import json


def get_neural_models(
    h: int, 
    hist_exog_list: Optional[List[str]] = None,
    hyperparameters_json_path: Optional[str] = None
) -> List[Any]:
    """
    Get neural forecasting models with default configurations, optionally enhanced 
    with best hyperparameters from JSON file.
    
    Args:
        h: Forecast horizon
        hist_exog_list: List of historical exogenous features
        hyperparameters_json_path: Path to best_hyperparameters.json file.
                                 If None, defaults to TUNING_RESULTS_DIR/best_hyperparameters.json
        
    Returns:
        List of configured neural model instances
    """
    
    # Define model configurations with their default parameters
    model_configs = _get_default_model_configs(h, hist_exog_list)
    
    # Load and apply hyperparameters if JSON path is provided
    if hyperparameters_json_path is not None:
        if hyperparameters_json_path == "":
            hyperparameters_json_path = os.path.join(TUNING_RESULTS_DIR, 'best_hyperparameters.json')
        
        best_hyperparams_map = load_best_hyperparameters_from_json(hyperparameters_json_path)
        if best_hyperparams_map:
            model_configs = _apply_hyperparameters(model_configs, best_hyperparams_map, h)
    
    # Instantiate all models
    return _instantiate_models(model_configs)


def _get_default_model_configs(h: int, hist_exog_list: Optional[List[str]] = None) -> Dict[str, Dict]:
    """
    Get default configurations for all neural models.
    
    Args:
        h: Forecast horizon
        hist_exog_list: List of historical exogenous features
        
    Returns:
        Dictionary mapping model names to their default configurations
    """
    # Base configuration shared across all models
    base_config = {
        'h': h,
        'input_size': h * 2,
        'loss': MAE(),
        'learning_rate': 1e-3,
        'max_steps': 1000,
        'val_check_steps': 50,
        'batch_size': 32,
        'windows_batch_size': 256,
        'scaler_type': LOCAL_SCALER_TYPE,
    }
    
    # Include exogenous features if available
    if hist_exog_list:
        base_config['hist_exog_list'] = hist_exog_list
    
    # Model-specific configurations
    model_configs = {
        'NHITS': {
            **base_config,
            'model_class': NHITS,
            'stack_types': ['identity', 'identity', 'identity'],
            'n_blocks': [1, 1, 1],
            'mlp_units': [[512, 512], [512, 512], [512, 512]],
            'n_pool_kernel_size': [2, 2, 1],
            'n_freq_downsample': [4, 2, 1],
            'interpolation_mode': 'linear',
        },
        
        'NBEATS': {
            **base_config,
            'model_class': NBEATS,
            'stack_types': ['trend', 'seasonality'],
            'n_blocks': [3, 3],
            'mlp_units': [[512, 512], [512, 512]],
        },
        
        'LSTM': {
            **base_config,
            'model_class': LSTM,
            'encoder_n_layers': 2,
            'encoder_hidden_size': 128,
            'decoder_hidden_size': 128,
            'decoder_layers': 2,
        },
        
        'TFT': {
            **base_config,
            'model_class': TFT,
            'hidden_size': 64,
            'lstm_layers': 2,
            'num_attention_heads': 4,
            'add_relative_index': True,
        },
        
        'GRU': {
            **base_config,
            'model_class': GRU,
            'encoder_n_layers': 2,
            'encoder_hidden_size': 128,
            'decoder_hidden_size': 128,
            'decoder_layers': 2,
        },
    }
    
    return model_configs


def _apply_hyperparameters(
    model_configs: Dict[str, Dict], 
    best_hyperparams_map: Dict[str, Dict], 
    h: int
) -> Dict[str, Dict]:
    """
    Apply best hyperparameters from JSON to model configurations.
    
    Args:
        model_configs: Default model configurations
        best_hyperparams_map: Best hyperparameters loaded from JSON
        h: Forecast horizon
        
    Returns:
        Updated model configurations
    """
    # Parameters that should not be overridden by HPO
    protected_params = ['loss', 'model_class']
    
    # Parameters used internally by HPO that should be filtered out
    internal_hpo_keys = ['backend', 'num_samples', 'config_id', 'trial_id', 'experiment_tag']
    
    for model_name, config in model_configs.items():
        # Try different naming conventions for the model in JSON
        possible_json_keys = [
            f'Auto{model_name}',  # AutoNHITS, AutoNBEATS, etc.
            model_name,           # NHITS, NBEATS, etc.
            f'{model_name}_Mock', # For mock versions if they exist
            f'Auto{model_name}_Mock'
        ]
        
        loaded_hpo_params = None
        matched_key = None
        
        # Find the matching hyperparameters in the JSON
        for json_key in possible_json_keys:
            if json_key in best_hyperparams_map:
                loaded_hpo_params = best_hyperparams_map[json_key]
                matched_key = json_key
                break
        
        if loaded_hpo_params:
            print(f"Applying HPO parameters for {model_name} (found as '{matched_key}'):")
            print(f"  HPO params: {loaded_hpo_params}")
            
            # Apply hyperparameters with validation
            for param_key, hpo_val in loaded_hpo_params.items():
                if param_key in protected_params:
                    print(f"    Skipping protected parameter '{param_key}' for {model_name}")
                    continue
                    
                if param_key in internal_hpo_keys:
                    print(f"    Filtering out internal HPO parameter '{param_key}' for {model_name}")
                    continue
                    
                if param_key == 'h' and hpo_val != h:
                    print(f"    Warning: HPO horizon '{hpo_val}' differs from function arg '{h}'. Using function arg.")
                    continue
                    
                # Apply the hyperparameter
                config[param_key] = hpo_val
                print(f"    Applied {param_key}: {hpo_val}")
        else:
            print(f"No HPO configuration found for {model_name} (tried: {possible_json_keys})")
            print(f"  Using default configuration for {model_name}")
    
    return model_configs


def _instantiate_models(model_configs: Dict[str, Dict]) -> List[Any]:
    """
    Instantiate model instances from configurations.
    
    Args:
        model_configs: Dictionary of model configurations
        
    Returns:
        List of instantiated model objects
    """
    models = []
    
    for model_name, config in model_configs.items():
        try:
            # Extract the model class
            ModelClass = config.pop('model_class')
            
            # Create model instance with remaining configuration
            print(f"Instantiating {model_name} with configuration:")
            for key, value in config.items():
                print(f"  {key}: {value} (type: {type(value)})")
            
            model_instance = ModelClass(**config)
            models.append(model_instance)
            print(f"Successfully created {model_name} model")
            
        except Exception as e:
            print(f"ERROR instantiating {model_name}: {e}")
            print(f"Problematic configuration for {model_name}:")
            for k, v in config.items():
                print(f"  {k}: {v} (type: {type(v)})")
            # Continue with other models instead of failing completely
            continue
    
    if not models:
        print("Warning: No models were successfully instantiated")
    else:
        print(f"Successfully instantiated {len(models)} models: {[type(m).__name__ for m in models]}")
    
    return models