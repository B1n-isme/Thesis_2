"""
Enhanced model configurations optimized for Bitcoin forecasting.
"""
from typing import List, Optional, Any, Dict
from neuralforecast.models import NHITS, NBEATS, TFT, LSTM, GRU, RNN, TCN, DeepAR
from neuralforecast.losses.pytorch import MAE
from lightning.pytorch.callbacks import RichProgressBar
from config.base import LOCAL_SCALER_TYPE, BEST_HYPERPARAMETERS_CSV
import numpy as np
import pandas as pd
import os
import json

trainer_kwargs = {
    'accelerator': 'gpu',
    'logger': False,
    'callbacks': RichProgressBar()
}


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
        
    Returns:
        List of configured neural model instances
    """
    
    # Define model configurations with their default parameters
    model_configs = _get_default_model_configs(h, hist_exog_list)
    
    print(f"Getting neural models with horizon= {h}")
    print(f"Exogenous features: {len(hist_exog_list)}")
    print(f"Hyperparameters path: {hyperparameters_json_path}")
    
    # Load and apply hyperparameters if JSON path is provided
    if hyperparameters_json_path is None:
        hyperparameters_json_path = BEST_HYPERPARAMETERS_CSV
        
    # Use the correct import from the refactored HPO system
    from src.pipelines.hyperparameter_tuning import load_best_hyperparameters
    
    best_hyperparams_map = load_best_hyperparameters(hyperparameters_json_path)
    if best_hyperparams_map:
        print(f"Loaded hyperparameters for {len(best_hyperparams_map)} models")
        model_configs = _apply_hyperparameters(model_configs, best_hyperparams_map, h)
    else:
        print("No hyperparameters loaded, using default configurations")
    
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
        'input_size': h * 6,  # Increased from h*2 for better performance
        'loss': MAE(),
        'learning_rate': 1e-3,
        'max_steps': 500,     # Reduced from 1000 for faster training
        'val_check_steps': 50,
        'batch_size': 32,
        'windows_batch_size': 256,
        'scaler_type': LOCAL_SCALER_TYPE,
        'random_seed': 42,    # Add for reproducibility
        'trainer_kwargs': trainer_kwargs
    }
    
    # Include exogenous features if available
    if hist_exog_list:
        base_config['hist_exog_list'] = hist_exog_list
    
    # Model-specific configurations matching HPO Auto models
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
            'dropout_prob_theta': 0.0,
        },
        
        # 'NBEATS': {
        #     **base_config,
        #     'model_class': NBEATS,
        #     'stack_types': ['trend', 'seasonality'],
        #     'n_blocks': [3, 3],
        #     'mlp_units': [[512, 512], [512, 512]],
        #     'sharing': [False, False],
        # },
        
        # 'LSTM': {
        #     **base_config,
        #     'model_class': LSTM,
        #     'encoder_n_layers': 2,
        #     'encoder_hidden_size': 128,
        #     'decoder_hidden_size': 128,
        #     'decoder_layers': 1,
        # },
        
        # 'TFT': {
        #     **base_config,
        #     'model_class': TFT,
        #     'hidden_size': 64,
        #     'n_rnn_layers': 2,
        #     'n_head': 4,
        #     'dropout': 0.1,
        # },
        
        # 'GRU': {
        #     **base_config,
        #     'model_class': GRU,
        #     'encoder_n_layers': 2,
        #     'encoder_hidden_size': 128,
        #     'decoder_hidden_size': 128,
        #     'decoder_layers': 1,
        # },
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
        best_hyperparams_map: Best hyperparameters loaded from JSON (from HPO system)
        h: Forecast horizon
        
    Returns:
        Updated model configurations
    """
    # Parameters that should not be overridden by HPO
    protected_params = {'loss', 'model_class', 'valid_loss'}
    
    # Parameters used internally by HPO that should be filtered out
    internal_hpo_keys = {
        'backend', 'num_samples', 'config_id', 'trial_id', 'experiment_tag',
        'model_name', 'best_valid_loss', 'training_iteration'
    }
    
    print(f"Available HPO configurations: {list(best_hyperparams_map.keys())}")
    
    for model_name, config in model_configs.items():
        # The HPO system saves models with 'Auto' prefix
        json_key = f'Auto{model_name}'

        loaded_hpo_params = None
        matched_key = None

        # Find the matching hyperparameters in the JSON
        if json_key in best_hyperparams_map:
            loaded_hpo_params = best_hyperparams_map[json_key].copy()  # Copy to avoid modifying original
            matched_key = json_key
        
        if loaded_hpo_params:
            print(f"\n=== Applying HPO parameters for {model_name} (found as '{matched_key}') ===")
            print(f"Original HPO params: {loaded_hpo_params}")
            
            # Clean up and validate hyperparameters
            cleaned_params = {}
            for param_key, hpo_val in loaded_hpo_params.items():
                if param_key in protected_params:
                    print(f"  → Skipping protected parameter '{param_key}'")
                    continue
                    
                if param_key in internal_hpo_keys:
                    print(f"  → Filtering out internal HPO parameter '{param_key}'")
                    continue
                    
                if param_key == 'h' and hpo_val != h:
                    print(f"  → Warning: HPO horizon '{hpo_val}' differs from function arg '{h}'. Using function arg.")
                    continue
                
                # Handle special parameter types
                cleaned_val = _process_hyperparameter_value(param_key, hpo_val)
                if cleaned_val is not None:
                    cleaned_params[param_key] = cleaned_val
                    print(f"  ✓ {param_key}: {hpo_val} → {cleaned_val} (type: {type(cleaned_val)})")
                else:
                    print(f"  ✗ Skipped invalid parameter '{param_key}': {hpo_val}")
            
            # Apply cleaned hyperparameters to model config
            config.update(cleaned_params)
            print(f"  Applied {len(cleaned_params)} hyperparameters to {model_name}")
            
        else:
            print(f"\n=== No HPO configuration found for {model_name} ===")
            print(f"  Tried keys: {json_key}")
            print(f"  Available keys: {list(best_hyperparams_map.keys())}")
            print(f"  Using default configuration for {model_name}")
    
    return model_configs


def _process_hyperparameter_value(param_key: str, value: Any) -> Any:
    """
    Process and validate hyperparameter values, handling special cases.
    
    Args:
        param_key: Parameter name
        value: Parameter value from HPO
        
    Returns:
        Processed value or None if invalid
    """
    # Handle None values
    if value is None:
        return None
    
    # Handle string representations of lists (from JSON serialization)
    if isinstance(value, str):
        # Try to parse JSON strings back to Python objects
        if value.startswith('[') and value.endswith(']'):
            try:
                parsed_value = json.loads(value)
                print(f"    Parsed JSON string '{value}' → {parsed_value}")
                return parsed_value
            except json.JSONDecodeError:
                print(f"    Failed to parse JSON string: {value}")
                return value
        # Try to parse tuples represented as strings
        elif value.startswith('(') and value.endswith(')'):
            try:
                # Convert string tuple representation to actual tuple
                parsed_value = eval(value)  # Use with caution, only for trusted data
                if isinstance(parsed_value, tuple):
                    return parsed_value
            except:
                pass
    
    # Handle specific parameter types that need special processing
    tuple_params = {'n_pool_kernel_size', 'n_freq_downsample'}
    if param_key in tuple_params and isinstance(value, list):
        return tuple(value)
    
    # Handle boolean strings
    if isinstance(value, str):
        if value.lower() == 'true':
            return True
        elif value.lower() == 'false':
            return False
    
    # Return value as-is if no special processing needed
    return value


def _instantiate_models(model_configs: Dict[str, Dict]) -> List[Any]:
    """
    Instantiate model instances from configurations.
    
    Args:
        model_configs: Dictionary of model configurations
        
    Returns:
        List of instantiated model objects
    """
    models = []
    
    print(f"\n=== Instantiating {len(model_configs)} models ===")
    
    for model_name, config in model_configs.items():
        try:
            # Extract the model class
            ModelClass = config.pop('model_class')
            
            # Validate required parameters
            if 'h' not in config:
                print(f"ERROR: Missing required parameter 'h' for {model_name}")
                continue
            
            print(f"\n--- Instantiating {model_name} ---")
            print(f"Model class: {ModelClass.__name__}")
            print(f"Configuration ({len(config)} params):")
            
            # Sort and display configuration for better readability
            for key in sorted(config.keys()):
                value = config[key]
                print(f"  {key}: {value} (type: {type(value).__name__})")
            
            # Create model instance with remaining configuration
            model_instance = ModelClass(**config)
            models.append(model_instance)
            print(f"✓ Successfully created {model_name} model")
            
        except Exception as e:
            print(f"✗ ERROR instantiating {model_name}: {e}")
            print(f"Configuration that caused error:")
            for k, v in config.items():
                print(f"    {k}: {v} (type: {type(v).__name__})")
            # Continue with other models instead of failing completely
            continue
    
    if not models:
        print("\n⚠️  WARNING: No models were successfully instantiated!")
        print("Check your hyperparameter configurations and model definitions.")
    else:
        model_names = [type(m).__name__ for m in models]
        print(f"\n✓ Successfully instantiated {len(models)} models: {model_names}")
    
    return models


if __name__ == "__main__":
    # Test the model loading with hyperparameters
    print("Testing neural model loading...")
    models = get_neural_models(h=7, hyperparameters_json_path=BEST_HYPERPARAMETERS_CSV)
    print(f"Loaded {len(models)} models: {[type(m).__name__ for m in models]}")