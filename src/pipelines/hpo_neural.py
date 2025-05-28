"""
Hyperparameter tuning module for neural forecasting models.

This module provides a comprehensive hyperparameter optimization pipeline
for neural forecasting models using Ray Tune and NeuralForecast.
"""

import os
import json
import traceback
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from neuralforecast import NeuralForecast
from utilsforecast.plotting import plot_series

from config.base import *
from src.models.model_registry import ModelRegistry


class HPOResultsProcessor:
    """Handles processing and extraction of HPO results."""
    
    @staticmethod
    def process_single_model_results(model, model_name: str) -> Optional[Dict[str, Any]]:
        """Process results for a single model."""
        try:
            results_df = model.results.get_dataframe()
            
            if results_df.empty:
                print(f"No tuning results found for {model_name}.")
                return None
            
            # Find the row with the lowest 'loss'
            # Reset index to avoid issues with unhashable types
            results_df_reset = results_df.reset_index(drop=True)
            best_idx = results_df_reset['loss'].idxmin()
            best_run = results_df_reset.loc[best_idx]
            
            # Extract the 'config/' columns to get the hyperparameters
            best_params = {}
            for col in results_df_reset.columns:
                if col.startswith('config/'):
                    val = best_run[col]
                    if isinstance(val, list):
                        val = json.dumps(val)
                    best_params[col.replace('config/', '')] = val
            
            # Add model name and best loss to the dictionary
            best_params['model_name'] = model_name
            best_params['best_valid_loss'] = best_run['loss']
            best_params['training_iteration'] = best_run['training_iteration']
            
            print(f"Best config for {model_name}: {best_params}")
            return best_params
            
        except Exception as e:
            print(f"Error processing results for {model_name}: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            if 'results_df' in locals():
                print(f"DataFrame columns: {list(results_df.columns)}")
                print(f"DataFrame shape: {results_df.shape}")
                print(f"DataFrame index type: {type(results_df.index)}")
            return None


class HPOConfigSerializer:
    """Handles serialization and deserialization of HPO configurations."""
    
    @staticmethod
    def make_json_serializable(value: Any) -> Any:
        """Convert a value to JSON-serializable format."""
        # Handle numpy types
        if hasattr(value, 'item'):
            return value.item()
        
        # Handle PyTorch loss objects
        if hasattr(value, '__class__') and hasattr(value.__class__, '__name__'):
            if 'loss' in str(type(value)).lower() and hasattr(value, '__module__'):
                return value.__class__.__name__
        
        # Handle tuples (convert to lists for JSON)
        if isinstance(value, tuple):
            return list(value)
        
        # Handle other non-serializable objects
        if not isinstance(value, (str, int, float, bool, list, dict, type(None))):
            try:
                return str(value)
            except Exception:
                return None
        
        return value
    
    @staticmethod
    def serialize_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize a single configuration dictionary."""
        serializable_config = {}
        for key, value in config.items():
            serialized_value = HPOConfigSerializer.make_json_serializable(value)
            if serialized_value is not None:
                serializable_config[key] = serialized_value
            else:
                print(f"Warning: Skipping non-serializable value for key '{key}': {type(value)}")
        return serializable_config
    
    @staticmethod
    def deserialize_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Deserialize a configuration dictionary for model use."""
        clean_config = config.copy()
        
        # Remove metadata fields
        metadata_fields = ['model_name', 'best_valid_loss', 'training_iteration', 'loss', 'valid_loss']
        for field in metadata_fields:
            clean_config.pop(field, None)
        
        # Convert lists back to tuples for specific parameters
        tuple_params = ['kernel_size', 'downsample']
        for key, value in clean_config.items():
            if isinstance(value, list) and any(param in key for param in tuple_params):
                clean_config[key] = tuple(value)
        
        return clean_config


class HPOConfigManager:
    """Manages saving and loading of HPO configurations."""
    
    @staticmethod
    def save_configurations(configs: List[Dict[str, Any]], filepath: str) -> bool:
        """Save configurations to JSON file."""
        try:
            output_dir = os.path.dirname(filepath)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            serializable_configs = [
                HPOConfigSerializer.serialize_config(config) 
                for config in configs
            ]
            
            with open(filepath, 'w') as f:
                json.dump(serializable_configs, f, indent=4)
            
            print(f"Best hyperparameters saved to {filepath}")
            return True
            
        except Exception as e:
            print(f"Error saving configurations to JSON {filepath}: {str(e)}")
            return False
    
    @staticmethod
    def load_configurations(filepath: str) -> Dict[str, Dict[str, Any]]:
        """Load configurations from JSON file."""
        if not os.path.exists(filepath):
            print(f"Warning: Hyperparameter JSON file not found at {filepath}. Model defaults will be used.")
            return {}
        
        try:
            with open(filepath, 'r') as f:
                configs_list = json.load(f)
            
            if not isinstance(configs_list, list):
                print(f"Warning: JSON file {filepath} is not a list of configs as expected.")
                return configs_list if isinstance(configs_list, dict) else {}
            
            configs_map = {}
            for config_item in configs_list:
                if 'model_name' not in config_item:
                    print(f"Warning: Found config item without 'model_name'. Skipping: {config_item}")
                    continue
                
                model_name = config_item['model_name']
                clean_config = HPOConfigSerializer.deserialize_config(config_item)
                configs_map[model_name] = clean_config
            
            print(f"Successfully loaded and parsed best hyperparameters from {filepath}")
            return configs_map
            
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {filepath}: {e}. Model defaults will be used.")
            return {}
        except Exception as e:
            print(f"Error loading hyperparameters from {filepath}: {e}. Model defaults will be used.")
            return {}


class HyperparameterTuner:
    """Main class for hyperparameter tuning operations."""
    
    def __init__(self, frequency: str = FREQUENCY, local_scaler_type: str = LOCAL_SCALER_TYPE):
        self.frequency = frequency
        self.local_scaler_type = local_scaler_type
        self.results_processor = HPOResultsProcessor()
        self.config_manager = HPOConfigManager()
    
    def run_optimization(self, train_df: pd.DataFrame, horizon: int, hist_exog_list: Optional[List[str]] = None, num_samples: int = NUM_SAMPLES_PER_MODEL) -> Tuple[pd.DataFrame, NeuralForecast]:
        """Run hyperparameter optimization using AutoModels."""
        print(f"\nStarting Hyperparameter Optimization with AutoModels...")
        print(f"Samples per model: {num_samples}")
        
        # Get auto models for HPO
        automodels = ModelRegistry.get_auto_models(
            horizon=horizon,
            num_samples=num_samples,
            hist_exog_list=hist_exog_list
        )
        
        # Create NeuralForecast instance
        nf_hpo = NeuralForecast(
            models=automodels,
            freq=self.frequency,
            local_scaler_type=self.local_scaler_type
        )
        
        # Perform fit
        # cv_df = nf_hpo.cross_validation(train_df, n_windows=CV_N_WINDOWS, verbose=False)
        cv_df = nf_hpo.fit(train_df, val_size=24)
        
        return cv_df, nf_hpo
    
    def extract_best_configurations(self, nf_hpo: NeuralForecast) -> List[Dict[str, Any]]:
        """Extract best configurations from HPO results."""
        all_best_configs = []
        
        for model in nf_hpo.models:
            if not (hasattr(model, 'results') and model.results is not None):
                print(f"Model {model.__class__.__name__} is not an Auto model or has no results.")
                continue
            
            model_name = model.__class__.__name__
            print(f"Processing results for {model_name}...")
            
            best_config = self.results_processor.process_single_model_results(model, model_name)
            if best_config:
                all_best_configs.append(best_config)
        
        return all_best_configs
    
    def save_best_configurations(self, configs: List[Dict[str, Any]], filepath: str) -> bool:
        """Save best configurations to file."""
        return self.config_manager.save_configurations(configs, filepath)
    
    def load_best_configurations(self, filepath: str) -> Dict[str, Dict[str, Any]]:
        """Load best configurations from file."""
        return self.config_manager.load_configurations(filepath)
    
    def run_complete_pipeline(self, train_df: pd.DataFrame, horizon: int, hist_exog_list: Optional[List[str]] = None, num_samples: int = NUM_SAMPLES_PER_MODEL, 
                            save_path: str = BEST_HYPERPARAMETERS_CSV) -> List[Dict[str, Any]]:
        """Run the complete hyperparameter optimization pipeline."""
        try:
            print("Step 1: Running hyperparameter optimization...")
            cv_df, nf_hpo = self.run_optimization(train_df, horizon, hist_exog_list, num_samples)
            print("Step 1 completed successfully.")
            
            print("Step 2: Extracting best configurations...")
            all_best_configs = self.extract_best_configurations(nf_hpo)
            print(f"Step 2 completed. Found {len(all_best_configs)} configurations.")
            
            print("Step 3: Saving best configurations...")
            success = self.save_best_configurations(all_best_configs, save_path)
            if success:
                print("Step 3 completed successfully.")
            else:
                print("Step 3 failed - configurations not saved.")
            
            return all_best_configs
            
        except Exception as e:
            print(f"Error in run_complete_hpo_pipeline: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            traceback.print_exc()
            raise


# Convenience functions for backward compatibility
def run_hyperparameter_optimization(train_df: pd.DataFrame, horizon: int = None, num_samples: int = None) -> Tuple[pd.DataFrame, NeuralForecast]:
    """Legacy function for backward compatibility."""
    tuner = HyperparameterTuner()
    return tuner.run_optimization(train_df, horizon, num_samples)


def extract_best_configurations(nf_hpo: NeuralForecast) -> List[Dict[str, Any]]:
    """Legacy function for backward compatibility."""
    tuner = HyperparameterTuner()
    return tuner.extract_best_configurations(nf_hpo)


def save_best_configurations(all_best_configs: List[Dict[str, Any]], filepath: str) -> bool:
    """Legacy function for backward compatibility."""
    return HPOConfigManager.save_configurations(all_best_configs, filepath)


def load_best_hyperparameters(json_filepath: str) -> Dict[str, Dict[str, Any]]:
    """Legacy function for backward compatibility."""
    return HPOConfigManager.load_configurations(json_filepath)


# def run_complete_hpo_pipeline(train_df: pd.DataFrame, horizon: int = None, num_samples: int = None) -> List[Dict[str, Any]]:
#     """Legacy function for backward compatibility."""
#     tuner = HyperparameterTuner()
#     return tuner.run_complete_pipeline(train_df, horizon, num_samples)


if __name__ == "__main__":
    from src.dataset.data_preparation import prepare_data
    
    # Prepare data
    train_df, test_df, hist_exog_list = prepare_data(
        horizon=HORIZON,
        test_length_multiplier=TEST_LENGTH_MULTIPLIER
    )
    
    # Create tuner instance
    tuner = HyperparameterTuner()
    
    # Run HPO pipeline
    all_best_configs = tuner.run_complete_pipeline(
        train_df=train_df,
        horizon=HORIZON,
        hist_exog_list=hist_exog_list,
        num_samples=NUM_SAMPLES_PER_MODEL
    )
    
    # Load and display best hyperparameters
    best_hyperparameters = tuner.load_best_configurations(BEST_HYPERPARAMETERS_CSV)
    print(f"Best hyperparameters: {best_hyperparameters}")