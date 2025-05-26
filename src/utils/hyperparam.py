import pandas as pd
import os
import json
from typing import List, Any, Optional, Dict


def save_best_configurations_to_json(all_best_configs: List[Dict[str, Any]], filepath: str):
    """
    Save the list of best hyperparameter configurations to a JSON file.

    Args:
        all_best_configs (list): A list of dictionaries, where each dictionary
                                 contains the best hyperparameters for a model.
        filepath (str): The full path to save the JSON file.
    """
    try:
        output_dir = os.path.dirname(filepath)
        if output_dir: # Ensure directory exists if filepath includes a path
            os.makedirs(output_dir, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(all_best_configs, f, indent=4)
        print(f"Best hyperparameters saved to {filepath}")
    except Exception as e:
        print(f"Error saving configurations to JSON {filepath}: {str(e)}")


def load_best_hyperparameters_from_json(json_filepath: str) -> Dict[str, Dict[str, Any]]:
    """
    Loads best hyperparameters from the JSON file.
    The JSON file is expected to be a list of dictionaries, where each dictionary
    represents a model's configuration and includes a 'model_name' key.

    Args:
        json_filepath: Path to the best_hyperparameters.json file.

    Returns:
        A dictionary where keys are model names and values are
        dictionaries of their best hyperparameters.
    """
    if not os.path.exists(json_filepath):
        print(f"Warning: Hyperparameter JSON file not found at {json_filepath}. Model defaults will be used.")
        return {}
    try:
        with open(json_filepath, 'r') as f:
            configs_list = json.load(f)
        
        best_configs_map = {}
        if isinstance(configs_list, list): # Expected format
            for config_item in configs_list:
                if 'model_name' in config_item:
                    model_name = config_item.pop('model_name') # Remove model_name to get only params
                    # Remove other metadata if present
                    config_item.pop('best_valid_loss', None)
                    config_item.pop('training_iteration', None)
                    best_configs_map[model_name] = config_item
                else:
                    print(f"Warning: Found a config item without 'model_name' in {json_filepath}. Skipping item: {config_item}")
        else:
            # If the JSON is already a map of model_name to config (less likely from extract_best_configurations)
            print(f"Warning: JSON file {json_filepath} is not a list of configs as expected. Assuming it's a direct map.")
            best_configs_map = configs_list

        print(f"Successfully loaded and parsed best hyperparameters from {json_filepath}")
        return best_configs_map
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {json_filepath}: {e}. Model defaults will be used.")
        return {}
    except Exception as e:
        print(f"Error loading hyperparameters from {json_filepath}: {e}. Model defaults will be used.")
        return {}
    

# --- Example HPO Extraction Function (Normally in a separate HPO script) ---
def extract_best_configurations_for_demo(nf_hpo_results_mock):
    """
    Simplified mock of extracting best configurations.
    In a real scenario, this function would interact with Ray Tune results.
    """
    all_best_configs = []
    for model_mock in nf_hpo_results_mock.models:
        if not hasattr(model_mock, 'results_data'): # Check if it's a mock with data
            print(f"Skipping mock model {model_mock.name if hasattr(model_mock, 'name') else 'Unknown'} as it has no results_data.")
            continue

        model_name = model_mock.name
        results_df = pd.DataFrame(model_mock.results_data)
        
        if not results_df.empty and 'loss' in results_df.columns:
            best_idx = results_df['loss'].idxmin()
            best_run = results_df.loc[best_idx]
            
            best_params = {}
            for col in results_df.columns:
                if col.startswith('config_'): # Assuming HPO params are prefixed this way
                    param_name = col.replace('config_', '')
                    val = best_run[col]
                    # For JSON, we want actual Python types, not JSON strings for lists
                    # If the source data already has lists/dicts, they are fine.
                    # If they were strings representing lists, they'd need json.loads here.
                    # For this demo, assume 'val' is already in the correct Python type.
                    best_params[param_name] = val
            
            best_params['model_name'] = model_name # This key is used by load_best_hyperparameters_from_json
            best_params['best_valid_loss'] = best_run['loss']
            best_params['training_iteration'] = best_run.get('training_iteration', 'N/A')
            all_best_configs.append(best_params)
        else:
            print(f"No valid results or 'loss' column for mock model {model_name}")
    return all_best_configs