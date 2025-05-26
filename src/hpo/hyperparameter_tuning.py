"""
Hyperparameter tuning module for neural forecasting models.
"""
import os
import pandas as pd
from neuralforecast import NeuralForecast
from utilsforecast.plotting import plot_series
from neuralforecast.losses.pytorch import MAE
from config.base import *
from config.base import CV_N_WINDOWS, LEVELS
from src.models.model_registry import ModelRegistry
import json


def run_hyperparameter_optimization(train_df, horizon=None, loss_fn=None, num_samples=None):
    """Run hyperparameter optimization using AutoModels."""
    if horizon is None:
        horizon = HORIZON
    if loss_fn is None:
        loss_fn = MAE()
    if num_samples is None:
        num_samples = NUM_SAMPLES_PER_MODEL
    
    print(f"\nStarting Hyperparameter Optimization with AutoModels...")
    print(f"Samples per model: {num_samples}")
    
    # Get auto models for HPO using the model registry
    automodels = ModelRegistry.get_auto_models(
        horizon=horizon,
        loss_fn=loss_fn,
        num_samples=num_samples
    )
    
    # Create NeuralForecast instance for HPO
    nf_hpo = NeuralForecast(
        models=automodels,
        freq=FREQUENCY,
        local_scaler_type=LOCAL_SCALER_TYPE
    )

    # nf_hpo.fit(train_df)

    # fcst_df = nf_hpo.predict()
    # fcst_df.columns = fcst_df.columns.str.replace('-median', '')

    # fig = plot_series(
    #     train_df,
    #     forecasts_df=fcst_df,
    #     levels=[80,90],
    # )
    # fig.savefig('results/hpo/hpo_forecasts.png')
    
    # Fit the models (performs HPO)
    cv_df = nf_hpo.cross_validation(train_df, n_windows=CV_N_WINDOWS)
    
    return cv_df, nf_hpo


def extract_best_configurations(nf_hpo):
    """Extract best configurations from HPO results."""
    all_best_configs = []
    
    for model in nf_hpo.models:
        # Check if the model is an Auto model and has results
        if hasattr(model, 'results') and model.results is not None:
            model_name = model.__class__.__name__
            print(f"Processing results for {model_name}...")
            
            try:
                # Get the DataFrame of all trials for this model
                results_df = model.results.get_dataframe()
                
                if not results_df.empty:
                    # Find the row with the lowest 'valid_loss'
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
                    
                    # Append to the list
                    all_best_configs.append(best_params)
                    print(f"Best config for {model_name}: {best_params}")
                else:
                    print(f"No tuning results found for {model_name}.")
                    
            except Exception as e:
                print(f"Error processing results for {model_name}: {str(e)}")
                print(f"Error type: {type(e).__name__}")
                if 'results_df' in locals():
                    print(f"DataFrame columns: {list(results_df.columns)}")
                    print(f"DataFrame index type: {type(results_df.index)}")
                    print(f"DataFrame shape: {results_df.shape}")
                continue
                
        else:
            print(f"Model {model.__class__.__name__} is not an Auto model or has no results.")
    
    return all_best_configs


def save_best_configurations(all_best_configs, output_dir=None):
    """Save best configurations to CSV file."""
    if output_dir is None:
        output_dir = TUNING_RESULTS_DIR
    
    if all_best_configs:
        try:
            best_configs_df = pd.DataFrame(all_best_configs)
        except Exception as e:
            print(f"Error creating DataFrame from configs: {str(e)}")
            print(f"Config data: {all_best_configs}")
            return None
        
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        csv_filename = os.path.join(output_dir, 'best_hyperparameters.csv')
        
        # Save to CSV
        best_configs_df.to_csv(csv_filename, index=False)
        print(f"\nBest hyperparameters saved to {csv_filename}")
        print("\nContent of best_hyperparameters.csv:")
        print(best_configs_df)
        
        return csv_filename
    else:
        print("No best configurations were found for any model.")
        return None


def run_complete_hpo_pipeline(train_df, horizon=None, loss_fn=None, num_samples=None):
    """Run the complete hyperparameter optimization pipeline."""
    try:
        print("Step 1: Running hyperparameter optimization...")
        # Run HPO
        cv_df, nf_hpo = run_hyperparameter_optimization(
            train_df, 
            horizon=horizon, 
            loss_fn=loss_fn, 
            num_samples=num_samples
        )
        print("Step 1 completed successfully.")
        
        print("Step 2: Extracting best configurations...")
        # Extract best configurations
        all_best_configs = extract_best_configurations(nf_hpo)
        print(f"Step 2 completed. Found {len(all_best_configs)} configurations.")
        
        print("Step 3: Saving best configurations...")
        # Save best configurations
        csv_filename = save_best_configurations(all_best_configs)
        print("Step 3 completed successfully.")
        
        return csv_filename
        
    except Exception as e:
        print(f"Error in run_complete_hpo_pipeline: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    from src.dataset.data_preparation import prepare_data
    
    # Prepare data
    train_df, test_df, hist_exog_list = prepare_data(
            horizon=HORIZON,
            test_length_multiplier=TEST_LENGTH_MULTIPLIER
        )
    
    # Run HPO
    csv_file = run_complete_hpo_pipeline(train_df=train_df,
                                        horizon=HORIZON,
                                        num_samples=NUM_SAMPLES_PER_MODEL)
    print(f"HPO completed. Results saved to: {csv_file}")