# Directory Structure
```
config/
  algo.txt
  base.py
  search_algo.py
results/
  final/
    summary_report_20250528_074647.txt
src/
  dataset/
    data_preparation.py
  models/
    mlforecast/
      auto_models.py
      models.py
    neuralforecast/
      auto_cfg.py
      auto_models.py
      models.py
    statsforecast/
      models.py
    model_registry.py
  pipelines/
    cross_validation.py
    cv.py
    evaluate.py
    final_fit_predict.py
    hpo_ml.py
    hpo_neural.py
    test.py
    train.py
  utils/
    hyperparam.py
    other.py
.gitignore
.repomixignore
main.py
output.txt
READEVERYTIMEOPENREPO.md
README.md
repomix.txt
requirements.txt
```

# Files

## File: config/algo.txt
````
BOHB (Bayesian Optimization HyperBand): 
- bayesian optimization (efficiency) + hyperband (resource adaptiveness)
-> require careful evaluate config

HEBO (Heteroscedastic Evolutionary Bayesian Optimization)
- suit with noisy data
-> hard to config 

OptunaSearch 
- TPE (Tree-structured Parzen Estimator): a robust Bayesian optimization technique
- CMA-ES (Covariance Matrix Adaptation Evolution Strategy): excellent for more complex, non-convex, or ill-conditioned search spaces.
-> start with TPE

Suggestion:
1. Start with Random Search or HyperOptSearch: Run a small number of trials (e.g., 20-30) to get a feel for the search space and establish a baseline.
2. Move to BOHB or Optuna (with TPE/HyperBand Pruner): These are often my first choices for serious tuning due to their balance of performance and efficiency. HEBO is a strong alternative if you anticipate a noisy or complex landscape.
````

## File: config/search_algo.py
````python
# For initial baseline establishment (as recommended in algo.txt)
BASELINE = ['random', 'hyperopt']
⋮----
# For efficient optimization (BOHB and Optuna with TPE)
EFFICIENT = ['bohb', 'optuna']
⋮----
# For noisy time series data
NOISY_DATA = ['hebo', 'random']
⋮----
def get_search_algorithm_class(algorithm: str) -> Any
````

## File: results/final/summary_report_20250528_074647.txt
````
============================================================
🎯 BITCOIN FORECASTING PIPELINE SUMMARY
============================================================
Execution Time: 2025-05-28 07:46:47
HPO: Skipped

📊 DATA INFORMATION:
  • Total samples: 2,922
  • Training samples: 2,887
  • Test samples: 35
  • Features: 77
  • Horizon: 7 days

🏆 MODEL COMPARISON:
  • Execution time: 275.7 seconds
  • CV models evaluated: 2
  • Final models evaluated: 2
  • Best model: AutoARIMA
  • Best test MAE: 5821.6465

🎯 RECOMMENDATIONS:
  • Review top-performing models for deployment
  • Consider ensemble methods for improved performance
  • Monitor model performance over time
  • Retrain periodically with new data

============================================================
````

## File: src/models/mlforecast/models.py
````python
lgb_params = {
⋮----
models={
````

## File: src/pipelines/cross_validation.py
````python
# src/pipelines/cross_validation.py
"""
Cross-validation evaluation framework for model performance comparison.
"""
⋮----
class CrossValidationEvaluator
⋮----
"""Cross-validation based model performance evaluation."""
⋮----
def __init__(self, results_dir: str = FINAL_DIR)
⋮----
def evaluate_neural_models_cv(self, models, train_df, horizon: int) -> List[Dict]
⋮----
"""Evaluate neural models using cross-validation."""
results = []
⋮----
model_name = model.__class__.__name__
⋮----
start_time = time.time()
⋮----
# Create NeuralForecast instance
nf = NeuralForecast(models=[model], freq='D')
⋮----
# Perform cross-validation
cv_results = nf.cross_validation(
⋮----
# Calculate metrics from CV results
metrics = self._calculate_metrics(cv_results, [model_name])
evaluation_method = "cross_validation"
⋮----
# Training time
training_time = time.time() - start_time
⋮----
# Check if metrics were calculated successfully
⋮----
result = {
⋮----
error_msg = str(e)
⋮----
successful_count = sum(1 for r in results if r.get('status') == 'success')
⋮----
def evaluate_statistical_models_cv(self, models, train_df, horizon: int) -> List[Dict]
⋮----
"""Evaluate statistical models using cross-validation."""
⋮----
# Prepare data for StatsForecast
df_train = train_df[['unique_id', 'ds', 'y']].copy()
⋮----
# Create StatsForecast instance
sf = StatsForecast(models=[model], freq='D', verbose=True)
⋮----
# Cross-validation for metrics
cv_results = sf.cross_validation(
⋮----
def _calculate_metrics(self, cv_results: pd.DataFrame, model_names: List[str]) -> Dict
⋮----
"""Calculate comprehensive metrics for cross-validation results."""
metrics = {}
⋮----
# Check for valid data
y_true = cv_results['y'].values
y_pred = cv_results[model_name].values
⋮----
# Remove NaN values
mask = ~(np.isnan(y_true) | np.isnan(y_pred))
y_true_clean = y_true[mask]
y_pred_clean = y_pred[mask]
⋮----
# Basic metrics using utilsforecast
⋮----
model_metrics = evaluate(
⋮----
# Extract metric values
mae_val = model_metrics[model_metrics['metric'] == 'mae'][model_name].iloc[0]
mse_val = model_metrics[model_metrics['metric'] == 'mse'][model_name].iloc[0]
rmse_val = model_metrics[model_metrics['metric'] == 'rmse'][model_name].iloc[0]
mape_val = model_metrics[model_metrics['metric'] == 'mape'][model_name].iloc[0]
smape_val = model_metrics[model_metrics['metric'] == 'smape'][model_name].iloc[0]
⋮----
# Fallback to manual calculation
mae_val = np.mean(np.abs(y_true_clean - y_pred_clean))
mse_val = np.mean((y_true_clean - y_pred_clean) ** 2)
rmse_val = np.sqrt(mse_val)
mape_val = np.mean(np.abs((y_true_clean - y_pred_clean) / y_true_clean)) * 100
smape_val = np.mean(2 * np.abs(y_true_clean - y_pred_clean) / (np.abs(y_true_clean) + np.abs(y_pred_clean))) * 100
⋮----
# Additional custom metrics
directional_accuracy = self._calculate_directional_accuracy(y_true_clean, y_pred_clean)
coverage = self._calculate_coverage(cv_results, model_name)
⋮----
def _calculate_directional_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float
⋮----
"""Calculate directional accuracy (correct prediction of up/down movement)."""
⋮----
true_direction = np.diff(y_true) > 0
pred_direction = np.diff(y_pred) > 0
⋮----
def _calculate_coverage(self, cv_results: pd.DataFrame, model_name: str) -> Dict
⋮----
"""Calculate prediction interval coverage."""
coverage = {}
⋮----
# Check for prediction intervals
⋮----
lo_col = f"{model_name}-lo-{level}"
hi_col = f"{model_name}-hi-{level}"
⋮----
y_true = cv_results['y']
y_lo = cv_results[lo_col]
y_hi = cv_results[hi_col]
⋮----
# Coverage = proportion of true values within prediction intervals
within_interval = (y_true >= y_lo) & (y_true <= y_hi)
⋮----
def compare_models_cv(self, neural_models, stat_models, train_df, horizon: int) -> pd.DataFrame
⋮----
"""Compare all models using cross-validation and return ranked results."""
⋮----
all_results = []
⋮----
# Evaluate neural models
⋮----
neural_results = self.evaluate_neural_models_cv(neural_models, train_df, horizon)
⋮----
# Evaluate statistical models
⋮----
stat_results = self.evaluate_statistical_models_cv(stat_models, train_df, horizon)
⋮----
# Create results DataFrame
⋮----
results_df = pd.DataFrame(all_results)
⋮----
# Filter out failed models
⋮----
successful_results = results_df[results_df['status'] == 'success'].copy()
⋮----
successful_results = results_df[results_df['error'].isna()].copy()
⋮----
# Assume all are successful if no status/error columns
successful_results = results_df.copy()
⋮----
# Rank models by MAE (lower is better)
⋮----
# Combined rank (simple average)
⋮----
# Sort by combined rank
successful_results = successful_results.sort_values('combined_rank')
⋮----
successful_results = successful_results.sort_values('mae')
⋮----
# Save results
⋮----
def save_cv_results(self, all_results: pd.DataFrame, successful_results: pd.DataFrame)
⋮----
"""Save cross-validation results."""
⋮----
# Use simple timestamp without timezone
timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
⋮----
# Save all results
⋮----
# Save successful results
⋮----
# Save summary
summary = {
⋮----
def get_top_models(self, results_df: pd.DataFrame, top_n: int = 5) -> List[str]
⋮----
"""Get top N model names based on CV performance."""
⋮----
actual_top_n = min(top_n, len(results_df))
top_models = results_df.head(actual_top_n)['model_name'].tolist()
````

## File: src/pipelines/cv.py
````python
"""
Cross-validation module for neural forecasting models.
"""
⋮----
def run_cross_validation(model_instance, train_df, n_windows=None, step_size=None)
⋮----
"""Run cross-validation with the best model configuration."""
⋮----
n_windows = CV_N_WINDOWS
⋮----
step_size = CV_STEP_SIZE
⋮----
# Create NeuralForecast object for Cross-Validation
nf_cv = NeuralForecast(
⋮----
# Run cross-validation
# test_size in cross_validation refers to the validation horizon for each fold (should be h)
# n_windows is the number of CV folds. step_size controls overlap.
cv_results_df = nf_cv.cross_validation(
⋮----
def evaluate_cv_results(cv_results_df)
⋮----
"""Evaluate cross-validation results and calculate metrics."""
# Convert pandas DataFrame to Polars DataFrame
df_pl = pl.from_pandas(cv_results_df)
⋮----
# Define columns to exclude
exclude_cols = ['unique_id', 'ds', 'cutoff', 'y']
⋮----
# Get the model columns dynamically
model_cols = [col for col in df_pl.columns if col not in exclude_cols]
⋮----
# Calculate metrics for each model
results = calculate_evaluation_metrics(df_pl, model_cols, exclude_cols)
⋮----
def run_complete_cv_pipeline(train_df, model_name='AutoNHITS', csv_path=None)
⋮----
"""Run the complete cross-validation pipeline."""
# Load best model configuration
⋮----
# Evaluate results
cv_metrics = evaluate_cv_results(cv_results_df)
⋮----
# Prepare data
⋮----
# Run CV pipeline
````

## File: src/pipelines/evaluate.py
````python
"""
Prediction and evaluation module for neural forecasting models.
"""
⋮----
def make_predictions(nf_final_train, test_df)
⋮----
"""Make predictions on the final holdout test set."""
⋮----
# NeuralForecast's predict method can take test_df directly.
# It will use the historical part of each series in test_df
# to generate the initial input window, and then predict 'h' steps.
#
# The predict() method forecasts h steps from the last timestamp in the training data for each unique_id.
# We need to ensure these forecasted ds values match our test_df.
⋮----
predictions_on_test = nf_final_train.predict(df=test_df)
⋮----
def evaluate_predictions(test_df, predictions_on_test, model_name='NHITS')
⋮----
"""Evaluate predictions against actual values."""
# For this example, assuming test_length was set up to align with forecasting h steps.
# If predict() output doesn't perfectly align or you need more control, consider predict(futr_df=...).
# Let's merge based on 'unique_id' and 'ds'.
⋮----
# final_evaluation_df = pd.merge(
#     test_df,
#     predictions_on_test,
#     on=['unique_id', 'ds'],
#     how='left'  # Use left to keep all test points; predictions might be shorter if h < test_length
# )
# final_evaluation_df.dropna(inplace=True)  # If some predictions couldn't be made or aligned.
⋮----
# print(f"Final evaluation dataframe columns: {final_evaluation_df.columns.tolist()}")
# print(f"Final evaluation dataframe shape: {final_evaluation_df.shape}")
⋮----
# if final_evaluation_df.empty:
#     print("Warning: No aligned predictions found for evaluation.")
#     return None
⋮----
# Calculate evaluation metrics
⋮----
# test_actuals = final_evaluation_df['y']
# test_preds = final_evaluation_df[model_name]
⋮----
# final_mae = mae(test_actuals, test_preds)
# final_rmse = rmse(test_actuals, test_preds)
⋮----
# print(f"\nFinal Evaluation on Holdout Test Set for {model_name}:")
# print(f"  Test MAE: {final_mae:.4f}")
# print(f"  Test RMSE: {final_rmse:.4f}")
⋮----
# return {
#     'model_name': model_name,
#     'test_mae': final_mae,
#     'test_rmse': final_rmse,
#     'evaluation_df': final_evaluation_df
# }
⋮----
evaluation_df = evaluate(predictions_on_test.drop(columns='cutoff'), metrics=[mse, mae, rmse])
⋮----
def run_prediction_evaluation(nf_final_train, test_df, model_name='NHITS')
⋮----
"""Run the complete prediction and evaluation pipeline."""
# Make predictions
predictions = make_predictions(nf_final_train, test_df)
⋮----
# Evaluate predictions
evaluation_results = evaluate_predictions(test_df, predictions, model_name)
⋮----
# Prepare data
⋮----
# Create and train final model
⋮----
# Run prediction and evaluation
⋮----
# print(f"Test MAE: {eval_results['test_mae']:.4f}")
# print(f"Test RMSE: {eval_results['test_rmse']:.4f}")
````

## File: src/pipelines/hpo_ml.py
````python
# difference helps remove seasonality
target_transforms =[
⋮----
@njit
def rolling_mean_48(x)
⋮----
fcst = MLForecast(
⋮----
freq=1,  # our series have integer timestamps, so we'll just add 1 in every timestep
⋮----
prep = fcst.preprocess(df)
````

## File: src/pipelines/hpo_neural.py
````python
"""
Hyperparameter tuning module for neural forecasting models.

This module provides a comprehensive hyperparameter optimization pipeline
for neural forecasting models using Ray Tune and NeuralForecast.
"""
⋮----
class HPOResultsProcessor
⋮----
"""Handles processing and extraction of HPO results."""
⋮----
@staticmethod
    def process_single_model_results(model, model_name: str) -> Optional[Dict[str, Any]]
⋮----
"""Process results for a single model."""
⋮----
results_df = model.results.get_dataframe()
⋮----
# Find the row with the lowest 'loss'
# Reset index to avoid issues with unhashable types
results_df_reset = results_df.reset_index(drop=True)
best_idx = results_df_reset['loss'].idxmin()
best_run = results_df_reset.loc[best_idx]
⋮----
# Extract the 'config/' columns to get the hyperparameters
best_params = {}
⋮----
val = best_run[col]
⋮----
val = json.dumps(val)
⋮----
# Add model name and best loss to the dictionary
⋮----
class HPOConfigSerializer
⋮----
"""Handles serialization and deserialization of HPO configurations."""
⋮----
@staticmethod
    def make_json_serializable(value: Any) -> Any
⋮----
"""Convert a value to JSON-serializable format."""
# Handle numpy types
⋮----
# Handle PyTorch loss objects
⋮----
# Handle tuples (convert to lists for JSON)
⋮----
# Handle other non-serializable objects
⋮----
@staticmethod
    def serialize_config(config: Dict[str, Any]) -> Dict[str, Any]
⋮----
"""Serialize a single configuration dictionary."""
serializable_config = {}
⋮----
serialized_value = HPOConfigSerializer.make_json_serializable(value)
⋮----
@staticmethod
    def deserialize_config(config: Dict[str, Any]) -> Dict[str, Any]
⋮----
"""Deserialize a configuration dictionary for model use."""
clean_config = config.copy()
⋮----
# Remove metadata fields
metadata_fields = ['model_name', 'best_valid_loss', 'training_iteration', 'loss', 'valid_loss']
⋮----
# Convert lists back to tuples for specific parameters
tuple_params = ['kernel_size', 'downsample']
⋮----
class HPOConfigManager
⋮----
"""Manages saving and loading of HPO configurations."""
⋮----
@staticmethod
    def save_configurations(configs: List[Dict[str, Any]], filepath: str) -> bool
⋮----
"""Save configurations to JSON file."""
⋮----
output_dir = os.path.dirname(filepath)
⋮----
serializable_configs = [
⋮----
@staticmethod
    def load_configurations(filepath: str) -> Dict[str, Dict[str, Any]]
⋮----
"""Load configurations from JSON file."""
⋮----
configs_list = json.load(f)
⋮----
configs_map = {}
⋮----
model_name = config_item['model_name']
clean_config = HPOConfigSerializer.deserialize_config(config_item)
⋮----
class HyperparameterTuner
⋮----
"""Main class for hyperparameter tuning operations."""
⋮----
def __init__(self, frequency: str = FREQUENCY, local_scaler_type: str = LOCAL_SCALER_TYPE)
⋮----
def run_optimization(self, train_df: pd.DataFrame, horizon: int, hist_exog_list: Optional[List[str]] = None, num_samples: int = NUM_SAMPLES_PER_MODEL) -> Tuple[pd.DataFrame, NeuralForecast]
⋮----
"""Run hyperparameter optimization using AutoModels."""
⋮----
# Get auto models for HPO
automodels = ModelRegistry.get_auto_models(
⋮----
# Create NeuralForecast instance
nf_hpo = NeuralForecast(
⋮----
# Perform fit
# cv_df = nf_hpo.cross_validation(train_df, n_windows=CV_N_WINDOWS, verbose=False)
cv_df = nf_hpo.fit(train_df, val_size=24)
⋮----
def extract_best_configurations(self, nf_hpo: NeuralForecast) -> List[Dict[str, Any]]
⋮----
"""Extract best configurations from HPO results."""
all_best_configs = []
⋮----
model_name = model.__class__.__name__
⋮----
best_config = self.results_processor.process_single_model_results(model, model_name)
⋮----
def save_best_configurations(self, configs: List[Dict[str, Any]], filepath: str) -> bool
⋮----
"""Save best configurations to file."""
⋮----
def load_best_configurations(self, filepath: str) -> Dict[str, Dict[str, Any]]
⋮----
"""Load best configurations from file."""
⋮----
"""Run the complete hyperparameter optimization pipeline."""
⋮----
all_best_configs = self.extract_best_configurations(nf_hpo)
⋮----
success = self.save_best_configurations(all_best_configs, save_path)
⋮----
# Convenience functions for backward compatibility
def run_hyperparameter_optimization(train_df: pd.DataFrame, horizon: int = None, num_samples: int = None) -> Tuple[pd.DataFrame, NeuralForecast]
⋮----
"""Legacy function for backward compatibility."""
tuner = HyperparameterTuner()
⋮----
def extract_best_configurations(nf_hpo: NeuralForecast) -> List[Dict[str, Any]]
⋮----
def save_best_configurations(all_best_configs: List[Dict[str, Any]], filepath: str) -> bool
⋮----
def load_best_hyperparameters(json_filepath: str) -> Dict[str, Dict[str, Any]]
⋮----
# def run_complete_hpo_pipeline(train_df: pd.DataFrame, horizon: int = None, num_samples: int = None) -> List[Dict[str, Any]]:
#     """Legacy function for backward compatibility."""
#     tuner = HyperparameterTuner()
#     return tuner.run_complete_pipeline(train_df, horizon, num_samples)
⋮----
# Prepare data
⋮----
# Create tuner instance
⋮----
# Run HPO pipeline
all_best_configs = tuner.run_complete_pipeline(
⋮----
# Load and display best hyperparameters
best_hyperparameters = tuner.load_best_configurations(BEST_HYPERPARAMETERS_CSV)
````

## File: src/pipelines/test.py
````python
# Get data
⋮----
# Get models
models = get_auto_statsmodels(HORIZON)
⋮----
# Instantiate StatsForecast
sf = StatsForecast(
⋮----
# forecasts_df = sf.forecast(df=train_df[['unique_id', 'ds', 'y']], X_df=test_df[['unique_id', 'ds']], h=HORIZON * TEST_LENGTH_MULTIPLIER)
forecasts_df = sf.forecast(df=train_df[['unique_id', 'ds', 'y'] + ['btc_sma_5', 'btc_trading_volume', 'Gold_Price']], X_df=test_df[['unique_id', 'ds'] + ['btc_sma_5', 'btc_trading_volume', 'Gold_Price']], h=HORIZON * TEST_LENGTH_MULTIPLIER)
⋮----
fig = sf.plot(test_df, forecasts_df, models=["AutoARIMA", "AutoETS", "CES", "Naive"])
````

## File: src/pipelines/train.py
````python
"""
Model training module for neural forecasting models.
"""
⋮----
def train_final_model(model_instance, train_df, val_size=0)
⋮----
"""Train the final model on the entire development set."""
⋮----
# Create NeuralForecast instance for final training
nf_final_train = NeuralForecast(
⋮----
# Train on the full development set
# val_size=0 ensures no further splitting here
⋮----
def create_and_train_final_model(train_df, model_name='AutoNHITS', csv_path=None)
⋮----
"""Create model from best config and train it on development data."""
# Load best model configuration
⋮----
# Train the final model
nf_final_train = train_final_model(model_instance, train_df)
⋮----
# Prepare data
⋮----
# Create and train final model
````

## File: .repomixignore
````
# Add patterns to ignore here, one per line
# Example:
# *.log
# tmp/
````

## File: output.txt
````
============================================================
🚀 BITCOIN FORECASTING PIPELINE
============================================================
Timestamp: 2025-05-28 07:46:47
Skip HPO: True
============================================================
2025-05-28 00:46:48,422 INFO worker.py:1888 -- Started a local Ray instance.
Pipeline execution started at: 2025-05-28 07:46:49 (Ho Chi Minh City Time)

📊 STEP 1: DATA PREPARATION
----------------------------------------
Loading and preparing data...
Forecast horizon (h) set to: 7 days

Total data shape: (2922, 80)
Train set shape: (2887, 80)
Test set shape: (35, 80)
  Train set covers: 2017-01-01 00:00:00 to 2024-11-26 00:00:00
  Test set covers: 2024-11-27 00:00:00 to 2024-12-31 00:00:00
✅ Data prepared successfully
   • Training samples: 2,887
   • Test samples: 35
   • Features: 77 exogenous variables
   • Forecast horizon: 7 days

⏭️  STEP 2: HYPERPARAMETER OPTIMIZATION (SKIPPED)
----------------------------------------
Using default configurations...

🏆 STEP 3: MODEL COMPARISON
----------------------------------------
Using default neural model configurations...
Getting neural models with horizon=7
Exogenous features: 77
Hyperparameters path: None
Successfully loaded and parsed best hyperparameters from results/best_hyperparameters.json
Loaded hyperparameters for 1 models
Available HPO configurations: ['AutoNHITS']

=== Applying HPO parameters for NHITS (found as 'AutoNHITS') ===
Original HPO params: {'input_size': 42, 'learning_rate': 0.0001, 'scaler_type': 'standard', 'max_steps': 100, 'batch_size': 64, 'windows_batch_size': 256, 'val_check_steps': 50, 'random_seed': 3, 'n_pool_kernel_size': (2, 2, 2), 'n_freq_downsample': (168, 24, 1), 'h': 7}
  ✓ input_size: 42 → 42 (type: <class 'int'>)
  ✓ learning_rate: 0.0001 → 0.0001 (type: <class 'float'>)
  ✓ scaler_type: standard → standard (type: <class 'str'>)
  ✓ max_steps: 100 → 100 (type: <class 'int'>)
  ✓ batch_size: 64 → 64 (type: <class 'int'>)
  ✓ windows_batch_size: 256 → 256 (type: <class 'int'>)
  ✓ val_check_steps: 50 → 50 (type: <class 'int'>)
  ✓ random_seed: 3 → 3 (type: <class 'int'>)
  ✓ n_pool_kernel_size: (2, 2, 2) → (2, 2, 2) (type: <class 'tuple'>)
  ✓ n_freq_downsample: (168, 24, 1) → (168, 24, 1) (type: <class 'tuple'>)
  ✓ h: 7 → 7 (type: <class 'int'>)
  Applied 11 hyperparameters to NHITS

=== Instantiating 1 models ===

--- Instantiating NHITS ---
Model class: NHITS
Configuration (18 params):
  batch_size: 64 (type: int)
  dropout_prob_theta: 0.0 (type: float)
  h: 7 (type: int)
  hist_exog_list: ['btc_sma_5', 'btc_ema_5', 'btc_sma_14', 'btc_ema_14', 'btc_sma_21', 'btc_ema_21', 'btc_sma_50', 'btc_ema_50', 'btc_sma_14_50_diff', 'btc_ema_14_50_diff', 'btc_sma_14_50_ratio', 'btc_sma_14_slope', 'btc_ema_14_slope', 'btc_sma_21_slope', 'btc_ema_21_slope', 'btc_sma_50_slope', 'btc_ema_50_slope', 'btc_close_ema_21_dist', 'btc_close_ema_21_dist_norm', 'btc_rsi_14', 'btc_macd', 'btc_macd_signal', 'btc_macd_diff', 'btc_bb_high', 'btc_bb_low', 'btc_bb_mid', 'btc_bb_width', 'btc_atr_14', 'btc_volatility_index', 'btc_trading_volume', 'active_addresses_blockchain', 'hash_rate_blockchain', 'miner_revenue_blockchain', 'difficulty_blockchain', 'estimated_transaction_volume_usd_blockchain', 'PiCycle_cbbi', 'RUPL_cbbi', 'RHODL_cbbi', 'Puell_cbbi', '2YMA_cbbi', 'Trolololo_cbbi', 'MVRV_cbbi', 'ReserveRisk_cbbi', 'Woobull_cbbi', 'Confidence_cbbi', 'Fear Greed', 'positive_sentiment', 'negative_sentiment', 'bullish_sentiment', 'bearish_sentiment', 'risk_uncertainty_sentiment', 'problem_malicious_sentiment', 'active_trading_sentiment', 'long_term_investment_sentiment', 'market_narrative_sentiment', 'core_technology_sentiment', 'development_ecosystem_sentiment', 'news_events_sentiment', 'regulations_sentiment', 'community_social_sentiment', 'price_sentiment', 'volume_sentiment', 'marketcap_sentiment', 'Gold_Price', 'Gold_Share', 'Gold_Volatility', 'Oil_Crude_Price', 'Oil_Brent_Price', 'Oil_Volatility', 'DJI', 'GSPC', 'IXIC', 'NYFANG', 'CBOE_Volatility', 'EM_ETF', 'DXY', 'EURUSD'] (type: list)
  input_size: 42 (type: int)
  interpolation_mode: linear (type: str)
  learning_rate: 0.0001 (type: float)
  loss: MAE() (type: MAE)
  max_steps: 100 (type: int)
  mlp_units: [[512, 512], [512, 512], [512, 512]] (type: list)
  n_blocks: [1, 1, 1] (type: list)
  n_freq_downsample: (168, 24, 1) (type: tuple)
  n_pool_kernel_size: (2, 2, 2) (type: tuple)
  random_seed: 3 (type: int)
  scaler_type: standard (type: str)
  stack_types: ['identity', 'identity', 'identity'] (type: list)
  val_check_steps: 50 (type: int)
  windows_batch_size: 256 (type: int)
Seed set to 3
✓ Successfully created NHITS model

✓ Successfully instantiated 1 models: ['NHITS']
   • Train data: 2,887 samples
   • Test data: 35 samples
   • Neural models: 1
   • Statistical models: 1

--- Step 3a: Cross-Validation Model Selection ---
=== Starting Cross-Validation Model Comparison ===
Train data shape: (2887, 80)
Horizon: 7
CV windows: 5
CV step size: 7
Neural models: 1
Statistical models: 1

--- Cross-Validating Neural Models ---
Evaluating 1 neural models with cross-validation...
[1/1] Cross-validating NHITS...
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
HPU available: False, using: 0 HPUs
Epoch 99: 100%|██████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 59.58it/s, v_num=50, train_loss_step=0.492, train_loss_epoch=0.492]`Trainer.fit` stopped: `max_steps=100` reached.                                                                                                                             
Epoch 99: 100%|██████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 57.16it/s, v_num=50, train_loss_step=0.492, train_loss_epoch=0.492]
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
HPU available: False, using: 0 HPUs
Predicting DataLoader 0: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 207.02it/s]
  ✓ NHITS CV completed (MAE: 4862.8310)
Neural models CV completed: 1/1 successful

--- Cross-Validating Statistical Models ---
Evaluating 1 statistical models with cross-validation...
[1/1] Cross-validating AutoARIMA...
Cross Validation Time Series 1: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [03:54<00:00, 46.93s/it]
  ✓ AutoARIMA CV completed (MAE: 4420.7214)
Statistical models CV completed: 1/1 successful

Total CV results: 2
Successful CV results: 2
✓ Models ranked successfully by CV performance
All CV results saved: 2 entries
Ranked CV results saved: 2 entries
CV results saved to results/final
CV Summary: {'timestamp': '20250528_005046', 'evaluation_type': 'cross_validation', 'cv_windows': 5, 'cv_step_size': 7, 'total_models': 2, 'successful_models': 2, 'failed_models': 0, 'best_model': 'AutoARIMA', 'best_mae': 4420.721415771486}
✓ Cross-validation completed: 2 models evaluated

🏅 TOP CV MODELS (2):
------------------------------
1. AutoARIMA            MAE: 4420.7214 (statsforecast)
2. NHITS                MAE: 4862.8310 (neuralforecast)

--- Step 3b: Final Fit-Predict Evaluation ---
=== Starting Final Fit-Predict Model Evaluation ===
Train data shape: (2887, 80)
Test data shape: (35, 80)
Horizon: 7
Neural models: 1
Statistical models: 1
Plots will be saved to: results/plot

--- Fit-Predict Neural Models ---
Fitting and predicting 1 neural models...
  Train data columns: ['unique_id', 'ds', 'y', 'btc_sma_5', 'btc_ema_5', 'btc_sma_14', 'btc_ema_14', 'btc_sma_21', 'btc_ema_21', 'btc_sma_50', 'btc_ema_50', 'btc_sma_14_50_diff', 'btc_ema_14_50_diff', 'btc_sma_14_50_ratio', 'btc_sma_14_slope', 'btc_ema_14_slope', 'btc_sma_21_slope', 'btc_ema_21_slope', 'btc_sma_50_slope', 'btc_ema_50_slope', 'btc_close_ema_21_dist', 'btc_close_ema_21_dist_norm', 'btc_rsi_14', 'btc_macd', 'btc_macd_signal', 'btc_macd_diff', 'btc_bb_high', 'btc_bb_low', 'btc_bb_mid', 'btc_bb_width', 'btc_atr_14', 'btc_volatility_index', 'btc_trading_volume', 'active_addresses_blockchain', 'hash_rate_blockchain', 'miner_revenue_blockchain', 'difficulty_blockchain', 'estimated_transaction_volume_usd_blockchain', 'PiCycle_cbbi', 'RUPL_cbbi', 'RHODL_cbbi', 'Puell_cbbi', '2YMA_cbbi', 'Trolololo_cbbi', 'MVRV_cbbi', 'ReserveRisk_cbbi', 'Woobull_cbbi', 'Confidence_cbbi', 'Fear Greed', 'positive_sentiment', 'negative_sentiment', 'bullish_sentiment', 'bearish_sentiment', 'risk_uncertainty_sentiment', 'problem_malicious_sentiment', 'active_trading_sentiment', 'long_term_investment_sentiment', 'market_narrative_sentiment', 'core_technology_sentiment', 'development_ecosystem_sentiment', 'news_events_sentiment', 'regulations_sentiment', 'community_social_sentiment', 'price_sentiment', 'volume_sentiment', 'marketcap_sentiment', 'Gold_Price', 'Gold_Share', 'Gold_Volatility', 'Oil_Crude_Price', 'Oil_Brent_Price', 'Oil_Volatility', 'DJI', 'GSPC', 'IXIC', 'NYFANG', 'CBOE_Volatility', 'EM_ETF', 'DXY', 'EURUSD']
  Test data columns: ['unique_id', 'ds', 'y', 'btc_sma_5', 'btc_ema_5', 'btc_sma_14', 'btc_ema_14', 'btc_sma_21', 'btc_ema_21', 'btc_sma_50', 'btc_ema_50', 'btc_sma_14_50_diff', 'btc_ema_14_50_diff', 'btc_sma_14_50_ratio', 'btc_sma_14_slope', 'btc_ema_14_slope', 'btc_sma_21_slope', 'btc_ema_21_slope', 'btc_sma_50_slope', 'btc_ema_50_slope', 'btc_close_ema_21_dist', 'btc_close_ema_21_dist_norm', 'btc_rsi_14', 'btc_macd', 'btc_macd_signal', 'btc_macd_diff', 'btc_bb_high', 'btc_bb_low', 'btc_bb_mid', 'btc_bb_width', 'btc_atr_14', 'btc_volatility_index', 'btc_trading_volume', 'active_addresses_blockchain', 'hash_rate_blockchain', 'miner_revenue_blockchain', 'difficulty_blockchain', 'estimated_transaction_volume_usd_blockchain', 'PiCycle_cbbi', 'RUPL_cbbi', 'RHODL_cbbi', 'Puell_cbbi', '2YMA_cbbi', 'Trolololo_cbbi', 'MVRV_cbbi', 'ReserveRisk_cbbi', 'Woobull_cbbi', 'Confidence_cbbi', 'Fear Greed', 'positive_sentiment', 'negative_sentiment', 'bullish_sentiment', 'bearish_sentiment', 'risk_uncertainty_sentiment', 'problem_malicious_sentiment', 'active_trading_sentiment', 'long_term_investment_sentiment', 'market_narrative_sentiment', 'core_technology_sentiment', 'development_ecosystem_sentiment', 'news_events_sentiment', 'regulations_sentiment', 'community_social_sentiment', 'price_sentiment', 'volume_sentiment', 'marketcap_sentiment', 'Gold_Price', 'Gold_Share', 'Gold_Volatility', 'Oil_Crude_Price', 'Oil_Brent_Price', 'Oil_Volatility', 'DJI', 'GSPC', 'IXIC', 'NYFANG', 'CBOE_Volatility', 'EM_ETF', 'DXY', 'EURUSD']
  Train data shape: (2887, 80)
  Test data shape: (35, 80)
  Using rolling forecast (horizon=7 < test_length=35)
[1/1] Fit-Predict NHITS...
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
HPU available: False, using: 0 HPUs
Epoch 99: 100%|██████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 54.53it/s, v_num=52, train_loss_step=0.483, train_loss_epoch=0.483]`Trainer.fit` stopped: `max_steps=100` reached.                                                                                                                             
Epoch 99: 100%|██████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 51.73it/s, v_num=52, train_loss_step=0.483, train_loss_epoch=0.483]
    Performing rolling forecast for NHITS (horizon=7, test_length=35)
      Exogenous features: ['btc_sma_5', 'btc_ema_5', 'btc_sma_14', 'btc_ema_14', 'btc_sma_21', 'btc_ema_21', 'btc_sma_50', 'btc_ema_50', 'btc_sma_14_50_diff', 'btc_ema_14_50_diff', 'btc_sma_14_50_ratio', 'btc_sma_14_slope', 'btc_ema_14_slope', 'btc_sma_21_slope', 'btc_ema_21_slope', 'btc_sma_50_slope', 'btc_ema_50_slope', 'btc_close_ema_21_dist', 'btc_close_ema_21_dist_norm', 'btc_rsi_14', 'btc_macd', 'btc_macd_signal', 'btc_macd_diff', 'btc_bb_high', 'btc_bb_low', 'btc_bb_mid', 'btc_bb_width', 'btc_atr_14', 'btc_volatility_index', 'btc_trading_volume', 'active_addresses_blockchain', 'hash_rate_blockchain', 'miner_revenue_blockchain', 'difficulty_blockchain', 'estimated_transaction_volume_usd_blockchain', 'PiCycle_cbbi', 'RUPL_cbbi', 'RHODL_cbbi', 'Puell_cbbi', '2YMA_cbbi', 'Trolololo_cbbi', 'MVRV_cbbi', 'ReserveRisk_cbbi', 'Woobull_cbbi', 'Confidence_cbbi', 'Fear Greed', 'positive_sentiment', 'negative_sentiment', 'bullish_sentiment', 'bearish_sentiment', 'risk_uncertainty_sentiment', 'problem_malicious_sentiment', 'active_trading_sentiment', 'long_term_investment_sentiment', 'market_narrative_sentiment', 'core_technology_sentiment', 'development_ecosystem_sentiment', 'news_events_sentiment', 'regulations_sentiment', 'community_social_sentiment', 'price_sentiment', 'volume_sentiment', 'marketcap_sentiment', 'Gold_Price', 'Gold_Share', 'Gold_Volatility', 'Oil_Crude_Price', 'Oil_Brent_Price', 'Oil_Volatility', 'DJI', 'GSPC', 'IXIC', 'NYFANG', 'CBOE_Volatility', 'EM_ETF', 'DXY', 'EURUSD']
      Window 1/5: forecasting 7 steps
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
HPU available: False, using: 0 HPUs
Predicting DataLoader 0: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 193.75it/s]
        Re-fitting model with 2894 samples...
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
HPU available: False, using: 0 HPUs
Epoch 99: 100%|██████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 60.02it/s, v_num=54, train_loss_step=0.319, train_loss_epoch=0.319]`Trainer.fit` stopped: `max_steps=100` reached.                                                                                                                             
Epoch 99: 100%|██████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 57.64it/s, v_num=54, train_loss_step=0.319, train_loss_epoch=0.319]
      Window 2/5: forecasting 7 steps
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
HPU available: False, using: 0 HPUs
Predicting DataLoader 0: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 289.94it/s]
        Re-fitting model with 2901 samples...
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
HPU available: False, using: 0 HPUs
Epoch 99: 100%|██████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 58.69it/s, v_num=56, train_loss_step=0.302, train_loss_epoch=0.302]`Trainer.fit` stopped: `max_steps=100` reached.                                                                                                                             
Epoch 99: 100%|██████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 56.23it/s, v_num=56, train_loss_step=0.302, train_loss_epoch=0.302]
      Window 3/5: forecasting 7 steps
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
HPU available: False, using: 0 HPUs
Predicting DataLoader 0: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 292.53it/s]
        Re-fitting model with 2908 samples...
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
HPU available: False, using: 0 HPUs
Epoch 99: 100%|██████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 58.83it/s, v_num=58, train_loss_step=0.231, train_loss_epoch=0.231]`Trainer.fit` stopped: `max_steps=100` reached.                                                                                                                             
Epoch 99: 100%|██████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 55.89it/s, v_num=58, train_loss_step=0.231, train_loss_epoch=0.231]
      Window 4/5: forecasting 7 steps
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
HPU available: False, using: 0 HPUs
Predicting DataLoader 0: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 268.25it/s]
        Re-fitting model with 2915 samples...
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
HPU available: False, using: 0 HPUs
Epoch 99: 100%|██████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 57.74it/s, v_num=60, train_loss_step=0.215, train_loss_epoch=0.215]`Trainer.fit` stopped: `max_steps=100` reached.                                                                                                                             
Epoch 99: 100%|██████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 54.96it/s, v_num=60, train_loss_step=0.215, train_loss_epoch=0.215]
      Window 5/5: forecasting 7 steps
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
HPU available: False, using: 0 HPUs
Predicting DataLoader 0: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 278.64it/s]
  ✓ NHITS rolling_forecast completed (Test MAE: 5823.4087)
Neural models fit-predict completed: 1/1 successful

--- Fit-Predict Statistical Models ---
Fitting and predicting 1 statistical models...
[1/1] Fit-Predict AutoARIMA...
Forecast: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:29<00:00, 29.20s/it]
  ✓ AutoARIMA fit-predict completed (Test MAE: 5821.6465)
Statistical models fit-predict completed: 1/1 successful

--- Creating Final Unified Forecast Plot ---
Creating unified forecast comparison plot...
✓ Final forecast plot saved: results/plot/final_forecast_comparison_20250528_005123.png
  • Plot shows 2 model predictions vs actual values
  • Continuous line from last 100 train values to 35 test values

Total fit-predict results: 2
Successful fit-predict results: 2
✓ Models ranked successfully by test set performance
All final results saved: 2 entries
Ranked final results saved: 2 entries
Final results saved to results/final
Final Summary: {'timestamp': '20250528_005124', 'evaluation_type': 'final_fit_predict', 'total_models': 2, 'successful_models': 2, 'failed_models': 0, 'best_model': 'AutoARIMA', 'best_test_mae': 5821.646466483697}

🏅 TOP FINAL MODELS (2):
------------------------------
1. AutoARIMA            MAE: 5821.6465 (statsforecast)
2. NHITS                MAE: 5823.4087 (neuralforecast)
✅ Model comparison completed in 275.7 seconds
   • CV models evaluated: 2
   • Final models evaluated: 2
   • Best model: AutoARIMA
   • Best test MAE: 5821.6465
   • Final results saved to: results/final/model_comparison_20250528_074647.csv

📈 STEP 4: FINAL EVALUATION
----------------------------------------

🏅 TOP 2 MODELS (2):
------------------------------
1. AutoARIMA            MAE: 5821.6465
2. NHITS                MAE: 5823.4087

📊 Ensemble Weights (inverse MAE):
   AutoARIMA: 0.500
   NHITS: 0.500

✅ Final evaluation completed
   • Results saved to: results/final/final_results_20250528_074647.json

============================================================
🎯 BITCOIN FORECASTING PIPELINE SUMMARY
============================================================
Execution Time: 2025-05-28 07:46:47
HPO: Skipped

📊 DATA INFORMATION:
  • Total samples: 2,922
  • Training samples: 2,887
  • Test samples: 35
  • Features: 77
  • Horizon: 7 days

🏆 MODEL COMPARISON:
  • Execution time: 275.7 seconds
  • CV models evaluated: 2
  • Final models evaluated: 2
  • Best model: AutoARIMA
  • Best test MAE: 5821.6465

🎯 RECOMMENDATIONS:
  • Review top-performing models for deployment
  • Consider ensemble methods for improved performance
  • Monitor model performance over time
  • Retrain periodically with new data

============================================================

📝 Summary report saved to: results/final/summary_report_20250528_074647.txt
````

## File: READEVERYTIMEOPENREPO.md
````markdown
- input_size: number of read past points

- step_size: slide step, if step_size >= input_size -> no overlap

- windows_batch_size: process input_size together in batch then average out mistake
    -> too small: noisy
    -> too large: takes longer learning

- max_steps: num of train iterations (num_epoch)
- val_check_steps: freq in making validate -> smaller is better since spot quick early stopping but can be computation overhead
    -> these 2 often go with early_stop_patience_steps

Include prediction interval to account for uncertainty
````

## File: README.md
````markdown
# Bitcoin Forecasting Pipeline - Clean Version

A streamlined, production-ready pipeline for Bitcoin price forecasting using the Nixtla ecosystem (statsforecast, mlforecast, neuralforecast).

## 🎯 Pipeline Overview

This pipeline implements a comprehensive workflow for Bitcoin daily close price forecasting:

1. **Data Preparation** - Load and preprocess Bitcoin price data with exogenous features
2. **Hyperparameter Optimization** - Use Auto Models with cross-validation to find optimal parameters
3. **Model Comparison** - Compare models from all three frameworks using best parameters
4. **Final Evaluation** - Generate comprehensive reports and recommendations

## 🚀 Quick Start

### Prerequisites
```bash
pip install statsforecast mlforecast neuralforecast utilsforecast
pip install pandas numpy matplotlib plotly
```

### Validation Test
```bash
python test_pipeline.py
```

### Quick Test Run
```bash
python main_clean.py --fast-mode --skip-hpo
```

### Full Pipeline
```bash
python main_clean.py
```

## 📊 Usage Options

### Command Line Arguments

- `--fast-mode`: Use fewer models and samples for quick testing
- `--skip-hpo`: Skip hyperparameter optimization step

### Examples

```bash
# Quick testing (3 models, 3 samples per model, skip HPO)
python main_clean.py --fast-mode --skip-hpo

# Fast with HPO (fewer models but with optimization)
python main_clean.py --fast-mode

# Full pipeline (all models, full HPO)
python main_clean.py

# Help
python main_clean.py --help
```

## 🏗️ Architecture

### Cleaned Structure

```
├── main_clean.py                    # Single entry point
├── test_pipeline.py                 # Validation tests
├── config/
│   └── base.py                      # Centralized configuration
├── src/
│   ├── models/
│   │   └── model_registry.py        # Unified model definitions
│   ├── dataset/
│   │   └── data_preparation.py      # Data loading and preprocessing
│   ├── hpo/
│   │   └── hyperparameter_tuning.py # HPO with auto models
│   ├── pipelines/
│   │   └── model_selection.py       # Model comparison framework
│   └── utils/
│       ├── other.py                 # Utilities
│       ├── plot.py                  # Visualization
│       └── hyperparam_loader.py     # Configuration loading
└── results/                         # Output directory
    ├── hpo/                         # HPO results
    ├── cv/                          # Cross-validation results
    ├── final/                       # Final model results
    └── models/                      # Saved models
```

### Key Improvements

1. **Single Entry Point**: Consolidated `main.py` and `main2.py` into `main_clean.py`
2. **Unified Model Registry**: All models from three frameworks in one place
3. **Consistent Configuration**: Centralized config with proper imports
4. **Clear Pipeline Steps**: Well-defined workflow with progress tracking
5. **Fast Mode Support**: Quick testing with reduced models/samples
6. **Error Handling**: Robust error handling with meaningful messages
7. **Comprehensive Logging**: Detailed progress and result reporting

## 📈 Model Registry

The `ModelRegistry` class provides unified access to all models:

### Statistical Models (statsforecast)
- **Primary**: AutoARIMA, AutoETS (multiple configurations), AutoTheta
- **Secondary**: AutoCES, RandomWalkWithDrift, WindowAverage, SeasonalNaive
- **Fast Mode**: Top 3 essential models only

### Neural Models (neuralforecast)
- **Full Set**: NHITS, NBEATS, LSTM, TFT, GRU
- **Fast Mode**: NHITS, NBEATS, LSTM only
- **Optimized**: Bitcoin-specific hyperparameters

### Auto Models (for HPO)
- **Full Set**: AutoNHITS, AutoNBEATS, AutoLSTM, AutoTFT
- **Fast Mode**: AutoNHITS, AutoNBEATS, AutoLSTM only
- **Adaptive Samples**: 3 samples in fast mode, 10+ in full mode

## 🔧 Configuration

Key settings in `config/base.py`:

```python
# Forecasting
HORIZON = 7                    # 7-day forecast
TEST_LENGTH_MULTIPLIER = 5     # 35 days test set
FREQUENCY = 'D'                # Daily frequency

# Cross-validation
CV_N_WINDOWS = 5               # 5 CV folds
CV_STEP_SIZE = HORIZON         # Non-overlapping windows

# HPO
NUM_SAMPLES_PER_MODEL = 10     # Hyperparameter samples
```

## 📊 Output

The pipeline generates:

1. **HPO Results**: Best hyperparameters for each auto model
2. **CV Results**: Cross-validation performance metrics
3. **Model Comparison**: Ranked comparison of all models
4. **Final Report**: Summary with recommendations
5. **Visualizations**: Forecast plots and performance charts

### Result Files

```
results/
├── hpo/
│   └── best_hyperparameters.csv
├── cv/
│   └── model_comparison_YYYYMMDD_HHMMSS.csv
└── final/
    ├── final_results_YYYYMMDD_HHMMSS.json
    └── summary_report_YYYYMMDD_HHMMSS.txt
```

## 🎯 Workflow Steps

### Step 1: Data Preparation
- Load Bitcoin price data from `data/final/dataset.parquet`
- Extract exogenous features (SMA, volume, etc.)
- Split into train/test sets (respecting temporal order)
- Format for Nixtla frameworks

### Step 2: Hyperparameter Optimization (Optional)
- Use Auto Models with cross-validation
- Search optimal hyperparameters for each model type
- Save best configurations for later use

### Step 3: Model Comparison
- Train models using best hyperparameters (or defaults)
- Compare performance across all frameworks
- Calculate comprehensive metrics (MAE, RMSE, directional accuracy)

### Step 4: Final Evaluation
- Rank models by performance
- Generate ensemble recommendations
- Create summary report with insights

## 🚀 Performance Tips

### Fast Mode Benefits
- **3x faster** execution time
- Suitable for development and testing
- Covers essential model types

### Full Mode Benefits
- Comprehensive model coverage
- Thorough hyperparameter search
- Production-ready results

### Hardware Recommendations
- **CPU**: 8+ cores for parallel processing
- **RAM**: 16GB+ for large datasets
- **GPU**: Optional but accelerates neural models

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**: Run `python test_pipeline.py` first
2. **Data Not Found**: Check `DATA_PATH` in config
3. **Memory Issues**: Use `--fast-mode` flag
4. **Ray Errors**: Reduce `RAY_NUM_CPUS` in config

### Debug Mode
```bash
python -u main_clean.py --fast-mode --skip-hpo > debug.log 2>&1
```

## 📝 Extending the Pipeline

### Adding New Models
1. Update `ModelRegistry` class in `src/models/model_registry.py`
2. Add model configurations for HPO
3. Test with `python test_pipeline.py`

### Custom Loss Functions
1. Add to `get_loss_functions()` in model registry
2. Update loss mapping in config

### New Evaluation Metrics
1. Extend `model_selection.py`
2. Add custom metric calculations

## 🎖️ Best Practices

1. **Always test first**: Use `--fast-mode --skip-hpo` for initial validation
2. **Version results**: Pipeline timestamps all outputs
3. **Monitor performance**: Check memory usage with large datasets
4. **Regular updates**: Retrain models with new data periodically
5. **Ensemble approaches**: Combine top models for better performance

## 📞 Support

For issues or questions:
1. Run validation tests: `python test_pipeline.py`
2. Check configuration: `config/base.py`
3. Review logs in results directory
4. Use fast mode for debugging

---

**Note**: This pipeline is optimized for Bitcoin forecasting but can be adapted for other time series by modifying the data preparation and model configurations.
````

## File: repomix.txt
````
repomix --ignore "**/*.csv,**/*.json,**/*.ipynb,**/__pycache__/,**/lightning_logs/**" --style markdown --compress
````

## File: src/models/mlforecast/auto_models.py
````python
def get_ml_models(self, model_names: List[str] = None) -> List[Tuple[str, Any]]
⋮----
"""
        Get MLForecast models with different algorithms.
        
        Parameters:
        -----------
        model_names : List[str], optional
            Specific model names to include. If None, returns all models.
            
        Returns:
        --------
        List[Tuple[str, Any]]: List of (model_name, sklearn_model) tuples
        """
n_estimators = 200
max_depth = 15
⋮----
all_models = {
⋮----
# Add XGBoost
````

## File: src/models/statsforecast/models.py
````python
"""
Bitcoin-optimized statistical models for forecasting.
"""
⋮----
def get_statistical_models(season_length: int = 7) -> List[Any]
⋮----
"""
    Get optimized statistical models for Bitcoin forecasting.
    
    Args:
        season_length: Seasonal period (default 7 for weekly patterns)
        
    Returns:
        List of statistical model instances
    """
# Full model set optimized for Bitcoin characteristics
all_models = [
⋮----
# PRIMARY: Best for Bitcoin's non-stationary, trending, volatile nature
⋮----
# AutoETS(season_length=season_length, model='ZZZ'),  # Auto-select
# AutoETS(season_length=season_length, model='MMM'),  # Multiplicative
# AutoETS(season_length=season_length, model='MAM'),  # Mixed
⋮----
# # THETA METHODS: Excellent for trending financial data
# AutoTheta(season_length=season_length),
⋮----
# # COMPLEX SMOOTHING: Handles complex patterns
# AutoCES(season_length=season_length),
⋮----
# # BASELINE MODELS: Simple but effective
# RandomWalkWithDrift(),
# WindowAverage(window_size=3),   # Very responsive
# WindowAverage(window_size=7),   # Weekly patterns
⋮----
# # SEASONAL PATTERNS
# SeasonalNaive(season_length=season_length),
# SeasonalWindowAverage(season_length=season_length, window_size=3),
````

## File: src/utils/hyperparam.py
````python

````

## File: config/base.py
````python
"""
Configuration settings.
"""
⋮----
# === Data Configuration ===
DATA_PATH = 'data/final/dataset.parquet'
DATE_COLUMN = 'Date'
DATE_RENAMED = 'ds'
TARGET_COLUMN = 'btc_close'
TARGET_RENAMED = 'y'
UNIQUE_ID_VALUE = 'Bitcoin'
⋮----
# === Forecasting Configuration ===
HORIZON = 7
LEVELS = [80, 90]
TEST_LENGTH_MULTIPLIER = 5
SEED = 42
⋮----
# === Rolling Forecast Configuration ===
ENABLE_ROLLING_FORECAST = True  # Enable rolling forecast for neural models when horizon < test_length
⋮----
# === Model Configuration ===
FREQUENCY = 'D'
SCALER_TYPE = ['standard']  # List for tune.choice()
LOCAL_SCALER_TYPE = 'standard'  # String for direct use
⋮----
# === Cross-validation Configuration ===
CV_N_WINDOWS = 5
CV_STEP_SIZE = HORIZON
⋮----
# === Hyperparameter Tuning Configuration ===
NUM_SAMPLES_PER_MODEL = 1
⋮----
# === Search Algorithm Configuration ===
DEFAULT_SEARCH_ALGORITHM = 'optuna'  # Default search algorithm
SEARCH_ALGORITHM_MAX_CONCURRENT = 4  # Max concurrent trials
SEARCH_ALGORITHM_REPEAT_TRIALS = None  # Number of repeated evaluations (None = no repeat)
FAST_SEARCH_ALGORITHM = 'hyperopt'  # Algorithm to use in fast mode
⋮----
# === Ray Configuration ===
RAY_ADDRESS = 'local'
RAY_NUM_CPUS = os.cpu_count()
RAY_NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0
⋮----
# === Loss function mapping ===
LOSS_MAP = {
⋮----
# === Columns to exclude when processing best hyperparameters ===
EXCLUDE_HYPERPARAMETER_KEYS = [
⋮----
# === JSON parseable hyperparameter keys ===
JSON_PARSEABLE_KEYS = [
⋮----
# === Output Directories ===
RESULTS_DIR: str = 'results'
HPO_DIR: str = f"{RESULTS_DIR}/hpo"
PLOT_DIR: str = f"{RESULTS_DIR}/plot"
FINAL_DIR: str = f"{RESULTS_DIR}/final"
MODELS_DIR: str = f"{RESULTS_DIR}/models"
BEST_HYPERPARAMETERS_CSV = f"{RESULTS_DIR}/best_hyperparameters.json"
⋮----
def __post_init__(self)
⋮----
"""Set default values that depend on other attributes."""
# Create output directories
````

## File: src/dataset/data_preparation.py
````python
"""
Data preparation module for neural forecasting pipeline.
"""
⋮----
def load_and_prepare_data()
⋮----
"""Load and prepare the dataset for forecasting."""
⋮----
# Load data
df = pd.read_parquet(DATA_PATH)
⋮----
# Rename columns
df = df.rename(columns={DATE_COLUMN: DATE_RENAMED, TARGET_COLUMN: TARGET_RENAMED})
⋮----
# Add unique_id and convert date
⋮----
def split_data(df, horizon, test_length_multiplier)
⋮----
"""Split data into train and test sets."""
⋮----
test_length = horizon * test_length_multiplier
⋮----
# print(f"Forecast horizon (h) set to: {horizon} days")
⋮----
# Validate data length
⋮----
# Split data
train_df = df.iloc[:-test_length].copy()
test_df = df.iloc[-test_length:].copy()
⋮----
# Print information
⋮----
def prepare_data(horizon, test_length_multiplier)
⋮----
"""Complete data preparation pipeline.
    
    Args:
        horizon (int): Forecast horizon in days
        test_length_multiplier (int): Multiplier to determine test set length
    """
# Load and prepare data
df = load_and_prepare_data()
⋮----
# Get historical exogenous features
hist_exog_list = get_historical_exogenous_features(df)
⋮----
# Move hist_exog_list to end of df
df = df[['unique_id', 'ds', 'y'] + hist_exog_list]
train_df = train_df[['unique_id', 'ds', 'y'] + hist_exog_list]
test_df = test_df[['unique_id', 'ds', 'y'] + hist_exog_list]
⋮----
# Example usage with sample values
````

## File: src/models/neuralforecast/auto_cfg.py
````python
def neural_auto_model_cfg(h: int) -> Dict[str, Dict]
⋮----
"""
    Get model configurations optimized for Bitcoin forecasting.

    Args:
        h: Forecast horizon (number of time steps to forecast)

    Returns:
        Dictionary of model configurations with Ray Tune choices for model-specific parameters only.
        Common parameters are handled in base_auto_config.
    """
⋮----
# NHITS config - only model-specific parameters
nhits_config = {
⋮----
),  # MaxPool's Kernelsize
⋮----
),  # Interpolation expressivity ratios
⋮----
# NBEATS config - only model-specific parameters
nbeats_config = {
⋮----
# 'stack_types': tune.choice([
#     ['trend', 'seasonality'],
#     ['trend', 'trend', 'seasonality'],
#     ['generic', 'generic']
# ]),
⋮----
# LSTM config - only model-specific parameters
lstm_config = {
⋮----
# TFT config - only model-specific parameters
tft_config = {
⋮----
# Legacy function for backward compatibility
def neural_auto_model_cfg_legacy(h: int) -> Dict[str, Dict]
⋮----
"""
    Get model configurations optimized for Bitcoin price forecasting.
    Bitcoin exhibits high volatility, trending behavior, and potential regime changes.
    """
⋮----
# Enhanced NHITS config for crypto volatility
⋮----
),  # Longer lookback for crypto
⋮----
# 'dropout_prob_theta': tune.choice([0.1, 0.2, 0.3]),
⋮----
# Enhanced NBEATS config
⋮----
# 'dropout_prob_theta': tune.choice([0.1, 0.2]),
⋮----
# TFT config for complex temporal patterns
⋮----
# 'dropout': tune.choice([0.1, 0.2, 0.3]),
````

## File: src/models/model_registry.py
````python
"""
Unified Model Registry for Bitcoin Forecasting

This module provides a centralized registry for all models across
statsforecast, mlforecast, and neuralforecast frameworks, eliminating
redundancy and ensuring consistency.
"""
⋮----
# Import from modular structure
⋮----
# Framework imports for loss functions
⋮----
# Configuration
⋮----
class ModelRegistry
⋮----
"""
    Centralized registry for all forecasting models.
    
    This class manages model definitions across all three frameworks
    and provides consistent interfaces for model selection.
    """
⋮----
@staticmethod
    def get_statistical_models(season_length: int = 7) -> List[Any]
⋮----
"""
        Get optimized statistical models for Bitcoin forecasting.
        
        Args:
            season_length: Seasonal period (default 7 for weekly patterns)
            
        Returns:
            List of statistical model instances
        """
⋮----
@staticmethod
    def get_neural_models(horizon: int, hist_exog_list: Optional[List[str]] = None, hyperparameters_json_path: Optional[str] = None) -> List[Any]
⋮----
"""
        Get neural forecasting models with fixed hyperparameters.
        
        Args:
            horizon: Forecast horizon
            hist_exog_list: List of historical exogenous features
            
        Returns:
            List of neural model instances
        """
⋮----
"""
        Get auto models for hyperparameter optimization.
        
        Args:
            horizon: Forecast horizon
            num_samples: Number of hyperparameter samples per model
            hist_exog_list: List of historical exogenous features
            
        Returns:
            List of auto model instances for HPO
        """
⋮----
@staticmethod
    def get_model_summary() -> Dict[str, int]
⋮----
"""
        Get summary of available models.
        
        Returns:
            Dictionary with model counts by category
        """
````

## File: src/pipelines/final_fit_predict.py
````python
# src/pipelines/final_fit_predict.py
"""
Final fit and predict framework for generating future forecasts and visualization.
"""
⋮----
class FinalFitPredictor
⋮----
"""Final fit and predict for generating future forecasts with visualization."""
⋮----
def __init__(self, results_dir: str = FINAL_DIR)
⋮----
self.all_forecasts = {}  # Store all forecasts for unified plotting
⋮----
def _rolling_forecast_neural(self, nf, train_df, test_df, horizon: int, model_name: str) -> pd.DataFrame
⋮----
"""Perform rolling forecast for neural models when horizon < test_df length."""
⋮----
all_forecasts = []
current_train_df = train_df.copy()  # Keep all columns from original training data
⋮----
# Identify exogenous columns (all columns except 'unique_id', 'ds', 'y')
exog_cols = [col for col in test_df.columns if col not in ['unique_id', 'ds', 'y']]
futr_cols = ['unique_id', 'ds'] + exog_cols  # Columns needed for prediction
⋮----
# Calculate number of rolling windows needed
test_length = len(test_df)
n_windows = (test_length + horizon - 1) // horizon  # Ceiling division
⋮----
start_idx = window * horizon
end_idx = min(start_idx + horizon, test_length)
⋮----
# Get the current test window
current_test_window = test_df.iloc[start_idx:end_idx].copy()
⋮----
# Predict for current window - include all exogenous features
futr_df = current_test_window[futr_cols].copy()
window_forecast = nf.predict(futr_df=futr_df)
⋮----
# Update training data with actual values from current window for next iteration
if window < n_windows - 1:  # Don't update after last window
# Add actual values to training data (keep all columns for consistency)
window_actual = current_test_window.copy()
current_train_df = pd.concat([current_train_df, window_actual], ignore_index=True)
⋮----
# Re-fit the model with updated training data
⋮----
# Combine all forecasts
combined_forecasts = pd.concat(all_forecasts, ignore_index=True)
⋮----
def fit_predict_neural_models(self, models, train_df, test_df, horizon: int) -> List[Dict]
⋮----
"""Fit neural models on full training data and predict on test data."""
results = []
⋮----
# # Debug: Show data structure
# print(f"  Train data columns: {list(train_df.columns)}")
# print(f"  Test data columns: {list(test_df.columns)}")
# print(f"  Train data shape: {train_df.shape}")
# print(f"  Test data shape: {test_df.shape}")
⋮----
# Determine if rolling forecast is needed
⋮----
# ENABLE_ROLLING_FORECAST = False
use_rolling = ENABLE_ROLLING_FORECAST and horizon < test_length
⋮----
model_name = model.__class__.__name__
⋮----
start_time = time.time()
⋮----
# Create NeuralForecast instance
nf = NeuralForecast(models=[model], freq='D')
⋮----
# Fit on full training data
⋮----
# Choose forecasting method based on horizon vs test length
⋮----
# Rolling forecast
forecasts = self._rolling_forecast_neural(nf, train_df, test_df, horizon, model_name)
evaluation_method = "rolling_forecast"
⋮----
# Direct multi-step forecast
# Include all exogenous features for prediction
⋮----
futr_cols = ['unique_id', 'ds'] + exog_cols
futr_df = test_df[futr_cols].copy()
forecasts = nf.predict(futr_df=futr_df)
evaluation_method = "direct_forecast"
⋮----
# Merge with actual test values for evaluation
eval_df = test_df.merge(forecasts, on=['unique_id', 'ds'], how='inner')
⋮----
# Store forecast for unified plotting
⋮----
'eval_df': eval_df,  # Store eval_df for plotting actual vs predicted
⋮----
# Calculate metrics from forecast results
metrics = self._calculate_test_metrics(eval_df, [model_name])
⋮----
# Training time
training_time = time.time() - start_time
⋮----
# Check if metrics were calculated successfully
⋮----
result = {
⋮----
error_msg = str(e)
⋮----
successful_count = sum(1 for r in results if r.get('status') == 'success')
⋮----
def fit_predict_statistical_models(self, models, train_df, test_df, horizon: int) -> List[Dict]
⋮----
"""Fit statistical models on training data and predict on test data."""
⋮----
# Prepare data for StatsForecast
df_train = train_df[['unique_id', 'ds', 'y']].copy()
df_test = test_df[['unique_id', 'ds']].copy()
⋮----
# Create StatsForecast instance
sf = StatsForecast(models=[model], freq='D', verbose=True)
⋮----
# Generate forecast on test data
forecasts = sf.forecast(df=df_train, X_df=df_test, h=HORIZON * TEST_LENGTH_MULTIPLIER)
⋮----
# Merge with test data for evaluation
⋮----
'eval_df': eval_df  # Store eval_df for plotting actual vs predicted
⋮----
def _calculate_test_metrics(self, test_results: pd.DataFrame, model_names: List[str]) -> Dict
⋮----
"""Calculate comprehensive metrics for test set predictions."""
metrics = {}
⋮----
# Check for valid data
y_true = test_results['y'].values
y_pred = test_results[model_name].values
⋮----
# Remove NaN values
mask = ~(np.isnan(y_true) | np.isnan(y_pred))
y_true_clean = y_true[mask]
y_pred_clean = y_pred[mask]
⋮----
# Basic metrics using utilsforecast
⋮----
model_metrics = evaluate(
⋮----
# Extract metric values
mae_val = model_metrics[model_metrics['metric'] == 'mae'][model_name].iloc[0]
mse_val = model_metrics[model_metrics['metric'] == 'mse'][model_name].iloc[0]
rmse_val = model_metrics[model_metrics['metric'] == 'rmse'][model_name].iloc[0]
mape_val = model_metrics[model_metrics['metric'] == 'mape'][model_name].iloc[0]
smape_val = model_metrics[model_metrics['metric'] == 'smape'][model_name].iloc[0]
⋮----
# Fallback to manual calculation
mae_val = np.mean(np.abs(y_true_clean - y_pred_clean))
mse_val = np.mean((y_true_clean - y_pred_clean) ** 2)
rmse_val = np.sqrt(mse_val)
mape_val = np.mean(np.abs((y_true_clean - y_pred_clean) / y_true_clean)) * 100
smape_val = np.mean(2 * np.abs(y_true_clean - y_pred_clean) / (np.abs(y_true_clean) + np.abs(y_pred_clean))) * 100
⋮----
# Additional custom metrics
directional_accuracy = self._calculate_directional_accuracy(y_true_clean, y_pred_clean)
⋮----
def _calculate_directional_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float
⋮----
"""Calculate directional accuracy (correct prediction of up/down movement)."""
⋮----
true_direction = np.diff(y_true) > 0
pred_direction = np.diff(y_pred) > 0
⋮----
def _create_unified_forecast_plot(self, train_df: pd.DataFrame, test_df: pd.DataFrame)
⋮----
"""Create a unified plot showing all model predictions with actual values from eval_df."""
⋮----
# Take last 100 values from train_df for continuity
train_tail = train_df.tail(50).copy()
⋮----
# Set up the plot
⋮----
# Convert ds to datetime for better plotting
⋮----
test_df = test_df.copy()
⋮----
# Create continuous actual values line (train tail + test)
# Combine train tail and test for continuous line
actual_dates = pd.concat([train_tail['ds'], test_df['ds']], ignore_index=True)
actual_values = pd.concat([train_tail['y'], test_df['y']], ignore_index=True)
⋮----
# Plot continuous actual values line
⋮----
# Highlight test period with markers
⋮----
# Define colors for different models
num_models = len(self.all_forecasts)
cmap = cm.get_cmap('tab20', num_models)
⋮----
# Plot forecasts from all models using eval_df data
⋮----
color = cmap(i)
⋮----
# Use eval_df data if available, otherwise fall back to forecast data
⋮----
eval_df = forecast_data['eval_df']
# Convert dates to datetime and sort by date
eval_df_sorted = eval_df.sort_values('ds')
forecast_dates = pd.to_datetime(eval_df_sorted['ds'])
predictions = eval_df_sorted[model_name].values
⋮----
# Remove any NaN predictions
mask = ~np.isnan(predictions)
forecast_dates = forecast_dates[mask]
predictions = predictions[mask]
⋮----
# Fallback to original forecast data
forecast_dates = pd.to_datetime(forecast_data['ds'])
predictions = forecast_data['predictions']
⋮----
# Plot model predictions
linestyle = '--' if forecast_data['framework'] == 'statistical' else '-'
alpha = 0.8
linewidth = 2.5 if forecast_data['framework'] == 'neural' else 2
⋮----
# Create label with forecast method info
method_info = ""
⋮----
method_info = f" - {forecast_data['forecast_method']}"
⋮----
# Customize the plot
⋮----
# Improve legend
legend = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
⋮----
# Format x-axis dates
⋮----
# plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(test_df)//8)))
⋮----
# Add vertical line to separate train and test with annotation
⋮----
split_date = test_df['ds'].iloc[0]
⋮----
# Add annotation for the split
⋮----
# Add text box with summary info
⋮----
info_text = f"Models Compared: {len(self.all_forecasts)}\n"
⋮----
# info_text += f"Train Context: {len(train_tail)} days\n"
⋮----
# Add rolling forecast info
rolling_models = [name for name, data in self.all_forecasts.items()
⋮----
# Save the plot
timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
plot_filename = f"final_forecast_comparison_{timestamp}.png"
plot_path = self.plot_dir / plot_filename
⋮----
def fit_predict_all_models(self, neural_models, stat_models, train_df, test_df, horizon: int) -> pd.DataFrame
⋮----
"""Fit all models and generate final predictions with unified plotting."""
⋮----
# print(f"Train data shape: {train_df.shape}")
# print(f"Test data shape: {test_df.shape}")
# print(f"Horizon: {horizon}")
⋮----
# Clear previous forecasts
⋮----
all_results = []
⋮----
# Fit-predict neural models
⋮----
neural_results = self.fit_predict_neural_models(neural_models, train_df, test_df, horizon)
⋮----
# Fit-predict statistical models
⋮----
stat_results = self.fit_predict_statistical_models(stat_models, train_df, test_df, horizon)
⋮----
# Create unified forecast plot
⋮----
# Create results DataFrame
⋮----
results_df = pd.DataFrame(all_results)
⋮----
# Filter out failed models
⋮----
successful_results = results_df[results_df['status'] == 'success'].copy()
⋮----
successful_results = results_df[results_df['error'].isna()].copy()
⋮----
# Assume all are successful if no status/error columns
successful_results = results_df.copy()
⋮----
# Rank models by test set MAE (lower is better)
⋮----
# Combined rank (simple average)
⋮----
# Sort by combined rank
successful_results = successful_results.sort_values('combined_rank')
⋮----
successful_results = successful_results.sort_values('mae')
⋮----
# Save results
⋮----
def save_final_results(self, all_results: pd.DataFrame, successful_results: pd.DataFrame)
⋮----
"""Save final fit-predict results."""
⋮----
# Use simple timestamp without timezone
⋮----
# Save all results
⋮----
# Save successful results
⋮----
# Save summary
summary = {
⋮----
def get_top_models(self, results_df: pd.DataFrame, top_n: int = 5) -> List[str]
⋮----
"""Get top N model names based on test set performance."""
⋮----
actual_top_n = min(top_n, len(results_df))
top_models = results_df.head(actual_top_n)['model_name'].tolist()
````

## File: src/utils/other.py
````python
"""
Utility functions for neural forecasting pipeline.
"""
⋮----
def seed_everything(seed=42)
⋮----
"""Set seeds for reproducibility across all random number generators."""
⋮----
def setup_environment(seed=42, ray_config=None)
⋮----
"""Setup the environment for neural forecasting."""
# Set seed for reproducibility
⋮----
# Setup logging
⋮----
# Initialize Ray
⋮----
ray_config = {
⋮----
def print_data_info(df, train_df, test_df)
⋮----
"""Print information about data splits."""
⋮----
def get_historical_exogenous_features(df, exclude_cols=None)
⋮----
"""Get list of historical exogenous features from dataframe."""
⋮----
exclude_cols = ['ds', 'unique_id', 'y']
⋮----
all_cols = df.columns.tolist()
hist_exog_list = [col for col in all_cols if col not in exclude_cols]
⋮----
def calculate_evaluation_metrics(df_pl, model_cols, exclude_cols=None)
⋮----
"""Calculate evaluation metrics for cross-validation results."""
⋮----
exclude_cols = ['unique_id', 'ds', 'cutoff', 'y']
⋮----
results = {}
⋮----
# Calculate MSE and MAE using utilsforecast
mse_val = mse(df=df_pl, models=[model], target_col='y').to_pandas()[model].values[0]
mae_val = mae(df=df_pl, models=[model], target_col='y').to_pandas()[model].values[0]
rmse_val = np.sqrt(mse_val)
````

## File: .gitignore
````
data/processed/
data/raw/
lightning_logs/
__pycache__/
.venv/
.DS_Store
````

## File: requirements.txt
````
matplotlib
numpy
pandas
seaborn
statsmodels
scikit-learn
scipy
torch
yfinance
pytrends
lightning
sktime
optuna
neuralforecast
hyperopt
torchvision
ray
HEBO
ConfigSpace
plotly
kaleido
window-ops
````

## File: src/models/neuralforecast/auto_models.py
````python
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
⋮----
configs = neural_auto_model_cfg(h)
search_alg = get_search_algorithm_class("hyperopt")
⋮----
init_config = {
⋮----
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
⋮----
models = [
⋮----
# Primary auto models for HPO
⋮----
# AutoNBEATS(**init_config, config=configs["nbeats"]),
# AutoLSTM(**init_config, config=configs["lstm"]),
# AutoTFT(**init_config, config=configs["tft"]),
````

## File: main.py
````python
"""
Bitcoin Forecasting Pipeline
==================================

A streamlined pipeline for Bitcoin price forecasting using the Nixtla ecosystem.
This pipeline implements the following workflow:
1. Data Preparation
2. Hyperparameter Tuning with Cross-Validation on Auto Models
3. Model Comparison using best parameters on normal models
4. Final evaluation and reporting

Usage:
    python main_clean.py [--skip-hpo]
"""
⋮----
# Configuration and utilities
⋮----
class BitcoinForecastingPipeline
⋮----
"""
    Clean, comprehensive Bitcoin forecasting pipeline.
    
    This pipeline coordinates the entire forecasting workflow with proper
    separation of concerns and clear execution steps.
    """
⋮----
def __init__(self,  skip_hpo: bool = False)
⋮----
"""
        Initialize the forecasting pipeline.
        
        Args:
            skip_hpo: Skip hyperparameter optimization step
        """
⋮----
# Create results directories
⋮----
def _setup_directories(self)
⋮----
"""Create necessary output directories."""
directories = [
⋮----
def setup_environment(self)
⋮----
"""Setup the forecasting environment."""
⋮----
# Setup environment with proper configuration
ray_config = {
⋮----
def step1_data_preparation(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]
⋮----
"""
        Step 1: Data Preparation
        
        Returns:
            Tuple of (train_df, test_df, exog_features)
        """
⋮----
# Load and prepare data
⋮----
# Store results
⋮----
def step2_hyperparameter_optimization(self, train_df: pd.DataFrame, hist_exog_list: List[str]) -> Optional[pd.DataFrame]
⋮----
"""
        Step 2: Hyperparameter Optimization using Auto Models
        
        Args:
            train_df: train dataset for HPO
            
        Returns:
            DataFrame with best hyperparameters or None if skipped
        """
⋮----
start_time = time.time()
⋮----
# Adjust samples based on mode
num_samples = NUM_SAMPLES_PER_MODEL
⋮----
tuner = HyperparameterTuner()
⋮----
# Run HPO pipeline
all_best_configs = tuner.run_complete_pipeline(
⋮----
# Load and display results
best_configs_map = tuner.load_best_configurations(BEST_HYPERPARAMETERS_CSV)
⋮----
execution_time = time.time() - start_time
⋮----
def _display_top_models(self, results_df: pd.DataFrame, title: str = "Top Models", top_n: int = 5, show_framework: bool = True, show_weights: bool = False) -> List[str]
⋮----
"""
        Helper method to display top models in a consistent format.
        
        Args:
            results_df: DataFrame with model results
            title: Title for the display section
            top_n: Number of top models to display
            show_framework: Whether to show framework information
            show_weights: Whether to calculate and show ensemble weights
            
        Returns:
            List of top model names
        """
⋮----
# Get top models
actual_top_n = min(top_n, len(results_df))
top_models_df = results_df.head(actual_top_n)
⋮----
top_model_names = []
⋮----
model_name = row['model_name']
mae_score = row['mae']
⋮----
display_parts = [f"{i}. {model_name:<20} MAE: {mae_score:.4f}"]
⋮----
# Show ensemble weights if requested
⋮----
mae_values = top_models_df['mae'].values
inverse_mae = 1 / mae_values
weights = inverse_mae / inverse_mae.sum()
⋮----
def step3_model_comparison(self, train_df: pd.DataFrame, df_test: pd.DataFrame, hist_exog_list: List[str], best_configs: Optional[pd.DataFrame]) -> pd.DataFrame
⋮----
"""
        Step 3: Comprehensive Model Comparison
        
        First perform cross-validation for model selection, then final fit-predict for test evaluation
        
        Args:
            train_df: train dataset
            df_test: Test dataset for final evaluation
            hist_exog_list: List of exogenous features
            best_configs: Best hyperparameters from HPO (if available)
            
        Returns:
            DataFrame with final model comparison results
        """
⋮----
# Get models for comparison
neural_models = []
stat_models = []
⋮----
# Statistical models (no HPO needed)
stat_models = ModelRegistry.get_statistical_models(season_length=7)
⋮----
# Neural models
⋮----
# Use optimized configurations from HPO
⋮----
neural_models = ModelRegistry.get_neural_models(HORIZON, hist_exog_list, BEST_HYPERPARAMETERS_CSV)
⋮----
# Use default configurations
⋮----
neural_models = ModelRegistry.get_neural_models(HORIZON, hist_exog_list)
⋮----
# print(f"   • Train data: {len(train_df):,} samples")
# print(f"   • Test data: {len(df_test):,} samples")
# print(f"   • Neural models: {len(neural_models)}")
# print(f"   • Statistical models: {len(stat_models)}")
⋮----
# Step 3a: Cross-validation for model selection
⋮----
cv_evaluator = CrossValidationEvaluator(results_dir=FINAL_DIR)
⋮----
cv_results_df = cv_evaluator.compare_models_cv(
⋮----
# Display top models from CV
⋮----
# Step 3b: Final fit-predict evaluation
⋮----
final_predictor = FinalFitPredictor(results_dir=FINAL_DIR)
⋮----
final_results_df = final_predictor.fit_predict_all_models(
⋮----
# Display top models using helper method
⋮----
# Store results
⋮----
# Save results
results_path = Path(FINAL_DIR) / f"model_comparison_{self.timestamp.strftime('%Y%m%d_%H%M%S')}.csv"
⋮----
# Return empty DataFrame as fallback
⋮----
def step4_final_evaluation(self, comparison_results: pd.DataFrame) -> Dict
⋮----
"""
        Step 4: Final Evaluation and Reporting
        
        Args:
            comparison_results: Results from model comparison
            
        Returns:
            Dictionary with final evaluation results
        """
⋮----
# Display top models with ensemble weights using helper method
top_n = min(5, len(comparison_results))
top_model_names = self._display_top_models(
⋮----
# Calculate ensemble weights for final results
top_models = comparison_results.head(top_n)
mae_values = top_models['mae'].values
⋮----
final_results = {
⋮----
# Save final results
results_path = Path(FINAL_DIR) / f"final_results_{self.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
⋮----
def generate_summary_report(self) -> str
⋮----
"""Generate a comprehensive summary report."""
report_lines = [
⋮----
data_info = self.results['data_info']
⋮----
hpo_info = self.results['hpo_info']
⋮----
comp_info = self.results['comparison_info']
⋮----
def run_complete_pipeline(self) -> Dict
⋮----
"""
        Execute the complete Bitcoin forecasting pipeline.
        
        Returns:
            Dictionary containing all pipeline results
        """
total_start_time = time.time()
⋮----
# Setup
⋮----
# Step 1: Data Preparation
⋮----
# Step 2: Hyperparameter Optimization
best_configs = self.step2_hyperparameter_optimization(train_df, hist_exog_list)
⋮----
# Step 3: Model Comparison
comparison_results = self.step3_model_comparison(train_df, df_test, hist_exog_list, best_configs)
⋮----
# Step 4: Final Evaluation
final_results = self.step4_final_evaluation(comparison_results)
⋮----
# Calculate total execution time
total_execution_time = time.time() - total_start_time
⋮----
# Generate and display summary report
summary_report = self.generate_summary_report()
⋮----
# Save summary report
report_path = Path(FINAL_DIR) / f"summary_report_{self.timestamp.strftime('%Y%m%d_%H%M%S')}.txt"
⋮----
def main()
⋮----
"""Main execution function with command line argument support."""
parser = argparse.ArgumentParser(
⋮----
args = parser.parse_args()
⋮----
# Initialize and run pipeline
pipeline = BitcoinForecastingPipeline(
⋮----
# Execute pipeline
results = pipeline.run_complete_pipeline()
⋮----
# Exit with appropriate code
````

## File: src/models/neuralforecast/models.py
````python
"""
Enhanced model configurations optimized for Bitcoin forecasting.
"""
⋮----
trainer_kwargs = {
⋮----
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
⋮----
# Define model configurations with their default parameters
model_configs = _get_default_model_configs(h, hist_exog_list)
⋮----
# Load and apply hyperparameters if JSON path is provided
⋮----
hyperparameters_json_path = BEST_HYPERPARAMETERS_CSV
⋮----
# Use the correct import from the refactored HPO system
⋮----
best_hyperparams_map = load_best_hyperparameters(hyperparameters_json_path)
⋮----
model_configs = _apply_hyperparameters(model_configs, best_hyperparams_map, h)
⋮----
# Instantiate all models
⋮----
def _get_default_model_configs(h: int, hist_exog_list: Optional[List[str]] = None) -> Dict[str, Dict]
⋮----
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
⋮----
'input_size': h * 6,  # Increased from h*2 for better performance
⋮----
'max_steps': 500,     # Reduced from 1000 for faster training
⋮----
'random_seed': 42,    # Add for reproducibility
⋮----
# Include exogenous features if available
⋮----
# Model-specific configurations matching HPO Auto models
model_configs = {
⋮----
# 'NBEATS': {
#     **base_config,
#     'model_class': NBEATS,
#     'stack_types': ['trend', 'seasonality'],
#     'n_blocks': [3, 3],
#     'mlp_units': [[512, 512], [512, 512]],
#     'sharing': [False, False],
# },
⋮----
# 'LSTM': {
⋮----
#     'model_class': LSTM,
#     'encoder_n_layers': 2,
#     'encoder_hidden_size': 128,
#     'decoder_hidden_size': 128,
#     'decoder_layers': 1,
⋮----
# 'TFT': {
⋮----
#     'model_class': TFT,
#     'hidden_size': 64,
#     'n_rnn_layers': 2,
#     'n_head': 4,
#     'dropout': 0.1,
⋮----
# 'GRU': {
⋮----
#     'model_class': GRU,
⋮----
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
⋮----
# Parameters used internally by HPO that should be filtered out
internal_hpo_keys = {
⋮----
# The HPO system saves models with 'Auto' prefix
json_key = f'Auto{model_name}'
⋮----
loaded_hpo_params = None
matched_key = None
⋮----
# Find the matching hyperparameters in the JSON
⋮----
loaded_hpo_params = best_hyperparams_map[json_key].copy()  # Copy to avoid modifying original
matched_key = json_key
⋮----
# Clean up and validate hyperparameters
cleaned_params = {}
⋮----
# Handle special parameter types
cleaned_val = _process_hyperparameter_value(param_key, hpo_val)
⋮----
# Apply cleaned hyperparameters to model config
⋮----
def _process_hyperparameter_value(param_key: str, value: Any) -> Any
⋮----
"""
    Process and validate hyperparameter values, handling special cases.
    
    Args:
        param_key: Parameter name
        value: Parameter value from HPO
        
    Returns:
        Processed value or None if invalid
    """
# Handle None values
⋮----
# Handle string representations of lists (from JSON serialization)
⋮----
# Try to parse JSON strings back to Python objects
⋮----
parsed_value = json.loads(value)
⋮----
# Try to parse tuples represented as strings
⋮----
# Convert string tuple representation to actual tuple
parsed_value = eval(value)  # Use with caution, only for trusted data
⋮----
# Handle specific parameter types that need special processing
tuple_params = {'n_pool_kernel_size', 'n_freq_downsample'}
⋮----
# Handle boolean strings
⋮----
# Return value as-is if no special processing needed
⋮----
def _instantiate_models(model_configs: Dict[str, Dict]) -> List[Any]
⋮----
"""
    Instantiate model instances from configurations.
    
    Args:
        model_configs: Dictionary of model configurations
        
    Returns:
        List of instantiated model objects
    """
models = []
⋮----
# Extract the model class
ModelClass = config.pop('model_class')
⋮----
# Validate required parameters
⋮----
# Sort and display configuration for better readability
⋮----
value = config[key]
⋮----
# Create model instance with remaining configuration
model_instance = ModelClass(**config)
⋮----
# Continue with other models instead of failing completely
⋮----
model_names = [type(m).__name__ for m in models]
⋮----
# Test the model loading with hyperparameters
⋮----
models = get_neural_models(h=7, hyperparameters_json_path=BEST_HYPERPARAMETERS_CSV)
````
