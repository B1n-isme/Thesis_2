This file is a merged representation of a subset of the codebase, containing files not matching ignore patterns, combined into a single document by Repomix.
The content has been processed where content has been compressed (code blocks are separated by ⋮---- delimiter).

# File Summary

## Purpose
This file contains a packed representation of the entire repository's contents.
It is designed to be easily consumable by AI systems for analysis, code review,
or other automated processes.

## File Format
The content is organized as follows:
1. This summary section
2. Repository information
3. Directory structure
4. Repository files (if enabled)
5. Multiple file entries, each consisting of:
  a. A header with the file path (## File: path/to/file)
  b. The full contents of the file in a code block

## Usage Guidelines
- This file should be treated as read-only. Any changes should be made to the
  original repository files, not this packed version.
- When processing this file, use the file path to distinguish
  between different files in the repository.
- Be aware that this file may contain sensitive information. Handle it with
  the same level of security as you would the original repository.

## Notes
- Some files may have been excluded based on .gitignore rules and Repomix's configuration
- Binary files are not included in this packed representation. Please refer to the Repository Structure section for a complete list of file paths, including binary files
- Files matching these patterns are excluded: **/*.csv, **/*.json, **/*.ipynb, **/__pycache__/, **/lightning_logs/**
- Files matching patterns in .gitignore are excluded
- Files matching default ignore patterns are excluded
- Content has been compressed - code blocks are separated by ⋮---- delimiter
- Files are sorted by Git change count (files with more changes are at the bottom)

# Directory Structure
```
config/
  base.py
src/
  dataset/
    data_preparation.py
  hpo/
    hyperparameter_tuning.py
  models/
    mlforecast/
      auto_models.py
    neuralforecast/
      auto_cfg.py
      auto_models.py
      models.py
    statsforecast/
      models.py
  pipelines/
    cv.py
    evaluate.py
    model_selection.py
    test.py
    train.py
  utils/
    hyperparam_loader.py
    other.py
    plot.py
.repomixignore
learn.md
main.py
main2.py
repomix.txt
```

# Files

## File: config/base.py
```python
"""
Configuration settings for neural forecasting pipeline.
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
# === Model Configuration ===
FREQUENCY = 'D'
LOCAL_SCALER_TYPE = 'standard'
⋮----
# === Cross-validation Configuration ===
CV_N_WINDOWS = 5
CV_STEP_SIZE = HORIZON  # Fixed: Using HORIZON instead of undefined HORIZON
⋮----
# === Hyperparameter Tuning Configuration ===
NUM_SAMPLES_PER_MODEL = 1
TUNING_RESULTS_DIR = 'tuning_results'
BEST_HYPERPARAMETERS_CSV = 'tuning_results/best_hyperparameters.csv'
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
RESULTS_DIR: str = 'enhanced_results'
HPO_DIR: str = f"{RESULTS_DIR}/hpo"
CV_DIR: str = f"{RESULTS_DIR}/cv"
FINAL_DIR: str = f"{RESULTS_DIR}/final"
MODELS_DIR: str = f"{RESULTS_DIR}/models"
⋮----
def __post_init__(self)
⋮----
"""Set default values that depend on other attributes."""
# Create output directories
```

## File: src/dataset/data_preparation.py
```python
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
"""Split data into development and final holdout test sets."""
⋮----
test_length = horizon * test_length_multiplier
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
```

## File: src/hpo/hyperparameter_tuning.py
```python
"""
Hyperparameter tuning module for neural forecasting models.
"""
⋮----
def run_hyperparameter_optimization(train_df, horizon=None, loss_fn=None, num_samples=None)
⋮----
"""Run hyperparameter optimization using AutoModels."""
⋮----
horizon = HORIZON
⋮----
loss_fn = MAE()
⋮----
num_samples = NUM_SAMPLES_PER_MODEL
⋮----
# Get auto models for HPO
automodels = get_auto_models(
⋮----
# Create NeuralForecast instance for HPO
nf_hpo = NeuralForecast(
⋮----
# Fit the models (performs HPO)
cv_df = nf_hpo.cross_validation(train_df)
⋮----
def extract_best_configurations(nf_hpo)
⋮----
"""Extract best configurations from HPO results."""
all_best_configs = []
⋮----
# Check if the model is an Auto model and has results
⋮----
model_name = model.__class__.__name__
⋮----
# Get the DataFrame of all trials for this model
results_df = model.results.get_dataframe()
⋮----
# Find the row with the lowest 'valid_loss'
best_run = results_df.loc[results_df['loss'].idxmin()]
⋮----
# Extract the 'config/' columns to get the hyperparameters
best_params = {
⋮----
# Add model name and best loss to the dictionary
⋮----
# Append to the list
⋮----
def save_best_configurations(all_best_configs, output_dir=None)
⋮----
"""Save best configurations to CSV file."""
⋮----
output_dir = TUNING_RESULTS_DIR
⋮----
best_configs_df = pd.DataFrame(all_best_configs)
⋮----
# Create directory if it doesn't exist
⋮----
csv_filename = os.path.join(output_dir, 'best_hyperparameters.csv')
⋮----
# Save to CSV
⋮----
def run_complete_hpo_pipeline(train_df, horizon=None, loss_fn=None, num_samples=None)
⋮----
"""Run the complete hyperparameter optimization pipeline."""
# Run HPO
nf_hpo = run_hyperparameter_optimization(
⋮----
# Extract best configurations
all_best_configs = extract_best_configurations(nf_hpo)
⋮----
# Save best configurations
csv_filename = save_best_configurations(all_best_configs)
⋮----
# Prepare data
```

## File: src/models/mlforecast/auto_models.py
```python
def get_mlforecast_models(self, model_names: List[str] = None) -> List[Tuple[str, Any]]
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
n_estimators = 50 if self.config.fast_mode else 200
max_depth = 8 if self.config.fast_mode else 15
⋮----
all_models = {
⋮----
# Add XGBoost if available
⋮----
# Return only essential models for fast testing
essential_models = ['RandomForest', 'LightGBM', 'Ridge']
```

## File: src/models/neuralforecast/auto_cfg.py
```python
def get_bitcoin_optimized_configs(h)
⋮----
"""
    Get model configurations optimized for Bitcoin price forecasting.
    Bitcoin exhibits high volatility, trending behavior, and potential regime changes.
    """
⋮----
# Enhanced NHITS config for crypto volatility
nhits_config = {
⋮----
'input_size': [h * 2, h * 3, h * 4, h * 6, h * 8],  # Longer lookback for crypto
⋮----
# Enhanced NBEATS config
nbeats_config = {
⋮----
# TFT config for complex temporal patterns
tft_config = {
⋮----
# LSTM config for sequential patterns
lstm_config = {
```

## File: src/models/neuralforecast/auto_models.py
```python
def get_auto_models(horizon: int, loss_fn=MAE(), num_samples_per_model: int = 10)
⋮----
"""
    Get set of auto models for Bitcoin forecasting.
    """
configs = get_bitcoin_optimized_configs(horizon)
⋮----
models = [
⋮----
# Trend-focused models
⋮----
# Complex pattern recognition
⋮----
# Sequential learning
⋮----
# Multiple loss functions for different aspects of Bitcoin forecasting
def get_loss_functions()
⋮----
"""
    Different loss functions for different forecasting objectives.
    """
⋮----
'mae': MAE(),           # Robust to outliers
'mse': MSE(),           # Penalizes large errors
'rmse': RMSE(),         # Scale-dependent, interpretable
'mape': MAPE(),         # Percentage errors
'smape': SMAPE(),       # Symmetric percentage errors
```

## File: src/models/neuralforecast/models.py
```python
# src/models/neuralforecast/enhanced_models.py
"""
Enhanced model configurations optimized for Bitcoin forecasting.
"""
⋮----
def get_neural_models(horizon: int, hist_exog_list=None)
⋮----
"""
    Get models with fixed hyperparameters for quick baseline.
    Useful for initial exploration or when computational resources are limited.
    """
models = [
⋮----
# Fast baseline models
⋮----
# Include exogenous variables if available
```

## File: src/models/statsforecast/models.py
```python
# src/models/statsforecast/enhanced_models.py
"""
Bitcoin-optimized statistical models for forecasting.
"""
⋮----
def get_bitcoin_optimized_models(season_length: int = 7)
⋮----
"""
    Optimized model selection for Bitcoin forecasting.
    
    Bitcoin characteristics:
    - Non-stationary (trending with potential structural breaks)
    - Non-linear (complex price dynamics)
    - High volatility (rapid price changes)
    
    Selected models based on their ability to handle these characteristics.
    """
⋮----
models = [
⋮----
# PRIMARY CANDIDATES - Best for Bitcoin characteristics
⋮----
# AutoARIMA: Excellent for non-stationary, trending data
# Can handle differencing, seasonality, and complex patterns
⋮----
stationary=False,  # Important for non-stationary Bitcoin
approximation=False,  # More thorough search
⋮----
# ETS Multiplicative models: Handle volatility and non-linearity better
AutoETS(season_length=season_length, model='ZZZ'),  # Auto-select (often picks multiplicative)
AutoETS(season_length=season_length, model='MMM'),  # Full multiplicative for high volatility
AutoETS(season_length=season_length, model='MAM'),  # Multiplicative trend, additive season
⋮----
# Theta methods: Excellent for trending, volatile financial data
⋮----
# Complex Exponential Smoothing: Handles complex patterns
⋮----
# SECONDARY CANDIDATES - Good baselines and complementary models
⋮----
# Random Walk with Drift: Natural for trending financial data
⋮----
# Short-term moving averages: Adapt quickly to volatility
WindowAverage(window_size=3),   # Very responsive to recent changes
WindowAverage(window_size=7),   # Weekly patterns
⋮----
# Seasonal patterns (if weekly seasonality exists)
⋮----
def get_bitcoin_ensemble_models(season_length: int = 7)
⋮----
"""
    Bitcoin-optimized ensemble components for forecasting.
    
    Returns the best performing individual models that complement each other
    in an ensemble approach for Bitcoin prediction.
    """
ensemble_components = [
⋮----
# Core trend-following models
⋮----
# Multiplicative ETS for volatility
⋮----
# Theta for trending behavior
⋮----
# Simple but effective baselines
⋮----
# REMOVED MODELS (not suitable for Bitcoin characteristics):
# - Naive(): Too simplistic for volatile financial data
# - HistoricAverage(): Ignores recent trends/volatility
# - Long window averages (14+): Too slow for high volatility
# - Croston family: Designed for intermittent demand, not financial data
# - ADIDA, IMAPA, TSB: For sparse/intermittent data
# - Simple additive ETS: Don't handle multiplicative volatility well
```

## File: src/pipelines/cv.py
```python
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
```

## File: src/pipelines/evaluate.py
```python
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
```

## File: src/pipelines/model_selection.py
```python
# src/pipelines/model_selection.py
"""
Comprehensive model selection and comparison framework.
"""
⋮----
class ModelComparison
⋮----
"""Comprehensive model comparison across different frameworks."""
⋮----
def __init__(self, results_dir: str = "model_comparison_results")
⋮----
def evaluate_neural_models(self, models, df_train, df_val, horizon: int) -> List[Dict]
⋮----
"""Evaluate neural models with cross-validation."""
results = []
⋮----
start_time = time.time()
⋮----
# Create NeuralForecast instance
nf = NeuralForecast(models=[model], freq='D')
⋮----
# Cross-validation
cv_results = nf.cross_validation(
⋮----
n_windows=3,  # Reduced for faster evaluation
⋮----
# Calculate metrics
metrics = self._calculate_metrics(cv_results, [model.__class__.__name__])
⋮----
# Training time
training_time = time.time() - start_time
⋮----
result = {
⋮----
def evaluate_statistical_models(self, models, df_train, df_val, horizon: int) -> List[Dict]
⋮----
"""Evaluate statistical models."""
⋮----
# Prepare data for StatsForecast
df_stats = df_train[['unique_id', 'ds', 'y']].copy()
⋮----
# Create StatsForecast instance
sf = StatsForecast(models=[model], freq='D', verbose=True)
⋮----
cv_results = sf.cross_validation(
⋮----
def _calculate_metrics(self, cv_results: pd.DataFrame, model_names: List[str]) -> Dict
⋮----
"""Calculate comprehensive metrics for cross-validation results."""
metrics = {}
⋮----
# Basic metrics
model_metrics = evaluate(
⋮----
# Additional custom metrics
y_true = cv_results['y'].values
y_pred = cv_results[model_name].values
⋮----
# Remove NaN values
mask = ~(np.isnan(y_true) | np.isnan(y_pred))
y_true_clean = y_true[mask]
y_pred_clean = y_pred[mask]
⋮----
# Directional accuracy
directional_accuracy = self._calculate_directional_accuracy(y_true_clean, y_pred_clean)
⋮----
# Prediction intervals coverage (if available)
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
def compare_all_models(self, neural_models, stat_models, df_train, df_val, horizon: int) -> pd.DataFrame
⋮----
"""Compare all models and return ranked results."""
⋮----
all_results = []
⋮----
# Evaluate neural models
⋮----
neural_results = self.evaluate_neural_models(neural_models, df_train, df_val, horizon)
⋮----
# Evaluate statistical models
⋮----
stat_results = self.evaluate_statistical_models(stat_models, df_train, df_val, horizon)
⋮----
# Create results DataFrame
results_df = pd.DataFrame(all_results)
⋮----
# Filter out failed models - fix indexing error
⋮----
successful_results = results_df[results_df['error'].isna()].copy()
⋮----
successful_results = results_df.copy()
⋮----
# Rank models by MAE (lower is better)
⋮----
# Combined rank (simple average)
⋮----
# Sort by combined rank
successful_results = successful_results.sort_values('combined_rank')
⋮----
# Save results
⋮----
def save_results(self, all_results: pd.DataFrame, successful_results: pd.DataFrame)
⋮----
"""Save comparison results."""
timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
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
"""Get top N model names for ensemble or further analysis."""
⋮----
# Usage example
def run_comprehensive_model_selection(df_train, df_val, horizon: int = 7)
⋮----
"""Run comprehensive model selection pipeline."""
⋮----
# Import model functions
⋮----
# Get models
neural_models = get_auto_models(horizon, num_samples_per_model=5)  # Reduced for faster testing
deterministic_models = get_neural_models(horizon)
stat_models = get_bitcoin_optimized_models()
⋮----
# Initialize comparison
comparison = ModelComparison()
⋮----
# Compare models
results = comparison.compare_all_models(
⋮----
# Get top models
top_models = comparison.get_top_models(results, top_n=5)
⋮----
row = results[results['model_name'] == model].iloc[0]
```

## File: src/pipelines/test.py
```python
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
```

## File: src/pipelines/train.py
```python
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
```

## File: src/utils/hyperparam_loader.py
```python
"""
Model configuration loader for neural forecasting models.
"""
⋮----
def load_best_hyperparameters(csv_path=None)
⋮----
"""Load best hyperparameters from CSV file."""
⋮----
csv_path = BEST_HYPERPARAMETERS_CSV
⋮----
loaded_best_configs_df = pd.read_csv(csv_path)
⋮----
def parse_hyperparameters(best_row, exclude_keys=None, json_keys=None)
⋮----
"""Parse hyperparameters from the best configuration row."""
⋮----
exclude_keys = EXCLUDE_HYPERPARAMETER_KEYS
⋮----
json_keys = JSON_PARSEABLE_KEYS
⋮----
best_params = {}
⋮----
# Attempt to parse values that are expected to be lists/nested lists
⋮----
def get_loss_object(best_row, loss_map=None)
⋮----
"""Get loss object from the loss string in the configuration."""
⋮----
loss_map = LOSS_MAP
⋮----
loss_string_from_csv = best_row.get('loss', 'MAE()')
final_loss_object = loss_map.get(loss_string_from_csv, MAE())
⋮----
def create_model_from_config(model_name, best_params, loss_object)
⋮----
"""Create a model instance from the best configuration."""
⋮----
def load_best_model_config(model_name='AutoNHITS', csv_path=None)
⋮----
"""Load the best model configuration for a specific model."""
# Load hyperparameters from CSV
loaded_best_configs_df = load_best_hyperparameters(csv_path)
⋮----
# Get the best configuration for the specified model
best_row = loaded_best_configs_df[loaded_best_configs_df['model_name'] == model_name].iloc[0]
⋮----
# Parse hyperparameters
best_params = parse_hyperparameters(best_row)
⋮----
# Get loss object
final_loss_object = get_loss_object(best_row)
⋮----
# Create model instance
model_instance = create_model_from_config(model_name, best_params, final_loss_object)
⋮----
# Test loading best model configuration
```

## File: src/utils/other.py
```python
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
```

## File: src/utils/plot.py
```python
class ForecastVisualizer
⋮----
"""
    A utility class for plotting and saving forecast visualizations using utilsforecast.plotting.plot_series.

    Attributes
    ----------
    Y_df : pd.DataFrame
        DataFrame of actual time series with columns ['unique_id', 'ds', 'y']
    Y_hat_df : pd.DataFrame
        DataFrame of forecasted values with columns ['unique_id', 'ds', 'y']
    """
⋮----
"""
        Generate a forecast vs. actual plot.

        Parameters
        ----------
        ids : List[int], optional
            List of series IDs to plot (default None = all)
        levels : List[int], optional
            Prediction intervals to display, e.g., [80, 95]
        max_insample_length : int, optional
            Number of past periods to display
        plot_anomalies : bool
            Whether to highlight anomalies outside intervals
        engine : str
            Backend engine ('matplotlib' or 'plotly')
        plot_random : bool
            If True, select random series to plot
        **kwargs
            Additional kwargs passed to plot_series

        Returns
        -------
        matplotlib.figure.Figure or plotly.graph_objs._figure.Figure
            The generated figure object
        """
fig = plot_series(
⋮----
"""
        Save the figure to disk.

        Parameters
        ----------
        fig : Figure
            The figure object returned by plot()
        save_dir : str
            Directory to save the figure
        filename : str
            File name for the saved image
        dpi : int
            Resolution in dots per inch

        Returns
        -------
        str
            Full path to the saved file
        """
⋮----
path = os.path.join(save_dir, filename)
⋮----
# Example
# --------
# from forecast_visualizer import ForecastVisualizer
# vis = ForecastVisualizer(Y_df, Y_hat_df)
# fig = vis.plot(ids=[0,1], levels=[80,95], max_insample_length=36, plot_anomalies=True)
# out_path = vis.save(fig, save_dir='outputs', filename='forecast.png')
# print(f"Plot saved to {out_path}")
```

## File: .repomixignore
```
# Add patterns to ignore here, one per line
# Example:
# *.log
# tmp/
```

## File: learn.md
```markdown
- input_size: number of read past points

- step_size: slide step, if step_size >= input_size -> no overlap

- windows_batch_size: process input_size together in batch then average out mistake
    -> too small: noisy
    -> too large: takes longer learning

- max_steps: num of train iterations (num_epoch)
- val_check_steps: freq in making validate -> smaller is better since spot quick early stopping but can be computation overhead
    -> these 2 often go with early_stop_patience_steps
```

## File: main.py
```python
"""
Main orchestration script for the neural forecasting pipeline.

This script coordinates the entire forecasting workflow:
1. Environment setup
2. Data preparation
3. Hyperparameter tuning (optional)
4. Cross-validation
5. Final model training
6. Prediction and evaluation

Usage:
    python forecasting_pipeline_main.py [--skip-hpo]
"""
⋮----
def main(skip_hpo=False)
⋮----
"""Run the complete neural forecasting pipeline."""
⋮----
# 1. Setup Environment
⋮----
ray_config = {
⋮----
# 2. Data Preparation
⋮----
# 3. Hyperparameter Tuning (Optional)
⋮----
# 4. Cross-Validation
⋮----
# 5. Final Model Training
⋮----
# 6. Prediction and Evaluation
⋮----
# 7. Final Summary
⋮----
# print(f"Model: {evaluation_results['model_name']}")
# print(f"Test MAE: {evaluation_results['test_mae']:.4f}")
# print(f"Test RMSE: {evaluation_results['test_rmse']:.4f}")
# print(f"Evaluation DataFrame shape: {evaluation_results['evaluation_df'].shape}")
⋮----
# 8. Plot visualization
vis = ForecastVisualizer(test_df, predictions_on_test)
fig = vis.plot(ids=[0,1], levels=[80,95], max_insample_length=36, plot_anomalies=True)
out_path = vis.save(fig, save_dir='outputs', filename='forecast.png')
⋮----
parser = argparse.ArgumentParser(description="Neural Forecasting Pipeline")
⋮----
args = parser.parse_args()
```

## File: main2.py
```python
# enhanced_main.py
"""
Enhanced main orchestration script for Bitcoin forecasting pipeline.

This improved pipeline includes:
1. Comprehensive model comparison across frameworks
2. Bitcoin-specific feature engineering
3. Systematic model selection
4. Ensemble methods
5. Advanced evaluation metrics
"""
⋮----
# Add src to path
⋮----
# Import configurations
⋮----
# Import enhanced modules
⋮----
class EnhancedBitcoinForecastingPipeline
⋮----
"""Enhanced Bitcoin forecasting pipeline with comprehensive model comparison."""
⋮----
def __init__(self, config: Dict = None)
⋮----
def _get_default_config(self) -> Dict
⋮----
"""Get default configuration."""
⋮----
'fast_mode': False,  # Use fewer models and samples for quick testing
⋮----
def setup_environment(self)
⋮----
"""Setup the forecasting environment."""
⋮----
# Create output directories
⋮----
def get_neural_models_for_comparison(self) -> Tuple[List, List]
⋮----
"""Get models for comprehensive comparison."""
neural_models = []
stat_models = []
⋮----
horizon = self.config['horizon']
num_samples = 3 if self.config['fast_mode'] else self.config['num_samples_per_model']
⋮----
# Neural models
⋮----
# Quick models for fast testing
neural_models = get_neural_models(horizon)
⋮----
# Comprehensive auto models
loss_functions = get_loss_functions()
⋮----
# Try different loss functions
for loss_name, loss_fn in list(loss_functions.items())[:2]:  # Limit for demo
models = get_comprehensive_auto_models(
⋮----
# Statistical models
⋮----
stat_models = get_bitcoin_optimized_models(season_length=7)
⋮----
# Reduce to essential statistical models
essential_stat_models = [
stat_models = essential_stat_models[:5]  # Limit for fast mode
⋮----
def run_model_selection(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> pd.DataFrame
⋮----
"""Run comprehensive model selection."""
⋮----
# Get models
⋮----
# Initialize comparison
comparison = ModelComparison(results_dir=HPO_DIR)
⋮----
# Run comparison
results_df = comparison.compare_all_models(
⋮----
# Store results
⋮----
"""Create ensemble from top performing models."""
⋮----
top_n = len(results_df)
⋮----
top_models = results_df.head(top_n)
⋮----
ensemble_info = {
⋮----
def _calculate_ensemble_weights(self, top_models: pd.DataFrame) -> List[float]
⋮----
"""Calculate ensemble weights based on performance."""
# Inverse MAE weighting (better models get higher weights)
mae_values = top_models['mae'].values
inverse_mae = 1 / mae_values
weights = inverse_mae / inverse_mae.sum()
⋮----
"""Evaluate final performance on holdout test set."""
⋮----
# This is a simplified version - in practice, you'd need to:
# 1. Retrain top models on full training data
# 2. Generate predictions on test set
# 3. Combine predictions using ensemble weights
# 4. Calculate final metrics
⋮----
# For now, return placeholder results
final_results = {
⋮----
'ensemble_mae': np.nan,  # Would calculate actual ensemble performance
⋮----
def generate_report(self) -> Dict
⋮----
"""Generate comprehensive analysis report."""
report = {
⋮----
def run_complete_pipeline(self) -> Dict
⋮----
"""Run the complete enhanced pipeline."""
⋮----
# 1. Setup
⋮----
# 2. Data preparation
⋮----
# 3. Model selection
# Create validation split from training data for model selection
val_split_size = min(len(train_df) // 4, self.config['horizon'] * 3)  # Use 25% or 3*horizon, whichever is smaller
train_for_selection = train_df.iloc[:-val_split_size].copy()
val_for_selection = train_df.iloc[-val_split_size:].copy()
⋮----
results_df = self.run_model_selection(train_for_selection, val_for_selection)
⋮----
# 4. Ensemble creation
⋮----
ensemble_info = self.create_ensemble(results_df, train_df)
⋮----
# 5. Final evaluation
final_performance = self.evaluate_final_performance(ensemble_info, train_df, test_df)
⋮----
# 6. Generate report
report = self.generate_report()
⋮----
def main()
⋮----
"""Main execution function."""
parser = argparse.ArgumentParser(description="Enhanced Bitcoin Forecasting Pipeline")
⋮----
args = parser.parse_args()
⋮----
# Configure pipeline
config = {
⋮----
# Run pipeline
pipeline = EnhancedBitcoinForecastingPipeline(config)
report = pipeline.run_complete_pipeline()
⋮----
# Print summary
```

## File: repomix.txt
```
repomix --ignore "**/*.csv,**/*.json,**/*.ipynb,**/__pycache__/,**/lightning_logs/**" --style markdown --compress
```
