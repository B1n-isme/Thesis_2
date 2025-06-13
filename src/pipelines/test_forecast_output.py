"""
Test file for validating the forecast processing logic from model_forecasting.py
Uses dummy data to test the exact dictionary construction logic.
"""
import pandas as pd
import numpy as np
import json
from datetime import datetime

# Dummy data setup - Copied from visualization.py
dates = pd.to_datetime(pd.date_range(start='2022-01-01', periods=100, freq='D'))
train_df = pd.DataFrame({
    'ds': dates[:80],
    'y': np.random.rand(80) * 100 + 1000,
    'unique_id': ['1']*80
})
test_df = pd.DataFrame({
    'ds': dates[80:],
    'y': np.random.rand(20) * 100 + 1000,
    'unique_id': ['1']*20
})

# Mock model outputs - make them smoother continuations of the actuals
# The new test_df has 20 data points, so adjust prediction generation accordingly
horizon_length_test = len(test_df)
arima_pred_base = np.linspace(test_df['y'].iloc[0], test_df['y'].iloc[-1] + 5, horizon_length_test) # Adjust trend continuation
arima_pred_noise = np.random.normal(0, 5, horizon_length_test) # Adjust noise level
arima_mean = arima_pred_base + arima_pred_noise

# Define interval spread
interval_spread_90 = 10.0 # Adjust as needed for visual spread
stat_forecasts_df = pd.DataFrame({
    'unique_id': ['1']*horizon_length_test,
    'ds': test_df['ds'],
    'AutoARIMA': arima_mean,
    'AutoARIMA-lo-90': arima_mean - interval_spread_90,
    'AutoARIMA-hi-90': arima_mean + interval_spread_90
})

# NHITS predictions
nhits_pred_base = np.linspace(test_df['y'].iloc[0] + 2, test_df['y'].iloc[-1] + 7, horizon_length_test) # Slightly different trend
nhits_pred_noise = np.random.normal(0, 6, horizon_length_test)
nhits_mean = nhits_pred_base + nhits_pred_noise

interval_spread_80 = 8.0 # Adjust as needed
neural_forecasts_df = pd.DataFrame({
    'unique_id': ['1']*horizon_length_test,
    'ds': test_df['ds'],
    'NHITS': nhits_mean,
    'NHITS-lo-80': nhits_mean - interval_spread_80,
    'NHITS-hi-80': nhits_mean + interval_spread_80
})

# Configuration constants (matching original and adjusted for new test_df length)
LEVELS = [80, 90]
HORIZON = horizon_length_test # Should be len(test_df)
ENABLE_ROLLING_FORECAST = False

# Initialize the results dictionary
final_plot_results_dict = {
    'common': {
        'ds': test_df['ds'].values,
        'actual': test_df['y'].values
    },
    'models': {}
}

# Mock consolidated forecasts (matches original merging logic)
consolidated_forecasts_df = test_df[['unique_id', 'ds', 'y']].copy()
consolidated_forecasts_df = consolidated_forecasts_df.merge(
    stat_forecasts_df, how='left', on=['unique_id', 'ds'])
consolidated_forecasts_df = consolidated_forecasts_df.merge(
    neural_forecasts_df, how='left', on=['unique_id', 'ds'])

# Extract model names (matches original processing)
model_columns = [col for col in consolidated_forecasts_df.columns 
                if col not in {'unique_id', 'ds', 'y'}]
model_names = set()
for col in model_columns:
    if '-lo-' in col or '-hi-' in col:
        base_name = col.split('-lo-')[0].split('-hi-')[0]
    else:
        base_name = col
    model_names.add(base_name)

# Process each model (EXACT copy from model_forecasting.py)
for model_name in model_names:
    # Extract prediction intervals
    lo_preds = {}
    hi_preds = {}
    for level in LEVELS:
        lo_col = f"{model_name}-lo-{level}"
        hi_col = f"{model_name}-hi-{level}"
        if lo_col in consolidated_forecasts_df.columns:
            lo_preds[str(level)] = consolidated_forecasts_df[lo_col].values
        if hi_col in consolidated_forecasts_df.columns:
            hi_preds[str(level)] = consolidated_forecasts_df[hi_col].values
    
    # Store in dictionary (matches original structure)
    final_plot_results_dict['models'][model_name] = {
        'predictions': {
            'mean': consolidated_forecasts_df[model_name].values,
            'lo': lo_preds,
            'hi': hi_preds
        },
        'forecast_method': 'rolling_forecast' if ENABLE_ROLLING_FORECAST else 'direct_forecast'
    }

# Validation
print("=== Dictionary Structure Validation ===")
print(final_plot_results_dict)
# print(f"Common keys: {list(final_plot_results_dict['common'].keys())}")
# print(f"Models: {list(final_plot_results_dict['models'].keys())}")
# print("\nSample Model Data (AutoARIMA):")
# print(final_plot_results_dict['models']['AutoARIMA'])
# print("\nSample Model Data (NHITS):")
# print(final_plot_results_dict['models']['NHITS']) 

# Convert the entire dictionary to be JSON serializable
def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic): # For single numpy values like np.float64
        return obj.item()
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, datetime):
        return obj.isoformat() # Handle standard datetime objects if any
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(x) for x in obj]
    return obj

serializable_dict = convert_to_serializable(final_plot_results_dict)

# Export to JSON
output_json_path = 'forecast_results.json'
with open(output_json_path, 'w') as f:
    json.dump(serializable_dict, f, indent=2)

print(f"Successfully exported to {output_json_path}")

# --- Demonstration of loading JSON and converting 'ds' back to datetime64[ns] ---
print("\n--- Testing JSON Load and Type Conversion ---")

# Load the JSON file back
with open(output_json_path, 'r') as f:
    loaded_data = json.load(f)

print("Type of 'ds' after JSON load (should be list of strings):", type(loaded_data['common']['ds'][0]))

# Convert 'ds' back to datetime64[ns]
if 'common' in loaded_data and 'ds' in loaded_data['common']:
    loaded_data['common']['ds'] = pd.to_datetime(loaded_data['common']['ds'])

print("Type of 'ds' after pd.to_datetime (should be numpy.datetime64):", type(loaded_data['common']['ds'][0]))
print("Converted 'ds' values:", loaded_data['common']['ds'])

print("--- Validation Complete ---")