"""
Test script to verify the functionality of the back_transform_log_returns function.
"""
import pandas as pd
import numpy as np

# Adjust the path to import from the project's source directory
import sys
from pathlib import Path
sys.path.append(str(Path.cwd()))

from src.pipelines.model_evaluation import back_transform_log_returns
from src.dataset.data_preparation import prepare_pipeline_data
from src.utils.utils import extract_model_names_from_columns

def main():
    """Main function to run the test."""
    print("--- Starting Test for back_transform_log_returns ---")

    # 1. Load the sample CV dataframe that contains raw prices
    try:
        original_cv_df_prices = pd.read_csv('results/results_7d/cv/cv_df_nf.csv')
        print("✅ Successfully loaded sample CV data (in price scale).")
    except FileNotFoundError:
        print("❌ Error: Could not find 'results/results_7d/cv/cv_df_nf.csv'.")
        print("Please ensure you have run the pipeline to generate this file.")
        return

    # 2. Get the original, untransformed data
    _, _, _, _, original_df = prepare_pipeline_data()
    print("✅ Successfully loaded original price data.")

    # 3. Simulate a CV dataframe with log returns from the price data
    # This is the crucial step to create a valid test case.
    print("\n--- Simulating a log-return CV dataframe ---")
    log_cv_df = original_cv_df_prices.copy()
    
    # Ensure date columns are datetime objects for correct lookup
    log_cv_df['ds'] = pd.to_datetime(log_cv_df['ds'])
    log_cv_df['cutoff'] = pd.to_datetime(log_cv_df['cutoff'])
    
    # Get model names
    model_names = extract_model_names_from_columns(log_cv_df.columns)
    
    cols_to_transform = ['y'] + model_names

    # Group by each forecasting window
    grouped = log_cv_df.groupby(['unique_id', 'cutoff'])

    for name, group in grouped:
        # Get the last true price at the cutoff date
        cutoff_date = name[1]
        last_true_price = original_df.loc[original_df['ds'] == cutoff_date, 'y'].iloc[0]
        
        # Prepend this price to each series for correct diff calculation
        for col in cols_to_transform:
            price_series = pd.concat([pd.Series([last_true_price]), group[col]], ignore_index=True)
            log_return_series = np.log(price_series).diff().dropna()
            
            # Update the dataframe at the correct indices
            log_cv_df.loc[group.index, col] = log_return_series.values

    print("✅ Successfully created a simulated CV dataframe in log-return scale.")
    
    # 4. Perform the back-transformation
    print("\n--- Performing back-transformation ---")
    back_transformed_df = back_transform_log_returns(log_cv_df, original_df, model_names)

    # 5. Validate the results
    print("\n--- Validation ---")
    print("Original CV predictions (prices):")
    print(original_cv_df_prices.head())
    
    print("\nSimulated CV predictions (log returns):")
    print(log_cv_df.head())

    print("\nBack-transformed CV predictions (prices):")
    print(back_transformed_df.head())

    # Calculate the difference
    validation_cols = ['y'] + model_names
    diff_df = np.abs(original_cv_df_prices[validation_cols] - back_transformed_df[validation_cols])

    print("\nMean Absolute Difference between original and back-transformed:")
    print(diff_df.mean())
    
    # Check if the transformation was successful
    if diff_df.mean().mean() < 1e-6:
        print("\n✅ Test Passed: The back-transformation was successful and accurate.")
    else:
        print("\n❌ Test Failed: The difference is larger than expected.")

if __name__ == "__main__":
    main() 