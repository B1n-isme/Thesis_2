"""
Script to back-transform a CV dataframe from log returns to prices.
"""
import pandas as pd
import numpy as np

# Adjust the path to import from the project's source directory
import sys
from pathlib import Path
sys.path.append(str(Path.cwd()))

from src.pipelines.model_evaluation import back_transform_log_returns
from src.dataset.data_preparation import prepare_pipeline_data
from src.utils.utils import extract_model_names_from_columns, process_cv_results

def main():
    """Main function to run the transformation."""
    print("--- Starting back-transformation of CV data ---")

    # 1. Load the sample CV dataframe that contains log-return forecasts
    cv_path = 'results/results_90d/cv'
    log_cv_df_path = f'{cv_path}/cv_df_nf.csv'
    log_cv_df = pd.read_csv(log_cv_df_path)
    print(f"✅ Successfully loaded CV data (log-return scale) from '{log_cv_df_path}'.")

    # 2. Get the original, untransformed price data
    # This data is used to get the last known price before each forecast window.
    _, _, _, _, original_df = prepare_pipeline_data()
    print("✅ Successfully loaded original price data.")

    # 3. Perform the back-transformation
    print("\n--- Performing back-transformation ---")
    
    # Extract model names from the dataframe columns
    model_names = extract_model_names_from_columns(log_cv_df.columns)
    if not model_names:
        print("❌ Error: No model columns found in the dataframe. Check column names for 'Auto' prefix.")
        return
        
    # Call the transformation function
    back_transformed_df = back_transform_log_returns(log_cv_df, original_df, model_names)

    # 4. Debug and process the back-transformed results
    print("\n--- Debugging and Processing Results ---")

    # Check for any remaining large or invalid values after clipping
    for model in model_names:
        inf_count = np.isinf(back_transformed_df[model]).sum()
        nan_count = np.isnan(back_transformed_df[model]).sum()
        if inf_count > 0 or nan_count > 0:
            print(f"⚠️  Model '{model}' still contains {inf_count} inf and {nan_count} NaN values.")

    # Calculate metrics on the back-transformed (price scale) data
    cv_metrics = process_cv_results(back_transformed_df)
    print("\n--- Calculated CV Metrics (Price Scale) ---")
    print(cv_metrics)

    # 5. Save the results
    print("\n--- Saving Results ---")
    
    # # Save the transformed dataframe for inspection
    # transformed_output_path = f'{cv_path}/cv_df_nf_prices.csv'
    # back_transformed_df.to_csv(transformed_output_path, index=False)
    # print(f"✅ Back-transformed data saved for inspection to '{transformed_output_path}'")
    
    # Save the final, corrected metrics
    metrics_output_path = f'{cv_path}/cv_metrics_nf.csv'
    cv_metrics.to_csv(metrics_output_path, index=False)
    print(f"✅ Corrected CV metrics saved to '{metrics_output_path}'")


if __name__ == "__main__":
    main() 