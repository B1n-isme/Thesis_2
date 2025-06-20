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
from src.utils.utils import extract_model_names_from_columns

def main():
    """Main function to run the transformation."""
    print("--- Starting back-transformation of CV data ---")

    # 1. Load the sample CV dataframe that contains log-return forecasts
    try:
        # This should be the path to your cross-validation data in log-return scale.
        log_cv_df_path = 'results/results_7d/cv/cv_df_nf.csv'
        log_cv_df = pd.read_csv(log_cv_df_path)
        print(f"✅ Successfully loaded CV data (log-return scale) from '{log_cv_df_path}'.")
    except FileNotFoundError:
        print(f"❌ Error: Could not find '{log_cv_df_path}'.")
        print("Please ensure the file exists and the path is correct.")
        return

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

    # 4. Display and save the results
    print("\n--- Results ---")
    print("Original CV predictions (log returns):")
    print(log_cv_df.head())

    print("\nBack-transformed CV predictions (prices):")
    print(back_transformed_df.head())

    # Save the transformed dataframe
    output_path = Path(log_cv_df_path).parent / f"{Path(log_cv_df_path).stem}_prices.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    back_transformed_df.to_csv(output_path, index=False)
    print(f"\n✅ Back-transformed data saved to '{output_path}'")


if __name__ == "__main__":
    main() 