import pandas as pd
import numpy as np
import warnings
from src.pipelines.feature_selection_pipeline import FeatureSelector

# Suppress the benign UserWarning from sklearn/lightgbm
warnings.filterwarnings("ignore", category=UserWarning)

def run_final_validation():
    """
    Loads the feature set after multicollinearity removal and runs only
    the final permutation importance validation step.
    """
    print("--- Loading Intermediate Feature Data ---")
    input_path = "data/final/final_feature_selected_data.parquet"
    try:
        df = pd.read_parquet(input_path)
    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_path}'")
        print("Please run the main feature selection pipeline first.")
        return

    print(f"Loaded data with {df.shape[1] - 3} features from '{input_path}'")

    # --- Recreate the exact same conditions as the main pipeline ---
    # 1. Add the temporary stationary target
    df['target_stationary'] = np.log(df['y']).diff()
    
    # 2. Recreate the train/validation split
    df = df.sort_values('ds').reset_index(drop=True)
    split_idx = int(len(df) * 0.9)
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]

    # 3. Get the list of features to validate
    features_to_validate = [
        col for col in df.columns if col not in ['unique_id', 'ds', 'y', 'target_stationary']
    ]

    print(f"\n--- Running Final Permutation Importance Validation ---")
    print(f"Validating {len(features_to_validate)} features...")

    # 4. Initialize the selector and run the validation method directly
    selector = FeatureSelector(random_state=42, verbose=True)
    
    pfi_results = selector.permutation_importance_validation(
        train_df=train_df,
        val_df=val_df,
        selected_features=features_to_validate,
        n_repeats=10
    )

    final_features = pfi_results.get('selected_features', [])

    # --- Print Final Results ---
    print("\n--- Permutation Importance Validation Complete ---")
    print(f"Final validated feature count: {len(final_features)}")
    print("Final feature list:")
    for i, feature in enumerate(final_features):
        print(f"{i+1}. {feature}")

    # --- Export the final dataset ---
    if final_features:
        output_path = "data/processed/final_feature_selected_data.parquet"
        print(f"\nExporting final validated dataset to '{output_path}'...")
        
        # Select the final features plus the identifier and target columns from the original df
        final_df = df[['unique_id', 'ds', 'y'] + final_features]
        
        # Save to parquet
        final_df.to_parquet(output_path, index=False)
        
        print(f"✅ Successfully saved final dataframe.")
        print(f"   • Shape of the saved data: {final_df.shape}")
        print(f"   • Columns: {final_df.columns.tolist()}")

if __name__ == "__main__":
    run_final_validation() 