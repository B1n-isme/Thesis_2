from src.pipelines.feature_selection_pipeline import FeatureSelector
from src.dataset.data_preparation import prepare_pipeline_data
import pandas as pd
from config.base import RAW_DATA_PATH, HORIZON
from pathlib import Path


def main():
    """
    Main function to run the feature selection pipeline.
    """
    # 1. Load data
    train_df, test_df, hist_exog_list, data_info = prepare_pipeline_data(data_path=RAW_DATA_PATH)

    # 2. Initialize the selector
    selector = FeatureSelector(random_state=42, verbose=True, use_gpu=True)
    
    # Define the directory for saving intermediate results
    results_dir = "src/pipelines/feature_results"
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    print(f"Intermediate feature selection results will be saved to: {results_dir}")

    # 3. Define parameters for the new workflow
    # Parameters are mainly for Stability Selection and PFI.
    selection_params = {
        'tree_methods': ['xgboost', 'lightgbm', 'random_forest'],
        'n_bootstrap': 50,          # For Stability Selection
        'selection_threshold': 0.6, # For Stability Selection
        'n_repeats_permutation': 10 # For PFI
    }

    # 4. Run the complete feature selection strategy
    # The method internally splits data into train/validation sets for robust selection
    # and ensures that all fitting is done only on the training portion.
    results = selector.run_complete_feature_selection_strategy(
        df=train_df,
        target_col='y',
        step3_params=selection_params,
        results_dir=results_dir
    )

    # 5. Print and review the results
    print("\n--- Feature Selection Results ---")
    print("Top consensus features recommended by the pipeline:")
    consensus_features = results.get('final_recommendations', {}).get('consensus_features', [])
    
    # Detailed feature counts
    recommendation_df = results.get('final_recommendations', {}).get('feature_counts')
    if recommendation_df is not None:
        print("\n--- Feature Selection Counts ---")
        print(recommendation_df.to_string())

    if not consensus_features:
        print("\nNo features met the consensus criteria.")
    else:
        for i, feature in enumerate(consensus_features):
            print(f"{i+1}. {feature}")
        
    print(f"\nTotal features selected: {len(consensus_features)}")
    
    # 6. Export the final dataframes with selected features
    print("\nExporting data with selected features...")
    
    # Filter the original train and test sets
    final_train_df = train_df[['unique_id', 'ds', 'y'] + consensus_features]
    final_test_df = test_df[['unique_id', 'ds', 'y'] + consensus_features]
    
    # Combine them for a final single file output
    final_df = pd.concat([final_train_df, final_test_df])
    
    # Define output path
    output_path = f"data/processed/feature_selection_{HORIZON}.parquet"
    
    # Save to parquet
    final_df.to_parquet(output_path, index=False)
    
    print(f"✅ Successfully saved final dataframe to '{output_path}'")
    print(f"   • Shape of the saved data: {final_df.shape}")
    print(f"   • Columns: {final_df.columns.tolist()}")

if __name__ == "__main__":
    main()
