from src.pipelines.feature_selection_pipeline import FeatureSelector
from src.dataset.data_preparation import prepare_pipeline_data
import pandas as pd

def main():
    """
    Main function to run the feature selection pipeline.
    """
    # 1. Load data
    train_df, test_df, hist_exog_list, data_info = prepare_pipeline_data()
    
    # For the purpose of feature selection, we can temporarily combine
    # train and test to have a complete dataframe, as the selector's
    # internal `run_complete_feature_selection_strategy` method
    # will handle the splitting internally to prevent leakage.
    full_df = pd.concat([train_df, test_df])

    # 2. Initialize the selector
    selector = FeatureSelector(random_state=42, verbose=True)

    # 3. Define enhanced parameters for a deeper search
    # Increase estimators for tree models and epochs for autoencoder for more robust results.
    deep_search_params = {
        'tree_n_estimators': 500,  # Increased from default 200
        'ae_epochs': 100,          # Increased from default 50
        'use_rfecv': True,         # Enable RFECV for optimal feature number detection
        'use_stability_selection': True # Enable Stability Selection for robustness check
    }

    # 4. Run the complete feature selection strategy
    # The method internally splits data into train/validation sets for robust selection
    # and ensures that all fitting is done only on the training portion.
    results = selector.run_complete_feature_selection_strategy(
        df=full_df,
        target_col='y',
        step3_params=deep_search_params
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
    output_path = "data/processed/final_feature_selected_data.parquet"
    
    # Save to parquet
    final_df.to_parquet(output_path, index=False)
    
    print(f"✅ Successfully saved final dataframe to '{output_path}'")
    print(f"   • Shape of the saved data: {final_df.shape}")
    print(f"   • Columns: {final_df.columns.tolist()}")

if __name__ == "__main__":
    main()
