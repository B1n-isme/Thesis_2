from src.pipelines.feature_selection_pipeline import FeatureSelector
from src.dataset.data_preparation import prepare_pipeline_data, load_and_prepare_data
import pandas as pd
from config.base import RAW_DATA_PATH, HORIZON
from pathlib import Path
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Feature Selection Pipeline')
    parser.add_argument('--tree_methods', nargs='+', default=['xgboost', 'lightgbm', 'random_forest'],
                       help='Tree-based methods to use for feature selection')
    parser.add_argument('--min_consensus_level', type=int, default=2,
                       help='Minimum consensus level for feature selection')
    parser.add_argument('--handle_multicollinearity', action='store_true',
                       help='Enable multicollinearity handling')
    parser.add_argument('--n_bootstrap', type=int, default=50,
                       help='Number of bootstrap samples for stability selection')
    parser.add_argument('--selection_threshold', type=float, default=0.6,
                       help='Threshold for feature selection')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Load data
    train_df, test_df, hist_exog_list, data_info, _ = prepare_pipeline_data(
        data_path=RAW_DATA_PATH,
        apply_transformations=True
    )

    # Data cleaning
    print("\nCleaning training data by removing constant columns...")
    cols_before = train_df.columns.tolist()
    non_constant_mask = train_df.nunique() > 1
    train_df = train_df.loc[:, non_constant_mask]
    cols_after = train_df.columns.tolist()
    removed_cols = sorted(list(set(cols_before) - set(cols_after)))

    if removed_cols:
        print(f"  - Removed {len(removed_cols)} constant column(s): {removed_cols}")
        test_df = test_df[cols_after]
        hist_exog_list = [col for col in hist_exog_list if col in cols_after]
    else:
        print("  - No constant columns found.")

    # 2. Initialize selector
    selector = FeatureSelector(random_state=42, verbose=True, use_gpu=True)
    results_dir = "src/pipelines/feature_results"
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    print(f"Intermediate results will be saved to: {results_dir}")

    # 3. Set selection params from args
    selection_params = {
        'tree_methods': args.tree_methods,
        'min_consensus_level': args.min_consensus_level,
        'handle_multicollinearity_flag': args.handle_multicollinearity,
        'n_bootstrap': args.n_bootstrap,
        'selection_threshold': args.selection_threshold
    }

    # 4. Run feature selection
    results = selector.run_complete_feature_selection_strategy(
        df=train_df,
        target_col='y',
        step3_params=selection_params,
        results_dir=results_dir
    )

    # 5. Print results
    print("\n--- Feature Selection Results ---")
    consensus_features = results.get('final_recommendations', {}).get('consensus_features', [])
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

    # 6. Export results
    print("\nExporting data with selected features...")
    full_original_df = load_and_prepare_data(data_path=RAW_DATA_PATH)
    final_df = full_original_df[['unique_id', 'ds', 'y'] + consensus_features]
    
    multicoll_postfix = "_mc" if selection_params['handle_multicollinearity_flag'] else ""
    output_path = f"data/processed/feature_selection_{HORIZON}{multicoll_postfix}.parquet"
    final_df.to_parquet(output_path, index=False)
    
    print(f"✅ Successfully saved final dataframe to '{output_path}'")
    print(f"   • Shape: {final_df.shape}")
    print(f"   • Columns: {final_df.columns.tolist()}")

if __name__ == "__main__":
    main()
