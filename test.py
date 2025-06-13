import pandas as pd

# Step 1: Create dummy DataFrame with 'cutoff'
cv_stat_df = pd.DataFrame({
    'unique_id': ['A', 'B', 'C'],
    'ds': ['2025-01-01', '2025-01-02', '2025-01-03'],
    'cutoff': ['2024-12-31', '2025-01-01', '2025-01-02'],
    'model_stat': [0.1, 0.2, 0.3]
})

# Step 2: Create dummy DataFrame without 'cutoff'
ml_cv_df = pd.DataFrame({
    'unique_id': ['A', 'B', 'C'],
    'ds': ['2025-01-01', '2025-01-02', '2025-01-03'],
    'model_ml': [0.15, 0.25, 0.35]
})

# Step 3: Combine the DataFrames
cv_stat_neural_dfs = [cv_stat_df]
all_dfs = cv_stat_neural_dfs.copy()

if not ml_cv_df.empty:
    all_dfs.append(ml_cv_df)

# Step 4: Consolidate the DataFrames on ['unique_id', 'ds']
consolidated_cv_df = pd.DataFrame()

if all_dfs:
    print(f"Consolidating {len(all_dfs)} CV dataframes...")

    merge_keys = ['unique_id', 'ds']
    consolidated_cv_df = all_dfs[0].copy()

    for i in range(1, len(all_dfs)):
        df_to_merge = all_dfs[i]

        # Identify new columns to bring
        cols_to_bring = [col for col in df_to_merge.columns
                         if col not in consolidated_cv_df.columns and col not in merge_keys]

        merge_subset = df_to_merge[merge_keys + cols_to_bring]

        consolidated_cv_df = pd.merge(
            consolidated_cv_df,
            merge_subset,
            on=merge_keys,
            how='inner'
        )

    print("✅ Consolidated CV DataFrame:")
    print(consolidated_cv_df)
else:
    print("⚠️ No DataFrames to consolidate.")
