import pandas as pd
from src.utils.utils import get_horizon_directories

cv_dir, _, _ = get_horizon_directories()

# Read the DataFrames
cv_df_nf = pd.read_csv(f"{cv_dir}/cv_df_nf.csv")
cv_df_stats = pd.read_csv(f"{cv_dir}/cv_df_stats.csv")
cv_metrics_nf = pd.read_csv(f"{cv_dir}/cv_metrics_nf.csv")
cv_metrics_stats = pd.read_csv(f"{cv_dir}/cv_metrics_stats.csv")

# Merge cv_df_nf and cv_df_stats on common columns, handling duplicate 'y' column
cv_df = pd.merge(
    cv_df_stats, 
    cv_df_nf.drop(columns=['y']), 
    on=["unique_id", "ds", "cutoff"], 
    how="outer"
)

# Concatenate cv_metrics_nf and cv_metrics_stats
cv_metrics = pd.concat([cv_metrics_nf, cv_metrics_stats], ignore_index=True)

# save the results
cv_df.to_csv(f"{cv_dir}/cv_df.csv", index=False)
cv_metrics.to_csv(f"{cv_dir}/cv_metrics.csv", index=False)