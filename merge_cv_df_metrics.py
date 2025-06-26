import pandas as pd
from src.utils.utils import get_horizon_directories
import glob

# cv_dir, _, _ = get_horizon_directories()
cv_dir = 'results/results_90d/cv'

# Read and merge all DataFrames
base_df = None

for f in glob.glob(f"{cv_dir}/cv_df_*.csv"):
    df = pd.read_csv(f)
    if base_df is None:
        base_df = df
    else:
        base_df = pd.merge(base_df, df.drop(columns=['y']),
                         on=["unique_id", "ds", "cutoff"],
                         how="outer")

cols = [c for c in base_df.columns if not c.startswith('Auto') and not c.endswith('-95')] + \
[c for c in base_df.columns if c.startswith('Auto') and c != 'AutoNBEATSx' and not c.endswith('-95')]
# print(cols)
base_df = base_df[cols]
print(base_df.columns)


base_df.to_csv(f"{cv_dir}/cv_df.csv", index=False)


# cv_df_nf = pd.read_csv(f"{cv_dir}/cv_df_0_20250620_202043.csv")
# cv_df_stats = pd.read_csv(f"{cv_dir}/cv_df_0_20250622_025002.csv")
# cv_metrics_nf = pd.read_csv(f"{cv_dir}/cv_metrics_nf.csv")
# cv_metrics_stats = pd.read_csv(f"{cv_dir}/cv_metrics_stats.csv")

# Merge cv_df_nf and cv_df_stats on common columns, handling duplicate 'y' column
# cv_df = pd.merge(
#     cv_df_stats, 
#     cv_df_nf.drop(columns=['y']), 
#     on=["unique_id", "ds", "cutoff"], 
#     how="outer"
# )


# Concatenate cv_metrics_nf and cv_metrics_stats
# cv_metrics = pd.concat([cv_metrics_nf, cv_metrics_stats], ignore_index=True)

# save the results
# cv_df.to_csv(f"{cv_dir}/cv_df.csv", index=False)
# cv_metrics.to_csv(f"{cv_dir}/cv_metrics.csv", index=False)
