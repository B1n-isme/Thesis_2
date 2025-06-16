import pandas as pd
from config.base import HORIZON
from src.utils.utils import get_horizon_directories

CV_DIR, FINAL_DIR, PLOT_DIR = get_horizon_directories()

# # Load the CSV files
# metrics_df = pd.read_csv(f'{FINAL_DIR}/metrics_results.csv')
# cv_df = pd.read_csv(f'{CV_DIR}/cv_metrics.csv')

# # Check if both dataframes have the same number of rows
# if len(metrics_df) != len(cv_df):
#     print(f"Warning: Different number of rows - metrics_df: {len(metrics_df)}, cv_df: {len(cv_df)}")
# else:
#     print(f"Both files have {len(metrics_df)} rows")

# # Add the training_time column row-wise
# metrics_df['training_time'] = metrics_df['training_time'] + cv_df['training_time']

# # Save the updated dataframe
# output_file = f'{FINAL_DIR}/metrics_results.csv'
# metrics_df.to_csv(output_file, index=False)

# print(f"Updated CSV saved to: {output_file}")
# print("\nFirst few rows of updated data:")
# print(metrics_df[['model_name', 'training_time']].head()) 

metrics_df = pd.read_csv('results/results_7d/final/metrics_results.csv')

print(metrics_df[['model_name', 'training_time']]) 