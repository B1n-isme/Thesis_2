import pandas as pd

from config.base import DATA_PATH
from src.utils.utils import process_cv_results, get_horizon_directories
from src.dataset.data_preparation import prepare_pipeline_data


def main():
    """
    Reads cross-validation data, calculates metrics, and saves them to a CSV file.
    """
    cv_dir, _, _ = get_horizon_directories()
    # Define file paths
    input_cv_path = f'{cv_dir}/cv_df.csv'
    input_metrics_path = f'{cv_dir}/cv_metrics.csv'
    output_metrics_path = f'{cv_dir}/cv_metrics_2.csv'

    # Read the cross-validation dataframe
    try:
        cv_df = pd.read_csv(input_cv_path)
        print(f"Successfully loaded CV data from '{input_cv_path}'.")
    except FileNotFoundError:
        print(f"Error: The file '{input_cv_path}' was not found.")
        return

    train_df, test_df, hist_exog_list, data_info, original_df = prepare_pipeline_data(
        data_path=DATA_PATH, 
        apply_transformations=False)

    # Process the CV results to get metrics
    metrics_df = process_cv_results(cv_df, train_df)

    # Merge training time from existing metrics
    if not metrics_df.empty:
        try:
            existing_metrics = pd.read_csv(input_metrics_path)
            metrics_df = metrics_df.merge(
                existing_metrics[['model_name', 'training_time']],
                on='model_name',
                how='left'
            )
        except FileNotFoundError:
            print(f"Warning: Could not merge training_time - '{input_metrics_path}' not found")

    print(metrics_df)
    # Save the metrics to a new CSV file
    if not metrics_df.empty:
        metrics_df.to_csv(output_metrics_path, index=False)
        print(f"Metrics successfully calculated and saved to '{output_metrics_path}'.")
    else:
        print("Metric calculation resulted in an empty DataFrame. No file was saved.")

    

if __name__ == "__main__":
    main() 