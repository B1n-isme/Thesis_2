"""
Script to generate final forecast visualizations from saved results.
"""
import glob
from pathlib import Path
import json

from src.utils.utils import get_horizon_directories, load_json_to_dict
from src.dataset.data_preparation import prepare_pipeline_data
from src.pipelines.visualization import create_separate_forecast_plots
from config.base import HORIZON


def main():
    """
    Main function to load forecast results and generate plots.
    """
    print("🚀 Starting Final Visualization Generation...")

    # Get directories based on current HORIZON configuration
    _, final_dir, plot_dir = get_horizon_directories()

    # Find the latest final_plot_results.json file
    search_pattern = str(final_dir / "final_plot_results*.json")
    list_of_files = glob.glob(search_pattern)
    
    if not list_of_files:
        print(f"❌ Error: No 'final_plot_results' JSON file found in {final_dir}")
        return

    latest_file = max(list_of_files, key=Path)
    print(f"   • Loading forecast data from: {latest_file}")

    # Load the forecast results data from the JSON file
    try:
        all_forecasts_data = load_json_to_dict(Path(latest_file))
    except json.JSONDecodeError:
        print(f"❌ Error: Could not decode JSON from {latest_file}. Check the file for corruption.")
        return

    # Load train and test dataframes for plotting context.
    # We use apply_transformations=False to ensure 'y' is in the original price scale for both dataframes.
    print("   • Loading training and test data for plot context...")
    train_df, test_df, _, _, _ = prepare_pipeline_data(
        horizon=HORIZON, 
        apply_transformations=False
    )

    # Create the unified forecast plot
    print("   • Generating unified forecast plot...")
    create_separate_forecast_plots(
        train_df=train_df,
        test_df=test_df,
        all_forecasts_data=all_forecasts_data,
        horizon=HORIZON,
        plot_dir=plot_dir
    )

    print("\n✅ Final visualization script completed successfully!")


if __name__ == "__main__":
    main() 