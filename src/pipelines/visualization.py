"""
Module for creating visualizations in the Auto Models workflow.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict
from pathlib import Path
import json
from datetime import datetime
import glob

# Configuration and local imports
from config.base import HORIZON
from src.utils.utils import get_horizon_directories
from src.dataset.data_preparation import prepare_pipeline_data

def create_unified_forecast_plot(
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame, 
    all_forecasts_data: Dict,
    horizon: int,
    plot_dir: Path
):
    """Create unified forecast plot showing all models.
    
    Args:
        all_forecasts_data: Dict[str, Dict] 
            Format: {
                "common": {
                    "ds": np.ndarray,
                    "actual": np.ndarray
                },
                "models": {
                    "ModelName": {
                        "predictions": {
                            "mean": np.ndarray,
                            "lo": {"level": np.ndarray},
                            "hi": {"level": np.ndarray}
                        },
                        "forecast_method": str
                    },
                    ...
                }
            }
        horizon: int - forecast horizon in days
    """
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(16, 10))
    
    train_tail = train_df.tail(40).copy()
    if 'ds' in train_tail.columns:
        train_tail['ds'] = pd.to_datetime(train_tail['ds'])
    if 'ds' in test_df.columns:
        test_df = test_df.copy()
        test_df['ds'] = pd.to_datetime(test_df['ds'])

    # Historical values
    common_data = all_forecasts_data.get('common', {})
    actual_dates = pd.to_datetime(common_data.get('ds'))
    actual_values = common_data.get('actual')
    
    # Combine train_tail and test_df for a single continuous actuals line
    combined_actuals = pd.concat([train_tail[['ds', 'y']], test_df[['ds', 'y']]], ignore_index=True)
    ax.plot(combined_actuals['ds'], combined_actuals['y'], color='black', linewidth=2, label='Actual Values', alpha=0.9, zorder=9)

    # Models Prediction values
    colors = plt.cm.Set3(np.linspace(0, 1, len(all_forecasts_data['models'])))
    for i, (model_name, forecast_data) in enumerate(all_forecasts_data['models'].items()):
        predictions = forecast_data['predictions']['mean']
        ds_values = actual_dates # Use common ds for all models
        
        # Removed framework as it's not in the new data structure
        # Use a default linestyle for now
        linestyle = '-'
        linewidth = 2
        ax.plot(ds_values, predictions, color=colors[i], linewidth=linewidth, linestyle=linestyle, label=f'{model_name}', alpha=0.8)
        
        # # Add prediction intervals (if available)
        # if 'lo' in forecast_data['predictions'] and forecast_data['predictions']['lo'] and \
        #    'hi' in forecast_data['predictions'] and forecast_data['predictions']['hi']:
        #     # Assuming the first key in 'lo' and 'hi' is the confidence level to use
        #     lo_key = list(forecast_data['predictions']['lo'].keys())[0]
        #     hi_key = list(forecast_data['predictions']['hi'].keys())[0]
        #     lo_predictions = forecast_data['predictions']['lo'][lo_key]
        #     hi_predictions = forecast_data['predictions']['hi'][hi_key]
        #     ax.fill_between(
        #         x=ds_values,
        #         y1=lo_predictions,
        #         y2=hi_predictions,
        #         color=colors[i],
        #         alpha=0.2,
        #         label=f'{model_name} Interval ({lo_key}%) '
        #     )
    
    ax.set_title(f'Bitcoin Price Forecasting - Auto Models Direct Workflow\nHorizon: {horizon} days | Test Period: {len(test_df)} days', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Bitcoin Price (USD)', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    legend = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    legend.set_title('Models & Actual', prop={'size': 10, 'weight': 'bold'})
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    if len(train_tail) > 0 and len(test_df) > 0:
        split_date = test_df['ds'].iloc[0]
        ax.axvline(x=split_date, color='red', linestyle=':', linewidth=2, alpha=0.7, zorder=8)
        ax.annotate('Train/Test Split', xy=(split_date, ax.get_ylim()[1]), xytext=(10, -10), textcoords='offset points', fontsize=10, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.1), ha='left', va='top')
    
    if all_forecasts_data and 'models' in all_forecasts_data:
        info_text = f"Models Compared: {len(all_forecasts_data['models'])} \nTest Period: {len(test_df)} days\n"
        rolling_models = [name for name, data in all_forecasts_data['models'].items() if data.get('forecast_method') == 'rolling_forecast']
        if rolling_models:
            info_text += f"Rolling Forecast: {len(rolling_models)} models"
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plot_path = plot_dir / f'unified_{horizon}d.png'
    plt.savefig(plot_path, dpi=400, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"   • Forecast plot saved: {plot_path}")

def create_separate_probabilistic_forecast_plots(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    all_forecasts_data: Dict,
    horizon: int,
    plot_dir: Path
):
    """Create separate probabilistic forecast plots for each model.
    
    Args:
        all_forecasts_data: Dict[str, Dict] 
            Format: {
                "common": {
                    "ds": np.ndarray,
                    "actual": np.ndarray
                },
                "models": {
                    "ModelName": {
                        "predictions": {
                            "mean": np.ndarray,
                            "lo": {"level": np.ndarray},
                            "hi": {"level": np.ndarray}
                        },
                        "forecast_method": str
                    },
                    ...
                }
            }
        horizon: int - forecast horizon in days
    """
    plt.style.use('default')

    train_tail = train_df.tail(40).copy()
    if 'ds' in train_tail.columns:
        train_tail['ds'] = pd.to_datetime(train_tail['ds'])
    if 'ds' in test_df.columns:
        test_df = test_df.copy()
        test_df['ds'] = pd.to_datetime(test_df['ds'])

    # Historical actual values for all plots
    common_data = all_forecasts_data.get('common', {})
    actual_dates = pd.to_datetime(common_data.get('ds'))
    actual_values = common_data.get('actual')

    for model_name, forecast_data in all_forecasts_data['models'].items():
        if 'predictions' in forecast_data and 'mean' in forecast_data['predictions'] and \
           'lo' in forecast_data['predictions'] and forecast_data['predictions']['lo'] and \
           'hi' in forecast_data['predictions'] and forecast_data['predictions']['hi']:
            fig, ax = plt.subplots(figsize=(16, 8))
            ds_values = actual_dates # Use common ds for all models
            mean_predictions = forecast_data['predictions']['mean']
            # Assuming the first key in 'lo' and 'hi' is the confidence level to use
            lo_key = list(forecast_data['predictions']['lo'].keys())[0]
            hi_key = list(forecast_data['predictions']['hi'].keys())[0]
            lo_predictions = forecast_data['predictions']['lo'][lo_key]
            hi_predictions = forecast_data['predictions']['hi'][hi_key]
            forecast_method = forecast_data.get('forecast_method', 'unknown')

            # Combine train_tail and test_df for a single continuous actuals line
            combined_actuals = pd.concat([train_tail[['ds', 'y']], test_df[['ds', 'y']]], ignore_index=True)
            ax.plot(combined_actuals['ds'], combined_actuals['y'], color='black', linewidth=2, label='Actual Values', alpha=0.7)

            # Plot mean prediction
            color = plt.cm.Set1(np.random.rand()) # Assign a random color for each model
            ax.plot(ds_values, mean_predictions, color=color, linewidth=2, linestyle='-', label=f'{model_name} (Mean)', alpha=0.8)

            # Plot prediction intervals
            ax.fill_between(
                x=ds_values,
                y1=lo_predictions,
                y2=hi_predictions,
                color=color,
                alpha=0.3,
                label=f'Prediction Interval ({lo_key}%)'
            )

            # Adjust Y-axis to focus on the mean and actuals if intervals are too wide
            all_y_values = np.concatenate([
                combined_actuals['y'].values,
                np.array(mean_predictions)
            ])
            np_lo_predictions = np.array(lo_predictions)
            np_hi_predictions = np.array(hi_predictions)

            interval_range = np_hi_predictions.max() - np_lo_predictions.min()
            main_range = all_y_values.max() - all_y_values.min()

            # Heuristic: if interval is > 20x larger than main data range, adjust axis
            if main_range > 0 and (interval_range / main_range) > 20:
                y_min = all_y_values.min()
                y_max = all_y_values.max()
                padding = (y_max - y_min) * 0.2  # 20% padding
                ax.set_ylim(y_min - padding, y_max + padding)
                print(f"   • Note: Y-axis for {model_name} was clipped to show detail due to wide prediction intervals.")

            ax.set_title(f'{model_name} Forecast ({forecast_method})\nHorizon: {horizon} days | Test Period: {len(test_df)} days', fontsize=14, fontweight='bold')
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Bitcoin Price (USD)', fontsize=12)
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            ax.legend(fontsize=10)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

            if len(train_tail) > 0 and len(test_df) > 0:
                split_date = test_df['ds'].iloc[0]
                ax.axvline(x=split_date, color='red', linestyle=':', linewidth=1.5, alpha=0.7, zorder=8)
                ax.annotate('Train/Test Split', xy=(split_date, ax.get_ylim()[1]), xytext=(10, -10), textcoords='offset points', fontsize=10, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.1), ha='left', va='top')

            plt.tight_layout()
            plot_path = plot_dir / f'{model_name}_{horizon}d.png'
            plt.savefig(plot_path, dpi=400, bbox_inches='tight', facecolor='white')
            plt.close(fig) # Close the figure to free up memory
            print(f"   • Probabilistic forecast plot for {model_name} saved: {plot_path}")
        else:
            print(f"Warning: Probabilistic forecast data (mean, lo, hi) not found for model: {model_name}. Skipping probabilistic plot.")

def create_visualizations_step(train_df: pd.DataFrame, test_df: pd.DataFrame, all_forecasts_data: Dict, horizon: int, plot_dir: Path):
    """Step 3: Create forecast visualizations wrapper."""
    if not all_forecasts_data:
        print("❌ No forecasts available for plotting")
        return
    try:
        plot_dir.mkdir(parents=True, exist_ok=True)
        print(f"   • Saving plots to: {plot_dir}")
        create_unified_forecast_plot(train_df, test_df, all_forecasts_data, horizon, plot_dir)
        create_separate_probabilistic_forecast_plots(train_df, test_df, all_forecasts_data, horizon, plot_dir)
        print("✅ Forecast visualizations created successfully")
    except Exception as e:
        print(f"❌ Error creating visualizations: {str(e)}")

def create_cv_individual_plots(cv_dir: Path, horizon: int):
    """
    Create individual cross-validation plots for each model from merged CV results.
    Uses complete actual values from prepare_pipeline_data() for better visual comparison.
    
    Args:
        cv_dir: Directory containing CV results
        horizon: Forecast horizon in days
    """
    print(f"📊 Creating cross-validation plots for {horizon}d horizon...")
    
    # Read merged CV results
    cv_file = cv_dir / 'cv_df.csv'
    
    if not cv_file.exists():
        print(f"❌ Required CV file not found: {cv_file}")
        return
    
    try:
        # Get complete training data for actual values
        print(f"   • Loading complete training data...")
        # We need the original, untransformed data for plotting.
        # The returned train_df would be transformed, so we use original_df instead.
        _, _, _, _, original_df = prepare_pipeline_data(horizon=horizon, apply_transformations=True)
        
        # Read CV data
        df_merged = pd.read_csv(cv_file)
        
        print(f"   • Loaded merged CV data: {cv_file.name}")
        print(f"   • Complete training data: {len(original_df):,} samples")
        print(f"   • CV data shape: {df_merged.shape}")
        
        # Convert date columns
        df_merged['ds'] = pd.to_datetime(df_merged['ds'])
        df_merged['cutoff'] = pd.to_datetime(df_merged['cutoff'])
        original_df['ds'] = pd.to_datetime(original_df['ds'])
        
        # Identify model columns (exclude metadata columns)
        exclude_cols = {'unique_id', 'ds', 'cutoff', 'y'}
        model_cols = [col for col in df_merged.columns if col not in exclude_cols and not col.endswith('-lo-95') and not col.endswith('-hi-95')]
        
        print(f"   • Found models: {model_cols}")
        
        # Create plots directory inside cv_dir
        cv_plot_dir = cv_dir / 'plot'
        cv_plot_dir.mkdir(parents=True, exist_ok=True)
        
        # Create individual plot for each model
        for model_name in model_cols:
            create_single_cv_plot(df_merged, model_name, cv_plot_dir, horizon, original_df)
            
        print(f"✅ Cross-validation plots created successfully in {cv_plot_dir}")
        
    except Exception as e:
        print(f"❌ Error creating CV plots: {str(e)}")

def create_single_cv_plot(df: pd.DataFrame, model_name: str, plot_dir: Path, horizon: int, all_data_df: pd.DataFrame):
    """
    Create a single cross-validation plot for a specific model.
    
    Args:
        df: Merged DataFrame with all CV results
        model_name: Name of the model to plot
        plot_dir: Directory to save plots
        horizon: Forecast horizon in days
        all_data_df: Complete DataFrame with all actual values (untransformed)
    """
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Get unique cutoff dates (CV windows)
    cutoff_dates = sorted(df['cutoff'].unique())
    
    # Use a color palette for different CV windows
    colors = plt.cm.tab20(np.linspace(0, 1, len(cutoff_dates)))
    
    # Get CV prediction date range
    cv_start_date = df['ds'].min()
    cv_end_date = df['ds'].max()
    
    # Get last 20 days of training data before CV starts
    context_days = 20
    context_start_date = cv_start_date - pd.Timedelta(days=context_days)
    
    # Filter training data for context period
    context_train = all_data_df[
        (all_data_df['ds'] >= context_start_date) & 
        (all_data_df['ds'] < cv_start_date)
    ].copy()
    
    # Get actual values for CV prediction period from train_df
    cv_actual = all_data_df[
        (all_data_df['ds'] >= cv_start_date) & 
        (all_data_df['ds'] <= cv_end_date)
    ].copy()
    
    # Combine context and CV actual data for continuous line
    combined_actual = pd.concat([context_train, cv_actual], ignore_index=True).sort_values('ds')
    
    # Plot focused actual values line (last 20 days + CV period)
    ax.plot(combined_actual['ds'], combined_actual['y'], 
           color='black', linewidth=2.5, label='Actual Values', alpha=0.9, zorder=10)
    
    # Plot each CV window predictions
    for i, cutoff_date in enumerate(cutoff_dates):
        window_data = df[df['cutoff'] == cutoff_date].sort_values('ds')
        
        # Plot model predictions
        ax.plot(window_data['ds'], window_data[model_name], 
               color=colors[i], linewidth=1.5, alpha=0.7, 
               label=f'CV Window {i+1}' if i < 5 else None)  # Only label first 5 windows
        
        # # Add prediction intervals if available
        # lo_col = f'{model_name}-lo-95'
        # hi_col = f'{model_name}-hi-95'
        # if lo_col in window_data.columns and hi_col in window_data.columns:
        #     ax.fill_between(window_data['ds'], window_data[lo_col], window_data[hi_col],
        #                    color=colors[i], alpha=0.1)
    
    # Styling
    ax.set_title(f'{model_name} - Cross-Validation Results\n{len(cutoff_dates)} CV Windows | Horizon: {horizon} days', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Bitcoin Price (USD)', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Legend (only show subset to avoid cluttering)
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) > 10:  # If too many handles, show only actual + first few CV windows
        ax.legend(handles[:6], labels[:6], bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    else:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # Format dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # Add info text
    info_text = f"Model: {model_name}\nCV Windows: {len(cutoff_dates)}\nHorizon: {horizon} days"
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10, 
           verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    plot_path = plot_dir / f'{model_name}_cv_{horizon}d.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"   • Saved CV plot: {plot_path.name}")

def create_cv_visualizations_main():
    """
    Main function to create cross-validation visualizations using current horizon configuration.
    """
    print("🚀 Starting Cross-Validation Visualization...")
    
    # Get directories based on current HORIZON
    cv_dir, final_dir, plot_dir = get_horizon_directories()
    
    print(f"   • Using horizon: {HORIZON} days")
    print(f"   • CV directory: {cv_dir}")
    
    # Create CV plots
    create_cv_individual_plots(cv_dir, HORIZON)
    
    print("✅ Cross-validation visualization complete!")

# def main():
#     # Dummy data for testing
#     dates = pd.to_datetime(pd.date_range(start='2022-01-01', periods=100, freq='D'))
#     train_df = pd.DataFrame({
#         'ds': dates[:80],
#         'y': np.random.rand(80) * 100 + 1000,
#         'unique_id': ['1']*80
#     })
#     test_df = pd.DataFrame({
#         'ds': dates[80:],
#         'y': np.random.rand(20) * 100 + 1000,
#         'unique_id': ['1']*20
#     })

#     # Load forecast_results.json
#     try:
#         with open('forecast_results.json', 'r') as f:
#             all_forecasts_data = json.load(f)
#     except FileNotFoundError:
#         print("Error: forecast_results.json not found. Please ensure it's in the same directory.")
#         return
#     except json.JSONDecodeError:
#         print("Error: Could not decode forecast_results.json. Check file format.")
#         return

#     if 'common' in all_forecasts_data and 'ds' in all_forecasts_data['common']:
#         all_forecasts_data['common']['ds'] = pd.to_datetime(all_forecasts_data['common']['ds'])

#     # Use HORIZON from config instead of timestamp
#     horizon = HORIZON

#     print("Starting visualization test with dummy data...")
#     create_visualizations_step(train_df, test_df, all_forecasts_data, horizon)
#     print("Visualization test complete.")

if __name__ == "__main__":
    create_cv_visualizations_main() 