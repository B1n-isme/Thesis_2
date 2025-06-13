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

# Configuration
from config.base import HORIZON, PLOT_DIR, FINAL_DIR

def create_unified_forecast_plot(
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame, 
    all_forecasts_data: Dict,
    horizon: int
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
        info_text = f"Models Compared: {len(all_forecasts_data['models'])}Test Period: {len(test_df)} days\n"
        rolling_models = [name for name, data in all_forecasts_data['models'].items() if data.get('forecast_method') == 'rolling_forecast']
        if rolling_models:
            info_text += f"Rolling Forecast: {len(rolling_models)} models"
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plot_path = FINAL_DIR / f'unified_{horizon}d.png'
    plt.savefig(plot_path, dpi=400, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"   • Forecast plot saved: {plot_path}")

def create_separate_probabilistic_forecast_plots(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    all_forecasts_data: Dict,
    horizon: int
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
            plot_path = PLOT_DIR / f'{model_name}_{horizon}d.png'
            plt.savefig(plot_path, dpi=400, bbox_inches='tight', facecolor='white')
            plt.close(fig) # Close the figure to free up memory
            print(f"   • Probabilistic forecast plot for {model_name} saved: {plot_path}")
        else:
            print(f"Warning: Probabilistic forecast data (mean, lo, hi) not found for model: {model_name}. Skipping probabilistic plot.")

def create_visualizations_step(train_df: pd.DataFrame, test_df: pd.DataFrame, all_forecasts_data: Dict, horizon: int):
    """Step 3: Create forecast visualizations wrapper."""
    print("\n📈 STEP 3: CREATING VISUALIZATIONS")
    print("-" * 40)
    if not all_forecasts_data:
        print("❌ No forecasts available for plotting")
        return
    try:
        create_unified_forecast_plot(train_df, test_df, all_forecasts_data, horizon)
        create_separate_probabilistic_forecast_plots(train_df, test_df, all_forecasts_data, horizon)
        print("✅ Forecast visualizations created successfully")
    except Exception as e:
        print(f"❌ Error creating visualizations: {str(e)}")

def main():
    # Dummy data for testing
    dates = pd.to_datetime(pd.date_range(start='2022-01-01', periods=100, freq='D'))
    train_df = pd.DataFrame({
        'ds': dates[:80],
        'y': np.random.rand(80) * 100 + 1000,
        'unique_id': ['1']*80
    })
    test_df = pd.DataFrame({
        'ds': dates[80:],
        'y': np.random.rand(20) * 100 + 1000,
        'unique_id': ['1']*20
    })

    # Load forecast_results.json
    try:
        with open('forecast_results.json', 'r') as f:
            all_forecasts_data = json.load(f)
    except FileNotFoundError:
        print("Error: forecast_results.json not found. Please ensure it's in the same directory.")
        return
    except json.JSONDecodeError:
        print("Error: Could not decode forecast_results.json. Check file format.")
        return

    if 'common' in all_forecasts_data and 'ds' in all_forecasts_data['common']:
        all_forecasts_data['common']['ds'] = pd.to_datetime(all_forecasts_data['common']['ds'])

    # Use HORIZON from config instead of timestamp
    horizon = HORIZON

    print("Starting visualization test with dummy data...")
    create_visualizations_step(train_df, test_df, all_forecasts_data, horizon)
    print("Visualization test complete.")

if __name__ == "__main__":
    main() 