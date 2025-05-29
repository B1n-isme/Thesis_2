"""
Module for creating visualizations in the Auto Models workflow.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict
from pathlib import Path

# Configuration
from config.base import HORIZON

def create_unified_forecast_plot(
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame, 
    all_forecasts_data: Dict, 
    plot_dir: Path, 
    pipeline_timestamp: pd.Timestamp
):
    """Create unified forecast plot showing all models."""
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(16, 10))
    
    train_tail = train_df.tail(60).copy()
    if 'ds' in train_tail.columns:
        train_tail['ds'] = pd.to_datetime(train_tail['ds'])
    if 'ds' in test_df.columns:
        test_df = test_df.copy()
        test_df['ds'] = pd.to_datetime(test_df['ds'])

    # Historical values
    actual_dates = pd.concat([train_tail['ds'], test_df['ds']], ignore_index=True)
    actual_values = pd.concat([train_tail['y'], test_df['y']], ignore_index=True)
    ax.plot(actual_dates, actual_values, color='black', linewidth=3, label='Actual Values', alpha=0.9, zorder=10)
    ax.plot(test_df['ds'], test_df['y'], color='black', linewidth=0, marker='o', markersize=5, label='Actual (Test Period)', alpha=0.9, zorder=11)

    # Models Prediction values
    colors = plt.cm.Set3(np.linspace(0, 1, len(all_forecasts_data)))
    for i, (model_name, forecast_data) in enumerate(all_forecasts_data.items()):
        predictions, ds_values = forecast_data['predictions']['mean'], forecast_data['ds']
        framework = forecast_data['framework']
        if isinstance(ds_values[0], str):
            ds_values = pd.to_datetime(ds_values)
        linestyle = '-' if framework == 'neural' else '--'
        linewidth = 2 if framework == 'neural' else 1.5
        ax.plot(ds_values, predictions, color=colors[i], linewidth=linewidth, linestyle=linestyle, label=f'{model_name} ({framework})', alpha=0.8)
    
    ax.set_title(f'Bitcoin Price Forecasting - Auto Models Direct Workflow\nHorizon: {HORIZON} days | Test Period: {len(test_df)} days', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Bitcoin Price (USD)', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    legend = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    legend.set_title('Models & Actual', prop={'size': 12, 'weight': 'bold'})
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    if len(train_tail) > 0 and len(test_df) > 0:
        split_date = test_df['ds'].iloc[0]
        ax.axvline(x=split_date, color='red', linestyle=':', linewidth=2, alpha=0.7, zorder=8)
        ax.annotate('Train/Test Split', xy=(split_date, ax.get_ylim()[1]), xytext=(10, -10), textcoords='offset points', fontsize=10, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.1), ha='left', va='top')
    
    if all_forecasts_data:
        info_text = f"Models Compared: {len(all_forecasts_data)}\nTest Period: {len(test_df)} days\n"
        rolling_models = [name for name, data in all_forecasts_data.items() if data.get('forecast_method') == 'rolling_forecast']
        if rolling_models:
            info_text += f"Rolling Forecast: {len(rolling_models)} models"
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plot_path = plot_dir / f'auto_models_forecast_{pipeline_timestamp.strftime("%Y%m%d_%H%M%S")}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"   • Forecast plot saved: {plot_path}")

def create_separate_probabilistic_forecast_plots(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    all_forecasts_data: Dict,
    plot_dir: Path,
    pipeline_timestamp: pd.Timestamp
):
    """Create separate probabilistic forecast plots for each model."""
    plt.style.use('default')

    train_tail = train_df.tail(60).copy()
    if 'ds' in train_tail.columns:
        train_tail['ds'] = pd.to_datetime(train_tail['ds'])
    if 'ds' in test_df.columns:
        test_df = test_df.copy()
        test_df['ds'] = pd.to_datetime(test_df['ds'])

    # Historical actual values for all plots
    actual_dates = pd.concat([train_tail['ds'], test_df['ds']], ignore_index=True)
    actual_values = pd.concat([train_tail['y'], test_df['y']], ignore_index=True)

    for model_name, forecast_data in all_forecasts_data.items():
        if 'predictions' in forecast_data and 'mean' in forecast_data['predictions'] and 'lo' in forecast_data['predictions'] and 'hi' in forecast_data['predictions']:
            fig, ax = plt.subplots(figsize=(16, 8))
            ds_values = forecast_data['ds']
            mean_predictions = forecast_data['predictions']['mean']
            lo_predictions = forecast_data['predictions']['lo']
            hi_predictions = forecast_data['predictions']['hi']
            framework = forecast_data['framework']
            forecast_method = forecast_data.get('forecast_method', 'unknown')

            if isinstance(ds_values[0], str):
                ds_values = pd.to_datetime(ds_values)

            # Plot actual values
            ax.plot(actual_dates, actual_values, color='black', linewidth=2, label='Actual Values', alpha=0.7)
            ax.plot(test_df['ds'], test_df['y'], color='black', linewidth=0, marker='o', markersize=4, alpha=0.7, label='Actual (Test Period)')

            # Plot mean prediction
            color = plt.cm.Set1(np.random.rand()) # Assign a random color for each model
            linestyle = '-' if framework == 'neural' else '--'
            linewidth = 2 if framework == 'neural' else 1.5
            ax.plot(ds_values, mean_predictions, color=color, linewidth=linewidth, linestyle=linestyle, label=f'{model_name} (Mean - {framework})', alpha=0.8)

            # Plot prediction intervals
            ax.fill_between(
                x=ds_values,
                y1=lo_predictions,
                y2=hi_predictions,
                color=color,
                alpha=0.3,
                label='Prediction Interval'
            )

            ax.set_title(f'{model_name} Forecast - {framework} ({forecast_method})\nHorizon: {HORIZON} days | Test Period: {len(test_df)} days', fontsize=14, fontweight='bold')
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
            plot_path = plot_dir / f'{model_name}_forecast_{pipeline_timestamp.strftime("%Y%m%d_%H%M%S")}.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close(fig) # Close the figure to free up memory
            print(f"   • Probabilistic forecast plot for {model_name} saved: {plot_path}")
        else:
            print(f"Warning: Probabilistic forecast data (mean, lo, hi) not found for model: {model_name}. Skipping probabilistic plot.")

def create_visualizations_step(train_df: pd.DataFrame, test_df: pd.DataFrame, all_forecasts_data: Dict, plot_dir: Path, pipeline_timestamp: pd.Timestamp):
    """Step 3: Create forecast visualizations wrapper."""
    print("\n📈 STEP 3: CREATING VISUALIZATIONS")
    print("-" * 40)
    if not all_forecasts_data:
        print("❌ No forecasts available for plotting")
        return
    try:
        create_unified_forecast_plot(train_df, test_df, all_forecasts_data, plot_dir, pipeline_timestamp)
        create_separate_probabilistic_forecast_plots(train_df, test_df, all_forecasts_data, plot_dir, pipeline_timestamp)
        print("✅ Forecast visualizations created successfully")
    except Exception as e:
        print(f"❌ Error creating visualizations: {str(e)}") 