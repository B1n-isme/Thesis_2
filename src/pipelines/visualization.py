"""
Module for creating visualizations in the Auto Models workflow.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict
from pathlib import Path
from config.base import HORIZON
from src.utils.utils import get_horizon_directories
from src.dataset.data_preparation import prepare_pipeline_data

def _prepare_plot_data(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple:
    """Helper to prepare common plotting data."""
    train_tail = train_df.tail(40).copy()
    if 'ds' in train_tail.columns:
        train_tail['ds'] = pd.to_datetime(train_tail['ds'])
    if 'ds' in test_df.columns:
        test_df = test_df.copy()
        test_df['ds'] = pd.to_datetime(test_df['ds'])
    return train_tail, test_df

def _save_plot(fig, plot_path: Path, dpi: int = 400):
    """Helper to save plots with consistent settings."""
    plt.tight_layout()
    fig.savefig(plot_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"   • Plot saved: {plot_path}")

def create_unified_forecast_plot(train_df: pd.DataFrame, test_df: pd.DataFrame, 
                               all_forecasts_data: Dict, horizon: int, plot_dir: Path):
    """Create unified forecast plot showing all models."""
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(16, 10))
    train_tail, test_df = _prepare_plot_data(train_df, test_df)

    # Plot actual values
    combined_actuals = pd.concat([train_tail[['ds', 'y']], test_df[['ds', 'y']]])
    ax.plot(combined_actuals['ds'], combined_actuals['y'], color='black', 
           linewidth=2, label='Actual Values', alpha=0.9, zorder=9)

    # Plot model predictions
    colors = plt.cm.Set3(np.linspace(0, 1, len(all_forecasts_data['models'])))
    for i, (model_name, forecast_data) in enumerate(all_forecasts_data['models'].items()):
        ds_values = pd.to_datetime(all_forecasts_data['common']['ds'])
        ax.plot(ds_values, forecast_data['predictions']['mean'], color=colors[i],
               linewidth=2, linestyle='-', label=model_name, alpha=0.8)

    # Plot styling
    ax.set_title(f'Bitcoin Price Forecasting\nHorizon: {horizon} days | Test Period: {len(test_df)} days', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Bitcoin Price (USD)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add train/test split line if data exists
    if len(train_tail) > 0 and len(test_df) > 0:
        split_date = test_df['ds'].iloc[0]
        ax.axvline(x=split_date, color='red', linestyle=':', linewidth=2, alpha=0.7, zorder=8)

    _save_plot(fig, plot_dir / f'unified_{horizon}d.png')

def create_cv_individual_plots(cv_dir: Path, horizon: int):
    """Create cross-validation plots for each model."""
    cv_file = cv_dir / 'cv_df.csv'
    if not cv_file.exists():
        print(f"❌ Required CV file not found: {cv_file}")
        return

    _, _, _, _, original_df = prepare_pipeline_data(horizon=horizon, apply_transformations=True)
    df_merged = pd.read_csv(cv_file)
    df_merged['ds'] = pd.to_datetime(df_merged['ds'])
    original_df['ds'] = pd.to_datetime(original_df['ds'])

    model_cols = [col for col in df_merged.columns 
                 if col not in {'unique_id', 'ds', 'cutoff', 'y'} 
                 and not col.endswith(('-lo-95', '-hi-95'))]

    cv_plot_dir = cv_dir / 'plot'
    cv_plot_dir.mkdir(parents=True, exist_ok=True)
    
    for model_name in model_cols:
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Plot actual values
        ax.plot(original_df['ds'], original_df['y'], color='black', 
               linewidth=2.5, label='Actual Values', alpha=0.9, zorder=10)
        
        # Plot model predictions
        cutoff_dates = sorted(df_merged['cutoff'].unique())
        colors = plt.cm.tab20(np.linspace(0, 1, len(cutoff_dates)))
        for i, cutoff_date in enumerate(cutoff_dates):
            window_data = df_merged[df_merged['cutoff'] == cutoff_date].sort_values('ds')
            ax.plot(window_data['ds'], window_data[model_name], color=colors[i],
                   linewidth=1.5, alpha=0.7, label=f'CV Window {i+1}' if i < 5 else None)

        ax.set_title(f'{model_name} - Cross-Validation Results\n{len(cutoff_dates)} CV Windows', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Bitcoin Price (USD)', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        _save_plot(fig, cv_plot_dir / f'{model_name}_cv_{horizon}d.png', dpi=300)

def create_cv_visualizations_main():
    """Main function to create cross-validation visualizations."""
    cv_dir, _, _ = get_horizon_directories()
    create_cv_individual_plots(cv_dir, HORIZON)

if __name__ == "__main__":
    create_cv_visualizations_main()