# src/pipelines/final_fit_predict.py
"""
Final fit and predict framework for generating future forecasts and visualization.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import time
import json
from pathlib import Path
import traceback
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

from neuralforecast import NeuralForecast
from statsforecast import StatsForecast
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mae, mse, rmse, mape, smape

from config.base import *

class FinalFitPredictor:
    """Final fit and predict for generating future forecasts with visualization."""
    
    def __init__(self, results_dir: str = FINAL_DIR):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.plot_dir = Path(PLOT_DIR)
        self.plot_dir.mkdir(exist_ok=True)
        self.all_forecasts = {}  # Store all forecasts for unified plotting
        
    def _rolling_forecast_neural(self, nf, train_df, test_df, horizon: int, model_name: str) -> pd.DataFrame:
        """Perform rolling forecast for neural models when horizon < test_df length."""
        print(f"    Performing rolling forecast for {model_name} (horizon={horizon}, test_length={len(test_df)})")
        
        all_forecasts = []
        current_train_df = train_df.copy()  # Keep all columns from original training data
        
        # Identify exogenous columns (all columns except 'unique_id', 'ds', 'y')
        exog_cols = [col for col in test_df.columns if col not in ['unique_id', 'ds', 'y']]
        futr_cols = ['unique_id', 'ds'] + exog_cols  # Columns needed for prediction
        
        print(f"      Exogenous features: {exog_cols}")
        
        # Calculate number of rolling windows needed
        test_length = len(test_df)
        n_windows = (test_length + horizon - 1) // horizon  # Ceiling division
        
        for window in range(n_windows):
            start_idx = window * horizon
            end_idx = min(start_idx + horizon, test_length)
            
            # Get the current test window
            current_test_window = test_df.iloc[start_idx:end_idx].copy()
            
            print(f"      Window {window + 1}/{n_windows}: forecasting {len(current_test_window)} steps")
            
            # Predict for current window - include all exogenous features
            futr_df = current_test_window[futr_cols].copy()
            window_forecast = nf.predict(futr_df=futr_df)
            all_forecasts.append(window_forecast)
            
            # Update training data with actual values from current window for next iteration
            if window < n_windows - 1:  # Don't update after last window
                # Add actual values to training data (keep all columns for consistency)
                window_actual = current_test_window.copy()
                current_train_df = pd.concat([current_train_df, window_actual], ignore_index=True)
                
                # Re-fit the model with updated training data
                print(f"        Re-fitting model with {len(current_train_df)} samples...")
                nf.fit(current_train_df, verbose=False)
        
        # Combine all forecasts
        combined_forecasts = pd.concat(all_forecasts, ignore_index=True)
        return combined_forecasts
    
    def fit_predict_neural_models(self, models, train_df, test_df, horizon: int) -> List[Dict]:
        """Fit neural models on full training data and predict on test data."""
        results = []
        
        if not models:
            print("No neural models provided for fit-predict")
            return results
        
        print(f"Fitting and predicting {len(models)} neural models...")
        
        # # Debug: Show data structure
        # print(f"  Train data columns: {list(train_df.columns)}")
        # print(f"  Test data columns: {list(test_df.columns)}")
        # print(f"  Train data shape: {train_df.shape}")
        # print(f"  Test data shape: {test_df.shape}")
        
        # Determine if rolling forecast is needed
        test_length = len(test_df)
        # ENABLE_ROLLING_FORECAST = False
        use_rolling = ENABLE_ROLLING_FORECAST and horizon < test_length
        
        if use_rolling:
            print(f"  Using rolling forecast (horizon={horizon} < test_length={test_length})")
        else:
            print(f"  Using direct multi-step forecast (horizon={horizon} >= test_length={test_length})")
        
        for i, model in enumerate(models, 1):
            model_name = model.__class__.__name__
            print(f"[{i}/{len(models)}] Fit-Predict {model_name}...")
            start_time = time.time()
            
            try:
                # Create NeuralForecast instance
                nf = NeuralForecast(models=[model], freq='D')
                
                # Fit on full training data
                nf.fit(train_df, verbose=True)
                
                # Choose forecasting method based on horizon vs test length
                if use_rolling:
                    # Rolling forecast
                    forecasts = self._rolling_forecast_neural(nf, train_df, test_df, horizon, model_name)
                    evaluation_method = "rolling_forecast"
                else:
                    # Direct multi-step forecast
                    # Include all exogenous features for prediction
                    exog_cols = [col for col in test_df.columns if col not in ['unique_id', 'ds', 'y']]
                    futr_cols = ['unique_id', 'ds'] + exog_cols
                    futr_df = test_df[futr_cols].copy()
                    forecasts = nf.predict(futr_df=futr_df)
                    evaluation_method = "direct_forecast"
                
                # Merge with actual test values for evaluation
                eval_df = test_df.merge(forecasts, on=['unique_id', 'ds'], how='inner')
                
                if eval_df.empty:
                    raise ValueError("No matching forecasts and actual values for evaluation")
                
                # Store forecast for unified plotting
                self.all_forecasts[model_name] = {
                    'framework': 'neural',
                    'predictions': forecasts[model_name].values,
                    'ds': forecasts['ds'].values,
                    'unique_id': forecasts['unique_id'].values,
                    'eval_df': eval_df,  # Store eval_df for plotting actual vs predicted
                    'forecast_method': evaluation_method
                }
                
                # Calculate metrics from forecast results
                metrics = self._calculate_test_metrics(eval_df, [model_name])
                
                # Training time
                training_time = time.time() - start_time
                
                # Check if metrics were calculated successfully
                if model_name not in metrics or 'error' in metrics[model_name]:
                    raise ValueError(f"Failed to calculate metrics for {model_name}")
                
                result = {
                    'model_name': model_name,
                    'framework': 'neuralforecast',
                    'training_time': training_time,
                    'evaluation_method': evaluation_method,
                    'is_auto': 'Auto' in model_name,
                    'status': 'success',
                    **metrics[model_name]
                }
                
                results.append(result)
                print(f"  ✓ {model_name} {evaluation_method} completed (Test MAE: {metrics[model_name].get('mae', 'N/A'):.4f})")
                
            except Exception as e:
                training_time = time.time() - start_time
                error_msg = str(e)
                print(f"  ✗ Error in fit-predict for {model_name}: {error_msg}")
                
                results.append({
                    'model_name': model_name,
                    'framework': 'neuralforecast',
                    'training_time': training_time,
                    'error': error_msg,
                    'status': 'failed',
                    'is_auto': 'Auto' in model_name,
                    'mae': np.nan,
                    'rmse': np.nan,
                    'mape': np.nan
                })
        
        successful_count = sum(1 for r in results if r.get('status') == 'success')
        print(f"Neural models fit-predict completed: {successful_count}/{len(models)} successful")
        
        return results
    
    def fit_predict_statistical_models(self, models, train_df, test_df, horizon: int) -> List[Dict]:
        """Fit statistical models on training data and predict on test data."""
        results = []
        
        if not models:
            print("No statistical models provided for fit-predict")
            return results
        
        print(f"Fitting and predicting {len(models)} statistical models...")
        
        # Prepare data for StatsForecast
        df_train = train_df[['unique_id', 'ds', 'y']].copy()
        df_test = test_df[['unique_id', 'ds']].copy()
        
        for i, model in enumerate(models, 1):
            model_name = model.__class__.__name__
            print(f"[{i}/{len(models)}] Fit-Predict {model_name}...")
            start_time = time.time()
            
            try:
                # Create StatsForecast instance
                sf = StatsForecast(models=[model], freq='D', verbose=True)
                
                # Generate forecast on test data
                forecasts = sf.forecast(df=df_train, X_df=df_test, h=HORIZON * TEST_LENGTH_MULTIPLIER)
                
                
                # Merge with test data for evaluation
                eval_df = test_df.merge(forecasts, on=['unique_id', 'ds'], how='inner')
                
                if eval_df.empty:
                    raise ValueError("No matching forecasts and actual values for evaluation")
                    
                # Store forecast for unified plotting
                if model_name in forecasts.columns:
                    self.all_forecasts[model_name] = {
                        'framework': 'statistical',
                        'predictions': forecasts[model_name].values,
                        'ds': forecasts['ds'].values,
                        'unique_id': forecasts['unique_id'].values,
                        'eval_df': eval_df  # Store eval_df for plotting actual vs predicted
                    }
                else:
                    raise ValueError(f"Model {model_name} not found in forecast columns: {list(forecasts.columns)}")
                
                # Calculate metrics from forecast results
                metrics = self._calculate_test_metrics(eval_df, [model_name])
                
                # Training time
                training_time = time.time() - start_time
                
                # Check if metrics were calculated successfully
                if model_name not in metrics or 'error' in metrics[model_name]:
                    raise ValueError(f"Failed to calculate metrics for {model_name}")
                
                result = {
                    'model_name': model_name,
                    'framework': 'statsforecast',
                    'training_time': training_time,
                    'evaluation_method': 'fit_predict',
                    'is_auto': 'Auto' in model_name,
                    'status': 'success',
                    **metrics[model_name]
                }
                
                results.append(result)
                print(f"  ✓ {model_name} fit-predict completed (Test MAE: {metrics[model_name].get('mae', 'N/A'):.4f})")
                
            except Exception as e:
                training_time = time.time() - start_time
                error_msg = str(e)
                print(f"  ✗ Error in fit-predict for {model_name}: {error_msg}")
                
                results.append({
                    'model_name': model_name,
                    'framework': 'statsforecast',
                    'training_time': training_time,
                    'error': error_msg,
                    'status': 'failed',
                    'is_auto': 'Auto' in model_name,
                    'mae': np.nan,
                    'rmse': np.nan,
                    'mape': np.nan
                })
        
        successful_count = sum(1 for r in results if r.get('status') == 'success')
        print(f"Statistical models fit-predict completed: {successful_count}/{len(models)} successful")
        
        return results
    
    def _calculate_test_metrics(self, test_results: pd.DataFrame, model_names: List[str]) -> Dict:
        """Calculate comprehensive metrics for test set predictions."""
        metrics = {}
        
        for model_name in model_names:
            try:
                if model_name not in test_results.columns:
                    print(f"  Warning: {model_name} not found in test results columns: {list(test_results.columns)}")
                    metrics[model_name] = {'error': f'Model {model_name} not found in results'}
                    continue
                
                # Check for valid data
                y_true = test_results['y'].values
                y_pred = test_results[model_name].values
                
                # Remove NaN values
                mask = ~(np.isnan(y_true) | np.isnan(y_pred))
                y_true_clean = y_true[mask]
                y_pred_clean = y_pred[mask]
                
                if len(y_true_clean) == 0:
                    metrics[model_name] = {'error': 'No valid predictions after removing NaN values'}
                    continue
                
                # Basic metrics using utilsforecast
                try:
                    model_metrics = evaluate(
                        test_results,
                        metrics=[mae, mse, rmse, mape, smape],
                        models=[model_name]
                    )
                    
                    # Extract metric values
                    mae_val = model_metrics[model_metrics['metric'] == 'mae'][model_name].iloc[0]
                    mse_val = model_metrics[model_metrics['metric'] == 'mse'][model_name].iloc[0]
                    rmse_val = model_metrics[model_metrics['metric'] == 'rmse'][model_name].iloc[0]
                    mape_val = model_metrics[model_metrics['metric'] == 'mape'][model_name].iloc[0]
                    smape_val = model_metrics[model_metrics['metric'] == 'smape'][model_name].iloc[0]
                    
                except Exception as e:
                    print(f"  Warning: utilsforecast evaluation failed for {model_name}, using manual calculation: {e}")
                    # Fallback to manual calculation
                    mae_val = np.mean(np.abs(y_true_clean - y_pred_clean))
                    mse_val = np.mean((y_true_clean - y_pred_clean) ** 2)
                    rmse_val = np.sqrt(mse_val)
                    mape_val = np.mean(np.abs((y_true_clean - y_pred_clean) / y_true_clean)) * 100
                    smape_val = np.mean(2 * np.abs(y_true_clean - y_pred_clean) / (np.abs(y_true_clean) + np.abs(y_pred_clean))) * 100
                
                # Additional custom metrics
                directional_accuracy = self._calculate_directional_accuracy(y_true_clean, y_pred_clean)
                
                metrics[model_name] = {
                    'mae': float(mae_val),
                    'mse': float(mse_val),
                    'rmse': float(rmse_val),
                    'mape': float(mape_val),
                    'smape': float(smape_val),
                    'directional_accuracy': float(directional_accuracy) if not np.isnan(directional_accuracy) else None,
                    'n_predictions': int(len(y_true_clean))
                }
                
            except Exception as e:
                print(f"  Error calculating test metrics for {model_name}: {str(e)}")
                metrics[model_name] = {'error': f'Test metric calculation failed: {str(e)}'}
        
        return metrics
    
    def _calculate_directional_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate directional accuracy (correct prediction of up/down movement)."""
        if len(y_true) < 2:
            return np.nan
            
        try:
            true_direction = np.diff(y_true) > 0
            pred_direction = np.diff(y_pred) > 0
            return np.mean(true_direction == pred_direction)
        except Exception:
            return np.nan
    
    def _create_unified_forecast_plot(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """Create a unified plot showing all model predictions with actual values from eval_df."""
        try:
            print("Creating unified forecast comparison plot...")
            
            # Take last 100 values from train_df for continuity
            train_tail = train_df.tail(100).copy()
            
            # Set up the plot
            plt.figure(figsize=(16, 10))
            
            # Convert ds to datetime for better plotting
            if 'ds' in train_tail.columns:
                train_tail['ds'] = pd.to_datetime(train_tail['ds'])
            if 'ds' in test_df.columns:
                test_df = test_df.copy()
                test_df['ds'] = pd.to_datetime(test_df['ds'])
            
            # Create continuous actual values line (train tail + test)
            # Combine train tail and test for continuous line
            actual_dates = pd.concat([train_tail['ds'], test_df['ds']], ignore_index=True)
            actual_values = pd.concat([train_tail['y'], test_df['y']], ignore_index=True)
            
            # Plot continuous actual values line
            plt.plot(actual_dates, actual_values, 
                    color='black', linewidth=3, label='Actual Values', alpha=0.9, zorder=10)
            
            # Highlight test period with markers
            plt.plot(test_df['ds'], test_df['y'], 
                    color='black', linewidth=0, marker='o', markersize=5, 
                    label='Actual (Test Period)', alpha=0.9, zorder=11)
            
            # Define colors for different models
            colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
            
            # Plot forecasts from all models using eval_df data
            for i, (model_name, forecast_data) in enumerate(self.all_forecasts.items()):
                color = colors[i % len(colors)]
                
                # Use eval_df data if available, otherwise fall back to forecast data
                if 'eval_df' in forecast_data and not forecast_data['eval_df'].empty:
                    eval_df = forecast_data['eval_df']
                    # Convert dates to datetime and sort by date
                    eval_df_sorted = eval_df.sort_values('ds')
                    forecast_dates = pd.to_datetime(eval_df_sorted['ds'])
                    predictions = eval_df_sorted[model_name].values
                    
                    # Remove any NaN predictions
                    mask = ~np.isnan(predictions)
                    forecast_dates = forecast_dates[mask]
                    predictions = predictions[mask]
                else:
                    # Fallback to original forecast data
                    forecast_dates = pd.to_datetime(forecast_data['ds'])
                    predictions = forecast_data['predictions']
                    
                    # Remove any NaN predictions
                    mask = ~np.isnan(predictions)
                    forecast_dates = forecast_dates[mask]
                    predictions = predictions[mask]
                
                # Plot model predictions
                linestyle = '--' if forecast_data['framework'] == 'statistical' else '-'
                alpha = 0.8
                linewidth = 2.5 if forecast_data['framework'] == 'neural' else 2
                
                # Create label with forecast method info
                method_info = ""
                if 'forecast_method' in forecast_data:
                    method_info = f" - {forecast_data['forecast_method']}"
                
                plt.plot(forecast_dates, predictions, 
                        color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha,
                        label=f"{model_name} ({forecast_data['framework']}{method_info})", zorder=5)
            
            # Customize the plot
            plt.title('Model Forecasting Comparison - Predictions vs Actual Values', fontsize=18, fontweight='bold', pad=20)
            plt.xlabel('Date', fontsize=14, fontweight='bold')
            plt.ylabel('Value', fontsize=14, fontweight='bold')
            
            # Improve legend
            legend = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
            legend.set_title('Models & Actual', prop={'size': 12, 'weight': 'bold'})
            
            plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            
            # Format x-axis dates
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(test_df)//8)))
            plt.xticks(rotation=45, fontsize=10)
            plt.yticks(fontsize=10)
            
            # Add vertical line to separate train and test with annotation
            if len(train_tail) > 0 and len(test_df) > 0:
                split_date = test_df['ds'].iloc[0]
                plt.axvline(x=split_date, color='red', linestyle=':', linewidth=2, alpha=0.7, zorder=8)
                
                # Add annotation for the split
                plt.annotate('Train/Test Split', 
                           xy=(split_date, plt.ylim()[1]), 
                           xytext=(10, -10), 
                           textcoords='offset points',
                           fontsize=10, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.1),
                           ha='left', va='top')
            
            # Add text box with summary info
            if self.all_forecasts:
                info_text = f"Models Compared: {len(self.all_forecasts)}\n"
                info_text += f"Test Period: {len(test_df)} days\n"
                info_text += f"Train Context: {len(train_tail)} days\n"
                
                # Add rolling forecast info
                rolling_models = [name for name, data in self.all_forecasts.items() 
                                if data.get('forecast_method') == 'rolling_forecast']
                if rolling_models:
                    info_text += f"Rolling Forecast: {len(rolling_models)} models"
                
                plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes, 
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
            
            plt.tight_layout()
            
            # Save the plot
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            plot_filename = f"final_forecast_comparison_{timestamp}.png"
            plot_path = self.plot_dir / plot_filename
            
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"✓ Final forecast plot saved: {plot_path}")
            print(f"  • Plot shows {len(self.all_forecasts)} model predictions vs actual values")
            print(f"  • Continuous line from last {len(train_tail)} train values to {len(test_df)} test values")
            
        except Exception as e:
            print(f"Warning: Failed to create final forecast plot: {str(e)}")
            traceback.print_exc()
    
    def fit_predict_all_models(self, neural_models, stat_models, train_df, test_df, horizon: int) -> pd.DataFrame:
        """Fit all models and generate final predictions with unified plotting."""
        
        print("=== Starting Final Fit-Predict Model Evaluation ===")
        # print(f"Train data shape: {train_df.shape}")
        # print(f"Test data shape: {test_df.shape}")
        # print(f"Horizon: {horizon}")
        print(f"Neural models: {len(neural_models) if neural_models else 0}")
        print(f"Statistical models: {len(stat_models) if stat_models else 0}")
        print(f"Plots will be saved to: {self.plot_dir}")
        
        # Clear previous forecasts
        self.all_forecasts = {}
        all_results = []
        
        # Fit-predict neural models
        if neural_models:
            print("\n--- Fit-Predict Neural Models ---")
            neural_results = self.fit_predict_neural_models(neural_models, train_df, test_df, horizon)
            all_results.extend(neural_results)
        
        # Fit-predict statistical models
        if stat_models:
            print("\n--- Fit-Predict Statistical Models ---")
            stat_results = self.fit_predict_statistical_models(stat_models, train_df, test_df, horizon)
            all_results.extend(stat_results)
        
        # Create unified forecast plot
        if self.all_forecasts:
            print("\n--- Creating Final Unified Forecast Plot ---")
            self._create_unified_forecast_plot(train_df, test_df)
        else:
            print("⚠️  No successful forecasts to plot")
        
        # Create results DataFrame
        if not all_results:
            print("⚠️  No results from fit-predict evaluation!")
            return pd.DataFrame()
        
        results_df = pd.DataFrame(all_results)
        print(f"\nTotal fit-predict results: {len(results_df)}")
        
        # Filter out failed models
        if 'status' in results_df.columns:
            successful_results = results_df[results_df['status'] == 'success'].copy()
        elif 'error' in results_df.columns:
            successful_results = results_df[results_df['error'].isna()].copy()
        else:
            # Assume all are successful if no status/error columns
            successful_results = results_df.copy()
        
        print(f"Successful fit-predict results: {len(successful_results)}")
        
        if len(successful_results) == 0:
            print("⚠️  No successful fit-predict evaluations!")
            self.save_final_results(results_df, pd.DataFrame())
            return pd.DataFrame()
        
        # Rank models by test set MAE (lower is better)
        try:
            successful_results['mae_rank'] = successful_results['mae'].rank()
            successful_results['rmse_rank'] = successful_results['rmse'].rank()
            successful_results['mape_rank'] = successful_results['mape'].rank()
            
            # Combined rank (simple average)
            successful_results['combined_rank'] = (
                successful_results['mae_rank'] + 
                successful_results['rmse_rank'] + 
                successful_results['mape_rank']
            ) / 3
            
            # Sort by combined rank
            successful_results = successful_results.sort_values('combined_rank')
            print(f"✓ Models ranked successfully by test set performance")
            
        except Exception as e:
            print(f"⚠️  Ranking failed, sorting by MAE only: {e}")
            successful_results = successful_results.sort_values('mae')
        
        # Save results
        self.save_final_results(results_df, successful_results)
        
        return successful_results
    
    def save_final_results(self, all_results: pd.DataFrame, successful_results: pd.DataFrame):
        """Save final fit-predict results."""
        try:
            # Use simple timestamp without timezone
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            
            # Save all results
            if not all_results.empty:
                all_results.to_csv(self.results_dir / f"final_all_results_{timestamp}.csv", index=False)
                print(f"All final results saved: {len(all_results)} entries")
            
            # Save successful results
            if not successful_results.empty:
                successful_results.to_csv(self.results_dir / f"final_ranked_results_{timestamp}.csv", index=False)
                print(f"Ranked final results saved: {len(successful_results)} entries")
            
            # Save summary
            summary = {
                'timestamp': timestamp,
                'evaluation_type': 'final_fit_predict',
                'total_models': len(all_results),
                'successful_models': len(successful_results),
                'failed_models': len(all_results) - len(successful_results),
                'best_model': successful_results.iloc[0]['model_name'] if len(successful_results) > 0 else None,
                'best_test_mae': float(successful_results.iloc[0]['mae']) if len(successful_results) > 0 else None
            }
            
            with open(self.results_dir / f"final_summary_{timestamp}.json", 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"Final results saved to {self.results_dir}")
            print(f"Final Summary: {summary}")
            
        except Exception as e:
            print(f"Error saving final results: {e}")
            traceback.print_exc()
    
    def get_top_models(self, results_df: pd.DataFrame, top_n: int = 5) -> List[str]:
        """Get top N model names based on test set performance."""
        if len(results_df) == 0:
            print("No final results available for top models selection")
            return []
        
        actual_top_n = min(top_n, len(results_df))
        top_models = results_df.head(actual_top_n)['model_name'].tolist()
        print(f"Selected top {len(top_models)} models from final evaluation: {top_models}")
        return top_models 