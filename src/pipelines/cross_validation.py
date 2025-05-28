# src/pipelines/cross_validation.py
"""
Cross-validation evaluation framework for model performance comparison.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import time
import json
from pathlib import Path
import traceback

from neuralforecast import NeuralForecast
from statsforecast import StatsForecast
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mae, mse, rmse, mape, smape

from config.base import *

class CrossValidationEvaluator:
    """Cross-validation based model performance evaluation."""
    
    def __init__(self, results_dir: str = FINAL_DIR):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.results = []
        
    def evaluate_neural_models_cv(self, models, train_df, horizon: int) -> List[Dict]:
        """Evaluate neural models using cross-validation."""
        results = []
        
        if not models:
            print("No neural models provided for evaluation")
            return results
        
        print(f"Evaluating {len(models)} neural models with cross-validation...")
        
        for i, model in enumerate(models, 1):
            model_name = model.__class__.__name__
            print(f"[{i}/{len(models)}] Cross-validating {model_name}...")
            start_time = time.time()
            
            try:
                # Create NeuralForecast instance
                nf = NeuralForecast(models=[model], freq='D')
                
                # Perform cross-validation
                cv_results = nf.cross_validation(
                    df=train_df,
                    n_windows=CV_N_WINDOWS,
                    step_size=CV_STEP_SIZE,
                    verbose=True
                )
                
                if cv_results.empty:
                    raise ValueError("Cross-validation returned empty results")
                
                # Calculate metrics from CV results
                metrics = self._calculate_metrics(cv_results, [model_name])
                evaluation_method = "cross_validation"
                
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
                print(f"  ✓ {model_name} CV completed (MAE: {metrics[model_name].get('mae', 'N/A'):.4f})")
                
            except Exception as e:
                training_time = time.time() - start_time
                error_msg = str(e)
                print(f"  ✗ Error in CV for {model_name}: {error_msg}")
                
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
        print(f"Neural models CV completed: {successful_count}/{len(models)} successful")
        
        return results
    
    def evaluate_statistical_models_cv(self, models, train_df, horizon: int) -> List[Dict]:
        """Evaluate statistical models using cross-validation."""
        results = []
        
        if not models:
            print("No statistical models provided for evaluation")
            return results
        
        print(f"Evaluating {len(models)} statistical models with cross-validation...")
        
        # Prepare data for StatsForecast
        df_train = train_df[['unique_id', 'ds', 'y']].copy()
        
        for i, model in enumerate(models, 1):
            model_name = model.__class__.__name__
            print(f"[{i}/{len(models)}] Cross-validating {model_name}...")
            start_time = time.time()
            
            try:
                # Create StatsForecast instance
                sf = StatsForecast(models=[model], freq='D', verbose=True)
                
                # Cross-validation for metrics
                cv_results = sf.cross_validation(
                    df=df_train,
                    h=horizon,
                    n_windows=CV_N_WINDOWS,
                    step_size=CV_STEP_SIZE
                )
                
                if cv_results.empty:
                    raise ValueError("Cross-validation returned empty results")
                
                # Calculate metrics from CV results
                metrics = self._calculate_metrics(cv_results, [model_name])
                
                # Training time
                training_time = time.time() - start_time
                
                # Check if metrics were calculated successfully
                if model_name not in metrics or 'error' in metrics[model_name]:
                    raise ValueError(f"Failed to calculate metrics for {model_name}")
                
                result = {
                    'model_name': model_name,
                    'framework': 'statsforecast',
                    'training_time': training_time,
                    'evaluation_method': 'cross_validation',
                    'is_auto': 'Auto' in model_name,
                    'status': 'success',
                    **metrics[model_name]
                }
                
                results.append(result)
                print(f"  ✓ {model_name} CV completed (MAE: {metrics[model_name].get('mae', 'N/A'):.4f})")
                
            except Exception as e:
                training_time = time.time() - start_time
                error_msg = str(e)
                print(f"  ✗ Error in CV for {model_name}: {error_msg}")
                
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
        print(f"Statistical models CV completed: {successful_count}/{len(models)} successful")
        
        return results
    
    def _calculate_metrics(self, cv_results: pd.DataFrame, model_names: List[str]) -> Dict:
        """Calculate comprehensive metrics for cross-validation results."""
        metrics = {}
        
        for model_name in model_names:
            try:
                if model_name not in cv_results.columns:
                    print(f"  Warning: {model_name} not found in results columns: {list(cv_results.columns)}")
                    metrics[model_name] = {'error': f'Model {model_name} not found in results'}
                    continue
                
                # Check for valid data
                y_true = cv_results['y'].values
                y_pred = cv_results[model_name].values
                
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
                        cv_results,
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
                coverage = self._calculate_coverage(cv_results, model_name)
                
                metrics[model_name] = {
                    'mae': float(mae_val),
                    'mse': float(mse_val),
                    'rmse': float(rmse_val),
                    'mape': float(mape_val),
                    'smape': float(smape_val),
                    'directional_accuracy': float(directional_accuracy) if not np.isnan(directional_accuracy) else None,
                    'coverage_80': coverage.get('80', None),
                    'coverage_90': coverage.get('90', None),
                    'n_predictions': int(len(y_true_clean))
                }
                
            except Exception as e:
                print(f"  Error calculating metrics for {model_name}: {str(e)}")
                metrics[model_name] = {'error': f'Metric calculation failed: {str(e)}'}
        
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
    
    def _calculate_coverage(self, cv_results: pd.DataFrame, model_name: str) -> Dict:
        """Calculate prediction interval coverage."""
        coverage = {}
        
        try:
            # Check for prediction intervals
            for level in [80, 90]:
                lo_col = f"{model_name}-lo-{level}"
                hi_col = f"{model_name}-hi-{level}"
                
                if lo_col in cv_results.columns and hi_col in cv_results.columns:
                    y_true = cv_results['y']
                    y_lo = cv_results[lo_col]
                    y_hi = cv_results[hi_col]
                    
                    # Coverage = proportion of true values within prediction intervals
                    within_interval = (y_true >= y_lo) & (y_true <= y_hi)
                    coverage[str(level)] = float(within_interval.mean())
        except Exception as e:
            print(f"  Warning: Coverage calculation failed for {model_name}: {e}")
        
        return coverage
    
    def compare_models_cv(self, neural_models, stat_models, train_df, horizon: int) -> pd.DataFrame:
        """Compare all models using cross-validation and return ranked results."""
        
        print("=== Starting Cross-Validation Model Comparison ===")
        print(f"Train data shape: {train_df.shape}")
        print(f"Horizon: {horizon}")
        print(f"CV windows: {CV_N_WINDOWS}")
        print(f"CV step size: {CV_STEP_SIZE}")
        print(f"Neural models: {len(neural_models) if neural_models else 0}")
        print(f"Statistical models: {len(stat_models) if stat_models else 0}")
        
        all_results = []
        
        # Evaluate neural models
        if neural_models:
            print("\n--- Cross-Validating Neural Models ---")
            neural_results = self.evaluate_neural_models_cv(neural_models, train_df, horizon)
            all_results.extend(neural_results)
        
        # Evaluate statistical models
        if stat_models:
            print("\n--- Cross-Validating Statistical Models ---")
            stat_results = self.evaluate_statistical_models_cv(stat_models, train_df, horizon)
            all_results.extend(stat_results)
        
        # Create results DataFrame
        if not all_results:
            print("⚠️  No results from cross-validation!")
            return pd.DataFrame()
        
        results_df = pd.DataFrame(all_results)
        print(f"\nTotal CV results: {len(results_df)}")
        
        # Filter out failed models
        if 'status' in results_df.columns:
            successful_results = results_df[results_df['status'] == 'success'].copy()
        elif 'error' in results_df.columns:
            successful_results = results_df[results_df['error'].isna()].copy()
        else:
            # Assume all are successful if no status/error columns
            successful_results = results_df.copy()
        
        print(f"Successful CV results: {len(successful_results)}")
        
        if len(successful_results) == 0:
            print("⚠️  No successful cross-validation evaluations!")
            self.save_cv_results(results_df, pd.DataFrame())
            return pd.DataFrame()
        
        # Rank models by MAE (lower is better)
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
            print(f"✓ Models ranked successfully by CV performance")
            
        except Exception as e:
            print(f"⚠️  Ranking failed, sorting by MAE only: {e}")
            successful_results = successful_results.sort_values('mae')
        
        # Save results
        self.save_cv_results(results_df, successful_results)
        
        return successful_results
    
    def save_cv_results(self, all_results: pd.DataFrame, successful_results: pd.DataFrame):
        """Save cross-validation results."""
        try:
            # Use simple timestamp without timezone
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            
            # Save all results
            if not all_results.empty:
                all_results.to_csv(self.results_dir / f"cv_all_results_{timestamp}.csv", index=False)
                print(f"All CV results saved: {len(all_results)} entries")
            
            # Save successful results
            if not successful_results.empty:
                successful_results.to_csv(self.results_dir / f"cv_ranked_results_{timestamp}.csv", index=False)
                print(f"Ranked CV results saved: {len(successful_results)} entries")
            
            # Save summary
            summary = {
                'timestamp': timestamp,
                'evaluation_type': 'cross_validation',
                'cv_windows': CV_N_WINDOWS,
                'cv_step_size': CV_STEP_SIZE,
                'total_models': len(all_results),
                'successful_models': len(successful_results),
                'failed_models': len(all_results) - len(successful_results),
                'best_model': successful_results.iloc[0]['model_name'] if len(successful_results) > 0 else None,
                'best_mae': float(successful_results.iloc[0]['mae']) if len(successful_results) > 0 else None
            }
            
            with open(self.results_dir / f"cv_summary_{timestamp}.json", 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"CV results saved to {self.results_dir}")
            print(f"CV Summary: {summary}")
            
        except Exception as e:
            print(f"Error saving CV results: {e}")
            traceback.print_exc()
    
    def get_top_models(self, results_df: pd.DataFrame, top_n: int = 5) -> List[str]:
        """Get top N model names based on CV performance."""
        if len(results_df) == 0:
            print("No CV results available for top models selection")
            return []
        
        actual_top_n = min(top_n, len(results_df))
        top_models = results_df.head(actual_top_n)['model_name'].tolist()
        print(f"Selected top {len(top_models)} models from CV: {top_models}")
        return top_models 