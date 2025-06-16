import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, TQDMProgressBar, ModelCheckpoint
import optuna
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import logging
import warnings
import os
from einops import rearrange

from iTransformer import iTransformer
# We'll assume this data preparation function is available from your project structure.
# It should return the training DataFrame and the holdout test DataFrame.
from src.dataset.data_preparation import prepare_pipeline_data
from src.utils.utils import calculate_metrics_1

# Suppress Optuna's trial info logging and other warnings for a cleaner output
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore', '.*does not have many workers.*')
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

class LitProgressBar(TQDMProgressBar):
    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.disable = True
        return bar

def create_sequences(data: np.ndarray, num_variates: int, lookback_len: int, pred_lengths: tuple[int, ...]):
    """
    Creates sequences of input data and corresponding target values.
    """
    xs, ys_per_pred_len = [], [[] for _ in pred_lengths]
    max_pred_len = max(pred_lengths)
    
    for i in range(len(data) - lookback_len - max_pred_len + 1):
        xs.append(data[i:(i + lookback_len)])
        for j, p_len in enumerate(pred_lengths):
            ys_per_pred_len[j].append(data[i + lookback_len : i + lookback_len + p_len])

    if not xs:
        return None, None

    xs_np = np.array(xs, dtype=np.float32)
    ys_np_tuple = tuple(np.array(y_list, dtype=np.float32) for y_list in ys_per_pred_len)

    return torch.from_numpy(xs_np), tuple(torch.from_numpy(y) for y in ys_np_tuple)

class LightningiTransformer(pl.LightningModule):
    """PyTorch Lightning wrapper for the iTransformer model."""
    def __init__(self, model_params: dict, learning_rate: float):
        super().__init__()
        self.save_hyperparameters()
        self.model = iTransformer(**model_params)
        self.learning_rate = learning_rate
        self.criterion = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> dict[int, torch.Tensor]:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, *y_tuple = batch
        # The model's forward pass with targets present calculates and returns the loss directly.
        # This requires the model to be in `train()` mode, which PTL handles for `training_step`.
        loss = self.model(x, targets=tuple(y_tuple))
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, *y_tuple = batch
        y_tuple = tuple(y_tuple)

        # In `validation_step`, PTL sets the model to `eval()` mode.
        # We get predictions and calculate the loss manually.
        predictions_dict = self.model(x)

        total_loss = 0
        num_losses = 0

        # The model may have a single or multiple prediction lengths.
        pred_lengths = self.model.pred_length
        if not isinstance(pred_lengths, (list, tuple)):
            pred_lengths = [pred_lengths]

        for i, pred_len in enumerate(pred_lengths):
            # The model output is already in the correct (batch, pred_len, variates) shape.
            pred = predictions_dict[pred_len]
            target = y_tuple[i]
            total_loss += self.criterion(pred, target)
            num_losses += 1
        
        if num_losses == 0:
             return None

        val_loss = total_loss / num_losses
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

def objective(trial: optuna.trial.Trial, train_df: pd.DataFrame, num_variates: int, pred_lengths: tuple[int, ...], horizon: int) -> float:
    """Optuna objective function for hyperparameter optimization."""
    # Define hyperparameter search space
    lookback_len = trial.suggest_categorical('lookback_len', [horizon * i for i in [2, 3, 4, 6]])
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 2e-3, log=True)
    epochs = trial.suggest_categorical('epoch', [50, 100, 150])
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    dim = trial.suggest_categorical('dim', [64, 128, 256])
    heads = trial.suggest_categorical('heads', [4, 8])
    dim_head = trial.suggest_categorical('dim_head', [32, 64])

    model_params = {
        'num_variates': num_variates,
        'lookback_len': lookback_len,
        'pred_length': pred_lengths,
        'dim': dim,
        'depth': 6, # Fixed as per the original architecture
        'heads': heads,
        'dim_head': dim_head,
        'use_reversible_instance_norm': True,
    }

    # Time Series Cross-Validation using a rolling window approach
    n_splits = 3
    total_len = len(train_df)
    fold_len = total_len // (n_splits + 1)
    
    val_losses = []

    for i in range(n_splits):
        train_end_idx = (i + 1) * fold_len
        val_end_idx = train_end_idx + fold_len
        
        train_data = train_df.iloc[:train_end_idx].values
        val_data = train_df.iloc[train_end_idx:val_end_idx].values

        scaler = StandardScaler()
        train_data_scaled = scaler.fit_transform(train_data)
        val_data_scaled = scaler.transform(val_data)

        X_train, y_train_tuple = create_sequences(train_data_scaled, num_variates, lookback_len, pred_lengths)
        X_val, y_val_tuple = create_sequences(val_data_scaled, num_variates, lookback_len, pred_lengths)

        if X_train is None or X_val is None:
            continue

        train_dataset = TensorDataset(X_train, *y_train_tuple)
        val_dataset = TensorDataset(X_val, *y_val_tuple)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)

        lightning_model = LightningiTransformer(model_params, learning_rate)
        
        trainer = pl.Trainer(
            max_epochs=epochs,
            accelerator='gpu',
            devices=1,
            callbacks=[EarlyStopping(monitor='val_loss', patience=10, mode='min'), LitProgressBar()],
            logger=False,
            enable_progress_bar=False,
            enable_checkpointing=False,
        )
        trainer.fit(lightning_model, train_loader, val_loader)
        
        val_loss = trainer.callback_metrics.get('val_loss')
        if val_loss is not None:
            val_losses.append(val_loss.item())
        
        # Pruning for Optuna
        trial.report(np.mean(val_losses) if val_losses else float('inf'), i)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return np.mean(val_losses) if val_losses else float('inf')

def main():
    # --- Configuration ---
    PRED_LENGTHS = (7, 14, 30)
    N_TRIALS_OPTUNA = 50
    # The primary forecast horizon for defining lookback ratios and data splits
    MAIN_HORIZON = max(PRED_LENGTHS)

    # --- 1. Load Data ---
    print("Step 1: Loading and preparing data...")
    train_df, test_df, _, _ = prepare_pipeline_data(horizon=MAIN_HORIZON, test_length_multiplier=1)
    # drop 'unique_id' column
    train_df = train_df.drop(columns=['unique_id'])
    test_df = test_df.drop(columns=['unique_id'])
    # set index to 'ds'
    train_df = train_df.set_index('ds')
    test_df = test_df.set_index('ds')
    num_variates = train_df.shape[1]
    print(f"Data loaded. Training set shape: {train_df.shape}, Test set shape: {test_df.shape}")

    # --- 2. HPO with Optuna ---
    print(f"\nStep 2: Starting Hyperparameter Optimization with Optuna ({N_TRIALS_OPTUNA} trials)...")
    study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda trial: objective(trial, train_df, num_variates, PRED_LENGTHS, MAIN_HORIZON), n_trials=N_TRIALS_OPTUNA)
    
    best_params = study.best_params
    print("\nBest hyperparameters found:")
    for key, value in best_params.items():
        print(f"  - {key}: {value}")

    # --- 3. Final Model Refit ---
    print("\nStep 3: Refitting the model with best hyperparameters on the full training data...")
    scaler = StandardScaler()
    train_data_scaled = scaler.fit_transform(train_df.values)

    best_lookback = best_params['lookback_len']
    X_train_full, y_train_full_tuple = create_sequences(train_data_scaled, num_variates, best_lookback, PRED_LENGTHS)
    
    train_dataset_full = TensorDataset(X_train_full, *y_train_full_tuple)
    train_loader_full = DataLoader(train_dataset_full, batch_size=best_params['batch_size'], shuffle=True, num_workers=4)

    final_model_params = {
        'num_variates': num_variates, 'lookback_len': best_lookback, 'pred_length': PRED_LENGTHS,
        'dim': best_params['dim'], 'depth': 6, 'heads': best_params['heads'],
        'dim_head': best_params['dim_head'], 'use_reversible_instance_norm': True,
    }

    final_model = LightningiTransformer(final_model_params, best_params['learning_rate'])
    
    # Configure checkpointing to save only the best model based on training loss
    output_dir = "results/itransformer"
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(output_dir, "checkpoints"),
        filename="best-model",
        save_top_k=1,
        monitor="train_loss",
        mode="min",
    )

    final_trainer = pl.Trainer(
        max_epochs=best_params['epoch'], accelerator='gpu', devices=1,
        callbacks=[TQDMProgressBar(refresh_rate=10), checkpoint_callback],
        logger=False # Consider adding a TensorBoardLogger here for tracking
    )
    final_trainer.fit(final_model, train_loader_full)
    print("Final model training complete.")

    # --- 4. Generate Final Forecast ---
    print("\nStep 4: Generating final forecast on the test set...")
    # Prepare the input for the test forecast
    last_window_scaled = train_data_scaled[-best_lookback:]
    input_tensor = torch.from_numpy(last_window_scaled).unsqueeze(0).float().to(final_model.device)

    # Generate predictions
    final_model.eval()
    with torch.no_grad():
        predictions_scaled = final_model(input_tensor)

    # Inverse transform and format predictions
    forecasts = {}
    for pred_len, tensor in predictions_scaled.items():
        forecast_scaled = tensor.squeeze(0).cpu().numpy()
        forecast_unscaled = scaler.inverse_transform(forecast_scaled)
        
        # Create a DataFrame for the forecast
        forecast_dates = pd.date_range(start=test_df.index[0], periods=pred_len, freq=test_df.index.freq)
        forecasts[pred_len] = pd.DataFrame(forecast_unscaled, index=forecast_dates, columns=train_df.columns)
        print(f"\n--- Forecast for {pred_len} steps ---")
        print(forecasts[pred_len].head())

    # --- 5. Evaluate Forecasts and Save Results ---
    print("\nStep 5: Evaluating forecasts and saving results...")
    metrics_results = {}
    os.makedirs(output_dir, exist_ok=True)

    for pred_len, forecast_df in forecasts.items():
        # Align true values with the forecast period
        true_df = test_df.iloc[:pred_len]
        
        # Prepare data for metric calculation
        y_true = true_df.values.flatten()
        y_pred = forecast_df.values.flatten()
        eval_df = pd.DataFrame({'y': y_true, 'pred': y_pred})
        
        # Calculate and store metrics
        metrics = calculate_metrics_1(eval_df)
        metrics_results[pred_len] = metrics
        print(f"\nMetrics for {pred_len}-step forecast:")
        print(pd.DataFrame([metrics]))

        # Save forecast to CSV
        forecast_path = os.path.join(output_dir, f"forecast_{pred_len}d.csv")
        forecast_df.to_csv(forecast_path)
        print(f"Saved {pred_len}d forecast to {forecast_path}")

    # Save all metrics to a single CSV
    metrics_df = pd.DataFrame.from_dict(metrics_results, orient='index')
    metrics_path = os.path.join(output_dir, "metrics.csv")
    metrics_df.to_csv(metrics_path)
    print(f"\nSaved all metrics to {metrics_path}")

    print("\nExperiment finished.")

if __name__ == '__main__':
    main() 