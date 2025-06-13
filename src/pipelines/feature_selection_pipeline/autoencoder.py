import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

class LSTMAutoEncoder(nn.Module):
    """LSTM-based AutoEncoder for time series feature compression."""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 2, 
                    encoding_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.encoding_dim = encoding_dim
        
        # Encoder
        self.encoder_lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                                    batch_first=True, dropout=dropout)
        self.encoder_fc = nn.Linear(hidden_size, encoding_dim)
        
        # Decoder
        self.decoder_fc = nn.Linear(encoding_dim, hidden_size)
        self.decoder_lstm = nn.LSTM(hidden_size, input_size, num_layers, 
                                    batch_first=True, dropout=dropout)
        
    def encode(self, x):
        """Encode input to latent representation."""
        lstm_out, _ = self.encoder_lstm(x)
        # Use the last time step
        encoded = self.encoder_fc(lstm_out[:, -1, :])
        return encoded
        
    def decode(self, encoded, seq_len):
        """Decode latent representation back to input space."""
        decoded_fc = self.decoder_fc(encoded)
        # Repeat for sequence length
        decoded_fc = decoded_fc.unsqueeze(1).repeat(1, seq_len, 1)
        decoded, _ = self.decoder_lstm(decoded_fc)
        return decoded
        
    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded, x.size(1))
        return decoded, encoded

class TransformerAutoEncoder(nn.Module):
    """Transformer-based AutoEncoder for time series feature compression."""
    
    def __init__(self, input_size: int, d_model: int = 128, nhead: int = 8, 
                    num_layers: int = 2, encoding_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.input_size = input_size
        self.d_model = d_model
        self.encoding_dim = encoding_dim
        
        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding (simple)
        self.pos_encoding = nn.Parameter(torch.randn(1000, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Encoding layer
        self.encoder_fc = nn.Linear(d_model, encoding_dim)
        
        # Decoder
        self.decoder_fc = nn.Linear(encoding_dim, d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, input_size)
        
    def encode(self, x):
        """Encode input to latent representation."""
        # Project to d_model dimensions
        x = self.input_projection(x)
        
        # Add positional encoding
        seq_len = x.size(1)
        x = x + self.pos_encoding[:seq_len, :].unsqueeze(0)
        
        # Transformer encoding
        encoded_output = self.transformer_encoder(x) # This is the full sequence output
        
        # Global average pooling and final encoding
        latent_vector = torch.mean(encoded_output, dim=1) # Pooling
        latent_vector = self.encoder_fc(latent_vector)
        
        return latent_vector, encoded_output # Return both the pooled vector and the full sequence

        
    def decode(self, encoded, seq_len):
        """Decode latent representation back to input space."""
        # Project back to d_model
        decoded = self.decoder_fc(encoded)
        
        # Repeat for sequence length
        decoded = decoded.unsqueeze(1).repeat(1, seq_len, 1)
        
        # Add positional encoding
        decoded = decoded + self.pos_encoding[:seq_len, :].unsqueeze(0)
        
        # Transformer decoding (using encoded as memory)
        memory = decoded  # Self-attention
        decoded = self.transformer_decoder(decoded, memory)
        
        # Project to output space
        decoded = self.output_projection(decoded)
        
        return decoded
        
    def forward(self, x):
        # Pass the encoder's output to the decoder
        latent_vector, encoded_output = self.encode(x) 
        decoded = self.decode(latent_vector, x.size(1))
        return decoded, latent_vector  # Return the latent representation

class AutoencoderSelectionMixin:
    """Mixin for autoencoder-based feature selection methods."""

    def _prepare_sequences_for_autoencoder(self, df: pd.DataFrame, 
                                         sequence_length: int = 20,
                                         scaler: StandardScaler = None,
                                         fit_scaler: bool = False) -> Tuple[np.ndarray, List[str], StandardScaler]:
        """
        Prepare sequences for autoencoder processing.
        
        Args:
            df: DataFrame with time series data
            sequence_length: Length of sequences to create
            scaler: A pre-fitted StandardScaler. If None, a new one is created.
            fit_scaler: If True, fit the scaler on the data. Otherwise, just transform.
            
        Returns:
            Tuple of (sequences, feature_names, scaler)
        """
        feature_cols = [col for col in df.columns if col not in ['unique_id', 'ds', 'y']]
        data = df[feature_cols].values
        
        if scaler is None:
            scaler = StandardScaler()

        if fit_scaler:
            data_scaled = scaler.fit_transform(data)
        else:
            data_scaled = scaler.transform(data)
            
        sequences = []
        for i in range(len(data_scaled) - sequence_length + 1):
            sequences.append(data_scaled[i:i + sequence_length])
        
        return np.array(sequences), feature_cols, scaler

    def _fit_autoencoder(self, train_df: pd.DataFrame, ae_method: str = 'lstm', **kwargs) -> Dict[str, Any]:
        """
        Fit an autoencoder on the training data, including the scaler.
        """
        self.print_info(f"Fitting {ae_method.upper()} AutoEncoder...")
        
        # Prepare sequences and FIT the scaler
        sequences, feature_names, scaler = self._prepare_sequences_for_autoencoder(
            train_df, 
            sequence_length=kwargs.get('sequence_length', 20),
            fit_scaler=True
        )
        
        # Store the fitted scaler
        self.fitted_parameters[f'{ae_method}_scaler'] = scaler
        
        # Train the model
        if ae_method == 'lstm':
            model_func = self.lstm_autoencoder_features
        elif ae_method == 'transformer':
            model_func = self.transformer_autoencoder_features
        else:
            raise ValueError(f"Unknown AE method: {ae_method}")
            
        # Call the training function
        result = model_func(train_df, **kwargs)
        
        # Store the trained model
        self.fitted_selectors[f'{ae_method}_autoencoder'] = result['model']
        
        self.print_info(f"Finished fitting {ae_method.upper()} AutoEncoder.")
        return result

    def _transform_autoencoder(self, df: pd.DataFrame, ae_method: str = 'lstm', **kwargs) -> pd.DataFrame:
        """
        Transform data using a pre-trained autoencoder and scaler.
        """
        self.print_info(f"Transforming data with fitted {ae_method.upper()} AutoEncoder...")
        
        # Load the pre-fitted scaler and model
        scaler = self.fitted_parameters.get(f'{ae_method}_scaler')
        model = self.fitted_selectors.get(f'{ae_method}_autoencoder')
        
        if scaler is None or model is None:
            raise RuntimeError(f"The {ae_method} autoencoder has not been fitted. Call a fit method first.")
            
        # Prepare sequences using the FITTED scaler (fit_scaler=False)
        sequences, _, _ = self._prepare_sequences_for_autoencoder(
            df, 
            sequence_length=kwargs.get('sequence_length', 20),
            scaler=scaler,
            fit_scaler=False
        )
        
        # Generate encoded features
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.eval()
        with torch.no_grad():
            all_sequences = torch.FloatTensor(sequences).to(device)
            _, encoded_features = model(all_sequences)
            encoded_features = encoded_features.cpu().numpy()
            
        # Create dataframe with encoded features
        encoding_dim = encoded_features.shape[1]
        encoded_feature_names = [f'{ae_method}_ae_dim_{i}' for i in range(encoding_dim)]
        
        encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names)
        
        # Align with original dataframe
        pad_length = len(df) - len(encoded_df)
        if pad_length > 0:
            # padding = pd.DataFrame(np.repeat(encoded_df.iloc[[0]].values, pad_length, axis=0), columns=encoded_feature_names)
            # encoded_df = pd.concat([padding, encoded_df]).reset_index(drop=True)
            # Pad at the beginning by repeating the first available encoded vector.
            # This assumes the first encoded vector is a reasonable fill-in for earlier, unencodable time steps.
            if not encoded_df.empty:
                padding_values = encoded_df.iloc[[0]].values
                padding = pd.DataFrame(np.repeat(padding_values, pad_length, axis=0), columns=encoded_feature_names)
                encoded_df = pd.concat([padding, encoded_df]).reset_index(drop=True)
            else: # If encoded_df is empty (e.g., input df was too short for any sequences)
                encoded_df = pd.DataFrame(columns=encoded_feature_names, index=df.index).fillna(0) # Or some other appropriate fill
            
        encoded_df.index = df.index
        
        # Combine with original data (or just return encoded)
        # For now, returning just the encoded features for simplicity in pipeline.
        # Can be concatenated with other features if needed.
        return encoded_df

    def lstm_autoencoder_features(self, train_df: pd.DataFrame,
                                 encoding_dim: int = 32,
                                 sequence_length: int = 20,
                                 hidden_size: int = 64,
                                 num_layers: int = 2,
                                 epochs: int = 50,
                                 batch_size: int = 32,
                                 learning_rate: float = 1e-3,
                                 verbose: bool = True) -> Dict[str, Any]:
        """
        Generate features using LSTM AutoEncoder.
        
        Args:
            train_df: Training dataframe
            encoding_dim: Dimensionality of encoded features
            sequence_length: Length of input sequences
            hidden_size: Hidden size of LSTM
            num_layers: Number of LSTM layers
            epochs: Training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            verbose: Whether to print training progress
            
        Returns:
            Dictionary with encoded features and model info
        """
        self.print_info("Starting LSTM AutoEncoder feature generation...")
        
        # Prepare sequences
        sequences, feature_names, scaler = self._prepare_sequences_for_autoencoder(
            train_df, sequence_length, fit_scaler=True # This will be called from _fit now
        )
        
        input_size = len(feature_names)
        
        # Create model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = LSTMAutoEncoder(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            encoding_dim=encoding_dim
        ).to(device)
        
        # Prepare data loader
        tensor_data = torch.FloatTensor(sequences).to(device)
        dataset = TensorDataset(tensor_data, tensor_data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        model.train()
        final_loss = 0
        for epoch in range(epochs):
            total_loss = 0
            for batch_input, batch_target in dataloader:
                optimizer.zero_grad()
                reconstructed, encoded = model(batch_input)
                loss = criterion(reconstructed, batch_target)
                loss.backward()
                
                # --- FIX: Add gradient clipping to prevent exploding gradients and NaN loss ---
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            final_loss = avg_loss
            if verbose and self.verbose and (epoch + 1) % 10 == 0:
                self.print_info(f"LSTM-AE Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")
        
        # Generate encoded features for the entire dataset
        model.eval()
        with torch.no_grad():
            all_sequences = torch.FloatTensor(sequences).to(device)
            _, encoded_features = model(all_sequences)
            encoded_features = encoded_features.cpu().numpy()
        
        # Create feature names for encoded dimensions
        encoded_feature_names = [f'lstm_ae_dim_{i}' for i in range(encoding_dim)]
        
        # Create dataframe with encoded features aligned to original timestamps
        # Take the last encoded feature for each position
        encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names)
        
        # Align with original dataframe (pad beginning with first values)
        pad_length = len(train_df) - len(encoded_df)
        if pad_length > 0:
            # Pad at the beginning by repeating the first available encoded vector
            padding = pd.DataFrame(np.repeat(encoded_df.iloc[[0]].values, pad_length, axis=0), columns=encoded_feature_names)
            encoded_df = pd.concat([padding, encoded_df]).reset_index(drop=True)
            
        encoded_df.index = train_df.index

        self.autoencoder_features['lstm'] = encoded_df
        
        return {
            'encoded_features': encoded_df,
            'model': model,
            'scaler': scaler,
            'feature_names': encoded_feature_names,
            'method': 'lstm_autoencoder',
            'loss': final_loss
        }
    
    def transformer_autoencoder_features(self, train_df: pd.DataFrame,
                                        encoding_dim: int = 32,
                                        sequence_length: int = 20,
                                        d_model: int = 128,
                                        nhead: int = 8,
                                        num_layers: int = 2,
                                        epochs: int = 50,
                                        batch_size: int = 32,
                                        learning_rate: float = 1e-3,
                                        verbose: bool = True) -> Dict[str, Any]:
        """
        Generate features using Transformer AutoEncoder.
        
        Args:
            train_df: Training dataframe
            encoding_dim: Dimensionality of encoded features
            sequence_length: Length of input sequences
            d_model: Model dimension for transformer
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            epochs: Training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            verbose: Whether to print training progress
            
        Returns:
            Dictionary with encoded features and model info
        """
        self.print_info("Starting Transformer AutoEncoder feature generation...")
        
        # Prepare sequences
        sequences, feature_names, scaler = self._prepare_sequences_for_autoencoder(
            train_df, sequence_length, fit_scaler=True # This will be called from _fit now
        )
        
        input_size = len(feature_names)
        
        # Create model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = TransformerAutoEncoder(
            input_size=input_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            encoding_dim=encoding_dim
        ).to(device)
        
        # Prepare data loader
        tensor_data = torch.FloatTensor(sequences).to(device)
        dataset = TensorDataset(tensor_data, tensor_data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        model.train()
        final_loss = 0
        for epoch in range(epochs):
            total_loss = 0
            for batch_input, batch_target in dataloader:
                optimizer.zero_grad()
                reconstructed, encoded = model(batch_input)
                loss = criterion(reconstructed, batch_target)
                loss.backward()
                
                # --- FIX: Add gradient clipping to prevent exploding gradients and NaN loss ---
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            final_loss = avg_loss
            if verbose and self.verbose and (epoch + 1) % 10 == 0:
                self.print_info(f"Transformer-AE Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")
        
        # Generate encoded features for the entire dataset
        model.eval()
        with torch.no_grad():
            all_sequences = torch.FloatTensor(sequences).to(device)
            _, encoded_features = model(all_sequences)
            encoded_features = encoded_features.cpu().numpy()
        
        # Create feature names for encoded dimensions
        encoded_feature_names = [f'transformer_ae_dim_{i}' for i in range(encoding_dim)]
        
        # Create dataframe with encoded features aligned to original timestamps
        # Take the last encoded feature for each position
        encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names)
        
        # Align with original dataframe (pad beginning with first values)
        pad_length = len(train_df) - len(encoded_df)
        if pad_length > 0:
            padding = pd.DataFrame(np.repeat(encoded_df.iloc[[0]].values, pad_length, axis=0), columns=encoded_feature_names)
            encoded_df = pd.concat([padding, encoded_df]).reset_index(drop=True)

        encoded_df.index = train_df.index
        
        self.autoencoder_features['transformer'] = encoded_df

        return {
            'encoded_features': encoded_df,
            'model': model,
            'scaler': scaler,
            'feature_names': encoded_feature_names,
            'method': 'transformer_autoencoder',
            'loss': final_loss
        }

    def autoencoder_selection_with_reconstruction_error(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
                                                    ae_method: str = 'lstm',
                                                    top_k_features: int = 20, **kwargs) -> Dict[str, Any]:
        """
        Selects features based on autoencoder reconstruction error.
        """
        self.print_info(f"Starting autoencoder feature selection based on reconstruction error...")

        # --- FIX: Drop NaN values to prevent issues during scaling and training ---
        train_df_clean = train_df.dropna().reset_index(drop=True)
        val_df_clean = val_df.dropna().reset_index(drop=True)

        if train_df_clean.empty:
            self.print_info("Warning: Training dataframe is empty after dropping NaNs. Skipping AE selection.")
            return {}

        feature_cols = [col for col in train_df_clean.columns if col not in ['unique_id', 'ds', 'y', 'target_stationary']]
        
        reconstruction_errors = {}

        for feature in feature_cols:
            # Create a temporary dataframe with only the target feature
            temp_train_df = train_df_clean[[feature]].copy()
            temp_val_df = val_df_clean[[feature]].copy()
            
            # Fit autoencoder
            ae_result = self._fit_autoencoder(
                temp_train_df, 
                ae_method=ae_method,
                **kwargs
            )
            model = ae_result['model']
            scaler = ae_result['scaler']

            # Evaluate on validation set
            val_sequences, _, _ = self._prepare_sequences_for_autoencoder(
                temp_val_df,
                scaler=scaler,
                fit_scaler=False,
                sequence_length=kwargs.get('sequence_length', 20)
            )
            
            if val_sequences.shape[0] == 0:
                reconstruction_errors[feature] = np.inf
                continue

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            model.eval()

            with torch.no_grad():
                val_tensor = torch.FloatTensor(val_sequences).to(device)
                reconstructed, _ = model(val_tensor)
                
                # Calculate loss
                loss = nn.MSELoss(reduction='mean')(reconstructed, val_tensor)
                reconstruction_errors[feature] = loss.item()

        if not reconstruction_errors:
            self.print_info("No reconstruction errors were calculated.")
            return {}

        # Sort features by reconstruction error (higher error = more complex/noisy)
        error_df = pd.DataFrame(
            list(reconstruction_errors.items()), 
            columns=['feature', 'reconstruction_error']
        ).sort_values('reconstruction_error', ascending=False)

        # Higher error means the feature is less predictable, potentially more noisy or complex.
        # We select features that are MORE predictable (lower error) as being more stable.
        selected_features = error_df.nsmallest(top_k_features, 'reconstruction_error')['feature'].tolist()
        
        self.print_info(f"Autoencoder selected top {len(selected_features)} features with the lowest reconstruction error.")

        return {
            'selected_features': selected_features,
            'feature_ranking': error_df
        } 