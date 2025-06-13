import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from src.pipelines.model_evaluation import prepare_pipeline_data

def main():
    # Load the dataset
    train_data, val_data, hist_exog_list, data_info = prepare_pipeline_data(horizon=7, test_length_multiplier=1)

    train_data = train_data.drop(columns=['unique_id'])
    train_data = train_data.set_index('ds')
    val_data = val_data.drop(columns=['unique_id'])
    val_data = val_data.set_index('ds')

    # Select features and target
    X_train = train_data.drop(columns=['y']).values
    y_train = train_data['y'].values

    X_val = val_data.drop(columns=['y']).values
    y_val = val_data['y'].values


    # Standardize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Fit and transform on the training data
    X_val_scaled = scaler.transform(X_val)  # Transform validation data using the same scaler

    # # Save the scaler
    # joblib.dump(scaler, "../models/scaler.pkl")

    # Fit PCA on the training set
    pca = PCA()
    pca.fit(X_train_scaled)

    # Determine the optimal number of components (k)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    optimal_k = np.argmax(cumulative_variance >= 0.95) + 1
    print(f"Optimal number of components (k): {optimal_k}")

    # Apply PCA with the optimal number of components
    pca = PCA(n_components=optimal_k)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)

    # Create DataFrames for the PCA-transformed data
    train_pca_df = pd.DataFrame(
        X_train_pca,
        columns=[f'pca_{i+1}' for i in range(optimal_k)],
        index=train_data.index
    )
    train_pca_df['y'] = y_train

    val_pca_df = pd.DataFrame(
        X_val_pca,
        columns=[f'pca_{i+1}' for i in range(optimal_k)],
        index=val_data.index
    )
    val_pca_df['y'] = y_val

    # Print results
    print("Train PCA DataFrame:")
    print(train_pca_df.head())

if __name__ == "__main__":
    main()