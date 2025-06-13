# Bitcoin Price Forecasting Feature Selection Workflow

This document provides a comprehensive workflow diagram for the feature selection strategy used in the Bitcoin price forecasting project.

## Complete Feature Selection Workflow

```mermaid
graph TD
    A[Start: main function] --> B{Load Data}
    B --> C[prepare_pipeline_data]
    C --> D{Combine train + test datasets}
    D --> E[Initialize FeatureSelector]
    E --> F[run_complete_feature_selection_strategy]
    
    subgraph Main["FeatureSelector Main Process"]
        F --> G{Sort DataFrame by ds}
        G --> H[Split into Train/Validation<br/>90% train, 10% validation]
        H --> I[robust_comprehensive_selection]
        I --> J[_generate_feature_recommendations]
        J --> K[Return results dictionary]
    end
    
    subgraph Selection["Robust Comprehensive Selection"]
        I --> L{SHAP-based Selection}
        I --> M{Autoencoder Selection}
        I --> N{Generate Robust Recommendations}
        I --> O{Handle Multicollinearity}
        I --> P{Permutation Importance Validation}
        
        L --> L1[XGBoost SHAP Analysis]
        L --> L2[LightGBM SHAP Analysis]
        L --> L3[Random Forest SHAP Analysis]
        
        M --> M1[LSTM Autoencoder]
        M --> M2[Transformer Autoencoder]
        M --> M3[Reconstruction Error Analysis]
        
        N --> N1[Aggregate Feature Counts]
        N --> N2[Calculate Combined Scores]
        N --> N3[Assess CV Stability]
        N --> N4[Apply Consensus Criteria]
        
        O --> O1[Calculate VIF]
        O --> O2[Hierarchical Clustering]
        O --> O3[Remove Collinear Features]
        
        P --> P1[Time Series Cross-Validation]
        P --> P2[Permutation Feature Importance]
        P --> P3[Statistical Significance Test]
    end
    
    subgraph SHAP["SHAP-based Feature Selection Details"]
        L1 --> S1[Time Series CV Split]
        S1 --> S2[Train XGBoost Models]
        S2 --> S3[Calculate SHAP Values]
        S3 --> S4[Aggregate across CV folds]
        S4 --> S5[Select top percentile features]
        S5 --> S6[Assess CV stability]
        
        L2 --> T1[Time Series CV Split]
        T1 --> T2[Train LightGBM Models]
        T2 --> T3[Calculate SHAP Values]
        T3 --> T4[Aggregate across CV folds]
        T4 --> T5[Select top percentile features]
        T5 --> T6[Assess CV stability]
        
        L3 --> U1[Time Series CV Split]
        U1 --> U2[Train Random Forest Models]
        U2 --> U3[Calculate Feature Importance]
        U3 --> U4[Aggregate across CV folds]
        U4 --> U5[Select top percentile features]
        U5 --> U6[Assess CV stability]
    end
    
    subgraph AE["Autoencoder Selection Details"]
        M1 --> AE1[Prepare sequence data]
        AE1 --> AE2[Train LSTM Autoencoder]
        AE2 --> AE3[Calculate reconstruction errors]
        AE3 --> AE4[Rank features by error contribution]
        AE4 --> AE5[Select top-k features]
        
        M2 --> AE6[Prepare sequence data]
        AE6 --> AE7[Train Transformer Autoencoder]
        AE7 --> AE8[Calculate reconstruction errors]
        AE8 --> AE9[Rank features by error contribution]
        AE9 --> AE10[Select top-k features]
    end
    
    K --> Q[Print Results Summary]
    Q --> R[Extract consensus_features]
    R --> S[Filter original datasets]
    S --> T[Save to Parquet file]
    T --> U[End]
    
    style A fill:#e1f5fe
    style U fill:#e8f5e8
    style F fill:#fff3e0
    style I fill:#fce4ec
    style L fill:#f3e5f5
    style M fill:#e0f2f1
    style N fill:#fff8e1
```

## Key Components Explained

### 1. Data Preparation Phase
- **Load Data**: Uses `prepare_pipeline_data()` to load training and testing datasets
- **Combine Datasets**: Temporarily combines train and test data for comprehensive feature analysis
- **Internal Splitting**: The FeatureSelector handles proper train/validation splitting internally to prevent data leakage

### 2. Robust Comprehensive Selection
This is the core of the feature selection strategy, implementing multiple complementary approaches:

#### A. SHAP-based Selection
- **XGBoost SHAP**: Tree-based model with SHAP value analysis
- **LightGBM SHAP**: Gradient boosting with SHAP interpretability
- **Random Forest SHAP**: Ensemble method with feature importance
- **Time Series CV**: Uses TimeSeriesSplit for temporal data integrity
- **Stability Assessment**: Evaluates feature consistency across CV folds

#### B. Autoencoder Selection
- **LSTM Autoencoder**: Captures temporal dependencies in feature reconstruction
- **Transformer Autoencoder**: Attention-based feature learning
- **Reconstruction Error**: Identifies features most critical for data reconstruction
- **Top-k Selection**: Selects features with highest reconstruction importance

#### C. Consensus Generation
- **Feature Aggregation**: Combines results from all selection methods
- **Scoring System**: Calculates combined importance scores
- **Stability Criteria**: Requires CV stability > 0.6
- **Consensus Threshold**: Features must be selected by ≥2 methods

### 3. Post-processing Steps

#### A. Multicollinearity Handling
- **VIF Calculation**: Identifies highly correlated features
- **Hierarchical Clustering**: Groups similar features
- **Intelligent Removal**: Keeps most important features from correlated groups

#### B. Permutation Importance Validation
- **Final Validation**: Tests selected features on held-out validation set
- **Statistical Significance**: Ensures features have meaningful predictive power
- **Robustness Check**: Confirms feature importance across multiple permutations

### 4. Output Generation
- **Results Summary**: Comprehensive analysis of feature selection process
- **Consensus Features**: Final list of recommended features
- **Data Export**: Filtered datasets saved to `data/final/final_feature_selected_data.parquet`

## Benefits of This Approach

1. **Multi-method Consensus**: Reduces selection bias by combining multiple techniques
2. **Time Series Aware**: Prevents data leakage through proper temporal splitting
3. **Robustness**: Cross-validation and stability assessments ensure reliable selections
4. **Interpretability**: SHAP values provide clear feature importance explanations
5. **Deep Learning Ready**: Autoencoder methods prepare features for neural network models
6. **Collinearity Handling**: Prevents redundant features from entering final models
7. **Validation**: Permutation importance provides final confirmation of feature utility

This comprehensive workflow ensures that only the most predictive, stable, and non-redundant features are selected for your Bitcoin price forecasting models.


