# Reproducibility Guide

This document provides detailed instructions for reproducing all results presented in the paper "Comparative Benchmarking of Machine Learning and Deep Learning Models for Solar Photovoltaic Power Forecasting."

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Software Environment](#software-environment)
3. [Data Preparation](#data-preparation)
4. [Model Training](#model-training)
5. [Evaluation and Metrics](#evaluation-and-metrics)
6. [Figure Generation](#figure-generation)
7. [Random Seeds](#random-seeds)
8. [Hyperparameter Configurations](#hyperparameter-configurations)

## System Requirements

### Hardware

Our experiments were conducted on the following hardware:
- **CPU**: Intel Core i7-10700K @ 3.80GHz (8 cores, 16 threads)
- **RAM**: 32 GB DDR4 @ 3200 MHz
- **GPU**: NVIDIA RTX 3080 (10GB VRAM) - Optional but recommended for deep learning models
- **Storage**: 500 GB SSD

**Note**: All models can run on CPU-only systems, though training times for deep learning models (LSTM, GRU, CNN-BiGRU-Attention) will be significantly longer (~10-15× slower).

### Operating System

- **Primary**: Windows 10/11 Professional (64-bit)
- **Tested on**: Ubuntu 20.04 LTS, macOS 12+

## Software Environment

### Python Version

```bash
Python 3.8.10 (or higher, up to 3.11.x)
```

### Create Virtual Environment

```bash
# Windows PowerShell
python -m venv .venv
.venv\Scripts\Activate.ps1

# Linux/macOS
python3 -m venv .venv
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Verify Installation

```python
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
import tensorflow as tf

print(f"NumPy: {np.__version__}")
print(f"Pandas: {pd.__version__}")
print(f"Scikit-learn: {sklearn.__version__}")
print(f"XGBoost: {xgb.__version__}")
print(f"TensorFlow: {tf.__version__}")
```

**Expected output**:
```
NumPy: 1.24.3
Pandas: 2.0.2
Scikit-learn: 1.3.0
XGBoost: 1.7.6
TensorFlow: 2.13.0
```

## Data Preparation

### Download NASA POWER Data

The dataset covers Hengsha Island, Shanghai, China (31.3403°N, 121.8389°E) from January 1, 2020 to December 31, 2024.

**Variables retrieved**:
- `ALLSKY_SFC_SW_DWN` - Global Horizontal Irradiance (GHI)
- `ALLSKY_SFC_SW_DNI` - Direct Normal Irradiance (DNI)
- `T2M` - Temperature at 2 meters (°C)
- `RH2M` - Relative Humidity at 2 meters (%)
- `WS10M` - Wind Speed at 10 meters (m/s)
- `PS` - Surface Pressure (kPa)

**Access NASA POWER API**:
```python
# Example API request
import requests
import pandas as pd

base_url = "https://power.larc.nasa.gov/api/temporal/hourly/point"
params = {
    "parameters": "ALLSKY_SFC_SW_DWN,T2M,RH2M,WS10M",
    "community": "RE",
    "longitude": 121.8389,
    "latitude": 31.3403,
    "start": "20200101",
    "end": "20241231",
    "format": "JSON"
}

response = requests.get(base_url, params=params)
data = response.json()
```

### Data Preprocessing Steps

1. **Load raw data**: `hengsha_hourly_2020_2024.csv`
2. **Filter daylight hours**: Keep only records where GHI > 20 W/m²
3. **Handle missing values**: Linear interpolation (< 0.1% of data)
4. **Calculate derived features**:
   - Solar position angles (zenith, azimuth) using NREL SPA algorithm
   - Extraterrestrial horizontal irradiance (G₀h)
   - Clearness index (kt = GHI / G₀h)
5. **Compute normalized PV power**:
   ```python
   eta_0 = 0.18  # Reference efficiency
   alpha = 0.005  # Temperature coefficient
   T_NOCT = 45  # Nominal Operating Cell Temperature
   
   T_c = T_a + GHI * (T_NOCT - 20) / 800
   PV_pu = eta_0 * (1 - alpha * (T_c - 25)) * (GHI / 1000)
   ```

### Data Partitioning

**Strict chronological splits** (no random shuffling):

```python
# Training: 2020-01-01 to 2022-12-31 (60%)
train_data = df[(df['datetime'] >= '2020-01-01') & 
                 (df['datetime'] <= '2022-12-31')]

# Validation: 2023-01-01 to 2023-12-31 (20%)
val_data = df[(df['datetime'] >= '2023-01-01') & 
              (df['datetime'] <= '2023-12-31')]

# Testing: 2024-01-01 to 2024-12-31 (20%)
test_data = df[(df['datetime'] >= '2024-01-01') & 
               (df['datetime'] <= '2024-12-31')]
```

**Result**: 
- Training: 26,294 hours
- Validation: 8,746 hours
- Testing: 8,784 hours
- **Total**: 43,824 hours

## Random Seeds

For reproducibility, the following random seeds were used across all experiments:

```python
import random
import numpy as np
import tensorflow as tf

RANDOM_SEED = 42

# Python random
random.seed(RANDOM_SEED)

# NumPy
np.random.seed(RANDOM_SEED)

# TensorFlow/Keras
tf.random.set_seed(RANDOM_SEED)

# Scikit-learn (pass to estimators)
# e.g., RandomForestRegressor(random_state=RANDOM_SEED)
```

## Hyperparameter Configurations

### XGBoost

```python
xgb_params = {
    'objective': 'reg:squarederror',
    'learning_rate': 0.05,
    'max_depth': 6,
    'n_estimators': 150,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 1,
    'gamma': 0,
    'reg_alpha': 0.1,  # L1 regularization
    'reg_lambda': 1.0,  # L2 regularization
    'random_state': 42,
    'n_jobs': -1
}
```

**GridSearchCV ranges**:
- `learning_rate`: [0.01, 0.05, 0.1]
- `max_depth`: [4, 6, 8]
- `n_estimators`: [100, 150, 200]
- `subsample`: [0.7, 0.8, 0.9]

### Random Forest

```python
rf_params = {
    'n_estimators': 200,
    'max_depth': 10,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_features': 'sqrt',
    'bootstrap': True,
    'random_state': 42,
    'n_jobs': -1
}
```

### ANFIS-SC

```python
anfis_params = {
    'n_inputs': 6,
    'membership_function': 'gaussmf',
    'cluster_method': 'subtractive',
    'cluster_radius': 0.28,  # Optimized via validation
    'max_iterations': 100,
    'error_criterion': 0.001
}
```

### GRU

```python
gru_architecture = {
    'input_shape': (24, 6),  # 24-hour sequence, 6 features
    'gru_units': [64],
    'dropout': 0.3,
    'recurrent_dropout': 0.2,
    'activation': 'tanh',
    'optimizer': 'adam',
    'learning_rate': 0.001,
    'batch_size': 64,
    'epochs': 50,
    'early_stopping_patience': 10
}

# Model structure
model = Sequential([
    GRU(64, dropout=0.3, recurrent_dropout=0.2, 
        input_shape=(24, 6)),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='linear')
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='mse',
              metrics=['mae'])
```

### LSTM

```python
lstm_architecture = {
    'input_shape': (24, 6),
    'lstm_units': [32],  # Smaller to reduce overfitting
    'dropout': 0.3,
    'recurrent_dropout': 0.2,
    'optimizer': 'adam',
    'learning_rate': 0.001,
    'batch_size': 64,
    'epochs': 50
}
```

### CNN-BiGRU-Attention v2

```python
cnn_bigru_params = {
    'cnn_filters': [32, 64],
    'kernel_size': 3,
    'pool_size': 2,
    'bigru_units': 64,
    'attention_units': 32,
    'dropout': 0.4,
    'learning_rate': 0.0005,
    'batch_size': 32,
    'epochs': 100,
    'early_stopping_patience': 15
}
```

## Model Training

### Training Time Benchmarks

Expected training times on our hardware (Intel i7 + RTX 3080):

| Model | CPU Only | With GPU |
|-------|----------|----------|
| XGBoost | 12.45s | N/A (CPU-optimized) |
| Random Forest | 18.67s | N/A (CPU-optimized) |
| ANFIS-SC | 8.92s | N/A |
| GRU | ~25 min | 145.33s (~2.4 min) |
| LSTM | ~30 min | 162.48s (~2.7 min) |
| CNN-BiGRU-AM | ~35 min | 198.75s (~3.3 min) |

## Evaluation and Metrics

All metrics computed on normalized PV power (0-1 scale):

```python
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def calculate_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    # sMAPE
    smape = np.mean(2 * np.abs(y_pred - y_true) / 
                    (np.abs(y_true) + np.abs(y_pred))) * 100
    
    # Skill Score (vs 24-hour persistence)
    persistence_error = mean_squared_error(y_true, y_persistence)
    model_error = mean_squared_error(y_true, y_pred)
    skill_score = 1 - (model_error / persistence_error)
    
    return {
        'R2': r2,
        'RMSE': rmse,
        'MAE': mae,
        'sMAPE': smape,
        'Skill_Score': skill_score
    }
```

## Figure Generation

All figures can be regenerated using:

```bash
python scripts/make_figs.py
```

This script generates:
- Taylor diagrams
- Parity plots
- Seasonal distribution plots
- Residual analysis plots
- Time series comparisons
- Feature importance plots

## Computational Resources

**Total computation time** (all models, including hyperparameter tuning):
- **With GPU**: ~8-10 hours
- **CPU only**: ~40-50 hours

**Storage requirements**:
- Dataset: ~50 MB
- Trained models: ~200 MB
- Figures: ~100 MB
- **Total**: ~350 MB

## Known Issues and Troubleshooting

1. **TensorFlow GPU issues**: Ensure CUDA 11.8 and cuDNN 8.6 are installed
2. **Memory errors with deep learning**: Reduce batch size if RAM < 16GB
3. **Scikit-fuzzy installation**: May require manual installation via conda

## Contact

For reproduction issues or questions:
- **Email**: your.email@institution.edu
- **GitHub Issues**: [Link to repository issues]

## Version History

- **v1.0** (January 2026): Initial release with paper publication
