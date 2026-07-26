# Predictive Maintenance and Failure Analysis in Industrial Engines

Predicting the **RUL (Remaining Useful Life)** — the remaining life cycles of an engine before failure — from telematics data (vibration, temperature, pressure), using the **NASA C-MAPSS** dataset.

## What does this project solve?

An unexpected failure on an assembly line or in a wind turbine costs thousands of euros per minute of downtime. Instead of waiting for the engine to fail (reactive maintenance) or inspecting it every X hours (blind preventive maintenance), this pipeline estimates in real-time how many life cycles each asset has left, so that intervention can be scheduled right before it fails.

## Repository structure

```text
predictive-maintenance-rul/
├── generate_synthetic_data.py   # generates test data in C-MAPSS format
├── requirements.txt
├── data/
│   └── raw/                     # train_FD001.txt, test_FD001.txt, RUL_FD001.txt go here
├── src/
│   ├── data/
│   │   ├── loader.py            # raw files loading
│   │   ├── preprocessing.py     # noise filtering, RUL calculation, normalisation
│   │   └── features.py          # window features + FFT (frequency domain)
│   ├── models/
│   │   ├── losses.py            # custom asymmetric loss (penalises false negatives more heavily)
│   │   ├── baseline.py          # XGBoost with the custom loss as objective
│   │   └── sequence_models.py   # LSTM and 1D-CNN in PyTorch
│   ├── eval/
│   │   └── metrics.py           # RMSE, business score, alert levels
│   └── train.py                 # orchestrates the entire pipeline
├── app/
│   └── dashboard.py             # Streamlit dashboard with per-engine alerts
└── models_saved/                # trained artefacts (generated when running train.py)
```

## Dataset: NASA C-MAPSS

The real dataset is downloaded from NASA's **PCoE Data Set Repository**:
https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/

Download `CMaps.zip`, and extract at least these three files from the FD001 subset (single engine, one operating condition, one failure mode) into `data/raw/`:

- `train_FD001.txt`
- `test_FD001.txt`
- `RUL_FD001.txt`

> If you haven't downloaded the dataset yet, you can generate synthetic data with the same format by running `python generate_synthetic_data.py`. This allows you to test the entire pipeline from start to finish.

## Installation (using uv)

```bash
uv venv          
source .venv/bin/activate # on Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

## How to run the complete pipeline

```bash
# 1. (optional, only if you don't have the real dataset) generate test data
python generate_synthetic_data.py

# 2. train baseline (XGBoost) + deep model (LSTM), evaluate and save artefacts
python -m src.train

# 3. launch the interactive dashboard
streamlit run app/dashboard.py
```

## Technical flow

**1. Pre-processing and features** (`src/data/`)
- Noise filtering with moving average per sensor and per engine.
- Calculation of the RUL label, capped (RUL cap) following standard C-MAPSS literature criteria: at the beginning of the engine's life, degradation is not observable, so it makes no sense to ask the model to predict it.
- Tabular features: mean, standard deviation, and slope in a sliding window + **spectral energy by frequency bands (FFT)** — the latter is the "frequency domain" part of the brief, and captures periodic vibratory patterns that simple statistics miss.
- Raw sequences normalised for deep models.

**2. Modelling** (`src/models/`)
- **Baseline**: XGBoost on the tabular features.
- **Deep model**: LSTM (and a 1D-CNN alternative included) in PyTorch on multivariate time windows.
- Both models are trained with the **same custom asymmetric loss function**, so the comparison is fair.

**3. Business metric** (`src/models/losses.py`)
- The loss penalises the `y_pred - y_true` error exponentially and asymmetrically:
  - If the model **overestimates the RUL** (says the engine will last longer than it actually will → false negative for failure, risk of catastrophic downtime) → harsh penalty.
  - If the model **underestimates the RUL** (sends for inspection prematurely → false positive, only costs an extra inspection) → mild penalty.
- Implemented both as an `nn.Module` for PyTorch and as a custom objective (gradient + hessian) for XGBoost.

**4. Dashboard** (`app/dashboard.py`)
- Comparative metrics XGBoost vs LSTM.
- Evolution of actual vs predicted RUL per engine.
- Alert traffic light system (🟢 normal / 🟡 inspect / 🔴 urgent) for the entire fleet.

## Extending the project

- Changing the subset (FD001 → FD002/FD003/FD004, with more operating conditions and failure modes) only requires passing `subset="FD002"` to the loader.
- Adding the 1D-CNN model to the dashboard: it is already implemented in `src/models/sequence_models.py`, it just needs to be trained in `train.py` exactly like the LSTM and have its predictions saved.
- Replacing the moving average with a Savitzky-Golay filter or wavelets if the sensor noise is more aggressive.
