"""
src/train.py
Orchestrates the complete pipeline:
1. Loads raw data (C-MAPSS)
2. Noise filtering + RUL calculation + normalization
3. Feature extraction (tabular for XGBoost, sequential for LSTM/CNN)
4. Baseline training (XGBoost, custom asymmetric loss)
5. Deep model training (LSTM, custom asymmetric loss)
6. Benchmarking and saving artifacts for the dashboard

Use:
    python -m src.train
"""

import os
import json
import joblib
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from src.data.loader import load_train, load_test, load_true_rul
from src.data.preprocessing import smooth_signals, add_rul_train, add_rul_test, MinMaxNormalizer, USEFUL_SENSORS
from src.data.features import build_tabular_features, build_sequences
from src.models.baseline import train_xgb_baseline, predict_xgb
from src.models.sequence_models import LSTMRegressor, train_torch_model
from src.models.losses import AsymmetricRULLoss
from src.eval.metrics import summarize

ROOT = os.path.dirname(os.path.dirname(__file__))
RAW_DIR = os.path.join(ROOT, "data", "raw")
SAVE_DIR = os.path.join(ROOT, "models_saved")
os.makedirs(SAVE_DIR, exist_ok=True)

WINDOW = 30
RUL_CAP = 125


def main():
    print("1) Loading data...")
    train_raw = load_train(RAW_DIR)
    test_raw = load_test(RAW_DIR)
    true_rul = load_true_rul(RAW_DIR)

    print("2) Noise filtering + RUL labels...")
    train_s = smooth_signals(train_raw)
    test_s = smooth_signals(test_raw)
    train_s = add_rul_train(train_s, rul_cap=RUL_CAP)
    test_s = add_rul_test(test_s, true_rul, rul_cap=RUL_CAP)

    print("3) Normalization (adjusted only with train)...")
    normalizer = MinMaxNormalizer()
    train_n = normalizer.fit_transform(train_s)
    test_n = normalizer.transform(test_s)
    joblib.dump(normalizer, os.path.join(SAVE_DIR, "normalizer.pkl"))

    # BASELINE: tabular feature + XGBoost
    print("4) Extracting tabular features (FFT + window statistics)...")
    train_feats = build_tabular_features(train_n, window=WINDOW)
    test_feats = build_tabular_features(test_n, window=WINDOW)

    feature_cols = [c for c in train_feats.columns if c not in ("unit", "cycle", "RUL")]
    X = train_feats[feature_cols].values
    y = train_feats["RUL"].values
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    print("5) Training baseline XGBoost with custom asymmetric loss...")
    xgb_model = train_xgb_baseline(X_train, y_train, X_val, y_val)
    xgb_model.save_model(os.path.join(SAVE_DIR, "xgb_baseline.json"))
    with open(os.path.join(SAVE_DIR, "feature_cols.json"), "w") as f:
        json.dump(feature_cols, f)

    X_test_tab = test_feats[feature_cols].values
    y_test_tab = test_feats["RUL"].values
    xgb_pred = predict_xgb(xgb_model, X_test_tab)
    xgb_metrics = summarize(y_test_tab, xgb_pred)
    print("  XGBoost results (test):", xgb_metrics)

    # DEEP MODEL: LSTM over sequences
    print("6) Building sequences for LSTM...")
    X_seq, y_seq, _, _ = build_sequences(train_n, window=WINDOW, feature_cols=USEFUL_SENSORS)
    Xtr, Xval, ytr, yval = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

    train_loader = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(ytr)), batch_size=128, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(Xval), torch.tensor(yval)), batch_size=128)

    print("7) Training LSTM with custom asymmetric loss...")
    model = LSTMRegressor(n_features=len(USEFUL_SENSORS))
    loss_fn = AsymmetricRULLoss(alpha_fn=13.0, alpha_fp=10.0)
    model, history = train_torch_model(model, train_loader, val_loader, loss_fn, epochs=15)
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "lstm_model.pt"))

    X_test_seq, y_test_seq, test_units, test_cycles = build_sequences(
        test_n, window=WINDOW, feature_cols=USEFUL_SENSORS
    )
    model.eval()
    with torch.no_grad():
        lstm_pred = model(torch.tensor(X_test_seq)).numpy()
    lstm_metrics = summarize(y_test_seq, lstm_pred)
    print("   LSTM results (test):", lstm_metrics)

    # Save results for the dashboard
    results = {
        "xgboost": xgb_metrics,
        "lstm": lstm_metrics,
        "window": WINDOW,
        "rul_cap": RUL_CAP,
    }
    with open(os.path.join(SAVE_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    np.savez(
        os.path.join(SAVE_DIR, "test_predictions.npz"),
        unit=test_units, cycle=test_cycles,
        y_true=y_test_seq, lstm_pred=lstm_pred,
    )

    print("\nPipeline completed. Artifacts saved in models_saved/")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
