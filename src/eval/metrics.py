"""
src/eval/metrics.py
Evaluation metrics for the RUL problem:
- RMSE (Regression Standard Error)
- Asymmetric Business Score (same function used as loss)
- Simplified "alert" matrix: classifies into green/yellow/red zones
according to the predicted RUL, to generate traffic light-type alerts on the dashboard.
"""

import numpy as np
from src.models.losses import asymmetric_score_numpy


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.array(y_pred) - np.array(y_true)) ** 2)))


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.array(y_pred) - np.array(y_true))))


def business_score(y_true, y_pred, alpha_fn=13.0, alpha_fp=10.0) -> float:
    return asymmetric_score_numpy(np.array(y_true), np.array(y_pred), alpha_fn, alpha_fp)


def alert_level(rul_pred: float, red_th: float = 20, yellow_th: float = 50) -> str:
    """Classifies the predicted RUL into operational alert levels."""
    if rul_pred <= red_th:
        return "RED - Urgent maintenance"
    elif rul_pred <= yellow_th:
        return "YELLOW - Schedule review"
    return "GREEN - Normal operation"


def summarize(y_true, y_pred) -> dict:
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "Business_Score": business_score(y_true, y_pred),
    }
