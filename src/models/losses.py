"""
src/models/losses.py

Business metric: A greater penalty should be applied when the model
incorrectly predicts a higher RUL (Remaining Life of Use) than the actual RUL 
(i.e., it predicts the engine has more life remaining than it actually does -> 
false negative for failure, the asset is inspected late) than when it predicts 
a lower RUL (false positive -> it is inspected prematurely, only 
costing one extra inspection).

error = y_pred - y_true
- error > 0 -> optimistic/dangerous prediction (FN) -> strong penalty
- error < 0 -> conservative prediction (FP) -> mild penalty

This philosophy is implemented as a trainable loss in 
both PyTorch and as a custom XGBoost target.
"""

import numpy as np
import torch
import torch.nn as nn


class AsymmetricRULLoss(nn.Module):
    """Loss for PyTorch. alpha_fn > alpha_fp implies penalizing FNs more."""

    def __init__(self, alpha_fn: float = 13.0, alpha_fp: float = 10.0):
        super().__init__()
        self.alpha_fn = alpha_fn
        self.alpha_fp = alpha_fp

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        error = y_pred - y_true
        loss = torch.where(
            error > 0,
            torch.exp(error / self.alpha_fn) - 1,   # Overestimation -> heavy penalty
            torch.exp(-error / self.alpha_fp) - 1,  # Underestimation -> mild penalty
        )
        return loss.mean()


def asymmetric_score_numpy(y_true: np.ndarray, y_pred: np.ndarray,
                            alpha_fn: float = 13.0, alpha_fp: float = 10.0) -> float:
    """Same function in pure NumPy, useful for evaluation and for XGBoost."""
    error = y_pred - y_true
    score = np.where(error > 0, np.exp(error / alpha_fn) - 1, np.exp(-error / alpha_fp) - 1)
    return float(np.mean(score))


def xgb_asymmetric_objective(alpha_fn: float = 13.0, alpha_fp: float = 10.0):
    """Returns a custom XGBoost-compatible objective function (grad, hess)
    derived analytically from the same asymmetric loss function as above."""

    def objective(y_pred: np.ndarray, dtrain) -> tuple:
        y_true = dtrain.get_label()
        error = y_pred - y_true
        grad = np.where(
            error > 0,
            np.exp(error / alpha_fn) / alpha_fn,
            -np.exp(-error / alpha_fp) / alpha_fp,
        )
        hess = np.where(
            error > 0,
            np.exp(error / alpha_fn) / (alpha_fn ** 2),
            np.exp(-error / alpha_fp) / (alpha_fp ** 2),
        )
        return grad, hess

    return objective
