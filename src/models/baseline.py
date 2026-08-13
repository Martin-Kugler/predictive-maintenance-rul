"""
src/models/baseline.py
Baseline regression model: XGBoost trained with the custom asymmetric loss.
"""

import xgboost as xgb
import numpy as np
from src.models.losses import xgb_asymmetric_objective, asymmetric_score_numpy


def train_xgb_baseline(X_train, y_train, X_val, y_val,
                        alpha_fn: float = 13.0, alpha_fp: float = 10.0,
                        num_boost_round: int = 300):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    params = {
        "max_depth": 6,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": 42,
    }

    objective = xgb_asymmetric_objective(alpha_fn, alpha_fp)

    def eval_rmse(y_pred, dmat):
        y_true = dmat.get_label()
        return "rmse", float(np.sqrt(np.mean((y_pred - y_true) ** 2)))

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        obj=objective,
        evals=[(dtrain, "train"), (dval, "val")],
        custom_metric=eval_rmse,
        early_stopping_rounds=20,
        verbose_eval=False,
    )
    return model


def predict_xgb(model, X):
    return model.predict(xgb.DMatrix(X))
