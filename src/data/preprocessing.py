"""
src/data/preprocessing.py
- Time-series noise filtering (moving average calculated independently per engine/sensor).
- Target label calculation for Remaining Useful Life (RUL) with "clipping", following 
  the methodology proposed in the original NASA C-MAPSS benchmark paper (initial degradation 
  is unobservable, so maximum RUL is capped at a ceiling value, e.g., 125 cycles).
- Min-Max feature scaling fitted strictly on the training set to prevent data leakage.
"""

import numpy as np
import pandas as pd

# Define feature column names for sensors and operational settings
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
SETTING_COLS = [f"setting_{i}" for i in range(1, 4)]

# Constant sensors in C-MAPSS FD001 dataset (zero variance, providing no predictive value)
CONSTANT_SENSORS = ["sensor_1", "sensor_5", "sensor_6", "sensor_10",
                     "sensor_16", "sensor_18", "sensor_19"]

# Filter out constant sensors to keep only informative input features
USEFUL_SENSORS = [c for c in SENSOR_COLS if c not in CONSTANT_SENSORS]


def smooth_signals(df: pd.DataFrame, window: int = 5, cols=None) -> pd.DataFrame:
    """Noise filtering using a moving average, calculated independently per engine unit."""
    df = df.copy()
    cols = cols or USEFUL_SENSORS
    
    # Group by engine unit ('unit') to ensure moving averages do not bleed across different engines
    df[cols] = (
        df.groupby("unit")[cols]
        .apply(lambda g: g.rolling(window, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )
    return df


def add_rul_train(df: pd.DataFrame, rul_cap: int = 125) -> pd.DataFrame:
    """For the training set: RUL = max_engine_cycle - current_cycle, capped at rul_cap."""
    df = df.copy()
    
    # Obtain total lifespan (maximum cycle reached) for each engine unit
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    
    # Calculate linear RUL countdown towards failure (cycle 0)
    df["RUL"] = max_cycle - df["cycle"]
    
    # Apply piece-wise linear upper cap to reflect stable early-life health state
    df["RUL"] = df["RUL"].clip(upper=rul_cap)
    return df


def add_rul_test(df: pd.DataFrame, true_rul: pd.Series, rul_cap: int = 125) -> pd.DataFrame:
    """
    For the test set: true RUL is only known at the LAST observed cycle of each 
    truncated engine sequence (provided via external ground-truth file RUL_FDxxx.txt). 
    Reconstructs RUL backwards for historical time steps and applies upper clipping.
    """
    df = df.copy()
    true_rul = true_rul.reset_index(drop=True)
    units = sorted(df["unit"].unique())
    
    # Map each engine unit ID to its remaining ground-truth lifespan at the end of the test sequence
    rul_map = dict(zip(units, true_rul))

    # Identify the last recorded cycle in the truncated test run per engine
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    df["final_rul"] = df["unit"].map(rul_map)
    
    # Reconstruct historic RUL target backwards across time steps
    df["RUL"] = (max_cycle - df["cycle"]) + df["final_rul"]
    
    # Enforce identical piece-wise capping ceiling used during training
    df["RUL"] = df["RUL"].clip(upper=rul_cap)
    df.drop(columns=["final_rul"], inplace=True)
    return df


class MinMaxNormalizer:
    """Normaliser fitted strictly on training data to prevent evaluation data leakage."""

    def __init__(self, cols=None):
        self.cols = cols or USEFUL_SENSORS + SETTING_COLS
        self.min_ = None
        self.max_ = None

    def fit(self, df: pd.DataFrame):
        """Compute feature minimums and maximums exclusively from the training set."""
        self.min_ = df[self.cols].min()
        self.max_ = df[self.cols].max()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rescale features to [0, 1] using parameters fitted on training data."""
        df = df.copy()
        
        # Replace zero denominator with 1 to avoid division-by-zero errors on static channels
        denom = (self.max_ - self.min_).replace(0, 1)
        df[self.cols]
