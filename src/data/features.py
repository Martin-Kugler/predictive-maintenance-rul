"""
src/data/features.py

Two types of feature representations:
1. Tabular features (for the baseline XGBoost/LightGBM model): sliding window statistics 
   (mean, std, slope) + spectral energy via FFT.
2. 3D Sequences (for LSTM / 1D-CNN in PyTorch): raw normalized sliding windows, 
   shape (n_samples, window_size, n_features).
"""

import numpy as np
import pandas as pd
from src.data.preprocessing import USEFUL_SENSORS


def fft_band_energy(signal: np.ndarray, n_bands: int = 3) -> np.ndarray:
    """
    Applies FFT to a signal window and returns the energy across n_bands frequency bands. 
    This provides the 'frequency domain' feature required to capture vibrational patterns 
    that standard time-domain metrics (mean/variance) miss.
    """
    # Subtract mean to remove the DC offset (0 Hz static baseline) before applying Real FFT
    spectrum = np.abs(np.fft.rfft(signal - signal.mean()))
    
    # Split the magnitude spectrum into equal frequency sub-bands
    bands = np.array_split(spectrum, n_bands)
    
    # Integrate energy (sum magnitudes) within each band
    return np.array([b.sum() for b in bands])


def build_tabular_features(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    For each row (engine unit, cycle), computes statistics over the preceding `window` cycles: 
    moving average, moving standard deviation, trend slope, and FFT band energy for each useful sensor.
    """
    frames = []
    
    # Group by engine unit to keep time series strictly isolated per asset
    for unit, g in df.groupby("unit"):
        # Ensure chronological ordering prior to windowing
        g = g.sort_values("cycle").reset_index(drop=True)
        feat = pd.DataFrame(index=g.index)
        feat["unit"] = unit
        feat["cycle"] = g["cycle"]

        for col in USEFUL_SENSORS:
            # Construct pandas rolling window object with left-side minimum period fallback
            roll = g[col].rolling(window, min_periods=1)
            feat[f"{col}_mean"] = roll.mean()
            feat[f"{col}_std"] = roll.std().fillna(0)

            # Simple slope: difference between current value and the reading `window` steps ago, scaled by window size
            feat[f"{col}_slope"] = g[col].diff(window).fillna(0) / window

            # Pre-allocate array to store FFT energy features across frequency bands
            fft_feats = np.zeros((len(g), 3))
            values = g[col].values
            
            # Slice sliding window segments row-by-row along the engine timeline
            for i in range(len(g)):
                start = max(0, i - window + 1)
                seg = values[start: i + 1]
                
                # Minimum sample threshold required to calculate meaningful Discrete Fourier Transform
                if len(seg) >= 4:
                    fft_feats[i] = fft_band_energy(seg, n_bands=3)
                    
            # Assign extracted spectral bands as scalar tabular columns
            feat[f"{col}_fft_low"] = fft_feats[:, 0]
            feat[f"{col}_fft_mid"] = fft_feats[:, 1]
            feat[f"{col}_fft_high"] = fft_feats[:, 2]

        # Preserve ground-truth RUL target if present in dataset
        if "RUL" in g.columns:
            feat["RUL"] = g["RUL"].values
        frames.append(feat)
        
    # Recombine engine DataFrames into a single tabular feature matrix
    return pd.concat(frames, ignore_index=True)


def build_sequences(df: pd.DataFrame, window: int = 30, feature_cols=None):
    """
    Generates raw sliding window 3D sequences per engine unit for deep sequential architectures.
    Returns X with shape (n_samples, window_size, n_features) and y with shape (n_samples,).
    """
    feature_cols = feature_cols or USEFUL_SENSORS
    X, y, units_out, cycles_out = [], [], [], []
    
    for unit, g in df.groupby("unit"):
        g = g.sort_values("cycle").reset_index(drop=True)
        values = g[feature_cols].values
        rul = g["RUL"].values if "RUL" in g.columns else None
        n = len(g)
        
        for i in range(n):
            start = max(0, i - window + 1)
            seg = values[start: i + 1]
            
            # Left-padding via Zero-Order Hold (repeating initial cycle reading) for early lifespan steps
            if len(seg) < window:
                pad = np.repeat(seg[:1], window - len(seg), axis=0)
                seg = np.vstack([pad, seg])
                
            X.append(seg)
            units_out.append(unit)
            cycles_out.append(g["cycle"].iloc[i])
            if rul is not None:
                y.append(rul[i])
                
    # Cast to float32 NumPy arrays ready for PyTorch tensor ingestion
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32) if y else None
    return X, y, np.array(units_out), np.array(cycles_out)
