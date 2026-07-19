"""
src/data/features.py

Dos tipos de features:
1. Features tabulares (para el modelo baseline XGBoost/LightGBM): estadísticos
   de ventana deslizante (media, std, pendiente) + energía espectral vía FFT.
2. Secuencias 3D (para LSTM / 1D-CNN en PyTorch): ventanas deslizantes crudas
   normalizadas, forma (n_muestras, window_size, n_sensores).
"""

import numpy as np
import pandas as pd
from src.data.preprocessing import USEFUL_SENSORS


def fft_band_energy(signal: np.ndarray, n_bands: int = 3) -> np.ndarray:
    """Aplica FFT a una ventana de señal y devuelve la energía en n_bands bandas
    de frecuencia. Es la feature de 'dominio de la frecuencia' pedida en el
    enunciado: capta patrones vibratorios que la media/varianza no ven."""
    spectrum = np.abs(np.fft.rfft(signal - signal.mean()))
    bands = np.array_split(spectrum, n_bands)
    return np.array([b.sum() for b in bands])


def build_tabular_features(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Para cada fila (motor, ciclo) calcula estadísticos de la ventana de los
    últimos `window` ciclos: media, std, pendiente (tendencia) y energía FFT
    por banda, para cada sensor útil."""
    frames = []
    for unit, g in df.groupby("unit"):
        g = g.sort_values("cycle").reset_index(drop=True)
        feat = pd.DataFrame(index=g.index)
        feat["unit"] = unit
        feat["cycle"] = g["cycle"]

        for col in USEFUL_SENSORS:
            roll = g[col].rolling(window, min_periods=1)
            feat[f"{col}_mean"] = roll.mean()
            feat[f"{col}_std"] = roll.std().fillna(0)

            # pendiente simple: diferencia entre el valor actual y el de hace `window` pasos
            feat[f"{col}_slope"] = g[col].diff(window).fillna(0) / window

            # energía FFT por banda, calculada sobre ventanas completas
            fft_feats = np.zeros((len(g), 3))
            values = g[col].values
            for i in range(len(g)):
                start = max(0, i - window + 1)
                seg = values[start: i + 1]
                if len(seg) >= 4:
                    fft_feats[i] = fft_band_energy(seg, n_bands=3)
            feat[f"{col}_fft_low"] = fft_feats[:, 0]
            feat[f"{col}_fft_mid"] = fft_feats[:, 1]
            feat[f"{col}_fft_high"] = fft_feats[:, 2]

        if "RUL" in g.columns:
            feat["RUL"] = g["RUL"].values
        frames.append(feat)
    return pd.concat(frames, ignore_index=True)


def build_sequences(df: pd.DataFrame, window: int = 30, feature_cols=None):
    """Genera ventanas deslizantes crudas por motor para los modelos secuenciales.
    Devuelve X con forma (n, window, n_features) e y con forma (n,)."""
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
            if len(seg) < window:
                pad = np.repeat(seg[:1], window - len(seg), axis=0)
                seg = np.vstack([pad, seg])
            X.append(seg)
            units_out.append(unit)
            cycles_out.append(g["cycle"].iloc[i])
            if rul is not None:
                y.append(rul[i])
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32) if y else None
    return X, y, np.array(units_out), np.array(cycles_out)
