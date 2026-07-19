"""
src/data/preprocessing.py
- Filtrado de ruido en series temporales (media móvil por motor/sensor).
- Cálculo de la etiqueta RUL (Remaining Useful Life), con "clipping" al estilo
  del paper original de C-MAPSS (la degradación no es visible al inicio de vida
  del motor, así que se acota el RUL máximo, p.ej. a 125 ciclos).
- Normalización Min-Max ajustada solo con el train.
"""

import numpy as np
import pandas as pd

SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
SETTING_COLS = [f"setting_{i}" for i in range(1, 4)]

# Sensores constantes en C-MAPSS FD001 (no aportan información, se descartan)
CONSTANT_SENSORS = ["sensor_1", "sensor_5", "sensor_6", "sensor_10",
                     "sensor_16", "sensor_18", "sensor_19"]

USEFUL_SENSORS = [c for c in SENSOR_COLS if c not in CONSTANT_SENSORS]


def smooth_signals(df: pd.DataFrame, window: int = 5, cols=None) -> pd.DataFrame:
    """Filtrado de ruido con media móvil, calculada por motor de forma independiente."""
    df = df.copy()
    cols = cols or USEFUL_SENSORS
    df[cols] = (
        df.groupby("unit")[cols]
        .apply(lambda g: g.rolling(window, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )
    return df


def add_rul_train(df: pd.DataFrame, rul_cap: int = 125) -> pd.DataFrame:
    """Para el set de train: RUL = ciclo_max_del_motor - ciclo_actual, acotado a rul_cap."""
    df = df.copy()
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    df["RUL"] = max_cycle - df["cycle"]
    df["RUL"] = df["RUL"].clip(upper=rul_cap)
    return df


def add_rul_test(df: pd.DataFrame, true_rul: pd.Series, rul_cap: int = 125) -> pd.DataFrame:
    """
    Para el set de test: solo conocemos el RUL real en el ÚLTIMO ciclo observado
    de cada motor (viene dado en RUL_FDxxx.txt). Reconstruimos el RUL para
    cada fila hacia atrás y lo acotamos igual que en train.
    """
    df = df.copy()
    true_rul = true_rul.reset_index(drop=True)
    units = sorted(df["unit"].unique())
    rul_map = dict(zip(units, true_rul))

    max_cycle = df.groupby("unit")["cycle"].transform("max")
    df["final_rul"] = df["unit"].map(rul_map)
    df["RUL"] = (max_cycle - df["cycle"]) + df["final_rul"]
    df["RUL"] = df["RUL"].clip(upper=rul_cap)
    df.drop(columns=["final_rul"], inplace=True)
    return df


class MinMaxNormalizer:
    """Normalizador ajustado únicamente con datos de train (evita fuga de información)."""

    def __init__(self, cols=None):
        self.cols = cols or USEFUL_SENSORS + SETTING_COLS
        self.min_ = None
        self.max_ = None

    def fit(self, df: pd.DataFrame):
        self.min_ = df[self.cols].min()
        self.max_ = df[self.cols].max()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        denom = (self.max_ - self.min_).replace(0, 1)
        df[self.cols] = (df[self.cols] - self.min_) / denom
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)
