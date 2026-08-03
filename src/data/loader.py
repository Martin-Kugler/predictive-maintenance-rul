"""
src/data/loader.py
Load the raw C-MAPSS files (train_FDxxx.txt, test_FDxxx.txt, RUL_FDxxx.txt)
and return pandas DataFrames with human-readable column names.
"""

import os
import pandas as pd

COLS = (
    ["unit", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def _read_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df = df.iloc[:, : len(COLS)]
    df.columns = COLS
    df["unit"] = df["unit"].astype(int)
    df["cycle"] = df["cycle"].astype(int)
    return df


def load_train(raw_dir: str, subset: str = "FD001") -> pd.DataFrame:
    return _read_raw(os.path.join(raw_dir, f"train_{subset}.txt"))


def load_test(raw_dir: str, subset: str = "FD001") -> pd.DataFrame:
    return _read_raw(os.path.join(raw_dir, f"test_{subset}.txt"))


def load_true_rul(raw_dir: str, subset: str = "FD001") -> pd.Series:
    return pd.read_csv(os.path.join(raw_dir, f"RUL_{subset}.txt"), sep=r"\s+", header=None).iloc[:, 0]


if __name__ == "__main__":
    raw_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
    train = load_train(raw_dir)
    print(train.head())
    print(f"Motores en train: {train['unit'].nunique()}, filas: {len(train)}")
