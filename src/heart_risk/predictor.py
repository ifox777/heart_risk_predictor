from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import joblib
import numpy as np
from sklearn.pipeline import Pipeline

from .io_utils import read_csv_strict


class PredictorService:
    def __init__(self, model_path: Path):
        self.pipeline = self._load_pipeline(model_path)

    @staticmethod
    def _load_pipeline(model_path: Path) -> Pipeline:
        if not model_path.exists():
            raise FileNotFoundError(f"Model pipeline not found at {model_path}")
        return joblib.load(model_path)

    def predict_df(self, df: pd.DataFrame) -> np.ndarray:
        # predict_proba возвращает вероятности для каждого класса,
        # нам нужна только вероятность для класса 1 (риск есть)
        return self.pipeline.predict_proba(df)[:, 1]

    def predict_path(self, path: Path) -> np.ndarray:
        df = read_csv_strict(path)
        return self.predict_df(df)
