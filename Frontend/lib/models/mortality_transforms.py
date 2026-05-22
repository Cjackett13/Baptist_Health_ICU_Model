"""Sklearn transformers for mortality XGBoost pipeline (stable pickle module path)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class Log1pSelectedColumns(BaseEstimator, TransformerMixin):
    """np.log1p on a fixed feature subset; clip at 0 first. NaN preserved for downstream XGBoost."""

    def __init__(self, columns: list[str] | None = None):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        out = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        cols = self.columns or []
        for c in cols:
            if c not in out.columns:
                continue
            v = pd.to_numeric(out[c], errors="coerce").astype(np.float64)
            v = np.clip(v, 0.0, None)
            out[c] = np.log1p(v)
        return out


class SqrtSelectedColumns(BaseEstimator, TransformerMixin):
    """np.sqrt on a fixed feature subset; clip at 0 first. NaN preserved for downstream XGBoost."""

    def __init__(self, columns: list[str] | None = None):
        self.columns = columns or []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        out = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        for c in self.columns:
            if c not in out.columns:
                continue
            v = pd.to_numeric(out[c], errors="coerce").astype(np.float64)
            v = np.clip(v, 0.0, None)
            out[c] = np.sqrt(v)
        return out
