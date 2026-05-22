"""
Train-fold-only LOS feature engineering: SCAI PCA (option D) + infusion residual within SCAI.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

SCAI_PCA_INPUT_COLUMNS: list[str] = [
    "scai_first_12h",
    "scai_last_12h",
    "scai_mean_12h",
    "scai_std_12h",
    "scai_slope_12h",
    "scai_prop_ge3_12h",
    "current_scai_12h",
]

SCAI_PCA_OUTPUT_COLUMNS: list[str] = ["scai_pc1_12h", "scai_pc2_12h"]

RESIDUAL_VALUE_COL = "med_infusion_mean_12h"
RESIDUAL_BIN_COL = "current_scai_12h"
RESIDUAL_OUT_COL = "infusion_residual_within_scai_12h"

LOG1P_COLUMNS: list[str] = ["clinical_event_n_12h"]
SQRT_COLUMNS: list[str] = [
    "med_infusion_mean_12h",
    "med_distinct_12h",
    "proc_n_12h",
    "med_per_clinical_event_12h",
    "proc_n_bucket_12h",
    "med_distinct_bucket_12h",
]

WINSORIZE_COLUMNS: list[str] = [
    "med_infusion_mean_12h",
    "clinical_event_n_12h",
    "med_per_clinical_event_12h",
]

CATEGORICAL_COLUMNS: list[str] = [
    "admit_type_cd",
    "admit_src_cd",
    "unit_cd",
    "race_cd",
    "ethnicity_cd",
]

DROP_ALWAYS: frozenset[str] = frozenset(
    {
        "infusion_mean_12h",
        "med_admin_rows_12h",
        "proc_distinct_nom_12h",
        "med_residual_within_scai_12h",
        "proc_residual_within_scai_12h",
        *SCAI_PCA_INPUT_COLUMNS,
    }
)


class LosTrainPreprocessor(BaseEstimator, TransformerMixin):
    """
    Fit on training rows only: SCAI PCA, infusion residual, impute, winsorize, sqrt/log1p, one-hot.
    """

    def __init__(
        self,
        *,
        base_feature_columns: list[str],
        weight_by_column: dict[str, float],
        scai_pca_n_components: int = 2,
        winsorize_quantiles: tuple[float, float] = (0.01, 0.99),
    ):
        self.base_feature_columns = list(base_feature_columns)
        self.weight_by_column = dict(weight_by_column)
        self.scai_pca_n_components = int(scai_pca_n_components)
        self.winsorize_quantiles = winsorize_quantiles
        self.feature_names_: list[str] = []
        self.feature_weights_: np.ndarray | None = None

    def _engineer(self, X: pd.DataFrame, *, fit: bool) -> pd.DataFrame:
        out = X.copy()
        if RESIDUAL_VALUE_COL in out.columns and RESIDUAL_BIN_COL in out.columns:
            bins = pd.to_numeric(out[RESIDUAL_BIN_COL], errors="coerce").round().astype("Int64")
            vals = pd.to_numeric(out[RESIDUAL_VALUE_COL], errors="coerce")
            if fit:
                frame = pd.DataFrame({"bin": bins, "val": vals}).dropna()
                self.global_mean_ = float(frame["val"].mean()) if len(frame) else 0.0
                self.bin_means_ = frame.groupby("bin")["val"].mean().to_dict()
            means = bins.map(getattr(self, "bin_means_", {})).astype(float)
            means = means.fillna(getattr(self, "global_mean_", 0.0))
            out[RESIDUAL_OUT_COL] = vals - means

        scai_cols = [c for c in SCAI_PCA_INPUT_COLUMNS if c in out.columns]
        if scai_cols:
            mat = out[scai_cols].apply(pd.to_numeric, errors="coerce")
            if fit:
                med = mat.median()
                self.scai_fill_ = med.to_dict()
                filled = mat.fillna(med)
                self.scai_scaler_ = StandardScaler()
                scaled = self.scai_scaler_.fit_transform(filled)
                self.scai_pca_ = PCA(n_components=self.scai_pca_n_components, random_state=42)
                comps = self.scai_pca_.fit_transform(scaled)
            else:
                filled = mat.fillna(pd.Series(self.scai_fill_))
                scaled = self.scai_scaler_.transform(filled)
                comps = self.scai_pca_.transform(scaled)
            for i, name in enumerate(SCAI_PCA_OUTPUT_COLUMNS[: comps.shape[1]]):
                out[name] = comps[:, i]

        drop_cols = [c for c in out.columns if c in DROP_ALWAYS]
        out = out.drop(columns=drop_cols, errors="ignore")

        keep = [c for c in self.base_feature_columns if c in out.columns]
        out = out[keep]

        num_cols = [c for c in out.columns if c not in CATEGORICAL_COLUMNS]
        for c in num_cols:
            out[c] = pd.to_numeric(out[c], errors="coerce")

        if fit:
            self.num_medians_ = {c: float(out[c].median()) for c in num_cols}
            self.cat_modes_ = {
                c: out[c].mode(dropna=True).iloc[0] if out[c].notna().any() else "UNKNOWN"
                for c in CATEGORICAL_COLUMNS
                if c in out.columns
            }
            self.winsor_bounds_ = {}
            for c in WINSORIZE_COLUMNS:
                if c not in out.columns:
                    continue
                lo, hi = out[c].quantile(self.winsorize_quantiles)
                self.winsor_bounds_[c] = (float(lo), float(hi))

        for c, med in getattr(self, "num_medians_", {}).items():
            out[c] = out[c].fillna(med)
        for c, mode in getattr(self, "cat_modes_", {}).items():
            if c in out.columns:
                out[c] = out[c].fillna(mode).astype("string")

        for c, (lo, hi) in getattr(self, "winsor_bounds_", {}).items():
            out[c] = out[c].clip(lo, hi)

        for c in LOG1P_COLUMNS:
            if c in out.columns:
                v = np.clip(out[c].astype(float), 0.0, None)
                out[c] = np.log1p(v)
        for c in SQRT_COLUMNS:
            if c in out.columns:
                v = np.clip(out[c].astype(float), 0.0, None)
                out[c] = np.sqrt(v)

        if fit:
            self.dummy_columns_ = {}
            for c in CATEGORICAL_COLUMNS:
                if c not in out.columns:
                    continue
                dummies = pd.get_dummies(out[c].astype("string"), prefix=c, dtype=float)
                self.dummy_columns_[c] = list(dummies.columns)
            parts = [out.drop(columns=[c for c in CATEGORICAL_COLUMNS if c in out.columns])]
            for c in CATEGORICAL_COLUMNS:
                if c in out.columns and c in self.dummy_columns_:
                    dummies = pd.get_dummies(out[c].astype("string"), prefix=c, dtype=float)
                    for col in self.dummy_columns_[c]:
                        if col not in dummies.columns:
                            dummies[col] = 0.0
                    parts.append(dummies[self.dummy_columns_[c]])
            out = pd.concat(parts, axis=1)
            self.feature_names_ = list(out.columns)
            self.feature_weights_ = self._build_weight_vector(self.feature_names_)
        else:
            parts = [out.drop(columns=[c for c in CATEGORICAL_COLUMNS if c in out.columns])]
            for c in CATEGORICAL_COLUMNS:
                if c not in out.columns or c not in getattr(self, "dummy_columns_", {}):
                    continue
                dummies = pd.get_dummies(out[c].astype("string"), prefix=c, dtype=float)
                for col in self.dummy_columns_[c]:
                    if col not in dummies.columns:
                        dummies[col] = 0.0
                parts.append(dummies[self.dummy_columns_[c]])
            out = pd.concat(parts, axis=1)
            out = out.reindex(columns=self.feature_names_, fill_value=0.0)

        return out

    def _build_weight_vector(self, names: list[str]) -> np.ndarray:
        weights: list[float] = []
        for col in names:
            if col in self.weight_by_column:
                weights.append(float(self.weight_by_column[col]))
                continue
            parent = None
            for cat in CATEGORICAL_COLUMNS:
                if col.startswith(f"{cat}_"):
                    parent = cat
                    break
            if parent and parent in self.weight_by_column:
                weights.append(float(self.weight_by_column[parent]))
            else:
                weights.append(1.0)
        return np.asarray(weights, dtype=np.float64)

    def fit(self, X: pd.DataFrame, y: Any = None) -> LosTrainPreprocessor:
        self._engineer(X, fit=True)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._engineer(X, fit=False)

    def fit_transform(self, X: pd.DataFrame, y: Any = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def get_feature_weights(self) -> np.ndarray:
        if self.feature_weights_ is None:
            raise RuntimeError("Call fit before get_feature_weights")
        return self.feature_weights_

    def explained_variance_ratio_(self) -> list[float] | None:
        pca = getattr(self, "scai_pca_", None)
        if pca is None:
            return None
        return [float(x) for x in pca.explained_variance_ratio_]
