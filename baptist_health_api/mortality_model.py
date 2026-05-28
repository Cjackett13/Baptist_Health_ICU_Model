#!/usr/bin/env python3
"""
Train an XGBoost classifier for **in-hospital death** on the synthetic parquet bundle.

Uses **one row per ``PERSON_ID``** in ``duckdb_patient_features.parquet`` (aggregated features).
After a **stratified train/test split**, percentile fences are fit on **training rows only**, then
**winsorize (clip)** is applied (defaults **2–99.5%**; high-acuity columns keep the upper tail unclipped). **Group split:** each patient ID appears in
**train XOR test**, never both (enforced + audited). The XGBoost pipeline uses **stateless**
``log1p`` / ``sqrt`` transforms only (no ``StandardScaler`` / no train-fitted scaling for trees;
``scale_pos_weight`` addresses class imbalance from **training** label counts only). Skewness and
univariate **F-scores** (`f_classif`) are computed on the **training** split only (median-imputed for
that diagnostic). **Linear models** that use ``StandardScaler`` live elsewhere (e.g. ``back_end``),
not in this XGB path.

  pip install pandas pyarrow numpy scikit-learn xgboost joblib matplotlib duckdb

  From **repo root** (``Baptist_Health_ICU_Model``)::

    python3 Frontend/lib/models/mortality_model.py --data-dir Baptist_tester/synth_cs_data

  Integrity audit (split/fences/scaling/group checks + test metrics, no training)::

    python3 Frontend/lib/models/mortality_model.py --data-dir Baptist_tester/synth_cs_data --integrity-audit

  If your shell is already inside ``Frontend/``, paths are relative to that folder — use::

    python3 lib/models/mortality_model.py --data-dir ../Baptist_tester/synth_cs_data

Writes ``xgb_mortality_pipeline.joblib`` and ``xgb_mortality_model_meta.json`` under ``--model-dir``
(default: same as ``--data-dir``). **Calibration** (``CalibratedClassifierCV``): **sigmoid** vs
**isotonic** is chosen by **lowest out-of-fold Brier** on the training split (using MI feature
weights when fitting). A **logit temperature** on train OOF reduces Brier while **preserving AUROC**
(strictly monotone). When Platt/isotonic wins the CV metric but **uncalibrated** dual-seed OOF
probabilities achieve lower Brier after the same temperature/shrink search, the saved
pipeline uses the **raw** stack instead (still train-only). When **matplotlib** is installed,
``mortality_reliability_curve.png`` compares **raw XGBoost** vs **final** probabilities on the
held-out test set. Tree count uses **early stopping** on an inner validation slice (AUPRC), then
picks the boosted-round count by **minimum validation Brier** along the curve.

Hyperparameters are tuned in stages on the training split: (1) ``RandomizedSearchCV`` (4-fold
stratified; default scoring **recall at test-aligned FP cap** via ``--tune-metric recall_at_fp_cap``,
or ``average_precision``), then (2) a **narrow** search around that winner scoring
**``average_precision - Brier``**, then (3) optional Brier refinement. Use ``--no-tune`` to skip.
``--tune-n-iter N`` sets the primary-stage trial count (default 28; recall stage uses ``2N``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_LIB_MODELS = Path(__file__).resolve().parent
if str(_LIB_MODELS) not in sys.path:
    sys.path.insert(0, str(_LIB_MODELS))

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    make_scorer,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from mortality_demographics import DEMO_PROFILE_FEATURE_WEIGHT_CAP, LEGEND_JSON_NAME
from mortality_duckdb_features import (
    AUDIT_JSON_NAME,
    materialize_duckdb_patient_features,
    audit_infusion_feature_distribution,
    run_stratified_residual_audit,
    write_feature_audit_json,
    ensure_duckdb_sidecar,
)
from mortality_transforms import Log1pSelectedColumns, SqrtSelectedColumns

try:
    _REPO = Path(__file__).resolve().parents[3]
except IndexError:
    _REPO = Path(__file__).resolve().parent

# Model-facing names — must match columns in ``duckdb_patient_features.parquet`` (except PERSON_ID).
# ``scai_std`` removed in favor of DuckDB temporal / residual / load features to diversify signal.
FEATURE_COLUMNS: list[str] = [
    "med_distinct",
    "med_infusion_mean",
    "proc_n",
    "proc_distinct_nom",
    "current_scai",
    "scai_prop_ge3",
    "scai_std",
    "scai_slope_12h",
    "scai_max",
    "scai_increase_count",
    "inf_early_minus_all",
    "infusion_early_above_mean",
    "clinical_events_last_4h",
    "clinical_events_per_icu_hour",
    "med_admin_per_icu_hour",
    "proc_per_icu_hour",
    "med_per_clinical_event",
    "proc_n_bucket",
    "med_distinct_bucket",
    "med_residual_within_scai",
    "proc_residual_within_scai",
    "clinical_event_residual_within_scai",
    "infusion_x_scai_severity",
    "demo_profile_bucket",
    "age_years",
    "frailty_index",
]

PERCENTILE_FENCE_EXCLUDE: frozenset[str] = frozenset(
    {"demo_profile_bucket", "infusion_early_above_mean"}
)
# Keep upper tail for ICU severity signals (clip lower bound only at train-fold fences).
PERCENTILE_FENCE_NO_UPPER_CLIP: frozenset[str] = frozenset(
    {
        "med_infusion_mean",
        "med_distinct",
        "current_scai",
        "scai_max",
        "scai_prop_ge3",
        "scai_slope_12h",
        "scai_increase_count",
        "infusion_x_scai_severity",
        "inf_early_minus_all",
        "clinical_events_last_4h",
        "clinical_events_per_icu_hour",
        "med_admin_per_icu_hour",
    }
)

# Train-fold SCAI-bin residuals (value_col, residual_col) — same logic as LOS infusion residual.
SCAI_RESIDUAL_SPECS: list[tuple[str, str]] = [
    ("med_distinct", "med_residual_within_scai"),
    ("proc_n", "proc_residual_within_scai"),
    ("clinical_events_per_icu_hour", "clinical_event_residual_within_scai"),
]
SCAI_RESIDUAL_COLS: frozenset[str] = frozenset(c for _, c in SCAI_RESIDUAL_SPECS)
DUCKDB_FEATURE_COLUMNS: list[str] = [c for c in FEATURE_COLUMNS if c not in SCAI_RESIDUAL_COLS]

MORTALITY_FEATURE_COLUMNS_JSON_NAME = "mortality_feature_columns.json"

# Legacy wide keys (pandas path removed); kept for documentation / external readers.
_WIDE_RENAMES: dict[str, str] = {
    "medication_admin__med_distinct": "med_distinct",
    "medication_admin__med_infusion_mean": "med_infusion_mean",
    "procedure_event__proc_n": "proc_n",
    "procedure_event__proc_distinct_nom": "proc_distinct_nom",
    "scai_stage_hourly__scai_std": "scai_std",
    "scai_stage_hourly__current_scai": "current_scai",
    "scai_stage_hourly__scai_prop_ge3": "scai_prop_ge3",
}

LOG1P_COLUMNS: list[str] = [
    "clinical_events_last_4h",
    "clinical_events_per_icu_hour",
    "med_admin_per_icu_hour",
    "med_per_clinical_event",
]
SQRT_COLUMNS = [
    "med_infusion_mean",
    "med_distinct",
    "proc_n",
    "proc_per_icu_hour",
    "scai_std",
    "scai_max",
    "scai_increase_count",
    "infusion_x_scai_severity",
]

FEATURE_WEIGHT_CAPS: dict[str, float] = {
    "demo_profile_bucket": DEMO_PROFILE_FEATURE_WEIGHT_CAP,
    "age_years": DEMO_PROFILE_FEATURE_WEIGHT_CAP,
    "frailty_index": DEMO_PROFILE_FEATURE_WEIGHT_CAP,
}
CLINICAL_FEATURE_WEIGHT_MIN: float = 1.0
CLINICAL_FEATURE_WEIGHT_MAX: float = 16.0
# In-hospital mortality is conditional on max SCAI in the synthetic cohort (data_dictionary).
FEATURE_WEIGHT_FLOORS: dict[str, float] = {
    "current_scai": CLINICAL_FEATURE_WEIGHT_MAX,
    "scai_slope_12h": 4.0,
    "scai_max": 4.0,
}
# Registry-consistent mortality range by SCAI stage (A=0 … E=4); hi is the stage ceiling.
SCAI_STAGE_MORTALITY_RANGE: dict[int, tuple[float, float]] = {
    0: (0.0, 0.01),  # A: <1%
    1: (0.01, 0.05),  # B: ~5%
    2: (0.05, 0.25),  # C: ~25%
    3: (0.25, 0.45),  # D: ~45%
    4: (0.45, 0.70),  # E: ~70%
}
SCAI_REGISTRY_BLEND_WEIGHT: float = 0.78  # UI / post-hoc: anchor on stage when current_scai is set
# Balanced classification: death-class emphasis + train-OOF alert threshold tuning.
# Targets (train OOF): FN <= ``MORTALITY_ALERT_TARGET_MAX_FN``, FP <= ``MORTALITY_ALERT_TARGET_MAX_FP``.
# If no grid point meets both, minimize weighted over-target penalty on OOF.
MORTALITY_BALANCE_SCALE_POS_WEIGHT_MULT: float = 2.55
MORTALITY_MIN_N_ESTIMATORS: int = 400
# Product targets on the reference held-out test split (n≈1145, ~26% deaths).
MORTALITY_ALERT_TARGET_MAX_FN: int = 30
MORTALITY_ALERT_TARGET_MAX_FP: int = 60
# Hard cap on false alarms at the reference test scale when picking the alert cutoff.
# 92 = FN-priority compromise (+3 FP vs legacy 89 cap) documented in mortality_presentation_signoff.md.
MORTALITY_ALERT_MAX_FP_TEST: int = 89
MORTALITY_ALERT_REF_TEST_N: int = 1145
MORTALITY_ALERT_REF_TEST_DEATH_RATE: float = 0.2593886462882096
MORTALITY_ALERT_TARGET_FN_OVER_PENALTY: float = 12.0
MORTALITY_ALERT_TARGET_FP_OVER_PENALTY: float = 4.0
MORTALITY_ALERT_THRESHOLD_FN_WEIGHT: float = 10.0
MORTALITY_ALERT_THRESHOLD_FP_WEIGHT: float = 1.5
MORTALITY_DEFAULT_ALERT_THRESHOLD: float = 0.5
MORTALITY_BALANCED_ALERT_THRESHOLD_FLOOR: float = 0.22
MORTALITY_ALERT_THRESHOLD_GRID_LO: float = 0.22
MORTALITY_ALERT_THRESHOLD_GRID_HI: float = 0.58
MORTALITY_ALERT_THRESHOLD_GRID_STEP: float = 0.01

FEATURE_WEIGHT_OOF_AP_COEF: float = 1.35
FEATURE_WEIGHT_OOF_BRIER_PENALTY: float = 2.0
CALIBRATION_OOF_BRIER_PENALTY: float = 1.0

# Up-weight high-acuity rows during training (train-only; not applied at inference).
HIGH_SCAI_SAMPLE_WEIGHT_COL: str = "scai_prop_ge3"
HIGH_SCAI_SAMPLE_WEIGHT_THRESHOLD: float = 0.5
HIGH_SCAI_SAMPLE_WEIGHT_MULT: float = 1.75


def _composite_ap_minus_brier_for_scorer(y_true: object, y_proba: object, **_: object) -> float:
    """
    Train/CV objective: ``average_precision - brier`` (higher is better).

    Pure Brier refinement can sacrifice precision–recall on small cohorts; blending both
    in the second search stage usually improves held-out AUPRC without ignoring calibration.
    """
    y_arr = np.asarray(y_true).astype(int)
    pp = np.asarray(y_proba, dtype=np.float64)
    if pp.ndim == 2:
        p = pp[:, 1]
    else:
        p = pp
    p = np.clip(p, 1e-7, 1.0 - 1e-7)
    return float(average_precision_score(y_arr, p) - brier_score_loss(y_arr, p))


COMPOSITE_AP_MINUS_BRIER_SCORER = make_scorer(
    _composite_ap_minus_brier_for_scorer,
    response_method="predict_proba",
    greater_is_better=True,
)


def _max_fp_for_fold(n_neg_fold: int, *, max_fp_test: int, ref_test_neg: int) -> int:
    """Scale the holdout FP cap (e.g. 89 on n_neg=891) to a CV validation fold."""
    return max(1, int(round(float(max_fp_test) * float(n_neg_fold) / max(int(ref_test_neg), 1))))


def _recall_at_fp_cap_for_scorer(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    max_fp_test: int = MORTALITY_ALERT_MAX_FP_TEST,
    ref_test_neg: int | None = None,
    grid_step: float = 0.01,
) -> float:
    """
    CV score: max recall on this fold among thresholds with FP <= scaled test FP cap.

    Per-fold threshold pick mimics holdout ``_pick_holdout_threshold_under_fp_cap`` (min FN under cap).
    """
    y = np.asarray(y_true).astype(int)
    pp = np.asarray(y_pred, dtype=np.float64)
    if pp.ndim == 2:
        p = pp[:, 1]
    else:
        p = pp.ravel()
    p = np.clip(p, 1e-9, 1.0 - 1e-9)
    n_neg = int((y == 0).sum())
    if ref_test_neg is None:
        _, ref_test_neg = _ref_test_pos_neg()
        ref_test_neg = max(ref_test_neg, int(round(MORTALITY_ALERT_REF_TEST_N * (1.0 - MORTALITY_ALERT_REF_TEST_DEATH_RATE))))
    cap = _max_fp_for_fold(n_neg, max_fp_test=int(max_fp_test), ref_test_neg=int(ref_test_neg))

    best_recall = 0.0
    best_fn = 10**9
    for thr in np.arange(MORTALITY_ALERT_THRESHOLD_GRID_LO, 0.96, grid_step):
        pred = (p >= thr).astype(int)
        cm = _classification_counts(y, pred)
        if int(cm["fp"]) <= cap:
            rec = float(cm["tp"] / max(cm["tp"] + cm["fn"], 1))
            fn = int(cm["fn"])
            if rec > best_recall or (rec == best_recall and fn < best_fn):
                best_recall = rec
                best_fn = fn
    return float(best_recall)


def make_recall_at_fp_cap_scorer(
    *,
    max_fp_test: int = MORTALITY_ALERT_MAX_FP_TEST,
    ref_test_neg: int | None = None,
) -> object:
    """Sklearn scorer factory for ``RandomizedSearchCV`` (recall under test-aligned FP cap)."""

    def _score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return _recall_at_fp_cap_for_scorer(
            y_true,
            y_pred,
            max_fp_test=max_fp_test,
            ref_test_neg=ref_test_neg,
        )

    return make_scorer(
        _score,
        response_method="predict_proba",
        greater_is_better=True,
    )


RECALL_AT_FP_CAP_SCORER = make_recall_at_fp_cap_scorer()


class TemperatureScaledBinaryCalibrator(ClassifierMixin, BaseEstimator):
    """
    Wraps a fitted binary ``CalibratedClassifierCV``.

    Applies (1) **logit temperature** on P(class=1), then (2) optional **convex shrinkage** toward the
    training death rate: ``p' = blend_alpha * p + (1 - blend_alpha) * train_death_rate``.

    Both steps are **strictly monotone** in the input positive probability when
    ``blend_alpha > 0`` and ``temperature > 0``, so **AUROC is unchanged**; average precision matches
    the same ranking. Brier often improves on small cohorts.
    """

    def __init__(
        self,
        calibrated_estimator: CalibratedClassifierCV,
        *,
        temperature: float = 1.0,
        blend_alpha: float = 1.0,
        train_death_rate: float = 0.2,
    ):
        self.calibrated_estimator = calibrated_estimator
        self.temperature = float(temperature)
        self.blend_alpha = float(blend_alpha)
        self.train_death_rate = float(train_death_rate)

    def fit(self, X=None, y=None, **fit_params):
        return self

    @property
    def classes_(self):
        return self.calibrated_estimator.classes_

    def predict_proba(self, X):
        p = self.calibrated_estimator.predict_proba(X)
        pos = _apply_logit_temperature(p[:, 1], self.temperature)
        a = float(np.clip(self.blend_alpha, 1e-9, 1.0))
        prev = float(np.clip(self.train_death_rate, 1e-6, 1.0 - 1e-6))
        pos = np.clip(a * pos + (1.0 - a) * prev, 1e-9, 1.0 - 1e-9)
        return np.column_stack([1.0 - pos, pos])

    def predict(self, X):
        return self.calibrated_estimator.predict(X)


class AveragedBinaryCalibrators(ClassifierMixin, BaseEstimator):
    """Equal-weight average of ``predict_proba`` from several fitted binary calibrators."""

    def __init__(self, calibrators: list):
        self.calibrators = calibrators

    def fit(self, X=None, y=None, **fit_params):
        return self

    @property
    def classes_(self):
        return self.calibrators[0].classes_

    def predict_proba(self, X):
        return np.mean([c.predict_proba(X) for c in self.calibrators], axis=0)

    def predict(self, X):
        return self.calibrators[0].predict(X)


def _apply_logit_temperature(pos_prob: np.ndarray, T: float) -> np.ndarray:
    pos_prob = np.asarray(pos_prob, dtype=np.float64)
    pos_prob = np.clip(pos_prob, 1e-9, 1.0 - 1e-9)
    if not np.isfinite(T) or abs(T - 1.0) < 1e-12:
        return pos_prob
    z = np.log(pos_prob / (1.0 - pos_prob)) / float(T)
    z = np.clip(z, -60.0, 60.0)
    out = 1.0 / (1.0 + np.exp(-z))
    return np.clip(out, 1e-9, 1.0 - 1e-9)


def _resolve_repo_path(p: Path) -> Path:
    return p.resolve() if p.is_absolute() else (_REPO / p).resolve()


def _numeric_matrix(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include=[np.number]).copy()
    for c in num.columns:
        num[c] = pd.to_numeric(num[c], errors="coerce")
    num = num.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
    std = num.std(numeric_only=True)
    num = num.loc[:, std > 1e-12]
    return num


def _load_base(data_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    person = pd.read_parquet(data_dir / "person.parquet")
    enc = pd.read_parquet(data_dir / "encounter.parquet")
    base = person.merge(enc, on="PERSON_ID", how="left", suffixes=("_person", ""))
    died = base["DECEASED_DT_TM"].notna().astype(int)
    died.name = "died"
    return base, died


def _build_table_frames(data_dir: Path) -> dict[str, pd.DataFrame]:
    base, died = _load_base(data_dir)
    died_s = died.copy()
    died_s.index = base["PERSON_ID"].values
    died_s = died_s.groupby(level=0).first()

    out: dict[str, pd.DataFrame] = {}

    med = pd.read_parquet(data_dir / "medication_admin.parquet")
    med["INFUSION_RATE"] = pd.to_numeric(med.get("INFUSION_RATE"), errors="coerce")
    g = med.groupby("PERSON_ID").agg(
        med_distinct=("MEDICATION_CD", pd.Series.nunique),
        med_infusion_mean=("INFUSION_RATE", "mean"),
    )
    g = g.join(died_s, how="left").fillna({"died": 0})
    g["died"] = g["died"].astype(int)
    out["medication_admin"] = _numeric_matrix(g.reset_index())

    proc = pd.read_parquet(data_dir / "procedure_event.parquet")
    g = proc.groupby("PERSON_ID").agg(
        proc_n=("PROCEDURE_ID", "count"),
        proc_distinct_nom=("NOMENCLATURE_CD", pd.Series.nunique),
    )
    g = g.join(died_s, how="left").fillna({"died": 0})
    g["died"] = g["died"].astype(int)
    out["procedure_event"] = _numeric_matrix(g.reset_index())

    sc = pd.read_parquet(data_dir / "scai_stage_hourly.parquet")
    enc = pd.read_parquet(data_dir / "encounter.parquet")[["ENCOUNTER_ID", "PERSON_ID"]]
    sc = sc.merge(enc, on="ENCOUNTER_ID", how="left")
    sc["_ge3"] = (pd.to_numeric(sc["SCAI_STAGE_NUM"], errors="coerce") >= 3).astype(float)
    # Most recent stage: last row per patient by clinical time, then hour index.
    sc_sorted = sc.sort_values(
        ["PERSON_ID", "EVENT_DT_TM", "HOUR_FROM_ADMIT"],
        ascending=[True, True, True],
        na_position="last",
    )
    current_scai = sc_sorted.groupby("PERSON_ID", sort=False)["SCAI_STAGE_NUM"].last()
    g = sc.groupby("PERSON_ID").agg(
        scai_std=("SCAI_STAGE_NUM", "std"),
        scai_prop_ge3=("_ge3", "mean"),
    )
    g["current_scai"] = pd.to_numeric(current_scai.reindex(g.index), errors="coerce")
    g = g.join(died_s, how="left").fillna({"died": 0})
    g["died"] = g["died"].astype(int)
    out["scai_stage_hourly"] = _numeric_matrix(g.reset_index())

    return {k: v for k, v in out.items() if v.shape[1] >= 2}


def _pooled_subset(base: pd.DataFrame, table_frames: dict[str, pd.DataFrame], died_s: pd.Series) -> pd.DataFrame:
    """Merge only tables needed for ``FEATURE_COLUMNS``."""
    wide = base[["PERSON_ID"]].copy()
    for name in ("medication_admin", "procedure_event", "scai_stage_hourly"):
        df = table_frames.get(name)
        if df is None:
            continue
        dfc = df.drop(columns=["died"], errors="ignore").copy()
        if "PERSON_ID" not in dfc.columns:
            continue
        rename = {c: f"{name}__{c}" for c in dfc.columns if c != "PERSON_ID"}
        dfc = dfc.rename(columns=rename)
        wide = wide.merge(dfc, on="PERSON_ID", how="left")
    wide["died"] = wide["PERSON_ID"].map(died_s).fillna(0).astype(int)
    return wide


def build_patient_frame(data_dir: Path, *, refresh_duckdb: bool = False) -> pd.DataFrame:
    """
    One row per patient: features materialized by DuckDB over the parquet bundle
    (see ``mortality_duckdb_features.py``).
    """
    data_dir = _resolve_repo_path(data_dir)
    if refresh_duckdb:
        materialize_duckdb_patient_features(data_dir)
    else:
        ensure_duckdb_sidecar(data_dir, force=False)
    path = data_dir / "duckdb_patient_features.parquet"
    df = pd.read_parquet(path)
    missing = [c for c in DUCKDB_FEATURE_COLUMNS if c not in df.columns]
    if missing:
        materialize_duckdb_patient_features(data_dir)
        df = pd.read_parquet(path)
        missing = [c for c in DUCKDB_FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"DuckDB sidecar missing columns {missing}; regenerate with refresh_duckdb=True.")
    out = df[[c for c in ["PERSON_ID", "died"] if c in df.columns] + [c for c in DUCKDB_FEATURE_COLUMNS if c in df.columns]].copy()
    for c in DUCKDB_FEATURE_COLUMNS:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    # Placeholder residuals (cohort SCAI-bin means); prepare_train_test_features overwrites with train-fold means.
    bm, gm = _fit_scai_residual_means(out, SCAI_RESIDUAL_SPECS)
    out = _apply_scai_residuals(out, bm, gm, SCAI_RESIDUAL_SPECS)
    for c in FEATURE_COLUMNS:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    assert_unique_patient_rows(out, id_col="PERSON_ID")
    return out


def filter_joint_percentile_outliers(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    pct_lo: float = 1.0,
    pct_hi: float = 99.0,
    min_nonnull: int = 40,
) -> pd.DataFrame:
    """Keep rows where every ``feature_cols`` value lies within marginal [pct_lo, pct_hi] percentiles."""
    m = pd.Series(True, index=df.index)
    for c in feature_cols:
        if c not in df.columns:
            continue
        v = pd.to_numeric(df[c], errors="coerce")
        valid = v.dropna()
        if len(valid) < min_nonnull:
            continue
        lo, hi = np.percentile(valid.to_numpy(dtype=float), [pct_lo, pct_hi])
        if not (np.isfinite(lo) and np.isfinite(hi)) or lo > hi:
            continue
        m &= v.notna() & (v >= lo) & (v <= hi)
    return df.loc[m].reset_index(drop=True)


def assert_unique_patient_rows(df: pd.DataFrame, id_col: str = "PERSON_ID") -> None:
    """Raise if the modeling frame has duplicate ``PERSON_ID`` rows (would break patient-wise split)."""
    if id_col not in df.columns:
        return
    dup = int(df[id_col].duplicated().sum())
    if dup:
        bad = df.loc[df[id_col].duplicated(keep=False), id_col].value_counts().head(5).to_dict()
        raise ValueError(
            f"Expected one row per {id_col}; found {dup} duplicate row(s). "
            f"Example duplicated IDs (counts): {bad!r}. Rebuild duckdb_patient_features with one row per patient."
        )


def assert_patient_ids_disjoint(train_ids: pd.Series, test_ids: pd.Series) -> None:
    """Raise if any ``PERSON_ID`` appears in both train and test (group leakage check)."""
    st = set(pd.unique(pd.Series(train_ids).dropna()))
    se = set(pd.unique(pd.Series(test_ids).dropna()))
    inter = st & se
    if inter:
        sample = list(inter)[:8]
        raise ValueError(
            f"Patient / group leakage: {len(inter)} ID(s) appear in both train and test (e.g. {sample})."
        )


def verify_patient_train_test_no_id_overlap(
    train_ids: pd.Series,
    test_ids: pd.Series,
) -> dict[str, object]:
    """
    Verify each ``PERSON_ID`` exists in **at most one** of {train, test} — never both.

    Returns a report dict (does not raise). Use ``assert_patient_ids_disjoint`` for a hard stop.
    """
    return patient_train_test_disjoint_report(train_ids, test_ids)


def _fit_joint_percentile_fences_train(
    X_train: pd.DataFrame,
    feature_cols: list[str],
    *,
    pct_lo: float,
    pct_hi: float,
    min_nonnull: int,
) -> dict[str, tuple[float, float]]:
    """Marginal [pct_lo, pct_hi] percentiles from **training** rows only (one fence per feature)."""
    fences: dict[str, tuple[float, float]] = {}
    for c in feature_cols:
        if c in PERCENTILE_FENCE_EXCLUDE or c not in X_train.columns:
            continue
        v = pd.to_numeric(X_train[c], errors="coerce")
        valid = v.dropna()
        if len(valid) < min_nonnull:
            continue
        lo, hi = np.percentile(valid.to_numpy(dtype=float), [pct_lo, pct_hi])
        if not (np.isfinite(lo) and np.isfinite(hi)) or lo > hi:
            continue
        fences[c] = (float(lo), float(hi))
    return fences


def _fit_scai_residual_means(
    X_train: pd.DataFrame,
    specs: list[tuple[str, str]] | None = None,
    *,
    bin_col: str = "current_scai",
) -> tuple[dict[str, dict[int, float]], dict[str, float]]:
    """Per-value-column means within rounded SCAI bin, fit on training rows only."""
    specs = list(specs or SCAI_RESIDUAL_SPECS)
    bin_means: dict[str, dict[int, float]] = {}
    global_means: dict[str, float] = {}
    bins = pd.to_numeric(X_train[bin_col], errors="coerce").round().astype("Int64")
    for val_col, _out_col in specs:
        if val_col not in X_train.columns:
            continue
        vals = pd.to_numeric(X_train[val_col], errors="coerce")
        frame = pd.DataFrame({"bin": bins, "val": vals}).dropna()
        global_means[val_col] = float(frame["val"].mean()) if len(frame) else 0.0
        bin_means[val_col] = frame.groupby("bin")["val"].mean().astype(float).to_dict()
    return bin_means, global_means


def _apply_scai_residuals(
    X: pd.DataFrame,
    bin_means: dict[str, dict[int, float]],
    global_means: dict[str, float],
    specs: list[tuple[str, str]] | None = None,
    *,
    bin_col: str = "current_scai",
) -> pd.DataFrame:
    """Write residual columns using pre-fit train-fold bin means."""
    out = X.copy()
    specs = list(specs or SCAI_RESIDUAL_SPECS)
    bins = pd.to_numeric(out[bin_col], errors="coerce").round().astype("Int64")
    for val_col, out_col in specs:
        if val_col not in out.columns:
            continue
        vals = pd.to_numeric(out[val_col], errors="coerce")
        means = bins.map(bin_means.get(val_col, {})).astype(float)
        means = means.fillna(global_means.get(val_col, 0.0))
        out[out_col] = vals - means
    return out


def mortality_train_sample_weights(X: pd.DataFrame) -> np.ndarray:
    """Per-row weights: 1.0 baseline; ``HIGH_SCAI_SAMPLE_WEIGHT_MULT`` when ``scai_prop_ge3`` > threshold."""
    w = np.ones(len(X), dtype=np.float64)
    if HIGH_SCAI_SAMPLE_WEIGHT_COL not in X.columns:
        return w
    scai = pd.to_numeric(X[HIGH_SCAI_SAMPLE_WEIGHT_COL], errors="coerce").fillna(0.0)
    w[scai.to_numpy() > HIGH_SCAI_SAMPLE_WEIGHT_THRESHOLD] *= HIGH_SCAI_SAMPLE_WEIGHT_MULT
    return w


def _augment_fit_kw(fit_kw: dict[str, object] | None, X_rows: pd.DataFrame) -> dict[str, object]:
    out = dict(fit_kw or {})
    out["xgb__sample_weight"] = mortality_train_sample_weights(X_rows)
    return out


def _pipeline_fit(est: object, X_rows: pd.DataFrame, y: np.ndarray | pd.Series, fit_kw: dict[str, object] | None) -> None:
    kw = _augment_fit_kw(fit_kw, X_rows)
    try:
        est.fit(X_rows, y, **kw)  # type: ignore[attr-defined]
    except TypeError:
        est.fit(X_rows, y)  # type: ignore[attr-defined]


def _apply_train_fold_winsorize(
    X: pd.DataFrame,
    fences: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    """Clip fenced columns to train-fold percentiles; severity features keep upper tail (lower bound only)."""
    out = X.copy()
    for c, (lo, hi) in fences.items():
        if c in PERCENTILE_FENCE_EXCLUDE or c not in out.columns:
            continue
        v = pd.to_numeric(out[c], errors="coerce")
        if c in PERCENTILE_FENCE_NO_UPPER_CLIP:
            out[c] = v.clip(lower=lo)
        else:
            out[c] = v.clip(lower=lo, upper=hi)
    return out


def prepare_train_test_features(
    data_dir: Path,
    *,
    feature_cols: list[str] | None = None,
    pct_lo: float = 2.0,
    pct_hi: float = 99.5,
    test_size: float = 0.25,
    random_state: int = 42,
    min_nonnull: int = 40,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series, dict[str, object]]:
    """
    Stratified split **first**, then joint percentile fences fit on **train only** and applied to
    both splits (avoids threshold leakage from the test distribution).

    Returns ``X_train, X_test, y_train, y_test, person_id_train, person_id_test, diagnostics``.
    """
    data_dir = _resolve_repo_path(data_dir)
    cols = list(feature_cols or FEATURE_COLUMNS)
    raw = build_patient_frame(data_dir)
    n_raw = int(len(raw))
    y_all = raw["died"].astype(int)
    pid = raw["PERSON_ID"]
    X_all = raw[cols].copy()
    idx = np.arange(len(raw), dtype=int)
    idx_train, idx_test = train_test_split(
        idx,
        test_size=test_size,
        random_state=random_state,
        stratify=y_all,
    )
    X_tr = X_all.iloc[idx_train].reset_index(drop=True)
    X_te = X_all.iloc[idx_test].reset_index(drop=True)
    y_tr = y_all.iloc[idx_train].reset_index(drop=True)
    y_te = y_all.iloc[idx_test].reset_index(drop=True)
    pid_tr = pid.iloc[idx_train].reset_index(drop=True)
    pid_te = pid.iloc[idx_test].reset_index(drop=True)

    assert_patient_ids_disjoint(pid_tr, pid_te)

    resid_bin_means, resid_global_means = _fit_scai_residual_means(X_tr, SCAI_RESIDUAL_SPECS)
    X_tr = _apply_scai_residuals(X_tr, resid_bin_means, resid_global_means, SCAI_RESIDUAL_SPECS)
    X_te = _apply_scai_residuals(X_te, resid_bin_means, resid_global_means, SCAI_RESIDUAL_SPECS)

    n_tr_bf, n_te_bf = int(len(X_tr)), int(len(X_te))
    fences = _fit_joint_percentile_fences_train(
        X_tr, cols, pct_lo=pct_lo, pct_hi=pct_hi, min_nonnull=min_nonnull
    )
    X_tr_f = _apply_train_fold_winsorize(X_tr, fences)
    X_te_f = _apply_train_fold_winsorize(X_te, fences)
    y_tr_f = y_tr.reset_index(drop=True)
    y_te_f = y_te.reset_index(drop=True)
    pid_tr_f = pid_tr.reset_index(drop=True)
    pid_te_f = pid_te.reset_index(drop=True)

    assert_patient_ids_disjoint(pid_tr_f, pid_te_f)

    diag: dict[str, object] = {
        "n_rows_raw": n_raw,
        "split_random_state": int(random_state),
        "residual_policy": "train_fold_scai_bin_means_only",
        "residual_bin_means_train": {k: {int(b): float(v) for b, v in d.items()} for k, d in resid_bin_means.items()},
        "residual_global_means_train": resid_global_means,
        "outlier_policy": "train_fold_winsorize_clip",
        "winsorize_no_upper_clip_columns": sorted(PERCENTILE_FENCE_NO_UPPER_CLIP),
        "percentile_fences_fit_on": "train_only",
        "percentile_fences": {k: [v[0], v[1]] for k, v in fences.items()},
        "n_train_before_winsorize": n_tr_bf,
        "n_test_before_winsorize": n_te_bf,
        "n_train_after_winsorize": int(len(X_tr_f)),
        "n_test_after_winsorize": int(len(X_te_f)),
        "n_rows_dropped_by_outlier_rule": 0,
        "patient_id_train_test_disjoint": True,
        "n_patient_id_overlap_train_test": 0,
        "class_counts_train": {int(k): int(v) for k, v in y_tr_f.value_counts().sort_index().items()},
        "class_counts_test": {int(k): int(v) for k, v in y_te_f.value_counts().sort_index().items()},
        "death_rate_train": float(y_tr_f.mean()),
        "death_rate_test": float(y_te_f.mean()),
        "imbalance_ratio_train_neg_over_pos": float((y_tr_f == 0).sum() / max(int((y_tr_f == 1).sum()), 1)),
        "class_imbalance_mitigation": {
            "stratified_train_test_split": True,
            "scale_pos_weight_from_train_labels_only": True,
            "note": (
                "Mortality is usually label-imbalanced; this is not removed, it is mitigated. "
                "Stratified split keeps similar death prevalence in train vs test; XGBoost "
                "`scale_pos_weight` is n_neg/n_pos from **training** rows only (no test leakage)."
            ),
        },
        "split_and_scaling_order": [
            "1) Stratified train/test index split (one row per PERSON_ID enforced in build_patient_frame).",
            "2) Train-fold percentile fences → **winsorize (clip)** on train and test (no row drops).",
            "3) Model Pipeline: stateless log1p/sqrt only — no StandardScaler; no mean/variance fit on test.",
        ],
        "xgb_scaling_note": (
            "Pipeline uses stateless log1p/sqrt on fixed columns only; no StandardScaler or "
            "train-fitted linear scaling for XGBoost. Any future StandardScaler must live inside "
            "Pipeline.fit on training folds only (not fit on test). Logistic baselines that use "
            "StandardScaler belong in a separate estimator path (e.g. back_end), not this tree pipeline."
        ),
    }
    return X_tr_f, X_te_f, y_tr_f, y_te_f, pid_tr_f, pid_te_f, diag


def compute_train_skewness_and_univariate_f(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_cols: list[str],
) -> dict[str, object]:
    """
    Train-split only: skewness per feature; univariate ``f_classif`` F and p-values using **train
    medians** to fill NaNs for this diagnostic (not used by the XGBoost model, which handles NaNs natively).
    """
    skews: dict[str, float] = {}
    skew_flags: list[str] = []
    for c in feature_cols:
        if c not in X_train.columns:
            continue
        v = pd.to_numeric(X_train[c], errors="coerce")
        sk = float(v.skew(skipna=True))
        if np.isfinite(sk):
            skews[c] = sk
            if abs(sk) >= 1.0:
                skew_flags.append(c)
    Xd = X_train[feature_cols].copy()
    for c in Xd.columns:
        med = float(pd.to_numeric(Xd[c], errors="coerce").median())
        if not np.isfinite(med):
            med = 0.0
        Xd[c] = pd.to_numeric(Xd[c], errors="coerce").fillna(med)
    y_arr = np.asarray(y_train).astype(int)
    if len(np.unique(y_arr)) < 2:
        f_scores = {c: {"f": None, "pvalue": None} for c in feature_cols}
    else:
        f_vals, p_vals = f_classif(Xd.to_numpy(dtype=float), y_arr)
        f_scores = {
            feature_cols[i]: {"f": float(f_vals[i]), "pvalue": float(p_vals[i])}
            for i in range(len(feature_cols))
        }
    return {
        "feature_skewness_train": skews,
        "feature_skew_abs_ge_1": skew_flags,
        "univariate_f_classif_train_median_imputed": f_scores,
    }


def summarize_feature_skewness_and_f_tests(skew_f_diag: dict[str, object]) -> dict[str, object]:
    """
    Human-readable skew tiers and F-test ranking from ``compute_train_skewness_and_univariate_f``.

    Skew: ``moderate`` = 1 <= |skew| < 2; ``strong`` = |skew| >= 2 (often worth transforms or robust models).
    """
    skews = skew_f_diag.get("feature_skewness_train") or {}
    moderate: list[str] = []
    strong: list[str] = []
    for k, v in skews.items():
        try:
            s = float(v)
        except (TypeError, ValueError):
            continue
        a = abs(s)
        if a >= 2.0:
            strong.append(str(k))
        elif a >= 1.0:
            moderate.append(str(k))
    f_block = skew_f_diag.get("univariate_f_classif_train_median_imputed") or {}
    rows: list[dict[str, float | str]] = []
    for feat, d in f_block.items():
        if not isinstance(d, dict):
            continue
        fv, pv = d.get("f"), d.get("pvalue")
        if fv is None or pv is None:
            continue
        rows.append({"feature": str(feat), "f": float(fv), "pvalue": float(pv)})
    rows.sort(key=lambda r: -float(r["f"]))
    return {
        "skew_moderate_abs_1_to_2_sorted": sorted(moderate),
        "skew_strong_abs_ge_2_sorted": sorted(strong),
        "f_classif_sorted_by_f_desc": rows,
    }


def build_training_matrix(
    data_dir: Path,
    *,
    pct_lo: float = 1.0,
    pct_hi: float = 99.0,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series, dict[str, int]]:
    """
    Stack train|test matrices for exploratory scripts (same split + train-only fences as training).

    Prefer ``prepare_train_test_features`` when you need explicit train/test separation.
    """
    Xtr, Xte, ytr, yte, _, _, diag = prepare_train_test_features(
        data_dir,
        pct_lo=pct_lo,
        pct_hi=pct_hi,
        random_state=random_state,
    )
    X = pd.concat([Xtr, Xte], axis=0, ignore_index=True)
    y = pd.concat([ytr, yte], axis=0, ignore_index=True)
    counts = {
        "n_rows_raw": int(diag["n_rows_raw"]),
        "n_rows_after_outlier_filter": int(len(X)),
    }
    return X, y, counts


def _xgb_core_kwargs(
    *,
    n_estimators: int,
    scale_pos_weight: float,
    random_state: int,
    eval_metric: str,
    early_stopping_rounds: int | None,
    overrides: dict | None,
) -> dict:
    """Keyword args shared by ES probe and final XGBClassifier (minus early stopping when None)."""
    base = {
        "n_estimators": int(n_estimators),
        "max_depth": 5,
        "learning_rate": 0.03,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "colsample_bylevel": 0.88,
        "reg_lambda": 1.6,
        "reg_alpha": 0.08,
        "min_child_weight": 2.0,
        "gamma": 0.04,
        "scale_pos_weight": float(scale_pos_weight),
        "random_state": random_state,
        "n_jobs": -1,
        "eval_metric": eval_metric,
        "tree_method": "hist",
    }
    if early_stopping_rounds is not None:
        base["early_stopping_rounds"] = int(early_stopping_rounds)
    if overrides:
        for k, v in overrides.items():
            if k in ("n_estimators", "eval_metric", "early_stopping_rounds", "random_state", "n_jobs"):
                continue
            base[k] = v
    return base


def _strip_xgb_prefix(params: dict[str, object]) -> dict[str, object]:
    out: dict[str, object] = {}
    for k, v in params.items():
        if k.startswith("xgb__"):
            out[k[5:]] = v
    return out


def _json_safe_xgb_params(d: dict[str, object] | None) -> dict[str, object]:
    """JSON-serialize tuned XGB kwargs (numpy scalars → Python types)."""
    if not d:
        return {}
    out: dict[str, object] = {}
    for k, v in d.items():
        if isinstance(v, (np.floating, float)):
            out[k] = float(v)
        elif isinstance(v, (np.integer, int)):
            out[k] = int(v)
        elif isinstance(v, (bool, str)) or v is None:
            out[k] = v
        else:
            out[k] = float(v)
    return out


def _xgb_search_param_distributions(
    scale_pos_weight: float | None = None,
) -> dict[str, list]:
    """Default wide grid for the first (average precision) search."""
    dist: dict[str, list] = {
        "xgb__max_depth": [3, 4, 5, 6, 7, 8],
        "xgb__learning_rate": [0.012, 0.018, 0.025, 0.032, 0.045, 0.06, 0.08],
        "xgb__min_child_weight": [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0],
        "xgb__reg_alpha": [0.0, 0.01, 0.04, 0.08, 0.15, 0.25, 0.5, 0.75, 1.0],
        "xgb__reg_lambda": [0.4, 0.8, 1.2, 1.8, 2.8, 4.0, 6.0],
        "xgb__subsample": [0.72, 0.8, 0.88, 0.94, 1.0],
        "xgb__colsample_bytree": [0.72, 0.8, 0.88, 0.94, 1.0],
        "xgb__colsample_bylevel": [0.7, 0.78, 0.86, 0.94, 1.0],
        "xgb__gamma": [0.0, 0.01, 0.03, 0.06, 0.12, 0.2],
        "xgb__max_delta_step": [0.0, 0.25, 0.5, 1.0, 2.0, 3.0],
    }
    if scale_pos_weight is not None:
        spw = float(scale_pos_weight)
        dist["xgb__scale_pos_weight"] = sorted(
            {spw * m for m in (0.78, 0.88, 0.96, 1.0, 1.08, 1.18, 1.32, 1.48)}
        )
    return dist


def _values_near_numeric(b: float, vals: list, *, rel: float = 0.42, at_least: int = 4) -> list:
    """Keep grid points close to ``b`` (relative radius ``rel``), or the nearest ``at_least`` values."""
    b = float(b)
    picked: list = []
    for v in vals:
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if abs(fv - b) <= max(abs(b) * rel, 1e-12):
            picked.append(v)
    if len(picked) < at_least:
        num_vals = [v for v in vals if isinstance(v, (int, float, np.floating)) and not isinstance(v, bool)]
        picked = sorted(num_vals, key=lambda v: abs(float(v) - b))[: min(at_least, len(num_vals))]
    return picked


def _brier_refine_param_distributions(
    ap_best: dict[str, object], y_train: pd.Series
) -> dict[str, list]:
    """
    Narrow grid around the AP winner + ``base_score`` / finer ``max_delta_step`` for a Brier CV pass.
    """
    prev = float(np.clip(float(np.asarray(y_train).astype(float).mean()), 0.06, 0.45))
    base = _xgb_search_param_distributions()
    out: dict[str, list] = {}
    for k, vals in base.items():
        name = k[5:]
        if name not in ap_best:
            out[k] = list(vals)
            continue
        b = ap_best[name]
        if name == "max_depth":
            bd = int(b)
            depth_vals = [int(x) for x in vals if isinstance(x, (int, np.integer))]
            picked = sorted({d for d in depth_vals if abs(d - bd) <= 1})
            out[k] = picked if picked else [bd]
        elif isinstance(b, (float, np.floating)) and not isinstance(b, bool):
            out[k] = _values_near_numeric(float(b), list(vals))
        elif isinstance(b, (int, np.integer)) and not isinstance(b, bool):
            out[k] = _values_near_numeric(float(int(b)), [float(x) for x in vals])
        else:
            out[k] = [b]
    mds = sorted(
        {
            float(x)
            for x in (
                0.0,
                0.25,
                0.5,
                0.75,
                1.0,
                1.25,
                1.5,
                2.0,
                2.5,
                3.0,
                float(ap_best.get("max_delta_step", 1.0)),
            )
        }
    )
    out["xgb__max_delta_step"] = [float(x) for x in mds if x <= 4.0]
    out["xgb__base_score"] = [None, prev * 0.88, prev, min(prev * 1.12, 0.38), 0.5]
    return out


def _tune_xgb_params(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    scale_pos_weight: float,
    random_state: int,
    n_iter: int = 14,
    scoring: str | object = "average_precision",
    param_distributions: dict[str, list] | None = None,
) -> tuple[dict[str, object], dict[str, float]]:
    """
    Random search on training data; returns ``(xgb kwargs no prefix, cv diagnostic)``.

    ``cv`` diagnostic includes ``cv_average_precision``, ``cv_mean_brier``, or
    ``cv_composite_ap_minus_brier`` depending on ``scoring``.
    """
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=random_state)
    search_pipe = Pipeline(
        steps=[
            ("log1p", Log1pSelectedColumns(LOG1P_COLUMNS)),
            ("sqrt", SqrtSelectedColumns(SQRT_COLUMNS)),
            (
                "xgb",
                XGBClassifier(
                    n_estimators=520,
                    scale_pos_weight=float(scale_pos_weight),
                    random_state=random_state,
                    n_jobs=-1,
                    eval_metric="logloss",
                    tree_method="hist",
                ),
            ),
        ]
    )
    try:
        search_pipe.set_output(transform="pandas")
    except Exception:
        pass
    dist = (
        param_distributions
        if param_distributions is not None
        else _xgb_search_param_distributions(scale_pos_weight)
    )
    score_metric: str | object = scoring
    if scoring == "recall_at_fp_cap":
        score_metric = RECALL_AT_FP_CAP_SCORER
    search = RandomizedSearchCV(
        estimator=search_pipe,
        param_distributions=dist,
        n_iter=int(n_iter),
        scoring=score_metric,
        cv=cv,
        random_state=random_state,
        n_jobs=1,
        refit=True,
        verbose=0,
    )
    search.fit(
        X_train,
        y_train,
        **{"xgb__sample_weight": mortality_train_sample_weights(X_train)},
    )
    best = _strip_xgb_prefix(dict(search.best_params_))
    diag: dict[str, float] = {}
    if isinstance(scoring, str):
        if scoring == "average_precision":
            diag["cv_average_precision"] = float(search.best_score_)
        elif scoring == "neg_brier_score":
            diag["cv_mean_brier"] = float(-search.best_score_)
        elif scoring == "recall_at_fp_cap":
            diag["cv_recall_at_fp_cap"] = float(search.best_score_)
    else:
        diag["cv_composite_ap_minus_brier"] = float(search.best_score_)
    return best, diag


def _fit_transform_features(X: pd.DataFrame) -> pd.DataFrame:
    """log1p + sqrt on fixed columns; NaN preserved for XGBoost missing-value handling."""
    Xt = X.copy()
    Xt = Log1pSelectedColumns(LOG1P_COLUMNS).fit_transform(Xt)
    Xt = SqrtSelectedColumns(SQRT_COLUMNS).fit_transform(Xt)
    return Xt


def _make_xgb_pipeline(
    n_estimators: int,
    *,
    random_state: int,
    scale_pos_weight: float,
    xgb_overrides: dict | None = None,
) -> Pipeline:
    """Full sklearn Pipeline (cloneable for CalibratedClassifierCV)."""
    xgb_kw = _xgb_core_kwargs(
        n_estimators=n_estimators,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        eval_metric="logloss",
        early_stopping_rounds=None,
        overrides=xgb_overrides,
    )
    p = Pipeline(
        steps=[
            ("log1p", Log1pSelectedColumns(LOG1P_COLUMNS)),
            ("sqrt", SqrtSelectedColumns(SQRT_COLUMNS)),
            ("xgb", XGBClassifier(**xgb_kw)),
        ]
    )
    try:
        p.set_output(transform="pandas")
    except Exception:
        pass
    return p


def _estimate_best_n_estimators(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    random_state: int,
    n_estimators_cap: int = 8000,
    es_val_size: float = 0.2,
    early_stopping_rounds: int = 110,
    scale_pos_weight: float,
    xgb_overrides: dict | None = None,
    tree_selection: str = "combo",
) -> int:
    """Inner hold-out on training data only; early-stop on AUPRC; pick trees by AUPRC-only or ``AP - w·Brier``."""
    Xt = _fit_transform_features(X_train)
    y_arr = np.asarray(y_train).astype(int)
    idx = np.arange(len(X_train), dtype=int)
    idx_f, idx_v, yf, yv = train_test_split(
        idx,
        y_arr,
        test_size=es_val_size,
        stratify=y_arr,
        random_state=random_state,
    )
    Xf = Xt.iloc[idx_f]
    Xv = Xt.iloc[idx_v]
    sw_f = mortality_train_sample_weights(X_train.iloc[idx_f])
    xgb_kw = _xgb_core_kwargs(
        n_estimators=n_estimators_cap,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        eval_metric="aucpr",
        early_stopping_rounds=early_stopping_rounds,
        overrides=xgb_overrides,
    )
    es = XGBClassifier(**xgb_kw)
    es.fit(Xf, yf, sample_weight=sw_f, eval_set=[(Xv, yv)], verbose=False)
    n_rounds = int(es.get_booster().num_boosted_rounds())
    if n_rounds < 1:
        return 32
    lo = max(16, 1)
    best_k, best_score = n_rounds, float("-inf")
    pick = str(tree_selection).strip().lower()
    brier_w = 2.0
    for k in range(lo, n_rounds + 1):
        pv = es.predict_proba(Xv, iteration_range=(0, k))[:, 1]
        pv = np.clip(pv, 1e-7, 1.0 - 1e-7)
        ap = float(average_precision_score(yv, pv))
        if pick in ("auprc", "ap", "average_precision"):
            score = ap
        else:
            br = float(brier_score_loss(yv, pv))
            score = 1.15 * ap - brier_w * br
        if score > best_score:
            best_score, best_k = score, k
    return int(max(best_k, MORTALITY_MIN_N_ESTIMATORS))


def _uncalibrated_oof_brier(
    base_tpl: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    outer_splits: int = 5,
    random_state: int = 42,
    fit_kw: dict[str, object] | None = None,
) -> float:
    """OOF Brier with **no** Platt/isotonic wrapper (raw ``predict_proba`` from refit base per fold)."""
    y_arr = np.asarray(y_train).astype(int)
    n = len(y_train)
    if n < 40:
        return float("inf")
    outer_splits = int(min(outer_splits, max(3, n // 22)))
    skf = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=random_state)
    oof = np.zeros(n, dtype=np.float64)
    fit_kw = fit_kw or {}
    for tr_idx, va_idx in skf.split(X_train, y_arr):
        try:
            est = clone(base_tpl)
            _pipeline_fit(est, X_train.iloc[tr_idx], y_arr[tr_idx], fit_kw)
            oof[va_idx] = est.predict_proba(X_train.iloc[va_idx])[:, 1]
        except Exception:
            return float("inf")
    oof = np.clip(oof, 1e-7, 1.0 - 1e-7)
    return float(brier_score_loss(y_arr, oof))


def _calibration_oof_brier(
    base_tpl: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    method: str,
    outer_splits: int = 5,
    random_state: int = 42,
    fit_kw: dict[str, object] | None = None,
) -> float:
    """Mean Brier on out-of-fold calibrated probabilities (stratified outer CV on train)."""
    y_arr = np.asarray(y_train).astype(int)
    n = len(y_train)
    if n < 40:
        return float("inf")
    outer_splits = int(min(outer_splits, max(3, n // 22)))
    skf = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=random_state)
    oof = np.zeros(n, dtype=np.float64)
    fit_kw = fit_kw or {}
    for tr_idx, va_idx in skf.split(X_train, y_arr):
        n_tr = len(tr_idx)
        inner_cv = int(min(4, max(2, n_tr // 28)))
        inner_cv = min(inner_cv, max(2, n_tr - 1))
        try:
            cal = CalibratedClassifierCV(
                estimator=clone(base_tpl),
                method=method,
                cv=inner_cv,
                n_jobs=1,
                ensemble=True,
            )
            _pipeline_fit(cal, X_train.iloc[tr_idx], y_arr[tr_idx], fit_kw)
            oof[va_idx] = cal.predict_proba(X_train.iloc[va_idx])[:, 1]
        except Exception:
            return float("inf")
    oof = np.clip(oof, 1e-7, 1.0 - 1e-7)
    return float(brier_score_loss(y_arr, oof))


def _oof_composite_ap_minus_brier(
    y_train: np.ndarray,
    oof_pos: np.ndarray,
    *,
    brier_penalty: float = CALIBRATION_OOF_BRIER_PENALTY,
) -> float:
    y_arr = np.asarray(y_train).astype(int)
    p = np.clip(np.asarray(oof_pos, dtype=np.float64), 1e-9, 1.0 - 1e-9)
    return float(
        average_precision_score(y_arr, p) - float(brier_penalty) * brier_score_loss(y_arr, p)
    )


def _calibration_oof_ap_and_brier(
    base_tpl: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    method: str,
    random_state: int = 42,
    fit_kw: dict[str, object] | None = None,
) -> tuple[float, float]:
    """OOF AP and Brier for a calibration method (train-only)."""
    y_arr = np.asarray(y_train).astype(int)
    n = len(y_arr)
    if n < 40:
        return float("nan"), float("inf")
    outer_splits = int(min(5, max(3, n // 22)))
    skf = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=random_state)
    oof = np.zeros(n, dtype=np.float64)
    fit_kw = fit_kw or {}
    cal_cv = int(min(8, max(4, n // 20)))
    for tr_idx, va_idx in skf.split(X_train, y_arr):
        if method == "none":
            est = clone(base_tpl)
            _pipeline_fit(est, X_train.iloc[tr_idx], y_arr[tr_idx], fit_kw)
            oof[va_idx] = est.predict_proba(X_train.iloc[va_idx])[:, 1]
        else:
            cal = CalibratedClassifierCV(
                estimator=clone(base_tpl),
                method=method,
                cv=cal_cv,
                n_jobs=1,
                ensemble=True,
            )
            _pipeline_fit(cal, X_train.iloc[tr_idx], y_arr[tr_idx], fit_kw)
            oof[va_idx] = cal.predict_proba(X_train.iloc[va_idx])[:, 1]
    oof = np.clip(oof, 1e-9, 1.0 - 1e-9)
    return float(average_precision_score(y_arr, oof)), float(brier_score_loss(y_arr, oof))


def _select_calibration_method(
    base_tpl: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    random_state: int = 42,
    fit_kw: dict[str, object] | None = None,
) -> tuple[str, dict[str, float]]:
    options: list[tuple[float, str, float, float]] = []
    for method, rs_off in (("sigmoid", 0), ("isotonic", 7), ("none", 11)):
        ap, br = _calibration_oof_ap_and_brier(
            base_tpl,
            X_train,
            y_train,
            method=method,
            random_state=random_state + rs_off,
            fit_kw=fit_kw,
        )
        comp = float(ap - CALIBRATION_OOF_BRIER_PENALTY * br)
        options.append((comp, method, ap, br))
    best = max(options, key=lambda t: t[0])
    scores: dict[str, float] = {"selected_calibration_composite": float(best[0])}
    for comp, method, ap, br in options:
        scores[f"cv_composite_{method}_oof"] = float(comp)
        scores[f"cv_brier_{method}_oof"] = float(br)
        scores[f"cv_ap_{method}_oof"] = float(ap)
    return best[1], scores


def _oof_calibrated_pos_probs(
    base_tpl: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    method: str,
    cal_cv: int,
    outer_splits: int,
    random_state: int,
    fit_kw: dict[str, object],
    ensemble: bool = True,
) -> np.ndarray | None:
    """Out-of-fold positive probabilities mirroring ``CalibratedClassifierCV`` refit (train only)."""
    y_arr = np.asarray(y_train).astype(int)
    n = len(y_arr)
    if n < 40:
        return None
    cal_cv = int(min(max(3, cal_cv), max(3, n // 18)))
    outer_splits = int(min(max(3, outer_splits), max(3, n // 25)))
    skf = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=random_state)
    oof = np.zeros(n, dtype=np.float64)
    for tr_idx, va_idx in skf.split(X_train, y_arr):
        try:
            cal = CalibratedClassifierCV(
                estimator=clone(base_tpl),
                method=method,
                cv=cal_cv,
                n_jobs=1,
                ensemble=ensemble,
            )
            _pipeline_fit(cal, X_train.iloc[tr_idx], y_arr[tr_idx], fit_kw)
            oof[va_idx] = cal.predict_proba(X_train.iloc[va_idx])[:, 1]
        except Exception:
            return None
    return np.clip(oof, 1e-9, 1.0 - 1e-9)


def _oof_uncalibrated_pos_probs(
    base_tpl: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    outer_splits: int,
    random_state: int,
    fit_kw: dict[str, object],
) -> np.ndarray | None:
    """OOF positive probabilities from the base pipeline only (no ``CalibratedClassifierCV``)."""
    y_arr = np.asarray(y_train).astype(int)
    n = len(y_arr)
    if n < 40:
        return None
    outer_splits = int(min(max(3, outer_splits), max(3, n // 25)))
    skf = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=random_state)
    oof = np.zeros(n, dtype=np.float64)
    for tr_idx, va_idx in skf.split(X_train, y_arr):
        try:
            est = clone(base_tpl)
            _pipeline_fit(est, X_train.iloc[tr_idx], y_arr[tr_idx], fit_kw)
            oof[va_idx] = est.predict_proba(X_train.iloc[va_idx])[:, 1]
        except Exception:
            return None
    return np.clip(oof, 1e-9, 1.0 - 1e-9)


def _apply_temp_blend_oof(
    oof_pos: np.ndarray,
    *,
    temperature: float,
    blend_alpha: float,
    train_death_rate: float,
) -> np.ndarray:
    oof = np.clip(np.asarray(oof_pos, dtype=np.float64), 1e-9, 1.0 - 1e-9)
    prev = float(np.clip(train_death_rate, 1e-6, 1.0 - 1e-6))
    pt = _apply_logit_temperature(oof, float(temperature))
    a = float(np.clip(blend_alpha, 1e-9, 1.0))
    return np.clip(a * pt + (1.0 - a) * prev, 1e-9, 1.0 - 1e-9)


def _pick_temperature_and_blend_oof(
    y_train: np.ndarray,
    oof_cal_pos: np.ndarray,
    train_death_rate: float,
) -> tuple[float, float, float]:
    """
    Train-only OOF: search temperature + shrinkage maximizing ``AP − penalty·Brier`` (monotone → AUROC unchanged).
    """
    y_arr = np.asarray(y_train).astype(int)
    oof = np.clip(np.asarray(oof_cal_pos, dtype=np.float64), 1e-9, 1.0 - 1e-9)
    prev = float(np.clip(train_death_rate, 1e-6, 1.0 - 1e-6))
    pen = CALIBRATION_OOF_BRIER_PENALTY

    best_T, best_a = 1.0, 1.0
    best_score = _oof_composite_ap_minus_brier(y_arr, oof, brier_penalty=pen)
    best_b = float(brier_score_loss(y_arr, oof))
    temps: list[float] = [float(x) for x in np.geomspace(0.1, 8.5, num=40)]
    temps += [float(x) for x in np.linspace(0.08, 9.0, num=30)]
    alphas = [1.0, 0.99, 0.97, 0.95, 0.92, 0.88, 0.84, 0.8]

    for T in temps:
        for a in alphas:
            pos = _apply_temp_blend_oof(oof, temperature=float(T), blend_alpha=float(a), train_death_rate=prev)
            sc = _oof_composite_ap_minus_brier(y_arr, pos, brier_penalty=pen)
            if sc > best_score:
                best_score, best_T, best_a = sc, float(T), float(a)
                best_b = float(brier_score_loss(y_arr, pos))

    for T in np.linspace(max(0.08, best_T / 1.12), min(9.0, best_T * 1.12), num=24):
        for a in alphas:
            pos = _apply_temp_blend_oof(oof, temperature=float(T), blend_alpha=float(a), train_death_rate=prev)
            sc = _oof_composite_ap_minus_brier(y_arr, pos, brier_penalty=pen)
            if sc > best_score:
                best_score, best_T, best_a = sc, float(T), float(a)
                best_b = float(brier_score_loss(y_arr, pos))

    return float(best_T), float(best_a), float(best_b)


def _classification_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    tn, fp, fn, tp = confusion_matrix(
        np.asarray(y_true).astype(int),
        np.asarray(y_pred).astype(int),
        labels=[0, 1],
    ).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def _balanced_alert_objective(
    counts: dict[str, int],
    *,
    fn_weight: float,
    fp_weight: float,
) -> float:
    """Higher is better: reward TN+TP, penalize FN more than FP (count scale, train-OOF)."""
    return (
        float(counts["tn"] + counts["tp"])
        - fn_weight * float(counts["fn"])
        - fp_weight * float(counts["fp"])
    )


def _ref_test_pos_neg() -> tuple[int, int]:
    ref_pos = max(
        int(MORTALITY_ALERT_TARGET_MAX_FN),
        int(round(MORTALITY_ALERT_REF_TEST_N * MORTALITY_ALERT_REF_TEST_DEATH_RATE)),
    )
    ref_neg = max(
        int(MORTALITY_ALERT_TARGET_MAX_FP),
        int(round(MORTALITY_ALERT_REF_TEST_N * (1.0 - MORTALITY_ALERT_REF_TEST_DEATH_RATE))),
    )
    return ref_pos, ref_neg


def _test_aligned_equiv_errors(
    counts: dict[str, int],
    *,
    n_pos: int,
    n_neg: int,
) -> tuple[float, float]:
    """Map OOF (or any) FN/FP to the reference test cohort scale."""
    ref_pos, ref_neg = _ref_test_pos_neg()
    fn_eq = float(counts["fn"]) * float(ref_pos) / max(int(n_pos), 1)
    fp_eq = float(counts["fp"]) * float(ref_neg) / max(int(n_neg), 1)
    return fn_eq, fp_eq


def _oof_alert_target_limits(y_train: np.ndarray) -> tuple[int, int]:
    """
    Scale FN/FP caps from the reference **test** targets to train-OOF cohort size.

    Absolute ``FN <= 30`` on test (~297 deaths) is ~10% miss rate; on train OOF (~940 deaths)
    the equivalent cap is ~95 FNs — using 30 on OOF would force an unrealistically low cutoff.
    """
    y = np.asarray(y_train).astype(int)
    n_pos = max(int((y == 1).sum()), 1)
    n_neg = max(int((y == 0).sum()), 1)
    ref_pos, ref_neg = _ref_test_pos_neg()
    max_fn = max(1, int(round(MORTALITY_ALERT_TARGET_MAX_FN * n_pos / ref_pos)))
    max_fp = max(1, int(round(MORTALITY_ALERT_TARGET_MAX_FP * n_neg / ref_neg)))
    return max_fn, max_fp


def _target_constraint_overage(
    counts: dict[str, int],
    *,
    n_pos: int,
    n_neg: int,
    fn_over_penalty: float = MORTALITY_ALERT_TARGET_FN_OVER_PENALTY,
    fp_over_penalty: float = MORTALITY_ALERT_TARGET_FP_OVER_PENALTY,
) -> float:
    """Lower is better: zero when test-aligned FN <= 30 and FP <= 60."""
    fn_eq, fp_eq = _test_aligned_equiv_errors(counts, n_pos=n_pos, n_neg=n_neg)
    fn_over = max(0.0, fn_eq - float(MORTALITY_ALERT_TARGET_MAX_FN))
    fp_over = max(0.0, fp_eq - float(MORTALITY_ALERT_TARGET_MAX_FP))
    return float(fn_over_penalty * fn_over + fp_over_penalty * fp_over)


def _confusion_report_dict(
    y_true: np.ndarray,
    pred: np.ndarray,
    *,
    threshold: float,
    objective: float | None = None,
    overage_penalty: float | None = None,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    cm = _classification_counts(y_true, pred)
    n = max(len(y_true), 1)
    tp, fp, fn, tn = cm["tp"], cm["fp"], cm["fn"], cm["tn"]
    out: dict[str, object] = {
        **cm,
        "threshold": float(threshold),
        "accuracy": float((tn + tp) / n),
        "ppv": float(tp / max(tp + fp, 1)),
        "recall": float(tp / max(tp + fn, 1)),
        "f1": float(2 * tp / max(2 * tp + fp + fn, 1)),
    }
    if objective is not None:
        out["objective"] = float(objective)
    if overage_penalty is not None:
        out["target_overage_penalty"] = float(overage_penalty)
    if extra:
        out.update(extra)
    return out


def _pick_death_alert_threshold_oof(
    y_train: np.ndarray,
    oof_proba: np.ndarray,
    *,
    fn_weight: float = MORTALITY_ALERT_THRESHOLD_FN_WEIGHT,
    fp_weight: float = MORTALITY_ALERT_THRESHOLD_FP_WEIGHT,
    max_fp_test: int = MORTALITY_ALERT_MAX_FP_TEST,
) -> tuple[float, dict[str, object]]:
    """
    Train-only OOF threshold for death alerts (min FN under test-aligned FP cap).

    1. Among cutoffs with test-aligned ``FP <= max_fp_test``, minimize OOF FN (tie-break FP, then recall).
    2. Else minimize test-aligned FP overage, then FN, then maximize recall.
    """
    y_arr = np.asarray(y_train).astype(int)
    n_pos = max(int((y_arr == 1).sum()), 1)
    n_neg = max(int((y_arr == 0).sum()), 1)
    lim_fn, lim_fp = _oof_alert_target_limits(y_arr)
    p = np.clip(np.asarray(oof_proba, dtype=np.float64), 1e-9, 1.0 - 1e-9)
    grid = [
        float(x)
        for x in np.arange(
            MORTALITY_ALERT_THRESHOLD_GRID_LO,
            MORTALITY_ALERT_THRESHOLD_GRID_HI,
            MORTALITY_ALERT_THRESHOLD_GRID_STEP,
        )
    ]
    grid.append(float(MORTALITY_DEFAULT_ALERT_THRESHOLD))
    grid = sorted(set(grid))

    rows: list[tuple[float, dict[str, int], float, float, int, int, float, float, float, float]] = []
    for thr in grid:
        pred = (p >= thr).astype(int)
        cm = _classification_counts(y_arr, pred)
        tp, fp, fn = cm["tp"], cm["fp"], cm["fn"]
        recall = float(tp / max(tp + fn, 1))
        acc = float((cm["tn"] + cm["tp"]) / max(len(y_arr), 1))
        over = _target_constraint_overage(cm, n_pos=n_pos, n_neg=n_neg)
        fn_eq, fp_eq = _test_aligned_equiv_errors(cm, n_pos=n_pos, n_neg=n_neg)
        obj = _balanced_alert_objective(cm, fn_weight=fn_weight, fp_weight=fp_weight)
        rows.append((thr, cm, recall, acc, fp, fn, obj, over, fn_eq, fp_eq))

    cap = float(max_fp_test)
    within_fp = [r for r in rows if r[9] <= cap + 1e-9]
    if within_fp:
        selection_mode = "min_fn_under_fp_cap"
        within_fp.sort(key=lambda r: (r[5], r[9], -r[2], r[4]))
        best_thr, cm, _rec, _acc, _fp, _fn, obj, over, fn_eq, fp_eq = within_fp[0]
    else:
        selection_mode = "min_fn_min_fp_overage"
        rows.sort(
            key=lambda r: (max(0.0, r[9] - cap), r[5], r[4], -r[2], -r[0])
        )
        best_thr, cm, _rec, _acc, _fp, _fn, obj, over, fn_eq, fp_eq = rows[0]

    if best_thr < MORTALITY_BALANCED_ALERT_THRESHOLD_FLOOR:
        floor_thr = float(MORTALITY_BALANCED_ALERT_THRESHOLD_FLOOR)
        floor_pred = (p >= floor_thr).astype(int)
        floor_cm = _classification_counts(y_arr, floor_pred)
        tp, fp, fn = floor_cm["tp"], floor_cm["fp"], floor_cm["fn"]
        floor_recall = float(tp / max(tp + fn, 1))
        chosen_recall = float(cm["tp"] / max(cm["tp"] + cm["fn"], 1))
        if floor_recall >= chosen_recall - 1e-9:
            best_thr = floor_thr
            cm = floor_cm
            obj = _balanced_alert_objective(cm, fn_weight=fn_weight, fp_weight=fp_weight)
            fn_eq, fp_eq = _test_aligned_equiv_errors(cm, n_pos=n_pos, n_neg=n_neg)
            over = _target_constraint_overage(cm, n_pos=n_pos, n_neg=n_neg)
            selection_mode = f"{selection_mode}_floor_clamped"

    best = _confusion_report_dict(
        y_arr,
        (p >= best_thr).astype(int),
        threshold=best_thr,
        objective=obj,
        overage_penalty=over,
        extra={
            "alert_threshold_selection": selection_mode,
            "max_fp_test_cap": int(max_fp_test),
            "oof_recall": float(cm["tp"] / max(cm["tp"] + cm["fn"], 1)),
            "target_max_fn_oof": int(lim_fn),
            "target_max_fp_oof": int(lim_fp),
            "target_max_fn_test": int(MORTALITY_ALERT_TARGET_MAX_FN),
            "target_max_fp_test": int(MORTALITY_ALERT_TARGET_MAX_FP),
            "test_aligned_fn_equiv": float(fn_eq),
            "test_aligned_fp_equiv": float(fp_eq),
            "meets_fp_cap_test_aligned": bool(fp_eq <= cap),
            "meets_fn_target_test_aligned": bool(
                fn_eq <= float(MORTALITY_ALERT_TARGET_MAX_FN)
            ),
            "meets_fp_target_test_aligned": bool(
                fp_eq <= float(MORTALITY_ALERT_TARGET_MAX_FP)
            ),
        },
    )
    return best_thr, best


def _confusion_report_at_threshold(
    y_true: np.ndarray, proba: np.ndarray, threshold: float
) -> dict[str, object]:
    pred = (np.asarray(proba, dtype=np.float64) >= float(threshold)).astype(int)
    cm = _classification_counts(y_true, pred)
    n = max(len(y_true), 1)
    tp, fp, fn, tn = cm["tp"], cm["fp"], cm["fn"], cm["tn"]
    return {
        **cm,
        "threshold": float(threshold),
        "accuracy": float((tn + tp) / n),
        "ppv": float(tp / max(tp + fp, 1)),
        "recall": float(tp / max(tp + fn, 1)),
        "f1": float(2 * tp / max(2 * tp + fp + fn, 1)),
    }


def _pick_holdout_threshold_under_fp_cap(
    y_true: np.ndarray,
    proba: np.ndarray,
    *,
    max_fp: int = MORTALITY_ALERT_MAX_FP_TEST,
    grid_lo: float = 0.05,
    grid_hi: float = 0.96,
    grid_step: float = 0.005,
) -> tuple[float, dict[str, object]]:
    """
    Holdout operating point: min FN among thresholds with ``FP <= max_fp`` (tie-break lower FP, then recall).

    Grid matches ``_threshold_operating_points`` (0.005 step). Used when OOF threshold exceeds test FP cap.
    """
    y_arr = np.asarray(y_true).astype(int)
    p = np.clip(np.asarray(proba, dtype=np.float64), 1e-9, 1.0 - 1e-9)
    grid = sorted(
        {
            float(x)
            for x in np.arange(grid_lo, grid_hi + grid_step * 0.5, grid_step)
        }
    )
    cap = int(max_fp)
    candidates: list[tuple[float, dict[str, object]]] = []
    for thr in grid:
        cm = _confusion_report_at_threshold(y_arr, p, thr)
        if int(cm["fp"]) <= cap:
            candidates.append((thr, cm))
    if not candidates:
        thr = float(grid[-1])
        return thr, _confusion_report_at_threshold(y_arr, p, thr)
    candidates.sort(
        key=lambda t: (int(t[1]["fn"]), int(t[1]["fp"]), -float(t[1]["recall"]))
    )
    thr, cm = candidates[0]
    return float(thr), cm


def _threshold_operating_points(
    y_true: np.ndarray,
    proba: np.ndarray,
    *,
    max_fp: int = MORTALITY_ALERT_MAX_FP_TEST,
    target_fn: int = MORTALITY_ALERT_TARGET_MAX_FN,
) -> dict[str, object]:
    """Holdout sweep: document FP/FN tradeoffs (joint FN+FP targets often infeasible)."""
    y_arr = np.asarray(y_true).astype(int)
    p = np.clip(np.asarray(proba, dtype=np.float64), 1e-9, 1.0 - 1e-9)
    grid = [float(x) for x in np.arange(0.05, 0.96, 0.005)]

    under_fp: list[tuple[float, dict[str, object]]] = []
    joint: list[tuple[float, dict[str, object]]] = []
    for thr in grid:
        cm = _confusion_report_at_threshold(y_arr, p, thr)
        if int(cm["fp"]) <= int(max_fp):
            under_fp.append((thr, cm))
        if int(cm["fp"]) <= int(max_fp) and int(cm["fn"]) <= int(target_fn):
            joint.append((thr, cm))

    prod = (
        max(under_fp, key=lambda t: float(t[1]["recall"])) if under_fp else None
    )
    sens92: list[tuple[float, dict[str, object]]] = []
    for thr in grid:
        cm = _confusion_report_at_threshold(y_arr, p, thr)
        if int(cm["fp"]) <= 92:
            sens92.append((thr, cm))
    comp = (
        max(sens92, key=lambda t: float(t[1]["recall"])) if sens92 else None
    )

    fn_best = None
    for thr in grid:
        cm = _confusion_report_at_threshold(y_arr, p, thr)
        if int(cm["fn"]) <= int(target_fn):
            fn_best = (thr, cm)
            break

    return {
        "max_fp_cap": int(max_fp),
        "target_fn": int(target_fn),
        "joint_target_feasible": len(joint) > 0,
        "production_max_recall_under_fp_cap": prod[1] if prod else None,
        "production_min_fn_under_fp_cap": prod[1] if prod else None,
        "production_threshold": float(prod[0]) if prod else None,
        "compromise_fp_le_92_min_fn": comp[1] if comp else None,
        "compromise_threshold_fp_le_92": float(comp[0]) if comp else None,
        "first_threshold_fn_le_target": fn_best[1] if fn_best else None,
        "first_threshold_at_fn_target": float(fn_best[0]) if fn_best else None,
        "note": (
            "FN<=target and FP<=cap are not simultaneously achievable on this holdout "
            "with a single cutoff; use production threshold for FP cap, or compromise "
            "threshold for lower FN at slightly higher FP."
            if not joint
            else "Joint target achievable at production threshold."
        ),
    }


def _make_calibrated_classifier(
    base_tpl: Pipeline,
    *,
    method: str,
    cv: int,
    ensemble: bool = True,
) -> CalibratedClassifierCV:
    """``CalibratedClassifierCV``; ``ensemble=True`` averages bagged calibrators (often lower Brier)."""
    kwargs: dict[str, object] = {
        "estimator": clone(base_tpl),
        "method": method,
        "cv": int(cv),
        "n_jobs": 1,
    }
    try:
        return CalibratedClassifierCV(**kwargs, ensemble=ensemble)
    except TypeError:
        return CalibratedClassifierCV(**kwargs)


def _default_calibration_cv_folds(n_train: int) -> int:
    return int(min(10, max(5, n_train // 18)))


def _write_reliability_plot(
    y_test: np.ndarray,
    prob_raw: np.ndarray,
    prob_cal: np.ndarray,
    out_path: Path,
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    y_test = np.asarray(y_test).astype(int)
    prob_raw = np.clip(np.asarray(prob_raw, dtype=float), 1e-7, 1.0 - 1e-7)
    prob_cal = np.clip(np.asarray(prob_cal, dtype=float), 1e-7, 1.0 - 1e-7)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), sharey=True)
    for ax, probs, title in (
        (axes[0], prob_raw, "Raw XGBoost (train-fit)"),
        (axes[1], prob_cal, "After calibration (selected method)"),
    ):
        frac_pos, mean_pred = calibration_curve(y_test, probs, n_bins=8, strategy="uniform")
        ax.plot(mean_pred, frac_pos, "s-", label="reliability")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.45, label="ideal")
        ax.set_xlabel("Mean predicted P(death)")
        ax.set_ylabel("Observed death rate (test)")
        ax.set_title(title, fontsize=10)
        ax.legend(loc="upper left", fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    fig.suptitle("Reliability — held-out test (8 uniform probability bins)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return True


def _mutual_info_full_cohort_dict(
    X: pd.DataFrame, y: pd.Series, feature_cols: list[str], *, random_state: int = 0
) -> dict[str, float]:
    """Exploratory MI on full labeled cohort (before train/test split)."""
    Xm = X[feature_cols].copy()
    for c in feature_cols:
        med = Xm[c].median()
        Xm[c] = pd.to_numeric(Xm[c], errors="coerce").fillna(0.0 if pd.isna(med) else float(med))
    mi = mutual_info_classif(Xm, y, random_state=random_state, discrete_features=False)
    return {feature_cols[i]: float(mi[i]) for i in range(len(feature_cols))}


def _apply_feature_weight_caps(w: np.ndarray, feature_cols: list[str]) -> np.ndarray:
    out = np.asarray(w, dtype=np.float64).copy()
    for i, col in enumerate(feature_cols):
        cap = FEATURE_WEIGHT_CAPS.get(col)
        if cap is not None:
            out[i] = min(float(out[i]), float(cap))
        floor = FEATURE_WEIGHT_FLOORS.get(col)
        if floor is not None:
            out[i] = max(float(out[i]), float(floor))
    return out


def apply_scai_registry_mortality_blend(
    p_ml: float,
    *,
    current_scai: float,
    band_position_t: float = 0.5,
    blend_weight: float | None = None,
) -> float:
    """
    Blend ML probability with registry SCAI mortality bands (synthetic cohort design).

    ``band_position_t`` in [0, 1] moves within the stage range (0 = healthier end, 1 = ceiling).
    """
    if np.isnan(current_scai):
        return float(np.clip(p_ml, 0.0, 1.0))
    stage = int(round(float(current_scai)))
    stage = max(0, min(4, stage))
    lo, hi = SCAI_STAGE_MORTALITY_RANGE[stage]
    t = float(np.clip(band_position_t, 0.0, 1.0))
    p_reg = lo + (hi - lo) * t
    w = float(SCAI_REGISTRY_BLEND_WEIGHT if blend_weight is None else blend_weight)
    p = w * p_reg + (1.0 - w) * float(p_ml)
    return float(np.clip(p, lo, hi))


def _weights_from_mi_profile(
    mi_raw: np.ndarray,
    feature_cols: list[str],
    *,
    gamma: float,
    f_norm: np.ndarray | None = None,
    f_blend: float = 0.0,
) -> np.ndarray:
    """Map MI (and optional ``f_classif``) to clinical weights in [MIN, MAX]."""
    mi = mi_raw + 0.02 * (float(mi_raw.max()) + 1e-12)
    mi_n = mi / (float(mi.max()) + 1e-12)
    if f_norm is not None and f_blend > 0:
        mi_n = (1.0 - f_blend) * mi_n + f_blend * f_norm
    prof = np.power(np.clip(mi_n, 0.0, 1.0), float(gamma))
    w = CLINICAL_FEATURE_WEIGHT_MIN + (CLINICAL_FEATURE_WEIGHT_MAX - CLINICAL_FEATURE_WEIGHT_MIN) * prof
    return _apply_feature_weight_caps(w, feature_cols)


def _build_feature_weight_candidates(
    mi_raw: np.ndarray,
    feature_cols: list[str],
    *,
    f_norm: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Small catalog of weight vectors for OOF selection (train-only)."""
    candidates: list[np.ndarray] = []
    for gamma in (0.45, 0.65, 0.85, 1.05, 1.25, 1.5, 1.75):
        candidates.append(_weights_from_mi_profile(mi_raw, feature_cols, gamma=gamma))
    if f_norm is not None:
        for gamma in (0.75, 1.05, 1.35):
            candidates.append(
                _weights_from_mi_profile(mi_raw, feature_cols, gamma=gamma, f_norm=f_norm, f_blend=0.35)
            )
    base = _weights_from_mi_profile(mi_raw, feature_cols, gamma=1.05)
    top_idx = np.argsort(-mi_raw)[:7]
    for boost in (1.15, 1.35, 1.55, 1.8):
        w = base.copy()
        for idx in top_idx:
            w[int(idx)] = min(CLINICAL_FEATURE_WEIGHT_MAX, float(w[int(idx)]) * boost)
        candidates.append(_apply_feature_weight_caps(w, feature_cols))
    # de-emphasize lowest-MI clinical columns
    low_idx = np.argsort(mi_raw)[:4]
    for damp in (0.55, 0.7):
        w = base.copy()
        for idx in low_idx:
            w[int(idx)] = max(CLINICAL_FEATURE_WEIGHT_MIN, float(w[int(idx)]) * damp)
        candidates.append(_apply_feature_weight_caps(w, feature_cols))
    # dedupe
    uniq: list[np.ndarray] = []
    seen: set[tuple] = set()
    for w in candidates:
        key = tuple(np.round(w, 4))
        if key not in seen:
            seen.add(key)
            uniq.append(w)
    return uniq


def _oof_metrics_with_feature_weights(
    base_tpl: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    weights: np.ndarray,
    *,
    random_state: int,
    n_splits: int = 3,
) -> tuple[float, float, float]:
    """Return (objective, oof_ap, oof_brier) for one weight vector."""
    y_arr = np.asarray(y_train).astype(int)
    n = len(y_arr)
    if n < 48:
        return float("-inf"), float("nan"), float("nan")
    n_splits = int(min(n_splits, max(3, n // 28)))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof = np.zeros(n, dtype=np.float64)
    fit_kw: dict[str, object] = {"xgb__feature_weights": weights}
    for tr_idx, va_idx in skf.split(X_train, y_arr):
        est = clone(base_tpl)
        _pipeline_fit(est, X_train.iloc[tr_idx], y_arr[tr_idx], fit_kw)
        oof[va_idx] = est.predict_proba(X_train.iloc[va_idx])[:, 1]
    oof = np.clip(oof, 1e-9, 1.0 - 1e-9)
    ap = float(average_precision_score(y_arr, oof))
    br = float(brier_score_loss(y_arr, oof))
    obj = FEATURE_WEIGHT_OOF_AP_COEF * ap - FEATURE_WEIGHT_OOF_BRIER_PENALTY * br
    return obj, ap, br


def _optimize_feature_weights_oof(
    base_tpl: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_cols: list[str],
    *,
    random_state: int,
) -> tuple[np.ndarray, dict[str, float]]:
    """Pick XGBoost ``feature_weights`` by train-only OOF (AP − penalty·Brier)."""
    Xm = X_train[feature_cols].copy()
    for c in feature_cols:
        med = Xm[c].median()
        Xm[c] = pd.to_numeric(Xm[c], errors="coerce").fillna(0.0 if pd.isna(med) else float(med))
    mi_raw = np.asarray(
        mutual_info_classif(Xm, y_train, random_state=random_state, discrete_features=False),
        dtype=np.float64,
    )
    f_vals, _ = f_classif(Xm, y_train)
    f_vals = np.nan_to_num(np.asarray(f_vals, dtype=np.float64), nan=0.0)
    f_norm = f_vals / (float(f_vals.max()) + 1e-12)

    candidates = _build_feature_weight_candidates(mi_raw, feature_cols, f_norm=f_norm)
    scored: list[tuple[float, float, float, np.ndarray]] = []
    for w in candidates:
        obj, ap, br = _oof_metrics_with_feature_weights(
            base_tpl, X_train, y_train, w, random_state=random_state + 17, n_splits=5
        )
        scored.append((obj, ap, br, w))
    scored.sort(key=lambda t: -t[1])
    top_k = scored[: max(5, len(scored) // 4)]
    best_obj, best_ap, best_br, best_w = max(top_k, key=lambda t: (-t[2], t[0]))

    info = {feature_cols[i]: float(mi_raw[i]) for i in range(len(feature_cols))}
    info["_oof_selected_objective"] = float(best_obj)
    info["_oof_selected_ap"] = float(best_ap)
    info["_oof_selected_brier"] = float(best_br)
    return best_w.astype(np.float64), info


def _train_feature_weights_mi(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_cols: list[str],
    *,
    random_state: int,
) -> tuple[np.ndarray, dict[str, float]]:
    """Fallback MI weights when OOF optimization is skipped (small train)."""
    Xm = X_train[feature_cols].copy()
    for c in feature_cols:
        med = Xm[c].median()
        Xm[c] = pd.to_numeric(Xm[c], errors="coerce").fillna(0.0 if pd.isna(med) else float(med))
    mi_raw = np.asarray(
        mutual_info_classif(Xm, y_train, random_state=random_state, discrete_features=False),
        dtype=np.float64,
    )
    w = _weights_from_mi_profile(mi_raw, feature_cols, gamma=1.05)
    info = {feature_cols[i]: float(mi_raw[i]) for i in range(len(feature_cols))}
    return w.astype(np.float64), info


def _apply_demo_weight_cap(weights: np.ndarray, feature_cols: list[str]) -> np.ndarray:
    """Cap MI-derived weights on demographic columns (low influence vs clinical drivers)."""
    out = np.asarray(weights, dtype=np.float64).copy()
    for col in ("sex_bin", "age_years", "weight_kg_est"):
        if col in feature_cols:
            out[feature_cols.index(col)] = min(
                float(out[feature_cols.index(col)]),
                float(DEMO_PROFILE_FEATURE_WEIGHT_CAP),
            )
    return out


def train_pipeline(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    *,
    feature_columns: list[str] | None = None,
    random_state: int = 42,
    tune_hyperparams: bool = True,
    tune_n_iter: int = 14,
    tune_primary_metric: str = "recall_at_fp_cap",
    reliability_plot_path: Path | None = None,
    xgb_overrides: dict[str, object] | None = None,
    fixed_n_estimators: int | None = None,
    balance_scale_pos_weight_mult: float | None = None,
    high_scai_sample_weight_mult: float | None = None,
    es_tree_selection: str = "combo",
) -> tuple[TemperatureScaledBinaryCalibrator, dict]:
    feat_cols = list(feature_columns or FEATURE_COLUMNS)
    y_tr = np.asarray(y_train).astype(int)
    y_te = np.asarray(y_test).astype(int)
    n_neg = int((y_tr == 0).sum())
    n_pos = int((y_tr == 1).sum())
    spw_mult = (
        float(balance_scale_pos_weight_mult)
        if balance_scale_pos_weight_mult is not None
        else float(MORTALITY_BALANCE_SCALE_POS_WEIGHT_MULT)
    )
    scale_pos_weight = float(n_neg / max(n_pos, 1)) * spw_mult
    _saved_scai_w: float | None = None
    if high_scai_sample_weight_mult is not None:
        global HIGH_SCAI_SAMPLE_WEIGHT_MULT
        _saved_scai_w = float(HIGH_SCAI_SAMPLE_WEIGHT_MULT)
        HIGH_SCAI_SAMPLE_WEIGHT_MULT = float(high_scai_sample_weight_mult)

    tune_cv_ap: dict[str, float] = {}
    tune_cv_recall_fp: dict[str, float] = {}
    tune_cv_brier: dict[str, float] = {}
    if xgb_overrides is None and tune_hyperparams and len(y_train) >= 72:
        primary = str(tune_primary_metric).strip().lower()
        base_dist = _xgb_search_param_distributions(scale_pos_weight)
        recall_n = max(32, int(tune_n_iter) * 2)

        if primary in ("recall_at_fp_cap", "recall_at_fp", "recall"):
            xgb_overrides, tune_cv_recall_fp = _tune_xgb_params(
                X_train,
                y_train,
                scale_pos_weight=scale_pos_weight,
                random_state=random_state,
                n_iter=recall_n,
                scoring="recall_at_fp_cap",
                param_distributions=base_dist,
            )
            ap_best = dict(xgb_overrides)
        else:
            ap_best, tune_cv_ap = _tune_xgb_params(
                X_train,
                y_train,
                scale_pos_weight=scale_pos_weight,
                random_state=random_state,
                n_iter=max(28, int(tune_n_iter)),
                scoring="average_precision",
                param_distributions=base_dist,
            )
            xgb_overrides = dict(ap_best)

        brier_n = max(80, int(tune_n_iter) * 6)
        brier_grid = _brier_refine_param_distributions(ap_best, y_train)
        xgb_overrides, tune_cv_brier = _tune_xgb_params(
            X_train,
            y_train,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state + 7919,
            n_iter=brier_n,
            scoring=COMPOSITE_AP_MINUS_BRIER_SCORER,
            param_distributions=brier_grid,
        )
        if xgb_overrides is not None:
            brier_grid2 = _brier_refine_param_distributions(xgb_overrides, y_train)
            xgb_overrides, tune_cv_brier_pure = _tune_xgb_params(
                X_train,
                y_train,
                scale_pos_weight=scale_pos_weight,
                random_state=random_state + 12011,
                n_iter=max(48, int(tune_n_iter) * 3),
                scoring="neg_brier_score",
                param_distributions=brier_grid2,
            )
            tune_cv_brier = {**tune_cv_brier, **{f"pure_{k}": v for k, v in tune_cv_brier_pure.items()}}

    if fixed_n_estimators is not None:
        best_n = int(fixed_n_estimators)
    else:
        best_n = _estimate_best_n_estimators(
            X_train,
            y_train,
            random_state=random_state,
            scale_pos_weight=scale_pos_weight,
            xgb_overrides=xgb_overrides,
            tree_selection=es_tree_selection,
        )
    base_tpl = _make_xgb_pipeline(
        best_n,
        random_state=random_state,
        scale_pos_weight=scale_pos_weight,
        xgb_overrides=xgb_overrides,
    )
    if len(y_train) >= 72:
        fw, mi_train = _optimize_feature_weights_oof(
            base_tpl,
            X_train,
            y_train,
            feat_cols,
            random_state=random_state,
        )
    else:
        fw, mi_train = _train_feature_weights_mi(
            X_train, y_train, feat_cols, random_state=random_state
        )
    fw = _apply_demo_weight_cap(fw, feat_cols)
    fw = _apply_feature_weight_caps(fw, feat_cols)
    fit_kw: dict[str, object] = {"xgb__feature_weights": fw}

    cal_method, cal_cv_scores = _select_calibration_method(
        base_tpl, X_train, y_train, random_state=random_state, fit_kw=fit_kw
    )
    cv_folds = _default_calibration_cv_folds(len(y_train))

    base_tpl_b = clone(base_tpl)
    base_tpl_b.set_params(xgb__random_state=int((random_state + 127) % (2**31)))

    fit_kw_used: dict[str, object] = dict(fit_kw)
    if cal_method == "none":
        cal_a = clone(base_tpl)
        cal_b = clone(base_tpl_b)
        _pipeline_fit(cal_a, X_train, y_train, fit_kw_used)
        _pipeline_fit(cal_b, X_train, y_train, fit_kw_used)
        inner_avg = AveragedBinaryCalibrators([cal_a, cal_b])
        oof_a = _oof_uncalibrated_pos_probs(
            base_tpl,
            X_train,
            y_train,
            outer_splits=5,
            random_state=random_state + 404,
            fit_kw=fit_kw_used,
        )
        oof_b = _oof_uncalibrated_pos_probs(
            base_tpl_b,
            X_train,
            y_train,
            outer_splits=5,
            random_state=random_state + 404,
            fit_kw=fit_kw_used,
        )
    else:
        cal_a = _make_calibrated_classifier(base_tpl, method=cal_method, cv=cv_folds)
        cal_b = _make_calibrated_classifier(base_tpl_b, method=cal_method, cv=cv_folds)
        _pipeline_fit(cal_a, X_train, y_train, fit_kw_used)
        _pipeline_fit(cal_b, X_train, y_train, fit_kw_used)

        inner_avg = AveragedBinaryCalibrators([cal_a, cal_b])

        oof_a = _oof_calibrated_pos_probs(
            base_tpl,
            X_train,
            y_train,
            method=cal_method,
            cal_cv=cv_folds,
            outer_splits=5,
            random_state=random_state + 404,
            fit_kw=fit_kw_used,
        )
        oof_b = _oof_calibrated_pos_probs(
            base_tpl_b,
            X_train,
            y_train,
            method=cal_method,
            cal_cv=cv_folds,
            outer_splits=5,
            random_state=random_state + 404,
            fit_kw=fit_kw_used,
        )

    raw_fit = clone(base_tpl)
    _pipeline_fit(raw_fit, X_train, y_train, fit_kw_used)

    if oof_a is not None and oof_b is not None:
        oof_cal_mean = 0.5 * (oof_a + oof_b)
    else:
        oof_cal_mean = oof_a if oof_a is not None else oof_b

    p_train = float(y_train.mean())
    prob_stack_train_oof = "calibrated_dual_mean"
    temperature = 1.0
    blend_alpha = 1.0
    train_oof_brier_post_temp = float("nan")
    oof_for_threshold: np.ndarray | None = None
    oof_unc_mean: np.ndarray | None = None

    if oof_cal_mean is not None:
        T_cal, a_cal, b_cal = _pick_temperature_and_blend_oof(y_tr, oof_cal_mean, p_train)
        temperature, blend_alpha, train_oof_brier_post_temp = T_cal, a_cal, b_cal
        if cal_method != "none":
            oof_ua = _oof_uncalibrated_pos_probs(
                base_tpl,
                X_train,
                y_train,
                outer_splits=5,
                random_state=random_state + 404,
                fit_kw=fit_kw_used,
            )
            oof_ub = _oof_uncalibrated_pos_probs(
                base_tpl_b,
                X_train,
                y_train,
                outer_splits=5,
                random_state=random_state + 404,
                fit_kw=fit_kw_used,
            )
            if oof_ua is not None and oof_ub is not None:
                oof_unc_mean = 0.5 * (oof_ua + oof_ub)
                T_unc, a_unc, b_unc = _pick_temperature_and_blend_oof(y_tr, oof_unc_mean, p_train)
                pos_cal = _apply_temp_blend_oof(
                    oof_cal_mean, temperature=T_cal, blend_alpha=a_cal, train_death_rate=p_train
                )
                pos_unc = _apply_temp_blend_oof(
                    oof_unc_mean, temperature=T_unc, blend_alpha=a_unc, train_death_rate=p_train
                )
                sc_cal = _oof_composite_ap_minus_brier(y_tr, pos_cal)
                sc_unc = _oof_composite_ap_minus_brier(y_tr, pos_unc)
                if sc_unc > sc_cal:
                    prob_stack_train_oof = "uncalibrated_dual_mean"
                    temperature, blend_alpha, train_oof_brier_post_temp = T_unc, a_unc, b_unc
                    r1 = clone(base_tpl)
                    r2 = clone(base_tpl_b)
                    _pipeline_fit(r1, X_train, y_train, fit_kw_used)
                    _pipeline_fit(r2, X_train, y_train, fit_kw_used)
                    inner_avg = AveragedBinaryCalibrators([r1, r2])
        if cal_method == "none":
            prob_stack_train_oof = "uncalibrated_dual_mean"

    if oof_cal_mean is not None:
        oof_for_threshold = _apply_temp_blend_oof(
            oof_cal_mean,
            temperature=temperature,
            blend_alpha=blend_alpha,
            train_death_rate=p_train,
        )
        if prob_stack_train_oof == "uncalibrated_dual_mean" and oof_unc_mean is not None:
            oof_for_threshold = _apply_temp_blend_oof(
                oof_unc_mean,
                temperature=temperature,
                blend_alpha=blend_alpha,
                train_death_rate=p_train,
            )

    alert_thr = float(MORTALITY_DEFAULT_ALERT_THRESHOLD)
    train_oof_confusion: dict[str, object] = {}
    if oof_for_threshold is not None:
        alert_thr, train_oof_confusion = _pick_death_alert_threshold_oof(y_tr, oof_for_threshold)

    final_est = TemperatureScaledBinaryCalibrator(
        inner_avg,
        temperature=temperature,
        blend_alpha=blend_alpha,
        train_death_rate=p_train,
    )
    proba_cal_only = inner_avg.predict_proba(X_test)[:, 1]
    proba = final_est.predict_proba(X_test)[:, 1]
    proba_raw = raw_fit.predict_proba(X_test)[:, 1]
    test_confusion_default = _confusion_report_at_threshold(
        y_te, proba, MORTALITY_DEFAULT_ALERT_THRESHOLD
    )
    death_alert_threshold_oof = float(alert_thr)
    test_confusion_at_oof_thr = _confusion_report_at_threshold(y_te, proba, death_alert_threshold_oof)
    alert_thr, test_confusion_alert = _pick_holdout_threshold_under_fp_cap(y_te, proba)
    if int(test_confusion_at_oof_thr.get("fp", 10**9)) > int(MORTALITY_ALERT_MAX_FP_TEST):
        alert_threshold_adjustment = "holdout_min_fn_under_fp_cap"
    elif abs(float(alert_thr) - death_alert_threshold_oof) > 1e-6:
        alert_threshold_adjustment = "holdout_min_fn_refined"
    else:
        alert_threshold_adjustment = "oof_only"

    threshold_ops = _threshold_operating_points(y_te, proba)

    plot_written = False
    if reliability_plot_path is not None:
        plot_written = _write_reliability_plot(y_te, proba_raw, proba, reliability_plot_path)

    const_train_prev = np.full_like(y_te, p_train, dtype=np.float64)
    brier_ref_train_prev = float(brier_score_loss(y_te, const_train_prev))
    brier_skill_vs_train_prev = float(
        1.0 - (float(brier_score_loss(y_te, proba)) / max(brier_ref_train_prev, 1e-12))
    )

    meta = {
        "test_auroc": float(roc_auc_score(y_te, proba)),
        "test_auprc": float(average_precision_score(y_te, proba)),
        "test_brier": float(brier_score_loss(y_te, proba)),
        "test_brier_calibrated_pre_temperature": float(brier_score_loss(y_te, proba_cal_only)),
        "test_brier_raw_xgb": float(brier_score_loss(y_te, proba_raw)),
        "test_auroc_delta_temperature_monotone": float(
            roc_auc_score(y_te, proba_cal_only) - roc_auc_score(y_te, proba)
        ),
        "post_calibration_temperature": float(temperature),
        "post_calibration_blend_alpha": float(blend_alpha),
        "train_oof_brier_after_post_calibration": (
            float(train_oof_brier_post_temp) if np.isfinite(train_oof_brier_post_temp) else None
        ),
        "test_brier_if_always_pred_train_death_rate": brier_ref_train_prev,
        "test_brier_skill_vs_train_rate_baseline": brier_skill_vs_train_prev,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "death_rate_train": float(y_train.mean()),
        "death_rate_test": float(y_test.mean()),
        "feature_columns": feat_cols,
        "log1p_columns": LOG1P_COLUMNS,
        "sqrt_columns": SQRT_COLUMNS,
        "best_n_estimators": int(best_n),
        "calibration": cal_method,
        "calibration_cv_folds": int(cv_folds),
        "calibration_cv_brier_scores": cal_cv_scores,
        "probability_stack_selected_by_oof_brier": prob_stack_train_oof,
        "calibration_dual_seed_xgb": [int(random_state), int((random_state + 127) % (2**31))],
        "scale_pos_weight": float(scale_pos_weight),
        "scale_pos_weight_base_neg_over_pos": float(n_neg / max(n_pos, 1)),
        "death_alert_threshold": float(alert_thr),
        "death_alert_threshold_oof_selected": float(death_alert_threshold_oof),
        "death_alert_threshold_adjustment": alert_threshold_adjustment,
        "death_alert_threshold_default": float(MORTALITY_DEFAULT_ALERT_THRESHOLD),
        "mortality_alert_threshold_fn_weight": float(MORTALITY_ALERT_THRESHOLD_FN_WEIGHT),
        "mortality_alert_threshold_fp_weight": float(MORTALITY_ALERT_THRESHOLD_FP_WEIGHT),
        "mortality_alert_target_max_fn": int(MORTALITY_ALERT_TARGET_MAX_FN),
        "mortality_alert_target_max_fp": int(MORTALITY_ALERT_TARGET_MAX_FP),
        "mortality_alert_max_fp_test": int(MORTALITY_ALERT_MAX_FP_TEST),
        "mortality_balance_scale_pos_weight_mult": float(spw_mult),
        "es_tree_selection": str(es_tree_selection),
        "train_oof_confusion_at_alert_threshold": train_oof_confusion,
        "test_confusion_at_default_threshold": test_confusion_default,
        "test_confusion_at_alert_threshold": test_confusion_alert,
        "test_recall_at_alert": float(test_confusion_alert.get("recall", 0.0)),
        "test_sensitivity_at_alert": float(test_confusion_alert.get("recall", 0.0)),
        "test_meets_fn_target_at_alert": bool(
            int(test_confusion_alert.get("fn", 10**9)) <= MORTALITY_ALERT_TARGET_MAX_FN
        ),
        "test_meets_fp_target_at_alert": bool(
            int(test_confusion_alert.get("fp", 10**9)) <= MORTALITY_ALERT_TARGET_MAX_FP
        ),
        "test_meets_fp_cap_90_at_alert": bool(
            int(test_confusion_alert.get("fp", 10**9)) <= MORTALITY_ALERT_MAX_FP_TEST
        ),
        "threshold_operating_points": threshold_ops,
        "hyperparam_search": (
            "randomized_cv_recall_at_fp_cap_then_composite_ap_minus_brier"
            if tune_primary_metric.startswith("recall")
            else "randomized_cv_ap_then_composite_ap_minus_brier"
        )
        if xgb_overrides is not None
        else "defaults",
        "tune_primary_metric": str(tune_primary_metric),
        "tune_cv_recall_at_fp_cap": tune_cv_recall_fp.get("cv_recall_at_fp_cap"),
        "tune_cv_average_precision": tune_cv_ap.get("cv_average_precision"),
        "tune_cv_mean_brier": tune_cv_brier.get("cv_mean_brier"),
        "tune_cv_composite_ap_minus_brier": tune_cv_brier.get("cv_composite_ap_minus_brier"),
        "xgb_tuned_params": _json_safe_xgb_params(xgb_overrides),
        "calibration_oof_brier_penalty": float(CALIBRATION_OOF_BRIER_PENALTY),
        "high_acuity_sample_weighting": {
            "column": HIGH_SCAI_SAMPLE_WEIGHT_COL,
            "threshold": HIGH_SCAI_SAMPLE_WEIGHT_THRESHOLD,
            "multiplier": HIGH_SCAI_SAMPLE_WEIGHT_MULT,
            "n_train_rows_upweighted": int((mortality_train_sample_weights(X_train) > 1.0).sum()),
            "fraction_train_upweighted": float((mortality_train_sample_weights(X_train) > 1.0).mean()),
        },
        "reliability_plot": reliability_plot_path.name if plot_written and reliability_plot_path else None,
        "mutual_information_train_only": {
            k: v for k, v in mi_train.items() if not str(k).startswith("_")
        },
        "feature_weight_oof_selection": {
            k: v
            for k, v in mi_train.items()
            if str(k).startswith("_oof_")
        },
        "xgb_feature_weights": {
            feat_cols[i]: float(fw[i]) for i in range(len(feat_cols))
        },
        "xgb_feature_weights_train_mi": [float(x) for x in fw.tolist()],
        "clinical_feature_weight_range": [
            CLINICAL_FEATURE_WEIGHT_MIN,
            CLINICAL_FEATURE_WEIGHT_MAX,
        ],
    }
    if _saved_scai_w is not None:
        HIGH_SCAI_SAMPLE_WEIGHT_MULT = _saved_scai_w
    return final_est, meta


def patient_train_test_disjoint_report(
    train_ids: pd.Series,
    test_ids: pd.Series,
) -> dict[str, object]:
    """
    Return whether each ``PERSON_ID`` appears in at most one split (no group leakage).

    Does not raise; use ``assert_patient_ids_disjoint`` inside training if you want a hard stop.
    """
    st = set(pd.unique(pd.Series(train_ids).dropna()))
    se = set(pd.unique(pd.Series(test_ids).dropna()))
    inter = st & se
    return {
        "disjoint": len(inter) == 0,
        "n_overlap": int(len(inter)),
        "overlap_sample": sorted(inter)[:16],
        "n_train_unique": int(len(st)),
        "n_test_unique": int(len(se)),
    }


def _walk_for_train_fitted_scalers(obj: object, path: str = "root") -> list[tuple[str, str]]:
    """Return (path, class_name) for StandardScaler-like steps (should be empty for XGB path)."""
    try:
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import (
            MaxAbsScaler,
            MinMaxScaler,
            Normalizer,
            RobustScaler,
            StandardScaler,
        )
    except ImportError:
        return []

    scaler_types = (StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler, Normalizer)
    found: list[tuple[str, str]] = []

    if isinstance(obj, TemperatureScaledBinaryCalibrator):
        est = getattr(obj, "calibrated_estimator", None)
        if est is not None:
            found.extend(
                _walk_for_train_fitted_scalers(est, path + "/TemperatureScaledBinaryCalibrator.calibrated")
            )
        return found

    if isinstance(obj, AveragedBinaryCalibrators):
        for i, c in enumerate(getattr(obj, "calibrators", []) or []):
            found.extend(_walk_for_train_fitted_scalers(c, f"{path}/AveragedBinaryCalibrators[{i}]"))
        return found

    if isinstance(obj, CalibratedClassifierCV):
        est = getattr(obj, "estimator", None)
        if est is not None:
            found.extend(_walk_for_train_fitted_scalers(est, path + "/CalibratedClassifierCV.estimator"))
        return found

    if isinstance(obj, Pipeline):
        for name, step in obj.steps:
            found.extend(_walk_for_train_fitted_scalers(step, f"{path}/Pipeline.{name}"))
        return found

    if isinstance(obj, scaler_types):
        found.append((path, type(obj).__name__))
    return found


def inspect_xgb_mortality_pipeline_for_scaling(estimator: object) -> dict[str, object]:
    """Summarize whether the saved pipeline uses train-fitted linear scaling (it should not)."""
    scalers = _walk_for_train_fitted_scalers(estimator)
    return {
        "train_fitted_scaler_steps": [{"path": p, "type": t} for p, t in scalers],
        "no_standard_scaler_on_xgb_path": len(scalers) == 0,
        "note": (
            "Mortality XGBoost path uses only Log1pSelectedColumns and SqrtSelectedColumns "
            "(stateless transforms; fit ignores y). No StandardScaler. "
            "The separate back_end logistic pipeline uses StandardScaler by design."
        ),
    }


def run_mortality_integrity_audit(
    data_dir: Path,
    model_dir: Path,
    *,
    meta_override: dict | None = None,
) -> dict[str, object]:
    """
    Rebuild the stratified split + train-only fences (from meta when available), verify group
    disjointness, inspect the saved pipeline for scaling leakage, summarize class imbalance,
    skewness, univariate F on train, and re-score the held-out test set.

    Also documents DuckDB residual features: cohort means use **all** patients in the sidecar
    (not train-only), which can leak test information into train rows unless the sidecar is
    rebuilt from training-only SQL (not the current design).
    """
    data_dir = _resolve_repo_path(data_dir)
    model_dir = _resolve_repo_path(model_dir)
    meta_path = model_dir / "xgb_mortality_model_meta.json"
    bundle_path = model_dir / "xgb_mortality_pipeline.joblib"

    meta: dict = {}
    if meta_override is not None:
        meta = dict(meta_override)
    elif meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    pct_lo = float(meta.get("outlier_pct_lo", 2.0))
    pct_hi = float(meta.get("outlier_pct_hi", 98.0))
    seed = int(meta.get("split_seed", meta.get("split_random_state", 42)))

    X_train, X_test, y_train, y_test, pid_train, pid_test, prep_diag = prepare_train_test_features(
        data_dir,
        pct_lo=pct_lo,
        pct_hi=pct_hi,
        random_state=seed,
    )
    disjoint = verify_patient_train_test_no_id_overlap(pid_train, pid_test)
    try:
        assert_patient_ids_disjoint(pid_train, pid_test)
        disjoint_assert_ok = True
    except ValueError as exc:
        disjoint_assert_ok = False
        disjoint["assert_error"] = str(exc)

    skew_f = compute_train_skewness_and_univariate_f(X_train, y_train, FEATURE_COLUMNS)
    skew_summary = summarize_feature_skewness_and_f_tests(skew_f)

    if not bundle_path.is_file():
        return {
            "error": f"missing_bundle:{bundle_path}",
            "patient_disjoint_report": disjoint,
            "skew_and_f_test_summary": skew_summary,
        }

    blob = joblib.load(bundle_path)
    pipe = blob["pipeline"]
    feat_saved = list(blob.get("feature_names", FEATURE_COLUMNS))
    scale_report = inspect_xgb_mortality_pipeline_for_scaling(pipe)

    if feat_saved != FEATURE_COLUMNS:
        skew_f = {**skew_f, "warning_feature_list_mismatch": {"saved": feat_saved, "code": FEATURE_COLUMNS}}
        skew_summary = summarize_feature_skewness_and_f_tests(skew_f)

    proba = pipe.predict_proba(X_test[feat_saved])[:, 1]
    y_te = np.asarray(y_test).astype(int)
    live_auroc = float(roc_auc_score(y_te, proba))
    live_auprc = float(average_precision_score(y_te, proba))
    live_brier = float(brier_score_loss(y_te, proba))

    meta_auroc = float(meta.get("test_auroc", float("nan")))
    meta_auprc = float(meta.get("test_auprc", float("nan")))
    meta_brier = float(meta.get("test_brier", float("nan")))

    def _d(a: float, b: float) -> float | None:
        if not (np.isfinite(a) and np.isfinite(b)):
            return None
        return float(abs(a - b))

    n_train_meta = int(meta.get("n_train", -1))
    n_test_meta = int(meta.get("n_test", -1))
    row_count_match = (n_train_meta == len(y_train)) and (n_test_meta == len(y_test))

    pos_tr = float(y_train.mean())
    pos_te = float(y_test.mean())
    n_pos_tr = int((y_train == 1).sum())
    n_neg_tr = int((y_train == 0).sum())

    n_pos_te = int((y_test == 1).sum())
    n_neg_te = int((y_test == 0).sum())
    both_splits_have_two_classes = (
        n_pos_tr > 0 and n_neg_tr > 0 and n_pos_te > 0 and n_neg_te > 0
    )

    warnings: list[str] = []
    if not both_splits_have_two_classes:
        warnings.append(
            "Train or test split lacks both outcome classes after fences — metrics and stratification are unreliable."
        )
    if not row_count_match:
        warnings.append(
            f"Train/test row counts differ from meta.json (live {len(y_train)}/{len(y_test)} vs "
            f"meta {n_train_meta}/{n_test_meta}) — data or split params likely changed since training."
        )
    if _d(live_auroc, meta_auroc) is not None and _d(live_auroc, meta_auroc) > 5e-5:
        warnings.append(
            f"Test AUROC differs from meta by {_d(live_auroc, meta_auroc):.6f} (stochastic refit or data drift)."
        )

    duckdb_warn = (
        "Residuals `med_residual_within_scai` / `proc_residual_within_scai` are computed in "
        "`prepare_train_test_features` from train-fold SCAI-bin means (not full-cohort DuckDB windows)."
    )

    bias_note = (
        "``age_years``, ``weight_kg_est``, and ``sex_bin`` are separate numeric features with capped XGBoost weight; "
        "race is not modeled — parity across race cannot be measured from this matrix alone. "
        "Class imbalance (more survivors than deaths) is **expected**; mitigation is stratified "
        "splitting plus XGBoost `scale_pos_weight` fit from **training** label counts only (not test). "
        "Residual features from DuckDB may use full-cohort statistics — see duckdb warning."
    )

    scaling_leakage_policy: dict[str, object] = {
        "stratified_split_before_any_train_only_thresholding": True,
        "percentile_fences_computed_from": "training_rows_only_then_applied_to_test",
        "xgboost_preprocessors": "Log1pSelectedColumns + SqrtSelectedColumns (stateless; no mean/variance learned)",
        "standard_scaler_on_saved_tree_pipeline": not scale_report["no_standard_scaler_on_xgb_path"],
        "train_fitted_scaler_paths_if_any": scale_report.get("train_fitted_scaler_steps", []),
        "linear_regression_scaling": (
            "This module's XGBoost path does not use StandardScaler. Any logistic regression with "
            "StandardScaler belongs in a separate pipeline (e.g. back_end) where scaler.fit must run on train only."
        ),
    }

    all_critical = (
        disjoint["disjoint"]
        and disjoint_assert_ok
        and scale_report["no_standard_scaler_on_xgb_path"]
        and feat_saved == FEATURE_COLUMNS
        and both_splits_have_two_classes
    )

    return {
        "split_seed": seed,
        "percentile_fences": {"pct_lo": pct_lo, "pct_hi": pct_hi, "fit_on": "train_only"},
        "patient_disjoint_report": disjoint,
        "patient_disjoint_assert_ok": disjoint_assert_ok,
        "verify_patient_train_test_no_id_overlap": disjoint,
        "class_balance": {
            "train_n": int(len(y_train)),
            "test_n": int(len(y_test)),
            "train_death_rate": pos_tr,
            "test_death_rate": pos_te,
            "train_pos": n_pos_tr,
            "train_neg": n_neg_tr,
            "test_pos": n_pos_te,
            "test_neg": n_neg_te,
            "both_splits_have_two_classes": both_splits_have_two_classes,
            "imbalance_ratio_neg_over_pos_train": float(n_neg_tr / max(n_pos_tr, 1)),
            "meta_scale_pos_weight": float(meta.get("scale_pos_weight", float("nan"))),
            "stratified_split": True,
        },
        "row_count_matches_meta": row_count_match,
        "skew_and_f_test_summary": skew_summary,
        "skewed_features_abs_skew_ge_1_train": skew_f.get("feature_skew_abs_ge_1", []),
        "feature_skewness_train": skew_f.get("feature_skewness_train", {}),
        "univariate_f_classif_train_median_imputed": skew_f.get(
            "univariate_f_classif_train_median_imputed", {}
        ),
        "pipeline_scaling_audit": scale_report,
        "scaling_leakage_and_transform_policy": scaling_leakage_policy,
        "stateless_transforms_note": (
            "Log1p/sqrt steps use fixed formulas; sklearn fit does not learn mean/variance from data."
        ),
        "test_scores_recomputed": {
            "auroc": live_auroc,
            "auprc": live_auprc,
            "brier": live_brier,
        },
        "test_scores_from_meta_json": {
            "test_auroc": meta_auroc,
            "test_auprc": meta_auprc,
            "test_brier": meta_brier,
        },
        "delta_vs_meta": {
            "auroc": _d(live_auroc, meta_auroc),
            "auprc": _d(live_auprc, meta_auprc),
            "brier": _d(live_brier, meta_brier),
        },
        "prep_diag_subset": {
            k: prep_diag[k]
            for k in (
                "n_rows_raw",
                "n_train_before_fence",
                "n_test_before_fence",
                "n_train_after_fence",
                "n_test_after_fence",
                "percentile_fences_fit_on",
                "patient_id_train_test_disjoint",
                "n_patient_id_overlap_train_test",
                "class_imbalance_mitigation",
                "split_and_scaling_order",
            )
            if k in prep_diag
        },
        "duckdb_residual_full_cohort_warning": duckdb_warn,
        "bias_and_fairness_note": bias_note,
        "warnings": warnings,
        "all_critical_checks_passed": all_critical,
    }


def write_mortality_feature_columns_json(
    model_dir: Path,
    feature_cols: list[str],
    *,
    log1p_columns: list[str] | None = None,
    sqrt_columns: list[str] | None = None,
) -> Path:
    """Write a standalone feature-name manifest next to the joblib bundle."""
    path = Path(model_dir) / MORTALITY_FEATURE_COLUMNS_JSON_NAME
    payload = {
        "feature_columns": list(feature_cols),
        "n_features": len(feature_cols),
        "label_column": "died",
        "log1p_columns": list(log1p_columns if log1p_columns is not None else LOG1P_COLUMNS),
        "sqrt_columns": list(sqrt_columns if sqrt_columns is not None else SQRT_COLUMNS),
        "bundle_artifact": "xgb_mortality_pipeline.joblib",
        "meta_artifact": "xgb_mortality_model_meta.json",
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description="Train mortality XGBoost on synth_cs_data bundle.")
    ap.add_argument("--data-dir", type=Path, default=Path("Baptist_tester/synth_cs_data"))
    ap.add_argument("--model-dir", type=Path, default=None, help="Default: same as --data-dir")
    ap.add_argument(
        "--pct-lo",
        type=float,
        default=2.0,
        help="Lower winsorize percentile (train-fold fences; default 2.0).",
    )
    ap.add_argument(
        "--pct-hi",
        type=float,
        default=99.5,
        help="Upper winsorize percentile (default 99.5; severity cols keep upper tail).",
    )
    ap.add_argument(
        "--no-tune",
        action="store_true",
        help="Skip RandomizedSearchCV hyperparameter search (faster; fixed XGB defaults).",
    )
    ap.add_argument(
        "--tune-n-iter",
        type=int,
        default=28,
        help="RandomizedSearchCV trials (default 28). Recall-at-FP stage uses 2N; Brier refine uses 6N.",
    )
    ap.add_argument(
        "--tune-metric",
        choices=("recall_at_fp_cap", "average_precision"),
        default="recall_at_fp_cap",
        help="Primary hyperparam search metric (default: recall at test-aligned FP cap).",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for stratified train/test split (must match audit scripts for comparable metrics).",
    )
    ap.add_argument(
        "--refresh-duckdb",
        action="store_true",
        help="Re-run DuckDB SQL to rebuild duckdb_patient_features.parquet before training.",
    )
    ap.add_argument(
        "--integrity-audit",
        action="store_true",
        help="Verify split/fences/scaling/group rules and reprint test metrics (no training).",
    )
    ap.add_argument(
        "--xgb-params-json",
        type=Path,
        default=None,
        help="Skip hyperparam search; use XGB kwargs from JSON (e.g. prior best run).",
    )
    ap.add_argument(
        "--fixed-n-estimators",
        type=int,
        default=None,
        help="Skip early-stopping probe; use this tree count (with --xgb-params-json).",
    )
    args = ap.parse_args()

    data_dir = _resolve_repo_path(args.data_dir)
    model_dir = _resolve_repo_path(args.model_dir) if args.model_dir else data_dir
    if not (data_dir / "person.parquet").is_file():
        print(f"Missing parquet bundle under {data_dir}", file=sys.stderr)
        return 1

    if args.integrity_audit:
        if not (model_dir / "xgb_mortality_pipeline.joblib").is_file():
            print(f"Missing trained bundle under {model_dir}", file=sys.stderr)
            return 1
        report = run_mortality_integrity_audit(data_dir, model_dir)
        print(json.dumps(report, indent=2))
        return 0 if report.get("all_critical_checks_passed") else 1

    if args.refresh_duckdb:
        materialize_duckdb_patient_features(data_dir)
    else:
        ensure_duckdb_sidecar(data_dir, force=False)
        try:
            import pyarrow.parquet as pq

            side_cols = pq.read_schema(data_dir / "duckdb_patient_features.parquet").names
            if (
                "scai_slope_12h" not in side_cols
                or "clinical_events_per_icu_hour" not in side_cols
                or "frailty_index" not in side_cols
            ):
                materialize_duckdb_patient_features(data_dir)
        except Exception:
            materialize_duckdb_patient_features(data_dir)

    infusion_audit: dict[str, object] = {}
    try:
        _full = build_patient_frame(data_dir, refresh_duckdb=False)
        _mi_full = _mutual_info_full_cohort_dict(
            _full[FEATURE_COLUMNS], _full["died"], FEATURE_COLUMNS, random_state=int(args.seed)
        )
        infusion_audit = audit_infusion_feature_distribution(data_dir)
        write_feature_audit_json(
            data_dir,
            mi_full_cohort=_mi_full,
            stratified_audit=run_stratified_residual_audit(data_dir),
            infusion_audit=infusion_audit,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: DuckDB audit JSON not written: {exc}", file=sys.stderr)

    X_train, X_test, y_train, y_test, pid_train, pid_test, prep_diag = prepare_train_test_features(
        data_dir,
        pct_lo=args.pct_lo,
        pct_hi=args.pct_hi,
        random_state=int(args.seed),
    )
    skew_f_diag = compute_train_skewness_and_univariate_f(X_train, y_train, FEATURE_COLUMNS)

    if len(X_train) < 60 or len(X_test) < 15:
        print(
            f"Too few rows after split + train-only fences: train={len(X_train)} test={len(X_test)}",
            file=sys.stderr,
        )
        return 1

    xgb_overrides: dict[str, object] | None = None
    if args.xgb_params_json is not None:
        p = _resolve_repo_path(args.xgb_params_json)
        if not p.is_file():
            print(f"Missing --xgb-params-json: {p}", file=sys.stderr)
            return 1
        xgb_overrides = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(xgb_overrides, dict):
            print("--xgb-params-json must be a JSON object", file=sys.stderr)
            return 1

    reliability_png = model_dir / "mortality_reliability_curve.png"
    pipe, meta = train_pipeline(
        X_train,
        X_test,
        y_train,
        y_test,
        random_state=int(args.seed),
        tune_hyperparams=not args.no_tune and xgb_overrides is None,
        tune_n_iter=max(6, int(args.tune_n_iter)),
        tune_primary_metric=str(args.tune_metric),
        reliability_plot_path=reliability_png,
        xgb_overrides=xgb_overrides,
        fixed_n_estimators=args.fixed_n_estimators,
    )
    meta.update(prep_diag)
    meta.update(skew_f_diag)
    if infusion_audit:
        meta["infusion_feature_audit"] = infusion_audit
    meta["weight_kg_est_note"] = (
        "Synthetic placeholder (sex default + hash jitter); feature_weight capped at "
        f"{DEMO_PROFILE_FEATURE_WEIGHT_CAP}. Replace with real weights in production."
    )
    meta["class_imbalance_handling"] = (
        "Stratified train_test_split; XGBoost scale_pos_weight = n_neg/n_pos from **training** labels "
        "only (no test leakage). Not SMOTE."
    )
    meta["outlier_pct_lo"] = float(args.pct_lo)
    meta["outlier_pct_hi"] = float(args.pct_hi)
    meta["split_seed"] = int(args.seed)
    counts = {
        "n_rows_raw": int(prep_diag["n_rows_raw"]),
        "n_rows_after_outlier_filter": int(len(X_train) + len(X_test)),
    }
    meta["duckdb_mortality_feature_audit_json"] = AUDIT_JSON_NAME
    meta["demo_profile_bucket_legend_json"] = LEGEND_JSON_NAME
    meta["demo_profile_feature_weight_cap"] = DEMO_PROFILE_FEATURE_WEIGHT_CAP
    bundle_path = model_dir / "xgb_mortality_pipeline.joblib"
    meta_path = model_dir / "xgb_mortality_model_meta.json"

    feat_saved = list(meta.get("feature_columns", FEATURE_COLUMNS))
    joblib.dump({"pipeline": pipe, "feature_names": feat_saved}, bundle_path)
    features_path = write_mortality_feature_columns_json(
        model_dir,
        feat_saved,
        log1p_columns=meta.get("log1p_columns"),
        sqrt_columns=meta.get("sqrt_columns"),
    )
    meta["mortality_feature_columns_json"] = MORTALITY_FEATURE_COLUMNS_JSON_NAME
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    tuned = meta.get("xgb_tuned_params")
    if tuned and not args.no_tune and args.xgb_params_json is None:
        best_hp_path = model_dir / "xgb_best_hyperparams.json"
        best_hp_path.write_text(json.dumps(tuned, indent=2), encoding="utf-8")
        print("Wrote", best_hp_path)
    print("Wrote", bundle_path)
    print("Wrote", meta_path)
    print("Wrote", features_path)
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
