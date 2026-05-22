"""Trained mortality XGBoost bundle inference (26 features)."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.fallback_mortality import MortalityScores, predict_fallback
from app.mortality_meta import death_alert_active, death_alert_threshold

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
_ARTIFACT_DIR = Path(__file__).resolve().parent / "artifact"
_MODELS_DIR = REPO_ROOT / "Frontend" / "lib" / "models"

_BUNDLE: dict[str, Any] | None = None
_FEATURE_NAMES: list[str] | None = None
_LOAD_ERROR: str | None = None


def _ensure_mortality_model_module() -> None:
    """Register custom calibrators for joblib unpickling."""
    if str(_MODELS_DIR) not in sys.path:
        sys.path.insert(0, str(_MODELS_DIR))
    import mortality_model  # noqa: F401

    sys.modules.setdefault("__main__", mortality_model)


def _bundle_path() -> Path:
    env = os.environ.get("MORTALITY_MODEL_PATH")
    if env:
        return Path(env)
    for name in ("mortality_pipeline.joblib", "xgb_mortality_pipeline.joblib"):
        p = _ARTIFACT_DIR / name
        if p.is_file():
            return p
    return _ARTIFACT_DIR / "xgb_mortality_pipeline.joblib"


def _feature_manifest_path() -> Path:
    env = os.environ.get("MORTALITY_FEATURE_COLUMNS_PATH")
    if env:
        return Path(env)
    return _ARTIFACT_DIR / "mortality_feature_columns.json"


def load_feature_columns() -> list[str]:
    global _FEATURE_NAMES
    if _FEATURE_NAMES is not None:
        return _FEATURE_NAMES
    path = _feature_manifest_path()
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        _FEATURE_NAMES = list(payload["feature_columns"])
        return _FEATURE_NAMES
    meta_path = _ARTIFACT_DIR / "xgb_mortality_model_meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        _FEATURE_NAMES = list(meta.get("feature_columns", []))
        return _FEATURE_NAMES or []
    raise FileNotFoundError("mortality_feature_columns.json not found in artifact/")


def _load_bundle() -> None:
    global _BUNDLE, _LOAD_ERROR
    if _BUNDLE is not None or _LOAD_ERROR is not None:
        return
    path = _bundle_path()
    if not path.is_file():
        _LOAD_ERROR = f"no bundle at {path}"
        logger.warning("%s — mortality fallback only", _LOAD_ERROR)
        return
    try:
        _ensure_mortality_model_module()
        loaded = joblib.load(path)
        if isinstance(loaded, dict) and "pipeline" in loaded:
            _BUNDLE = loaded
        else:
            _BUNDLE = {"pipeline": loaded, "feature_names": load_feature_columns()}
        logger.info("Loaded mortality bundle from %s", path)
    except Exception as exc:  # noqa: BLE001
        _LOAD_ERROR = str(exc)
        logger.exception("Mortality bundle load failed: %s", exc)


def predict_mortality_from_features(
    features: dict[str, Any],
) -> tuple[MortalityScores, str, str, float, bool]:
    """Returns (scores, model_version, source, alert_threshold, death_alert)."""
    _load_bundle()
    try:
        alert_thr = death_alert_threshold()
    except FileNotFoundError:
        alert_thr = 0.5

    if _BUNDLE is None:
        fb = predict_fallback(
            num_medications=int(features.get("num_medications", features.get("med_distinct", 8))),
            number_inpatient=int(features.get("number_inpatient", 1)),
            num_lab_procedures=int(features.get("num_lab_procedures", 35)),
            time_in_hospital=int(features.get("time_in_hospital", 3)),
            number_diagnoses=int(features.get("number_diagnoses", 4)),
            number_emergency=int(features.get("number_emergency", 1)),
            number_outpatient=int(features.get("number_outpatient", 0)),
            age_mid=float(features.get("age_mid", features.get("age_years", 65))),
        )
        return fb, "fallback-v1", "fallback", alert_thr, death_alert_active(fb.hospital_mortality)

    feat_cols = list(_BUNDLE.get("feature_names") or load_feature_columns())
    row = {c: features.get(c, np.nan) for c in feat_cols}
    X = pd.DataFrame([row])
    for col in X.columns:
        if X[col].dtype == object:
            X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.astype(float)
    pipe = _BUNDLE["pipeline"]
    try:
        proba = float(pipe.predict_proba(X)[0, 1])
    except Exception as exc:  # noqa: BLE001
        logger.exception("Trained mortality inference failed: %s", exc)
        fb = predict_fallback(
            num_medications=8,
            number_inpatient=1,
            num_lab_procedures=35,
            time_in_hospital=3,
            number_diagnoses=4,
            number_emergency=1,
            number_outpatient=0,
            age_mid=float(features.get("age_years", 65)),
        )
        return fb, "fallback-after-error", "fallback", alert_thr, death_alert_active(fb.hospital_mortality)

    proba = float(np.clip(proba, 0.01, 0.85))
    source = "trained"

    # Anchor displayed mortality to registry SCAI bands (A <1% … E ~70%).
    registry_scai = features.get("_registry_scai")
    if registry_scai is None and features.get("current_scai") is not None:
        registry_scai = features.get("current_scai")
    if registry_scai is not None and not (
        isinstance(registry_scai, float) and np.isnan(registry_scai)
    ):
        _ensure_mortality_model_module()
        import mortality_model as mm  # noqa: WPS433

        proba = mm.apply_scai_registry_mortality_blend(
            proba,
            current_scai=float(registry_scai),
            band_position_t=0.5,
        )
        source = "trained+scai-registry"

    icu = float(np.clip(proba * 1.12 + 0.018, 0.01, 0.92))
    expiry = float(np.clip(proba * 0.90 + 0.01, 0.01, 0.88))
    scores = MortalityScores(
        hospital_mortality=round(proba, 4),
        icu_mortality=round(icu, 4),
        in_hospital_expiry=round(expiry, 4),
    )
    alert = death_alert_active(scores.hospital_mortality)
    return scores, "xgb-mortality-bundle", source, alert_thr, alert


def predict_mortality_legacy_eight(
    *,
    num_medications: int,
    number_inpatient: int,
    num_lab_procedures: int,
    time_in_hospital: int,
    number_diagnoses: int,
    number_emergency: int,
    number_outpatient: int,
    age_mid: float,
) -> tuple[MortalityScores, str, str, float, bool]:
    """Backward-compatible 8-field API used by ``MortalityFeaturesRequest``."""
    return predict_mortality_from_features(
        {
            "num_medications": num_medications,
            "number_inpatient": number_inpatient,
            "num_lab_procedures": num_lab_procedures,
            "time_in_hospital": time_in_hospital,
            "number_diagnoses": number_diagnoses,
            "number_emergency": number_emergency,
            "number_outpatient": number_outpatient,
            "age_mid": age_mid,
            "age_years": age_mid,
        }
    )
