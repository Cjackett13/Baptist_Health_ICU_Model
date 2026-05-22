"""LOS regression inference (14 raw features → hours)."""

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

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
_ARTIFACT_DIR = Path(__file__).resolve().parent / "artifact"
_DEFAULT_LOS_DIR = REPO_ROOT / "data" / "los_model"

_PIPELINE: Any = None
_META: dict[str, Any] | None = None
_RAW_COLUMNS: list[str] | None = None
_LOAD_ERROR: str | None = None


def _los_model_dir() -> Path:
    return Path(os.environ.get("LOS_MODEL_DIR", str(_DEFAULT_LOS_DIR))).resolve()


def _bundle_path() -> Path:
    env = os.environ.get("LOS_MODEL_PATH")
    if env:
        return Path(env)
    for base in (_ARTIFACT_DIR, _los_model_dir()):
        p = base / "xgb_los_pipeline.joblib"
        if p.is_file():
            return p
    return _los_model_dir() / "xgb_los_pipeline.joblib"


def _meta_path() -> Path:
    env = os.environ.get("LOS_META_PATH")
    if env:
        return Path(env)
    for base in (_ARTIFACT_DIR, _los_model_dir()):
        p = base / "xgb_los_model_meta.json"
        if p.is_file():
            return p
    return _los_model_dir() / "xgb_los_model_meta.json"


def load_los_meta() -> dict[str, Any]:
    global _META
    if _META is not None:
        return _META
    path = _meta_path()
    if not path.is_file():
        raise FileNotFoundError(f"LOS meta not found at {path}")
    _META = json.loads(path.read_text(encoding="utf-8"))
    return _META


def raw_feature_columns() -> list[str]:
    global _RAW_COLUMNS
    if _RAW_COLUMNS is not None:
        return _RAW_COLUMNS
    meta = load_los_meta()
    _RAW_COLUMNS = list(meta.get("feature_columns_raw_included", []))
    if not _RAW_COLUMNS:
        raise ValueError("feature_columns_raw_included missing from LOS meta")
    return _RAW_COLUMNS


def _load_pipeline() -> None:
    global _PIPELINE, _LOAD_ERROR
    if _PIPELINE is not None or _LOAD_ERROR is not None:
        return
    path = _bundle_path()
    if not path.is_file():
        _LOAD_ERROR = f"no LOS pipeline at {path}"
        logger.warning("%s", _LOAD_ERROR)
        return
    try:
        repo = str(REPO_ROOT)
        if repo not in sys.path:
            sys.path.insert(0, repo)
        import models.length_of_stay.transforms  # noqa: F401 — LosTrainPreprocessor for unpickle

        _PIPELINE = joblib.load(path)
        load_los_meta()
        logger.info("Loaded LOS pipeline from %s", path)
    except Exception as exc:  # noqa: BLE001
        _LOAD_ERROR = str(exc)
        logger.exception("LOS pipeline load failed: %s", exc)


def predict_los_from_features(features: dict[str, Any]) -> tuple[float, float, str, str]:
    """
    Returns (hospital_los_hours, icu_los_hours, model_version, source).
    ICU LOS uses a simple fraction of hospital LOS when not modeled separately.
    """
    _load_pipeline()
    if _PIPELINE is None:
        days = float(features.get("days_admitted", features.get("time_in_hospital", 3)))
        h = max(0.5, days * 24.0 * 0.4)
        return round(h, 2), round(h * 0.85, 2), "heuristic-days", "fallback"

    cols = raw_feature_columns()
    row = {c: features.get(c, np.nan) for c in cols}
    X = pd.DataFrame([row])
    pred_log = float(_PIPELINE.predict(X)[0])
    hospital_h = float(np.expm1(np.clip(pred_log, 0, None)))
    hospital_h = max(0.5, hospital_h)
    icu_h = max(0.5, hospital_h * 0.85)
    return (
        round(hospital_h, 2),
        round(icu_h, 2),
        "xgb-los-regressor",
        "trained",
    )
