"""
Model loader — called once at startup via FastAPI lifespan.

Expected model_files/ layout:
  model_files/
  ├── scai/
  │   └── xgboost_unified_final.pkl        (bundle: xgboost_model, platt_scaler, feature_cols)
  ├── vasopressor/
  │   ├── binary_alert_model.pkl           (IsotonicCalibratedModel)
  │   ├── ordinal_count_model.pkl          (XGBClassifier)
  │   ├── high_severity_classifier.pkl     (PlattXGB)
  │   ├── knn_imputer.pkl                  (imputer state dict)
  │   ├── iqr_bounds.json
  │   ├── log_transform_features.json
  │   └── feature_manifest.json            (contains feature_columns list)
  ├── johnathan/
  │   ├── mcs_model.pkl
  │   └── va_ecmo_model.pkl
  └── lilly/
      ├── mortality_pipeline.pkl
      ├── los_pipeline.pkl
      ├── mortality_feature_columns.json
      ├── xgb_mortality_model_meta.json
      └── xgb_los_model_meta.json

GCS: set MODEL_DIR env var to a local staging path and call download_from_gcs() first.
"""

import json
import pickle
import logging
from pathlib import Path

import joblib

from app.config import settings

# Wrapper classes must be importable before any vasopressor pkl is loaded.
# The pkls were saved with these classes defined in __main__, so we inject them
# into sys.modules['__main__'] so pickle.load() can resolve them.
import sys
from app.models.wrapper_classes import IsotonicCalibratedModel, PlattXGB  # noqa: F401
sys.modules["__main__"].IsotonicCalibratedModel = IsotonicCalibratedModel
sys.modules["__main__"].PlattXGB = PlattXGB

# Johnathan's bundle classes must be importable before his pkls are deserialized.
import mcs_escalation_inference  # noqa: F401
import ecmo_escalation_inference  # noqa: F401

# Lilly's transformer classes must be importable before her pkls are deserialized.
import mortality_transforms  # noqa: F401
import models.length_of_stay.transforms  # noqa: F401

log = logging.getLogger(__name__)

_registry: dict = {}

_MODEL_PATHS = {
    # Christie
    "scai":                 ("scai",       "xgboost_unified_final.pkl"),
    "vasopressor_binary":   ("vasopressor","binary_alert_model.pkl"),
    "vasopressor_count":    ("vasopressor","ordinal_count_model.pkl"),
    "vasopressor_severity": ("vasopressor","high_severity_classifier.pkl"),
    # Johnathan
    "mcs":                  ("johnathan",  "mcs_seed42.bundle.pkl"),
    "va_ecmo":              ("johnathan",  "ecmo_seed42.bundle.pkl"),
    # Lilly
    "mortality":            ("lilly",      "mortality_pipeline.pkl"),
    "los":                  ("lilly",      "xgb_los_pipeline.joblib"),
}

# Models that must be present for the API to be considered healthy.
_REQUIRED = {"scai", "vasopressor_binary", "vasopressor_count", "vasopressor_severity", "vasopressor_prep"}

# Non-model artifacts for vasopressor preprocessing (loaded into a single bundle)
_VASO_PREP_FILES = {
    "knn_imputer":        ("vasopressor", "knn_imputer.pkl",               "pkl"),
    "iqr_bounds":         ("vasopressor", "iqr_bounds.json",               "json"),
    "log_features":       ("vasopressor", "log_transform_features.json",   "json"),
    "feature_manifest":   ("vasopressor", "feature_manifest.json",         "json"),
}

# Additional artifact paths to download from GCS (JSON / pkl files, not model PKLs)
_ARTIFACT_PATHS = {
    (subfolder, filename)
    for _, (subfolder, filename, _) in _VASO_PREP_FILES.items()
}


def load_all_models() -> None:
    model_dir = Path(settings.model_dir)

    if settings.gcs_bucket:
        _download_from_gcs(model_dir)

    for name, (subfolder, filename) in _MODEL_PATHS.items():
        path = model_dir / subfolder / filename
        loader = _load_joblib if filename.endswith(".joblib") else _load_pkl
        _registry[name] = loader(path, required=(name in _REQUIRED))

    _registry["vasopressor_prep"] = _load_vasopressor_prep(model_dir)

    loaded  = [k for k, v in _registry.items() if v is not None]
    missing = [k for k, v in _registry.items() if v is None]
    log.info("Models loaded: %s", loaded)
    if missing:
        log.warning("Models not yet available (stubs): %s", missing)


def get(name: str):
    return _registry.get(name)


def is_healthy() -> bool:
    return all(_registry.get(k) is not None for k in _REQUIRED)


def status() -> dict[str, bool]:
    return {k: v is not None for k, v in _registry.items()}


def _load_pkl(path: Path, required: bool = False):
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required model file not found: {path}")
        return None
    with open(path, "rb") as f:
        model = pickle.load(f)
    log.info("Loaded %s", path)
    return model


def _load_joblib(path: Path, required: bool = False):
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required model file not found: {path}")
        return None
    model = joblib.load(path)
    log.info("Loaded %s", path)
    return model


def _load_vasopressor_prep(model_dir: Path) -> dict | None:
    """Load the vasopressor preprocessing artifact bundle."""
    paths = {
        key: model_dir / subfolder / filename
        for key, (subfolder, filename, _) in _VASO_PREP_FILES.items()
    }
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        log.warning("Vasopressor prep artifacts not found (feature engineering disabled): %s", missing)
        return None

    bundle: dict = {}
    for key, (subfolder, filename, fmt) in _VASO_PREP_FILES.items():
        path = model_dir / subfolder / filename
        if fmt == "pkl":
            with open(path, "rb") as f:
                bundle[key] = pickle.load(f)
        else:
            with open(path) as f:
                bundle[key] = json.load(f)

    bundle["feature_cols"] = bundle["feature_manifest"]["feature_columns"]
    log.info("Loaded vasopressor preprocessing artifacts")
    return bundle


def _download_from_gcs(local_dir: Path) -> None:
    """Download all model files and artifacts from GCS bucket to local_dir."""
    import gcsfs  # type: ignore
    fs = gcsfs.GCSFileSystem()
    local_dir.mkdir(parents=True, exist_ok=True)

    all_files = list(_MODEL_PATHS.values()) + [
        (subfolder, filename)
        for _, (subfolder, filename, _) in _VASO_PREP_FILES.items()
    ]
    for subfolder, filename in all_files:
        remote = f"{settings.gcs_bucket}/{subfolder}/{filename}"
        local  = local_dir / subfolder / filename
        local.parent.mkdir(parents=True, exist_ok=True)
        if not local.exists():
            log.info("Downloading %s → %s", remote, local)
            fs.get(remote, str(local))
