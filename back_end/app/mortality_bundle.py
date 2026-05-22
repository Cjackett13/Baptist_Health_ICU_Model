"""Load exported SHAP JSON + predictions PKL (same artifacts `export_shap_predictions_bundle.py` writes)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifact"


def shap_bundle_path() -> Path:
    return Path(
        os.environ.get(
            "MORTALITY_SHAP_JSON_PATH",
            str(ARTIFACT_DIR / "mortality_shap_bundle.json"),
        )
    )


def predictions_pkl_path() -> Path:
    return Path(
        os.environ.get(
            "MORTALITY_PREDICTIONS_PKL_PATH",
            str(ARTIFACT_DIR / "mortality_predictions_all.pkl"),
        )
    )


def load_shap_bundle_dict() -> dict[str, Any]:
    path = shap_bundle_path()
    if not path.is_file():
        raise FileNotFoundError(path.name)
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_predictions_payload() -> dict[str, Any]:
    path = predictions_pkl_path()
    if not path.is_file():
        raise FileNotFoundError(path.name)
    data = joblib.load(path)
    if not isinstance(data, dict) or "predictions_df" not in data:
        raise ValueError("Invalid predictions PKL (expected dict with predictions_df)")
    return data


def predictions_page_as_json(
    page: int,
    page_size: int,
) -> dict[str, Any]:
    payload = load_predictions_payload()
    df: pd.DataFrame = payload["predictions_df"]
    total = len(df)
    page_size = max(1, min(page_size, 500))
    page = max(0, page)
    start = page * page_size
    end = min(start + page_size, total)
    subset = df.iloc[start:end]
    rows = json.loads(subset.to_json(orient="records", date_format="iso"))
    return {
        "model_version": payload.get("model_version", "unknown"),
        "total": int(total),
        "page": int(page),
        "page_size": int(page_size),
        "columns": list(df.columns),
        "rows": rows,
    }
