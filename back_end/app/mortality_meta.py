"""Load mortality model card JSON from the FastAPI artifact directory."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifact"
META_FILENAME = "xgb_mortality_model_meta.json"


def meta_path() -> Path:
    return Path(
        os.environ.get(
            "MORTALITY_META_PATH",
            str(ARTIFACT_DIR / META_FILENAME),
        )
    )


@lru_cache(maxsize=1)
def load_mortality_meta() -> dict[str, Any]:
    path = meta_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"{META_FILENAME} not found at {path}. "
            "Run: python back_end/scripts/sync_mortality_artifacts_from_data.py"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def death_alert_threshold() -> float:
    meta = load_mortality_meta()
    return float(meta.get("death_alert_threshold", 0.5))


def death_alert_active(probability: float) -> bool:
    return float(probability) >= death_alert_threshold()
