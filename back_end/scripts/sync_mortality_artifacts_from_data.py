#!/usr/bin/env python3
"""
Copy canonical mortality artifacts from ``data/`` into ``back_end/app/artifact/``.

Run after retraining or holdout threshold refresh:

  python back_end/scripts/sync_mortality_artifacts_from_data.py
  python back_end/scripts/sync_mortality_artifacts_from_data.py --source-dir /path/to/data
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

BACK_END = Path(__file__).resolve().parents[1]
REPO = BACK_END.parent
DEFAULT_SOURCE = REPO / "data"
ARTIFACT_DIR = BACK_END / "app" / "artifact"

FILES = (
    "xgb_mortality_model_meta.json",
    "mortality_feature_columns.json",
    "xgb_mortality_pipeline.joblib",
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
    args = ap.parse_args()

    source = args.source_dir.resolve()
    dest = args.artifact_dir.resolve()
    dest.mkdir(parents=True, exist_ok=True)

    missing: list[str] = []
    for name in FILES:
        src = source / name
        if not src.is_file():
            missing.append(name)
            continue
        shutil.copy2(src, dest / name)
        print(f"Copied {src} -> {dest / name}")

    # FastAPI inference default path (see app/inference.py)
    xgb_joblib = source / "xgb_mortality_pipeline.joblib"
    if xgb_joblib.is_file():
        shutil.copy2(xgb_joblib, dest / "mortality_pipeline.joblib")
        print(f"Copied {xgb_joblib} -> {dest / 'mortality_pipeline.joblib'}")

    meta_path = dest / "xgb_mortality_model_meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        thr = meta.get("death_alert_threshold")
        print(f"death_alert_threshold in artifact meta: {thr}")

    if missing:
        print("Missing (skipped):", ", ".join(missing), file=__import__("sys").stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
