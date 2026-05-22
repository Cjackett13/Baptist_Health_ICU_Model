#!/usr/bin/env python3
"""Sync mortality + LOS model artifacts into back_end/app/artifact/."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

BACK_END = Path(__file__).resolve().parents[1]
REPO = BACK_END.parent
DEFAULT_DATA = REPO / "data"
ARTIFACT = BACK_END / "app" / "artifact"

MORTALITY_FILES = (
    "xgb_mortality_model_meta.json",
    "mortality_feature_columns.json",
    "xgb_mortality_pipeline.joblib",
)

LOS_FILES = (
    "xgb_los_model_meta.json",
    "xgb_los_pipeline.joblib",
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--artifact-dir", type=Path, default=ARTIFACT)
    args = ap.parse_args()

    source = args.data_dir.resolve()
    dest = args.artifact_dir.resolve()
    dest.mkdir(parents=True, exist_ok=True)

    missing: list[str] = []
    for name in MORTALITY_FILES:
        src = source / name
        if not src.is_file():
            missing.append(name)
            continue
        shutil.copy2(src, dest / name)
        print(f"Copied {src} -> {dest / name}")

    xgb = source / "xgb_mortality_pipeline.joblib"
    if xgb.is_file():
        shutil.copy2(xgb, dest / "mortality_pipeline.joblib")

    los_dir = source / "los_model"
    for name in LOS_FILES:
        src = los_dir / name
        if not src.is_file():
            missing.append(f"los_model/{name}")
            continue
        shutil.copy2(src, dest / name)
        print(f"Copied {src} -> {dest / name}")

    if missing:
        print("Missing:", ", ".join(missing))
        return 1
    print("Artifact sync complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
