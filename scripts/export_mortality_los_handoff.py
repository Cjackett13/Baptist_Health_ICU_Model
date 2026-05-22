#!/usr/bin/env python3
"""Export mortality + LOS trained artifacts for Mortality_LOS_Models GitHub handoff."""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
from pathlib import Path

import joblib

REPO = Path(__file__).resolve().parents[1]
DEFAULT_DATA = REPO / "data"
DEFAULT_OUT = REPO / "Mortality_LOS_Models"


def _copy(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    print(f"  copied {src.name} -> {dest.relative_to(dest.parent.parent)}")


def _pickle_bundle(obj: object, path: Path) -> None:
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  wrote pickle {path.name} ({path.stat().st_size // 1024} KiB)")


def _export_los_xgb_json(pipeline_path: Path, out_json: Path) -> bool:
    pipeline = joblib.load(pipeline_path)
    xgb = pipeline.named_steps.get("xgb")
    if xgb is None or not hasattr(xgb, "save_model"):
        return False
    xgb.save_model(str(out_json))
    print(f"  wrote native XGBoost JSON {out_json.name}")
    return True


def _export_mortality_xgb_json(bundle_path: Path, out_json: Path) -> bool:
    """Best-effort reference booster; mortality production path uses joblib/pickle only."""
    bundle = joblib.load(bundle_path)
    cal = bundle["pipeline"]
    cals = getattr(getattr(cal, "calibrated_estimator", None), "calibrators", None)
    if not cals:
        print("  skipped native mortality JSON (ensemble has no exportable single booster)")
        return False
    for calibrator in cals:
        for attr in ("calibrated_classifiers_", "estimators_"):
            members = getattr(calibrator, attr, None)
            if not members:
                continue
            for member in members:
                est = getattr(member, "estimator", member)
                if est is None or not hasattr(est, "named_steps"):
                    continue
                xgb = est.named_steps.get("xgb")
                if xgb is None:
                    continue
                try:
                    booster = xgb.get_booster()
                    booster.save_model(str(out_json))
                    print(
                        f"  wrote reference XGBoost JSON {out_json.name} "
                        "(trees only; use joblib/pickle for calibrated inference)"
                    )
                    return True
                except Exception as exc:
                    print(f"  skipped native mortality JSON ({exc})")
                    return False
    print("  skipped native mortality JSON (calibrated ensemble)")
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    data = args.data_dir.resolve()
    out = args.out_dir.resolve()
    mort_dir = out / "mortality"
    los_dir = out / "los"

    mort_joblib = data / "xgb_mortality_pipeline.joblib"
    los_joblib = data / "los_model" / "xgb_los_pipeline.joblib"
    missing = [p for p in (mort_joblib, los_joblib) if not p.is_file()]
    if missing:
        print("Missing trained artifacts:", ", ".join(str(p) for p in missing))
        print("Train/sync first: python3 back_end/scripts/sync_all_artifacts.py")
        return 1

    print(f"Exporting to {out}")
    out.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(REPO / "Frontend" / "lib" / "models"))
    import mortality_model  # noqa: F401

    sys.path.insert(0, str(REPO))
    from models.length_of_stay.transforms import LosTrainPreprocessor  # noqa: F401

    print("\n[mortality]")
    _copy(mort_joblib, mort_dir / "xgb_mortality_pipeline.joblib")
    _copy(data / "mortality_feature_columns.json", mort_dir / "mortality_feature_columns.json")
    _copy(data / "xgb_mortality_model_meta.json", mort_dir / "xgb_mortality_model_meta.json")
    mort_bundle = joblib.load(mort_joblib)
    _pickle_bundle(mort_bundle, mort_dir / "mortality_pipeline.pkl")
    _export_mortality_xgb_json(mort_joblib, mort_dir / "xgb_mortality_booster_reference.json")

    print("\n[los]")
    _copy(los_joblib, los_dir / "xgb_los_pipeline.joblib")
    _copy(data / "los_model" / "xgb_los_model_meta.json", los_dir / "xgb_los_model_meta.json")
    los_pipeline = joblib.load(los_joblib)
    _pickle_bundle(los_pipeline, los_dir / "los_pipeline.pkl")
    _export_los_xgb_json(los_joblib, los_dir / "xgb_los_booster.json")

    meta = {
        "source_repo": "Baptist_Health_ICU_Model",
        "data_dir": str(data),
        "mortality": {
            "task": "binary_classification",
            "label": "died",
            "n_features": 26,
            "primary_artifact": "xgb_mortality_pipeline.joblib",
            "pickle_artifact": "mortality_pipeline.pkl",
            "alert_threshold": 0.18,
        },
        "los": {
            "task": "regression",
            "label_hours": "los_hours_total",
            "n_raw_features": 14,
            "primary_artifact": "xgb_los_pipeline.joblib",
            "pickle_artifact": "los_pipeline.pkl",
            "native_booster": "xgb_los_booster.json",
        },
    }
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {manifest_path.relative_to(REPO)}")
    print("Handoff export complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
