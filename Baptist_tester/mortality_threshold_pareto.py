#!/usr/bin/env python3
"""Holdout threshold Pareto: FN vs FP tradeoff and feasibility of joint targets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np

_REPO = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data/cleaned"))
    ap.add_argument("--model-dir", type=Path, default=Path("data"))
    ap.add_argument("--max-fp", type=int, default=89)
    ap.add_argument("--target-fn", type=int, default=30)
    args = ap.parse_args()

    models = _REPO / "Frontend" / "lib" / "models"
    sys.path.insert(0, str(models))
    import mortality_model as mm  # noqa: E402

    sys.modules["__main__"] = mm
    from mortality_model import prepare_train_test_features  # noqa: E402

    data_dir = (_REPO / args.data_dir).resolve()
    model_dir = (_REPO / args.model_dir).resolve()
    _, X_test, _, y_test, _, _, _ = prepare_train_test_features(data_dir, random_state=42)
    blob = joblib.load(model_dir / "xgb_mortality_pipeline.joblib")
    proba = blob["pipeline"].predict_proba(X_test[blob["feature_names"]])[:, 1]
    y = y_test.to_numpy().astype(int)

    rows = []
    best_under_fp: dict | None = None
    for thr in np.round(np.arange(0.05, 0.96, 0.005), 3):
        pred = (proba >= thr).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        rows.append(
            {
                "threshold": float(thr),
                "fp": fp,
                "fn": fn,
                "tp": tp,
                "recall": round(tp / max(tp + fn, 1), 4),
                "ppv": round(tp / max(tp + fp, 1), 4),
            }
        )
        if fp <= args.max_fp:
            if best_under_fp is None or fn < best_under_fp["fn"]:
                best_under_fp = rows[-1]

    joint_feasible = [r for r in rows if r["fp"] <= args.max_fp and r["fn"] <= args.target_fn]
    report = {
        "n_test": int(len(y)),
        "death_rate": round(float(y.mean()), 4),
        "max_fp_cap": args.max_fp,
        "target_fn": args.target_fn,
        "joint_target_feasible": len(joint_feasible) > 0,
        "best_under_fp_cap": best_under_fp,
        "fn_at_target_with_min_fp": next(
            (r for r in rows if r["fn"] <= args.target_fn),
            None,
        ),
        "note": (
            "No single threshold achieves both caps on this holdout; lower FN requires "
            "a lower threshold and more FP, or model retraining for better separation."
            if not joint_feasible
            else "Joint target achievable."
        ),
        "threshold_sweep_sample": rows[::4],
    }
    out = model_dir / "mortality_threshold_pareto.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
