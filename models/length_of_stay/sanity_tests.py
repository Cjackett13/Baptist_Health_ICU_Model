#!/usr/bin/env python3
"""Direction sweeps (p5/p95) and edge-case inference tests for LOS regression."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from models.length_of_stay.config import BUNDLE_NAME, REPO_ROOT
from models.length_of_stay.features import (
    MODEL_NUMERIC_FEATURES,
    load_training_feature_spec,
    raw_matrix_for_pipeline,
)
from models.length_of_stay.train import load_training_frame, split_by_manifest

# Five clinical drivers: higher value → longer predicted stay (synthetic + clinical prior).
FEATURES_TO_TEST: list[tuple[str, str]] = [
    ("clinical_event_n_12h", "more events → longer stay"),
    ("med_infusion_mean_12h", "more infusions → longer stay"),
    ("scai_prop_ge3_12h", "more time in high acuity → longer stay"),
    ("med_per_clinical_event_12h", "higher med burden → longer stay"),
    ("current_scai_12h", "higher severity → longer stay"),
]

EDGE_PRED_MIN_HOURS = 0.0
EDGE_PRED_MAX_HOURS = 10_000.0


def _resolve(p: Path) -> Path:
    return p.resolve() if p.is_absolute() else (REPO_ROOT / p).resolve()


def _pred_hours(pipe, df: pd.DataFrame, included: list[str]) -> float:
    pred_log = pipe.predict(raw_matrix_for_pipeline(df, included))
    return float(np.expm1(np.clip(pred_log[0], 0, None)))


def _median_patient_row(te: pd.DataFrame, included: list[str]) -> pd.DataFrame:
    med = te.median(numeric_only=True)
    row = te.iloc[[0]].copy()
    for c in included:
        if c in row.columns and c in med.index:
            row[c] = med[c]
    return row


def _run_direction_sweeps(
    pipe: object,
    te: pd.DataFrame,
    base: pd.DataFrame,
    included: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for feat, clinical_note in FEATURES_TO_TEST:
        if feat not in included or feat not in te.columns:
            rows.append(
                {
                    "feature": feat,
                    "status": "SKIP",
                    "clinical_expectation": clinical_note,
                    "error": "column missing from manifest or test frame",
                }
            )
            continue
        p05 = float(te[feat].quantile(0.05))
        p95 = float(te[feat].quantile(0.95))
        low_row = base.copy()
        high_row = base.copy()
        low_row[feat] = p05
        high_row[feat] = p95
        p_low = _pred_hours(pipe, low_row, included)
        p_high = _pred_hours(pipe, high_row, included)
        delta = round(p_high - p_low, 2)
        direction_ok = p_high >= p_low
        rows.append(
            {
                "feature": feat,
                "status": "PASS" if direction_ok else "FAIL",
                "clinical_expectation": clinical_note,
                "p05_value": round(p05, 4),
                "p95_value": round(p95, 4),
                "pred_at_p05_hours": round(p_low, 2),
                "pred_at_p95_hours": round(p_high, 2),
                "delta_hours_p95_minus_p05": delta,
                "direction_correct": direction_ok,
                "note": "Median patient template; only this feature moved to cohort p5/p95.",
            }
        )
    return rows


def _run_edge_cases(
    pipe: object,
    template: pd.DataFrame,
    included: list[str],
) -> dict[str, Any]:
    numeric_cols = [c for c in MODEL_NUMERIC_FEATURES if c in template.columns]
    results: dict[str, Any] = {"numeric_columns_tested": numeric_cols}

    def _check(name: str, row: pd.DataFrame) -> dict[str, Any]:
        try:
            pred = _pred_hours(pipe, row, included)
            ok = (
                not np.isnan(pred)
                and EDGE_PRED_MIN_HOURS < pred < EDGE_PRED_MAX_HOURS
            )
            return {
                "pred_hours": round(pred, 2),
                "crashed": False,
                "finite_output": not np.isnan(pred),
                "in_reasonable_range": EDGE_PRED_MIN_HOURS < pred < EDGE_PRED_MAX_HOURS,
                "pass": ok,
            }
        except Exception as exc:  # noqa: BLE001
            return {"crashed": True, "error": str(exc), "pass": False}

    extreme_high = template.copy()
    for col in numeric_cols:
        extreme_high[col] = 999.0
    results["extreme_high_all_numeric_999"] = _check("extreme_high", extreme_high)

    extreme_low = template.copy()
    for col in numeric_cols:
        extreme_low[col] = 0.0
    results["extreme_low_all_numeric_0"] = _check("extreme_low", extreme_low)

    median_case = template.copy()
    results["median_template"] = _check("median", median_case)

    preds = [round(_pred_hours(pipe, template, included), 4) for _ in range(20)]
    results["prediction_consistency_20x"] = {
        "unique_predictions": len(set(preds)),
        "deterministic": len(set(preds)) == 1,
    }

    results["all_edge_pass"] = all(
        results[k].get("pass") for k in ("extreme_high_all_numeric_999", "extreme_low_all_numeric_0", "median_template")
    )
    return results


def run_sanity_tests(data_dir: Path, model_dir: Path) -> dict[str, Any]:
    data_dir = _resolve(data_dir)
    model_dir = _resolve(model_dir)
    pipe = joblib.load(model_dir / BUNDLE_NAME)
    included, _ = load_training_feature_spec(model_dir, data_dir)
    df = load_training_frame(data_dir)
    _, _, te = split_by_manifest(df, data_dir)

    base = _median_patient_row(te, included)
    base_pred = _pred_hours(pipe, base, included)

    direction_rows = _run_direction_sweeps(pipe, te, base, included)
    tested = [r for r in direction_rows if r.get("status") != "SKIP"]
    n_correct = sum(1 for r in tested if r.get("direction_correct"))
    all_direction_pass = len(tested) == len(FEATURES_TO_TEST) and n_correct == len(tested)

    edge = _run_edge_cases(pipe, base, included)

    return {
        "baseline_median_patient_pred_hours": round(base_pred, 2),
        "direction_tests": direction_rows,
        "direction_test_pass": all_direction_pass,
        "direction_pass_count": f"{n_correct}/{len(FEATURES_TO_TEST)}",
        "direction_pass_rate": round(n_correct / max(len(tested), 1), 3),
        "edge_cases": edge,
        "edge_case_pass": bool(edge.get("all_edge_pass")),
        "summary": (
            "Direction: p5→p95 sweeps on median patient for 5 clinical features; "
            "PASS if all deltas are non-negative. "
            "Edge: extreme 0/999 on numeric manifest cols must return finite hours in (0, 10000)."
        ),
    }


def main() -> int:
    import argparse

    from models.length_of_stay.config import DATA_DIR_DEFAULT

    ap = argparse.ArgumentParser(description="LOS direction + edge-case sanity tests")
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR_DEFAULT)
    ap.add_argument("--model-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    model_dir = _resolve(args.model_dir) if args.model_dir else _resolve(args.data_dir) / "los_model"
    report = run_sanity_tests(args.data_dir, model_dir)
    out = args.out or (model_dir / "validation" / "los_sanity_tests.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report.get("direction_test_pass") and report.get("edge_case_pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
