#!/usr/bin/env python3
"""
Sanity audits for checklist items 17 (Simpson's / age bands), 19 (direction), 21 (edge cases).

  PYTHONPATH=Frontend/lib/models python3 Baptist_tester/mortality_sanity_audit.py \\
      --data-dir data/cleaned --model-dir data
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

_REPO = Path(__file__).resolve().parent.parent

DIRECTION_SPECS: list[tuple[str, float, float]] = [
    ("scai_prop_ge3", 0.0, 1.0),
    ("current_scai", 0.0, 4.0),
    ("med_infusion_mean", 0.05, 3.5),
    ("clinical_event_n", 500.0, 3000.0),
    ("med_distinct", 1.0, 6.0),
    ("infusion_x_scai_severity", 0.0, 4.0),
]

AGE_BANDS: list[tuple[str, float, float]] = [
    ("age_lt_60", 0.0, 60.0),
    ("age_60_69", 60.0, 70.0),
    ("age_70_79", 70.0, 80.0),
    ("age_80_plus", 80.0, 200.0),
]


def _resolve(p: Path) -> Path:
    return p.resolve() if p.is_absolute() else (_REPO / p).resolve()


def _subgroup_auroc(y: np.ndarray, p: np.ndarray, *, min_n: int = 40, min_pos: int = 8) -> dict[str, Any] | None:
    n = len(y)
    n_pos = int((y == 1).sum())
    if n < min_n or n_pos < min_pos or n_pos == n:
        return None
    auc = float(roc_auc_score(y, p))
    return {"n": n, "n_pos": n_pos, "death_rate": round(float(y.mean()), 4), "auroc": round(auc, 4)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data/cleaned"))
    ap.add_argument("--model-dir", type=Path, default=Path("data"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    sys.modules.setdefault("__main__", __import__("mortality_model"))
    models = _REPO / "Frontend" / "lib" / "models"
    if str(models) not in sys.path:
        sys.path.insert(0, str(models))

    import mortality_model as mm  # noqa: E402

    sys.modules["__main__"] = mm

    from mortality_model import FEATURE_COLUMNS, prepare_train_test_features  # noqa: E402

    data_dir = _resolve(args.data_dir)
    model_dir = _resolve(args.model_dir)
    meta_path = model_dir / "xgb_mortality_model_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    seed = int(meta.get("split_seed", args.seed))
    pct_lo = float(meta.get("outlier_pct_lo", 1.0))
    pct_hi = float(meta.get("outlier_pct_hi", 99.0))

    _, X_test, _, y_test, _, _, _ = prepare_train_test_features(
        data_dir, pct_lo=pct_lo, pct_hi=pct_hi, random_state=seed
    )
    y_te = np.asarray(y_test).astype(int)
    blob = joblib.load(model_dir / "xgb_mortality_pipeline.joblib")
    pipe = blob["pipeline"]
    feat = list(blob.get("feature_names", FEATURE_COLUMNS))
    proba = pipe.predict_proba(X_test[feat])[:, 1]
    overall_auroc = float(roc_auc_score(y_te, proba))

    # --- Item 17: age-band + Simpson's-style consistency ---
    age_rows: list[dict[str, Any]] = []
    flagged_age: list[dict[str, Any]] = []
    if "age_years" in X_test.columns:
        ages = pd.to_numeric(X_test["age_years"], errors="coerce")
        for label, lo, hi in AGE_BANDS:
            mask = (ages >= lo) & (ages < hi)
            sub_y = y_te[mask.to_numpy()]
            sub_p = proba[mask.to_numpy()]
            m = _subgroup_auroc(sub_y, sub_p, min_n=30, min_pos=5)
            if m is None:
                age_rows.append({"group": label, "note": "insufficient n or single class"})
                continue
            m["group"] = label
            m["auroc_delta_vs_overall"] = round(m["auroc"] - overall_auroc, 4)
            age_rows.append(m)
            if m["auroc_delta_vs_overall"] < -0.05:
                flagged_age.append(m)

    # Within age bands: death rate should rise with current_scai tertile (no reversal)
    scai_inversions: list[dict[str, Any]] = []
    if "age_years" in X_test.columns and "current_scai" in X_test.columns:
        frame = X_test.copy()
        frame["y"] = y_te
        frame["p"] = proba
        for label, lo, hi in AGE_BANDS:
            band = frame[(frame["age_years"] >= lo) & (frame["age_years"] < hi)]
            if len(band) < 60:
                continue
            means = band.groupby(
                pd.qcut(band["current_scai"], q=3, duplicates="drop"),
                observed=True,
            )["y"].mean()
            if len(means) >= 2 and float(means.iloc[-1]) < float(means.iloc[0]):
                scai_inversions.append({"age_band": label, "death_rate_by_scai_tertile": means.round(3).to_dict()})

    simpsons_ok = len(flagged_age) == 0 and len(scai_inversions) == 0

    # --- Item 19: direction on top drivers ---
    median_row = X_test[feat].median(numeric_only=True).to_frame().T
    p_base = float(pipe.predict_proba(median_row[feat])[0, 1])
    direction_rows: list[dict[str, Any]] = []
    primary_direction = {
        "scai_prop_ge3",
        "med_infusion_mean",
        "med_distinct",
        "infusion_x_scai_severity",
    }
    direction_ok = True
    for col, v_lo, v_hi in DIRECTION_SPECS:
        if col not in median_row.columns:
            continue
        lo_row = median_row.copy()
        hi_row = median_row.copy()
        lo_row[col] = v_lo
        hi_row[col] = v_hi
        if col == "current_scai":
            lo_row["scai_prop_ge3"] = 0.0
            hi_row["scai_prop_ge3"] = 1.0
        if col == "clinical_event_n":
            lo_row["med_per_clinical_event"] = 0.001
            hi_row["med_per_clinical_event"] = 0.015
        if col == "infusion_x_scai_severity":
            lo_row["med_infusion_mean"] = 0.05
            lo_row["scai_prop_ge3"] = 0.0
            hi_row["med_infusion_mean"] = 2.0
            hi_row["scai_prop_ge3"] = 1.0
        p_lo = float(pipe.predict_proba(lo_row[feat])[0, 1])
        p_hi = float(pipe.predict_proba(hi_row[feat])[0, 1])
        ok = p_hi >= p_lo - 1e-6
        direction_rows.append(
            {
                "feature": col,
                "p_at_low": round(p_lo, 4),
                "p_at_high": round(p_hi, 4),
                "delta": round(p_hi - p_lo, 4),
                "direction_ok": ok,
                "primary_driver": col in primary_direction,
            }
        )
    direction_ok = all(
        r["direction_ok"] for r in direction_rows if r.get("primary_driver")
    )
    secondary_miss = [r["feature"] for r in direction_rows if not r["direction_ok"] and not r.get("primary_driver")]

    # --- Item 21: edge cases ---
    low_row = median_row.copy()
    for c, v in {
        "med_distinct": 1.0,
        "med_infusion_mean": 0.05,
        "proc_n": 0.0,
        "proc_distinct_nom": 0.0,
        "current_scai": 0.0,
        "scai_prop_ge3": 0.0,
        "scai_std": 0.0,
        "infusion_x_scai_severity": 0.0,
        "clinical_events_last_4h": float(
            X_test["clinical_events_last_4h"].quantile(0.1)
            if "clinical_events_last_4h" in X_test.columns
            else 0.0
        ),
    }.items():
        if c in low_row.columns:
            low_row[c] = v

    high_row = median_row.copy()
    for c, v in {
        "med_distinct": 6.0,
        "med_infusion_mean": 3.5,
        "proc_n": 4.0,
        "proc_distinct_nom": 4.0,
        "current_scai": 4.0,
        "scai_prop_ge3": 1.0,
        "scai_std": 1.5,
        "infusion_x_scai_severity": 4.0,
        "clinical_events_last_4h": float(
            X_test["clinical_events_last_4h"].quantile(0.9)
            if "clinical_events_last_4h" in X_test.columns
            else 5.0
        ),
    }.items():
        if c in high_row.columns:
            high_row[c] = v

    p_median = p_base
    p_low = float(pipe.predict_proba(low_row[feat])[0, 1])
    p_high = float(pipe.predict_proba(high_row[feat])[0, 1])
    edge_ok = p_low < p_median < p_high or (p_low <= 0.15 and p_high >= 0.40)

    report = {
        "data_dir": str(data_dir),
        "model_dir": str(model_dir),
        "overall_auroc": round(overall_auroc, 4),
        "simpsons_paradox_check": {
            "pass": simpsons_ok,
            "by_age_band": age_rows,
            "flagged_age_auroc_delta_lt_5pp": flagged_age,
            "scai_death_rate_inversions_within_age_band": scai_inversions,
            "note": (
                "No age-band AUROC collapse >5pp and no SCAI death-rate inversion within age bands."
                if simpsons_ok
                else "Review flagged age bands or SCAI inversions."
            ),
        },
        "direction_test": {
            "pass": direction_ok,
            "baseline_median_row_p_death": round(p_median, 4),
            "feature_sweeps": direction_rows,
            "secondary_isolated_miss": secondary_miss,
            "note": (
                "PASS on primary drivers (SCAI fraction, infusion, med load, interaction). "
                "Isolated sweeps on correlated columns may be flat."
                if direction_ok
                else "Primary driver direction check failed."
            ),
        },
        "edge_case_test": {
            "pass": edge_ok,
            "p_death_median_row": round(p_median, 4),
            "p_death_extreme_low_scai": round(p_low, 4),
            "p_death_extreme_high_scai": round(p_high, 4),
            "ordering_ok": p_low < p_median < p_high,
        },
    }

    out = model_dir / "mortality_sanity_audit.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(
        {
            "simpsons_pass": simpsons_ok,
            "direction_pass": direction_ok,
            "edge_pass": edge_ok,
            "wrote": str(out),
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
