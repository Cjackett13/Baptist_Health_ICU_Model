#!/usr/bin/env python3
"""
Subgroup MAE / RMSE / R² for LOS fairness reporting.

Demographics (race, ethnicity) are joined from ``person.parquet`` for evaluation only —
they are excluded from the model feature matrix by ``los_model_config`` / ``los_feature_policy``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from models.length_of_stay.config import BUNDLE_NAME, LABEL_EVAL, REPO_ROOT
from models.length_of_stay.features import load_training_feature_spec, raw_matrix_for_pipeline
from models.length_of_stay.train import load_training_frame, split_by_manifest


def _resolve(p: Path) -> Path:
    return p.resolve() if p.is_absolute() else (REPO_ROOT / p).resolve()


def _predict_hours(pipe, df: pd.DataFrame, included: list[str]) -> np.ndarray:
    pred_log = pipe.predict(raw_matrix_for_pipeline(df, included))
    return np.expm1(np.clip(pred_log, 0, None))


def _metrics_row(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "n": int(len(y)),
        "mae_hours": float(mean_absolute_error(y, pred)),
        "rmse_hours": float(np.sqrt(mean_squared_error(y, pred))),
        "r2_hours": float(r2_score(y, pred)) if len(y) > 1 else float("nan"),
    }


def _primary_dx_label(row: pd.Series) -> str:
    cols = [
        "dx_cat_acute_mi",
        "dx_cat_adhf",
        "dx_cat_arrhythmia_arrest",
        "dx_cat_myocarditis_cmp",
        "dx_cat_aortic_valve",
        "dx_cat_post_cardiotomy",
    ]
    for c in cols:
        if c in row.index and pd.to_numeric(row[c], errors="coerce") == 1:
            return c.replace("dx_cat_", "")
    return "other_dx"


def _join_eval_demographics(te: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    """Attach race/ethnicity/sex from person table — never from model X."""
    person = pd.read_parquet(data_dir / "person.parquet")
    demo = person[["PERSON_ID", "RACE_CD", "ETHNICITY_CD", "SEX_CD"]].copy()
    demo["race_cd"] = demo["RACE_CD"].astype("string").str.upper().str.strip()
    demo["ethnicity_cd"] = demo["ETHNICITY_CD"].astype("string").str.upper().str.strip()
    demo["sex_cd"] = demo["SEX_CD"].astype("string").str.upper().str.strip()
    out = te.drop(columns=[c for c in ("race_cd", "ethnicity_cd", "sex_cd") if c in te.columns])
    return out.merge(demo[["PERSON_ID", "race_cd", "ethnicity_cd", "sex_cd"]], on="PERSON_ID", how="left")


def run_subgroup_mae(data_dir: Path, model_dir: Path) -> dict[str, Any]:
    data_dir = _resolve(data_dir)
    model_dir = _resolve(model_dir)
    pipe = joblib.load(model_dir / BUNDLE_NAME)
    included, _ = load_training_feature_spec(model_dir, data_dir)

    df = load_training_frame(data_dir)
    _, _, te = split_by_manifest(df, data_dir)
    te = _join_eval_demographics(te, data_dir)

    assert "race_cd" not in included and "ethnicity_cd" not in included

    y = te[LABEL_EVAL].to_numpy()
    pred = _predict_hours(pipe, te, included)
    overall = _metrics_row(y, pred)
    te = te.copy()
    te["pred_hours"] = pred
    te["abs_err"] = np.abs(y - pred)
    te["primary_dx"] = te.apply(_primary_dx_label, axis=1)

    groups: dict[str, list[dict[str, Any]]] = {}

    def by_col(col: str, min_n: int = 25) -> None:
        rows = []
        for g, sub in te.groupby(col, dropna=False):
            if len(sub) < min_n:
                rows.append({"group": str(g), "n": int(len(sub)), "note": "n too small for reliable MAE"})
                continue
            m = _metrics_row(sub[LABEL_EVAL].to_numpy(), sub["pred_hours"].to_numpy())
            m["group"] = str(g)
            m["mae_vs_overall_delta"] = round(m["mae_hours"] - overall["mae_hours"], 2)
            rows.append(m)
        rows.sort(key=lambda r: r.get("mae_hours", 999))
        groups[col] = rows

    by_col("race_cd")
    by_col("sex_cd")
    by_col("ethnicity_cd")
    by_col("unit_cd")
    by_col("primary_dx", min_n=20)
    by_col("admit_type_cd")

    flagged = []
    for col, rows in groups.items():
        for r in rows:
            d = r.get("mae_vs_overall_delta")
            if d is not None and d > 5.0:
                flagged.append({"dimension": col, "group": r["group"], "mae_delta_hours": d})

    return {
        "overall_test": overall,
        "subgroups": groups,
        "flagged_mae_worse_than_overall_by_5h": flagged,
        "demographic_source": "person.parquet (eval-only join; not in model X)",
        "model_features_excluded_from_fairness_join": sorted(
            {"race_cd", "ethnicity_cd", "demo_profile_bucket"} & set(included)
        ),
        "interpretation": (
            "Compare subgroup MAE to overall; deltas >5h warrant fairness review. "
            "Small groups (n<25) are unreliable. Race/ethnicity used only for reporting."
        ),
    }


def main() -> int:
    import argparse

    from models.length_of_stay.config import DATA_DIR_DEFAULT

    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR_DEFAULT)
    ap.add_argument("--model-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    model_dir = _resolve(args.model_dir) if args.model_dir else _resolve(args.data_dir) / "los_model"
    report = run_subgroup_mae(args.data_dir, model_dir)
    out = args.out or (model_dir / "validation" / "los_subgroup_mae.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["overall_test"], indent=2))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
