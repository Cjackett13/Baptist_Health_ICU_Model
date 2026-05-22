#!/usr/bin/env python3
"""Feature ablation for mortality XGBoost (always keeps demo_profile_bucket at min weight)."""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "Frontend" / "lib" / "models"))

import joblib  # noqa: E402
import mortality_model as mm  # noqa: E402

DATA = _REPO / "Baptist_tester" / "synth_cs_data"
LEGACY = json.loads((DATA / "xgb_best_hyperparams.json").read_text(encoding="utf-8"))

BASE_CLINICAL = [c for c in mm.FEATURE_COLUMNS if c != "demo_profile_bucket"]
DEMO = "demo_profile_bucket"

DROP_SINGLES = [
    "clinical_event_n",
    "proc_distinct_nom",
    "proc_n_bucket",
    "med_distinct_bucket",
    "infusion_residual_within_scai",
    "scai_slope_12h",
]

DROP_COMBOS: dict[str, list[str]] = {
    "redundant_bundle": [
        "proc_distinct_nom",
        "proc_n_bucket",
        "med_distinct_bucket",
        "clinical_event_n",
    ],
    "weak_bundle": [
        "clinical_event_n",
        "infusion_residual_within_scai",
        "scai_slope_12h",
    ],
    "all_candidates": list(DROP_SINGLES),
}


def _feat_list(drop: set[str]) -> list[str]:
    clinical = [c for c in BASE_CLINICAL if c not in drop]
    return clinical + [DEMO]


def _rank(meta: dict) -> tuple[float, float, float]:
    return (
        -float(meta["test_brier"]),
        float(meta["test_auprc"]),
        float(meta["test_auroc"]),
    )


def _run(name: str, drop: set[str]) -> dict:
    cols = _feat_list(drop)
    X_tr, X_te, y_tr, y_te, _, _, _ = mm.prepare_train_test_features(
        DATA, feature_cols=cols, pct_lo=1.0, pct_hi=99.0, random_state=42
    )
    _, meta = mm.train_pipeline(
        X_tr,
        X_te,
        y_tr,
        y_te,
        feature_columns=cols,
        random_state=42,
        tune_hyperparams=False,
        xgb_overrides=LEGACY,
        fixed_n_estimators=225,
        reliability_plot_path=None,
    )
    row = {
        "name": name,
        "dropped": sorted(drop),
        "n_features": len(cols),
        "features": cols,
        "test_auroc": float(meta["test_auroc"]),
        "test_auprc": float(meta["test_auprc"]),
        "test_brier": float(meta["test_brier"]),
        "n_train": int(meta["n_train"]),
        "n_test": int(meta["n_test"]),
    }
    print(
        f"{name:40s}  AUROC={row['test_auroc']:.4f}  "
        f"AUPRC={row['test_auprc']:.4f}  Brier={row['test_brier']:.4f}  "
        f"n_feat={row['n_features']}",
        flush=True,
    )
    return row


def main() -> int:
    trials: list[dict] = []
    trials.append(_run("baseline_all_17", set()))

    for feat in DROP_SINGLES:
        trials.append(_run(f"drop_{feat}", {feat}))

    for combo_name, feats in DROP_COMBOS.items():
        trials.append(_run(combo_name, set(feats)))

  # pairwise among weakest four
    weak4 = ["clinical_event_n", "proc_distinct_nom", "infusion_residual_within_scai", "scai_slope_12h"]
    for a, b in combinations(weak4, 2):
        trials.append(_run(f"drop_{a}+{b}", {a, b}))

    trials.sort(key=lambda r: _rank(r), reverse=True)
    best = trials[0]
    out_path = DATA / "mortality_ablation_results.json"
    out_path.write_text(json.dumps({"trials": trials, "best": best}, indent=2), encoding="utf-8")

    print("\n=== TOP 5 (by lower Brier, then AUPRC, then AUROC) ===")
    for r in trials[:5]:
        print(
            f"  {r['name']}: AUROC={r['test_auroc']:.4f} "
            f"AUPRC={r['test_auprc']:.4f} Brier={r['test_brier']:.4f}"
        )

    print(f"\n=== BEST: {best['name']} ===")
    cols = best["features"]
    X_tr, X_te, y_tr, y_te, _, _, prep = mm.prepare_train_test_features(
        DATA, feature_cols=cols, pct_lo=1.0, pct_hi=99.0, random_state=42
    )
    pipe, meta = mm.train_pipeline(
        X_tr,
        X_te,
        y_tr,
        y_te,
        feature_columns=cols,
        random_state=42,
        tune_hyperparams=False,
        xgb_overrides=LEGACY,
        fixed_n_estimators=225,
        reliability_plot_path=DATA / "mortality_reliability_curve.png",
    )
    meta.update(prep)
    meta["ablation_winner"] = best["name"]
    meta["ablation_dropped"] = best["dropped"]
    joblib.dump({"pipeline": pipe, "feature_names": cols}, DATA / "xgb_mortality_pipeline.joblib")
    (DATA / "xgb_mortality_model_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote best model ({best['name']}) to {DATA}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
