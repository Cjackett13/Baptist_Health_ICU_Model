#!/usr/bin/env python3
"""
Sweep mortality training strategies; evaluate FN/FP/sensitivity at thr=0.17 and under FP cap.

Usage (repo root)::

  PYTHONPATH=Frontend/lib/models python3 Baptist_tester/mortality_strategy_sweep.py \\
    --data-dir data/cleaned --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "Frontend" / "lib" / "models"))

import mortality_model as mm  # noqa: E402
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score  # noqa: E402

ALERT_THR = 0.17
FP_CAP = int(mm.MORTALITY_ALERT_MAX_FP_TEST)
MIN_AUROC = 0.75

BASE_HP = json.loads((_REPO / "data" / "xgb_best_hyperparams.json").read_text(encoding="utf-8"))


def _eval(y, proba, thr: float) -> dict[str, float | int]:
    cm = mm._confusion_report_at_threshold(y, proba, thr)
    return {
        "threshold": float(thr),
        "fn": int(cm["fn"]),
        "fp": int(cm["fp"]),
        "tp": int(cm["tp"]),
        "tn": int(cm["tn"]),
        "recall": float(cm["recall"]),
        "ppv": float(cm["ppv"]),
        "accuracy": float(cm["accuracy"]),
        "f1": float(cm["f1"]),
    }


def _objective(row: dict) -> float:
    if row["test_auroc"] < MIN_AUROC:
        return 1e6
    u = row["under_fp_cap"]
    a = row["at_017"]
    return float(u["fn"] + u["fp"]) + 0.35 * float(a["fn"] + a["fp"])


STRATEGIES: list[dict] = [
    {
        "name": "baseline_v3_hp",
        "spw_mult": 2.55,
        "scai_mult": 1.75,
        "es_tree": "combo",
        "tune": False,
        "xgb_overrides": deepcopy(BASE_HP),
        "tune_metric": "recall_at_fp_cap",
    },
    {
        "name": "spw3_scai225",
        "spw_mult": 3.0,
        "scai_mult": 2.25,
        "es_tree": "combo",
        "tune": False,
        "xgb_overrides": deepcopy(BASE_HP),
        "tune_metric": "recall_at_fp_cap",
    },
    {
        "name": "spw35_scai225",
        "spw_mult": 3.5,
        "scai_mult": 2.25,
        "es_tree": "combo",
        "tune": False,
        "xgb_overrides": deepcopy(BASE_HP),
        "tune_metric": "recall_at_fp_cap",
    },
    {
        "name": "spw255_scai225_auprc_es",
        "spw_mult": 2.55,
        "scai_mult": 2.25,
        "es_tree": "auprc",
        "tune": False,
        "xgb_overrides": deepcopy(BASE_HP),
        "tune_metric": "recall_at_fp_cap",
    },
    {
        "name": "reg_strong_mcw10",
        "spw_mult": 2.55,
        "scai_mult": 2.25,
        "es_tree": "combo",
        "tune": False,
        "xgb_overrides": {**deepcopy(BASE_HP), "min_child_weight": 10.0, "reg_alpha": 0.5},
        "tune_metric": "recall_at_fp_cap",
    },
    {
        "name": "reg_strong_mcw12",
        "spw_mult": 3.0,
        "scai_mult": 2.25,
        "es_tree": "combo",
        "tune": False,
        "xgb_overrides": {**deepcopy(BASE_HP), "min_child_weight": 12.0, "reg_alpha": 0.75},
        "tune_metric": "recall_at_fp_cap",
    },
    {
        "name": "tune_ap_spw30_scai225",
        "spw_mult": 3.0,
        "scai_mult": 2.25,
        "es_tree": "auprc",
        "tune": True,
        "xgb_overrides": None,
        "tune_metric": "average_precision",
        "tune_n_iter": 16,
    },
    {
        "name": "tune_recall_fp_spw30_scai225",
        "spw_mult": 3.0,
        "scai_mult": 2.25,
        "es_tree": "combo",
        "tune": True,
        "xgb_overrides": None,
        "tune_metric": "recall_at_fp_cap",
        "tune_n_iter": 16,
    },
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=_REPO / "data" / "cleaned")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=_REPO / "data" / "mortality_strategy_sweep.json")
    ap.add_argument("--only", type=str, default="", help="Comma-separated strategy names")
    args = ap.parse_args()

    data_dir = args.data_dir.resolve()
    mm.ensure_duckdb_sidecar(data_dir, force=False)
    X_train, X_test, y_train, y_test, *_ = mm.prepare_train_test_features(
        data_dir, random_state=int(args.seed)
    )
    y_te = y_test.to_numpy().astype(int)

    only = {s.strip() for s in args.only.split(",") if s.strip()}
    results: list[dict] = []

    for spec in STRATEGIES:
        if only and spec["name"] not in only:
            continue
        print(f"=== {spec['name']} ===", flush=True)
        pipe, meta = mm.train_pipeline(
            X_train,
            X_test,
            y_train,
            y_test,
            random_state=int(args.seed),
            tune_hyperparams=bool(spec.get("tune")),
            tune_n_iter=int(spec.get("tune_n_iter", 14)),
            tune_primary_metric=str(spec.get("tune_metric", "recall_at_fp_cap")),
            xgb_overrides=spec.get("xgb_overrides"),
            balance_scale_pos_weight_mult=float(spec["spw_mult"]),
            high_scai_sample_weight_mult=float(spec["scai_mult"]),
            es_tree_selection=str(spec["es_tree"]),
        )
        proba = pipe.predict_proba(X_test)[:, 1]
        thr_cap, cm_cap = mm._pick_holdout_threshold_under_fp_cap(y_te, proba, max_fp=FP_CAP)
        row = {
            "name": spec["name"],
            "spec": {k: v for k, v in spec.items() if k != "xgb_overrides"},
            "test_auroc": float(roc_auc_score(y_te, proba)),
            "test_auprc": float(average_precision_score(y_te, proba)),
            "test_brier": float(brier_score_loss(y_te, proba)),
            "best_n_estimators": int(meta.get("best_n_estimators", 0)),
            "at_017": _eval(y_te, proba, ALERT_THR),
            "under_fp_cap": {
                "threshold": float(thr_cap),
                **_eval(y_te, proba, thr_cap),
            },
            "xgb_params": meta.get("xgb_tuned_params") or spec.get("xgb_overrides"),
        }
        row["objective"] = _objective(row)
        results.append(row)
        print(
            f"  AUROC={row['test_auroc']:.3f} AUPRC={row['test_auprc']:.3f} "
            f"@0.17 FN={row['at_017']['fn']} FP={row['at_017']['fp']} sens={row['at_017']['recall']:.3f} "
            f"cap FN={row['under_fp_cap']['fn']} FP={row['under_fp_cap']['fp']}",
            flush=True,
        )

    results.sort(key=lambda r: r["objective"])
    best = results[0] if results else None
    out_doc = {
        "alert_threshold_fixed": ALERT_THR,
        "fp_cap": FP_CAP,
        "min_auroc_gate": MIN_AUROC,
        "results": results,
        "best_by_objective": best["name"] if best else None,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_doc, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")
    if best:
        print(f"Best: {best['name']} objective={best['objective']:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
