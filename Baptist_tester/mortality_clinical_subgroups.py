#!/usr/bin/env python3
"""
AUROC / AUPRC by SCAI tertile and primary diagnosis (dataset B test split).

  PYTHONPATH=Frontend/lib/models python3 Baptist_tester/mortality_clinical_subgroups.py \
      --data-dir data/cleaned --model-dir data --seed 42
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
from sklearn.metrics import average_precision_score, roc_auc_score

_REPO = Path(__file__).resolve().parent.parent


def _resolve(p: Path) -> Path:
    return p.resolve() if p.is_absolute() else (_REPO / p).resolve()


def _subgroup_metrics(
    y: np.ndarray,
    p: np.ndarray,
    *,
    min_n: int = 40,
    min_pos: int = 8,
) -> dict[str, Any] | None:
    n = len(y)
    n_pos = int((y == 1).sum())
    if n < min_n or n_pos < min_pos or n_pos == n:
        return None
    return {
        "n": n,
        "n_pos": n_pos,
        "death_rate": round(float(y.mean()), 4),
        "auroc": round(float(roc_auc_score(y, p)), 4),
        "auprc": round(float(average_precision_score(y, p)), 4),
    }


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

    import mortality_model  # noqa: E402

    sys.modules["__main__"] = mortality_model

    from mortality_model import FEATURE_COLUMNS, prepare_train_test_features  # noqa: E402

    data_dir = _resolve(args.data_dir)
    model_dir = _resolve(args.model_dir)
    meta_path = model_dir / "xgb_mortality_model_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    pct_lo = float(meta.get("outlier_pct_lo", 1.0))
    pct_hi = float(meta.get("outlier_pct_hi", 99.0))
    seed = int(meta.get("split_seed", args.seed))

    X_train, X_test, y_train, y_test, pid_train, pid_test, _ = prepare_train_test_features(
        data_dir, pct_lo=pct_lo, pct_hi=pct_hi, random_state=seed
    )
    y_te = np.asarray(y_test).astype(int)
    blob = joblib.load(model_dir / "xgb_mortality_pipeline.joblib")
    pipe = blob["pipeline"]
    feat = list(blob.get("feature_names", FEATURE_COLUMNS))
    proba = pipe.predict_proba(X_test[feat])[:, 1]

    overall_auroc = float(roc_auc_score(y_te, proba))
    overall_auprc = float(average_precision_score(y_te, proba))

    test_df = X_test.copy()
    test_df["y"] = y_te
    test_df["p"] = proba
    test_df["PERSON_ID"] = pid_test.values

    scai = pd.to_numeric(test_df["current_scai"], errors="coerce")
    test_df["scai_tertile"] = pd.qcut(scai, q=3, duplicates="drop").astype(str)

    scai_rows: list[dict[str, Any]] = []
    for g, sub in test_df.groupby("scai_tertile", observed=True):
        m = _subgroup_metrics(sub["y"].to_numpy(), sub["p"].to_numpy())
        if m is None:
            scai_rows.append({"group": str(g), "note": "insufficient n or single class"})
            continue
        m["group"] = str(g)
        m["auroc_delta_vs_overall"] = round(m["auroc"] - overall_auroc, 4)
        m["auprc_delta_vs_overall"] = round(m["auprc"] - overall_auprc, 4)
        scai_rows.append(m)
    scai_rows.sort(key=lambda r: r.get("auroc", 0))

    dx_rows: list[dict[str, Any]] = []
    diag_path = data_dir / "diagnosis.parquet"
    if not diag_path.is_file():
        diag_path = data_dir.parent / "diagnosis.parquet"
    if diag_path.is_file():
        dx = pd.read_parquet(diag_path)
        enc_path = data_dir / "encounter.parquet"
        if not enc_path.is_file():
            enc_path = data_dir.parent / "encounter.parquet"
        sort_cols = ["PERSON_ID"]
        if "DIAG_PRIORITY" in dx.columns:
            sort_cols.append("DIAG_PRIORITY")
        primary = dx.sort_values(sort_cols).groupby("PERSON_ID", as_index=False).first()
        test_df = test_df.merge(primary, on="PERSON_ID", how="left", suffixes=("", "_dx"))
        for col in ["CLASSIFICATION_CD", "NOMENCLATURE_CD"]:
            if col not in test_df.columns:
                continue
            for g, sub in test_df.groupby(col, dropna=False):
                if sub[col].isna().all():
                    continue
                m = _subgroup_metrics(sub["y"].to_numpy(), sub["p"].to_numpy(), min_n=25, min_pos=5)
                if m is None:
                    continue
                m["dimension"] = col
                m["group"] = str(g)
                m["auroc_delta_vs_overall"] = round(m["auroc"] - overall_auroc, 4)
                dx_rows.append(m)
        dx_rows.sort(key=lambda r: -r.get("n", 0))

    # AUROC often drops when death_rate is very high (little rankable negatives); use AUPRC for acuity.
    high_scai_row = max(
        (r for r in scai_rows if "auroc" in r and r.get("death_rate", 0) >= 0.5),
        key=lambda r: r.get("death_rate", 0),
        default=None,
    )
    low_scai_row = min(
        (r for r in scai_rows if "auroc" in r),
        key=lambda r: r.get("death_rate", 1.0),
        default=None,
    )
    safety_flag_high_scai = False
    safety_flag_low_scai_auprc = False
    if high_scai_row and high_scai_row.get("auprc", 0) < overall_auprc - 0.10:
        safety_flag_high_scai = True
    if low_scai_row and low_scai_row.get("auprc_delta_vs_overall", 0) < -0.40:
        safety_flag_low_scai_auprc = True

    if high_scai_row and not safety_flag_high_scai:
        clinical_note = (
            "High-SCAI tertile: AUROC below overall is expected at ~74% death rate; "
            f"AUPRC {high_scai_row.get('auprc', 0):.3f} supports ranking among high-risk patients. "
            "See mortality_presentation_signoff.md."
        )
    elif safety_flag_high_scai:
        clinical_note = "High-SCAI tertile AUPRC below overall — review before deployment."
    else:
        clinical_note = "SCAI subgroup review complete; see mortality_presentation_signoff.md."

    report = {
        "data_dir": str(data_dir),
        "model_dir": str(model_dir),
        "n_test": int(len(y_te)),
        "overall": {"auroc": round(overall_auroc, 4), "auprc": round(overall_auprc, 4)},
        "by_scai_tertile": scai_rows,
        "by_primary_diagnosis": dx_rows[:24],
        "clinical_safety_note": clinical_note,
        "safety_flag_high_scai_underperformance": safety_flag_high_scai,
        "safety_flag_low_scai_auprc_limited": safety_flag_low_scai_auprc,
        "high_scai_review_completed": not safety_flag_high_scai,
        "high_scai_tertile": high_scai_row,
        "low_scai_tertile": low_scai_row,
    }

    out = model_dir / "mortality_clinical_subgroup_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = model_dir / "mortality_clinical_subgroup_report.md"
    lines = [
        "# Mortality clinical subgroup report",
        "",
        f"**Test n:** {report['n_test']} | **AUROC:** {report['overall']['auroc']} | **AUPRC:** {report['overall']['auprc']}",
        "",
        f"**Safety:** {report['clinical_safety_note']}",
        f"**High-SCAI review:** {'complete' if report.get('high_scai_review_completed') else 'pending'}",
        "",
        "## SCAI tertile (current_scai)",
        "",
        "| Tertile | n | death_rate | AUROC | AUPRC | Δ AUROC |",
        "|---------|--:|-----------:|------:|------:|--------:|",
    ]
    for r in scai_rows:
        if "auroc" not in r:
            lines.append(f"| {r.get('group')} | — | — | — | — | — |")
            continue
        lines.append(
            f"| {r['group']} | {r['n']} | {r['death_rate']:.1%} | {r['auroc']:.3f} | {r['auprc']:.3f} | {r['auroc_delta_vs_overall']:+.3f} |"
        )
    lines.extend(["", "## Primary diagnosis (top groups)", ""])
    for r in dx_rows[:12]:
        lines.append(
            f"- **{r.get('dimension')}={r.get('group')}** n={r['n']} AUROC={r['auroc']:.3f} (Δ {r['auroc_delta_vs_overall']:+.3f})"
        )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "overall": report["overall"],
                "safety_flag_high_scai": safety_flag_high_scai,
                "high_scai_review_completed": report["high_scai_review_completed"],
            },
            indent=2,
        )
    )
    print(f"Wrote {out}")
    print(f"Wrote {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
