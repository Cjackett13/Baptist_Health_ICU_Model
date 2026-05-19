#!/usr/bin/env python3
"""
LOS parquet bundle audit — **does not modify mortality pipeline files.**

Loads the 8 core synthetic tables + ``code_value`` from the parquet bundle (default: ``data/``, 5000 patients).
runs LOS-only string cleaning, and writes:

  - ``los_bundle_audit_report.json``
  - ``eda/los_bundle_audit_report.html`` (includes LOS_HOURS correlation heatmap)

  python3 Baptist_tester/los_bundle_audit.py
  python3 Baptist_tester/los_bundle_audit.py --data-dir Baptist_tester/synth_cs_data --no-open
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

_REPO = Path(__file__).resolve().parent.parent

# 8 core tables + code map (exclude mortality sidecars)
BUNDLE_TABLES: dict[str, str] = {
    "person": "person.parquet",
    "encounter": "encounter.parquet",
    "diagnosis": "diagnosis.parquet",
    "clinical_event": "clinical_event.parquet",
    "medication_admin": "medication_admin.parquet",
    "procedure_event": "procedure_event.parquet",
    "scai_stage_hourly": "scai_stage_hourly.parquet",
    "code_value": "code_value.parquet",
}

PRIMARY_KEYS: dict[str, list[str]] = {
    "person": ["PERSON_ID"],
    "encounter": ["ENCOUNTER_ID"],
    "diagnosis": ["DIAGNOSIS_ID"],
    "clinical_event": ["EVENT_ID"],
    "medication_admin": ["MED_ADMIN_ID"],
    "procedure_event": ["PROCEDURE_ID"],
    "scai_stage_hourly": ["ENCOUNTER_ID", "HOUR_FROM_ADMIT"],
    "code_value": ["EVENT_CD"],
}

FK_CHECKS: list[dict[str, Any]] = [
    {
        "child_table": "encounter",
        "child_cols": ["PERSON_ID"],
        "parent_table": "person",
        "parent_cols": ["PERSON_ID"],
    },
    {
        "child_table": "diagnosis",
        "child_cols": ["ENCOUNTER_ID"],
        "parent_table": "encounter",
        "parent_cols": ["ENCOUNTER_ID"],
    },
    {
        "child_table": "diagnosis",
        "child_cols": ["PERSON_ID"],
        "parent_table": "person",
        "parent_cols": ["PERSON_ID"],
    },
    {
        "child_table": "clinical_event",
        "child_cols": ["ENCOUNTER_ID"],
        "parent_table": "encounter",
        "parent_cols": ["ENCOUNTER_ID"],
    },
    {
        "child_table": "clinical_event",
        "child_cols": ["PERSON_ID"],
        "parent_table": "person",
        "parent_cols": ["PERSON_ID"],
    },
    {
        "child_table": "medication_admin",
        "child_cols": ["ENCOUNTER_ID"],
        "parent_table": "encounter",
        "parent_cols": ["ENCOUNTER_ID"],
    },
    {
        "child_table": "medication_admin",
        "child_cols": ["PERSON_ID"],
        "parent_table": "person",
        "parent_cols": ["PERSON_ID"],
    },
    {
        "child_table": "procedure_event",
        "child_cols": ["ENCOUNTER_ID"],
        "parent_table": "encounter",
        "parent_cols": ["ENCOUNTER_ID"],
    },
    {
        "child_table": "procedure_event",
        "child_cols": ["PERSON_ID"],
        "parent_table": "person",
        "parent_cols": ["PERSON_ID"],
    },
    {
        "child_table": "scai_stage_hourly",
        "child_cols": ["ENCOUNTER_ID"],
        "parent_table": "encounter",
        "parent_cols": ["ENCOUNTER_ID"],
    },
    {
        "child_table": "clinical_event",
        "child_cols": ["EVENT_CD"],
        "parent_table": "code_value",
        "parent_cols": ["EVENT_CD"],
        "optional": True,
    },
]

CORR_EXCLUDE_FROM_FEATURES = frozenset(
    {
        "LOS_HOURS",
        "los_hours_calc",
        "los_hours_reconciled",
        "los_reconcile_delta_h",
    }
)


def _resolve(p: Path) -> Path:
    return p.resolve() if p.is_absolute() else (_REPO / p).resolve()


def _load_clean_bundle(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Load 8+1 tables; apply LOS-only sentinel cleaning from los_data_prep."""
    import importlib.util

    prep_path = _REPO / "Baptist_tester" / "los_data_prep.py"
    spec = importlib.util.spec_from_file_location("los_data_prep", prep_path)
    prep = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(prep)

    tables: dict[str, pd.DataFrame] = {}
    for name, fname in BUNDLE_TABLES.items():
        raw = pd.read_parquet(data_dir / fname)
        cleaned, _ = prep._clean_table_strings(raw)
        if name == "encounter":
            cleaned = prep.reconcile_los_hours(cleaned)
        tables[name] = cleaned
    return tables


def _table_inventory(tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, df in tables.items():
        out[name] = {
            "n_rows": int(len(df)),
            "n_columns": int(df.shape[1]),
            "columns": df.columns.tolist(),
            "dtypes": {c: str(df[c].dtype) for c in df.columns},
        }
    return out


def _pk_audit(tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for name, pk_cols in PRIMARY_KEYS.items():
        df = tables[name]
        if not all(c in df.columns for c in pk_cols):
            results[name] = {"ok": False, "error": f"Missing PK columns {pk_cols}"}
            continue
        sub = df[pk_cols].dropna(how="any")
        n = len(df)
        n_unique = int(sub.drop_duplicates().shape[0])
        dup = int(n - n_unique - int(df[pk_cols].isna().any(axis=1).sum()))
        null_pk = int(df[pk_cols].isna().any(axis=1).sum())
        results[name] = {
            "ok": dup == 0 and null_pk == 0,
            "primary_key": pk_cols,
            "n_rows": n,
            "n_unique_pk": n_unique,
            "duplicate_pk_rows": max(0, dup),
            "null_pk_rows": null_pk,
        }
    return results


def _fk_coverage(tables: dict[str, pd.DataFrame]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    coverage: dict[str, Any] = {}
    orphans: list[dict[str, Any]] = []

    for spec in FK_CHECKS:
        child_name = spec["child_table"]
        parent_name = spec["parent_table"]
        child = tables[child_name]
        parent = tables[parent_name]
        cc, pc = spec["child_cols"], spec["parent_cols"]
        key = f"{child_name}.{'.'.join(cc)} -> {parent_name}.{'.'.join(pc)}"

        parent_df = parent[pc].drop_duplicates()
        merged = child[cc].merge(parent_df, left_on=cc, right_on=pc, how="left", indicator=True)
        has_fk = child[cc].notna().all(axis=1) if len(cc) > 1 else child[cc[0]].notna()
        orphan_n = int((has_fk & (merged["_merge"] == "left_only")).sum())
        n_child = int(has_fk.sum())
        pct = 100.0 * (1.0 - orphan_n / n_child) if n_child else 100.0
        ok = orphan_n == 0 or spec.get("optional", False)

        if len(cc) == 1:
            orphan_sample = (
                child.loc[has_fk & (merged["_merge"] == "left_only"), cc[0]].drop_duplicates().head(20).tolist()
            )
        else:
            orphan_sample = (
                child.loc[has_fk & (merged["_merge"] == "left_only"), cc]
                .drop_duplicates()
                .head(20)
                .to_dict(orient="records")
            )

        coverage[key] = {
            "child_table": child_name,
            "parent_table": parent_name,
            "child_columns": cc,
            "parent_columns": pc,
            "child_rows_with_fk": n_child,
            "orphan_n": orphan_n,
            "coverage_pct": round(pct, 4),
            "ok": ok,
            "orphan_sample": orphan_sample,
        }
        if orphan_n > 0 and not spec.get("optional", False):
            orphans.append({"relationship": key, "orphan_n": orphan_n, "sample": orphan_sample})
    return coverage, orphans


def _patient_level_wide(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Encounter-level numeric aggregates for LOS_HOURS correlation (exploratory)."""
    enc = tables["encounter"][["PERSON_ID", "ENCOUNTER_ID", "LOS_HOURS"]].copy()
    wide = enc.set_index("PERSON_ID")

    dx = tables["diagnosis"].groupby("PERSON_ID").size().rename("diagnosis__n")
    med = tables["medication_admin"]
    med_agg = pd.DataFrame(
        {
            "medication_admin__n": med.groupby("PERSON_ID").size(),
            "medication_admin__med_distinct": med.groupby("PERSON_ID")["MEDICATION_CD"].nunique(),
            "medication_admin__infusion_mean": med.groupby("PERSON_ID")["INFUSION_RATE"].mean(),
        }
    )
    proc = tables["procedure_event"]
    proc_agg = pd.DataFrame(
        {
            "procedure_event__n": proc.groupby("PERSON_ID").size(),
            "procedure_event__proc_distinct": proc.groupby("PERSON_ID")["NOMENCLATURE_CD"].nunique(),
        }
    )
    ce = tables["clinical_event"]
    ce_agg = pd.DataFrame(
        {
            "clinical_event__n": ce.groupby("PERSON_ID").size(),
            "clinical_event__result_mean": ce.groupby("PERSON_ID")["RESULT_VAL"].mean(),
            "clinical_event__result_std": ce.groupby("PERSON_ID")["RESULT_VAL"].std(),
        }
    )
    scai = tables["scai_stage_hourly"].merge(
        tables["encounter"][["ENCOUNTER_ID", "PERSON_ID"]], on="ENCOUNTER_ID", how="left"
    )
    scai_agg = pd.DataFrame(
        {
            "scai__n_hours": scai.groupby("PERSON_ID").size(),
            "scai__stage_mean": scai.groupby("PERSON_ID")["SCAI_STAGE_NUM"].mean(),
            "scai__stage_max": scai.groupby("PERSON_ID")["SCAI_STAGE_NUM"].max(),
            "scai__stage_std": scai.groupby("PERSON_ID")["SCAI_STAGE_NUM"].std(),
        }
    )

    for block in (dx, med_agg, proc_agg, ce_agg, scai_agg):
        wide = wide.join(block, how="left")

    wide = wide.rename(columns={"LOS_HOURS": "encounter__LOS_HOURS"})
    return wide.reset_index()


def _los_hours_correlation(wide: pd.DataFrame) -> dict[str, Any]:
    num = wide.select_dtypes(include=[np.number]).copy()
    for c in list(num.columns):
        if c in CORR_EXCLUDE_FROM_FEATURES or c == "ENCOUNTER_ID":
            num = num.drop(columns=[c], errors="ignore")
    if "encounter__LOS_HOURS" not in num.columns and "LOS_HOURS" in wide.columns:
        num["LOS_HOURS"] = wide["LOS_HOURS"]

    target = "LOS_HOURS"
    if target not in num.columns:
        target = "encounter__LOS_HOURS"
    num = num.replace([np.inf, -np.inf], np.nan)
    std = num.std(numeric_only=True)
    num = num.loc[:, std > 1e-12]
    if target not in num.columns:
        return {"error": "LOS_HOURS not in numeric frame"}

    corr = num.corr(numeric_only=True)
    los_corr = corr[target].drop(target, errors="ignore").sort_values(key=lambda s: s.abs(), ascending=False)
    top = [
        {"feature": str(k), "pearson_r": float(v)}
        for k, v in los_corr.head(20).items()
        if np.isfinite(v)
    ]
    return {
        "n_patients": int(len(wide)),
        "n_numeric_features": int(num.shape[1]),
        "target_column": target,
        "note": "Full-stay aggregates; exploratory only. Training uses censored 0_to_t features.",
        "top_correlations_with_LOS_HOURS": top,
        "correlation_matrix": {
            "columns": corr.columns.tolist(),
            "values": corr.values.tolist(),
        },
    }


def _corr_heatmap_fig(corr_payload: dict[str, Any]) -> go.Figure | None:
    cm = corr_payload.get("correlation_matrix")
    if not cm:
        return None
    cols = cm["columns"]
    z = np.array(cm["values"])
    los_last = [c for c in cols if c != "LOS_HOURS" and c != "encounter__LOS_HOURS"]
    for t in ("LOS_HOURS", "encounter__LOS_HOURS"):
        if t in cols:
            los_last.append(t)
    idx = [cols.index(c) for c in los_last if c in cols]
    z = z[np.ix_(idx, idx)]
    labels = [los_last[i] for i in range(len(idx))]
    return go.Figure(
        data=go.Heatmap(
            z=z,
            x=labels,
            y=labels,
            zmin=-1,
            zmax=1,
            colorscale="RdBu",
            hovertemplate="%{y} vs %{x}<br>r=%{z:.3f}<extra></extra>",
        ),
        layout=dict(
            title="Pearson correlation with LOS_HOURS (patient-level aggregates)",
            template="plotly_white",
            height=max(480, 28 * len(labels) + 120),
            width=max(560, 22 * len(labels) + 160),
            xaxis=dict(tickangle=-45),
            margin=dict(l=140, r=40, t=80, b=140),
        ),
    )


def _html_report(inventory: dict, pk: dict, fk: dict, orphans: list, corr: dict, fig: go.Figure | None) -> str:
    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>LOS bundle audit</title>",
        "<style>body{font-family:system-ui,sans-serif;margin:24px}table{border-collapse:collapse}",
        "td,th{border:1px solid #ccc;padding:6px 10px}code{background:#f4f4f4;padding:2px 4px}</style>",
        "</head><body><h1>LOS parquet bundle audit</h1>",
        f"<p>Tables: {', '.join(BUNDLE_TABLES)}</p>",
        "<h2>Row / column counts</h2><table><tr><th>Table</th><th>Rows</th><th>Cols</th></tr>",
    ]
    for name, meta in inventory.items():
        parts.append(f"<tr><td>{name}</td><td>{meta['n_rows']}</td><td>{meta['n_columns']}</td></tr>")
    parts.append("</table><h2>Primary key uniqueness (after cleaning)</h2><table>")
    parts.append("<tr><th>Table</th><th>PK</th><th>OK</th><th>Duplicates</th><th>Null PK</th></tr>")
    for name, meta in pk.items():
        parts.append(
            f"<tr><td>{name}</td><td><code>{meta.get('primary_key')}</code></td>"
            f"<td>{meta.get('ok')}</td><td>{meta.get('duplicate_pk_rows')}</td>"
            f"<td>{meta.get('null_pk_rows')}</td></tr>"
        )
    parts.append("</table><h2>FK join coverage</h2><table>")
    parts.append("<tr><th>Relationship</th><th>Orphans</th><th>Coverage %</th><th>OK</th></tr>")
    for _k, meta in fk.items():
        parts.append(
            f"<tr><td><code>{meta['child_table']}</code> → <code>{meta['parent_table']}</code> "
            f"({', '.join(meta['child_columns'])})</td>"
            f"<td>{meta['orphan_n']}</td><td>{meta['coverage_pct']}</td><td>{meta['ok']}</td></tr>"
        )
    parts.append("</table>")
    if orphans:
        parts.append(f"<h2>Orphan rows ({len(orphans)} relationships)</h2><pre>")
        parts.append(json.dumps(orphans, indent=2))
        parts.append("</pre>")
    else:
        parts.append("<h2>Orphan rows</h2><p>None (all required FKs satisfied).</p>")
    if corr.get("top_correlations_with_LOS_HOURS"):
        parts.append("<h2>Top |r| with LOS_HOURS</h2><table><tr><th>Feature</th><th>r</th></tr>")
        for row in corr["top_correlations_with_LOS_HOURS"][:15]:
            parts.append(f"<tr><td><code>{row['feature']}</code></td><td>{row['pearson_r']:.4f}</td></tr>")
        parts.append("</table>")
    if fig is not None:
        parts.append("<h2>Correlation heatmap</h2>")
        parts.append(fig.to_html(full_html=False, include_plotlyjs="cdn"))
    parts.append("</body></html>")
    return "\n".join(parts)


def run_audit(data_dir: Path) -> dict[str, Any]:
    data_dir = _resolve(data_dir)
    tables = _load_clean_bundle(data_dir)
    inventory = _table_inventory(tables)
    pk = _pk_audit(tables)
    fk, orphans = _fk_coverage(tables)
    wide = _patient_level_wide(tables)
    corr = _los_hours_correlation(wide)
    fig = _corr_heatmap_fig(corr)

    report: dict[str, Any] = {
        "data_dir": str(data_dir),
        "bundle": "synth_cs_data (8 core tables + code_value)",
        "tables": inventory,
        "primary_key_audit_after_cleaning": pk,
        "fk_join_coverage": fk,
        "orphan_rows": orphans,
        "referential_summary": {
            "all_fk_ok": all(v.get("ok", False) for v in fk.values()),
            "orphan_table_count": len(orphans),
        },
        "los_hours_correlation": corr,
    }

    out_json = data_dir / "los_bundle_audit_report.json"
    out_json.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    eda_dir = data_dir / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)
    out_html = eda_dir / "los_bundle_audit_report.html"
    out_html.write_text(_html_report(inventory, pk, fk, orphans, corr, fig), encoding="utf-8")

    report["_outputs"] = {"json": str(out_json), "html": str(out_html)}
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="LOS-only parquet bundle audit.")
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Parquet bundle (default: data/ — 5000-patient cohort; synth_cs_data is 500-patient dev sample)",
    )
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    report = run_audit(args.data_dir)
    html = report["_outputs"]["html"]
    print(f"Wrote {report['_outputs']['json']}")
    print(f"Wrote {html}")
    print(f"all_fk_ok={report['referential_summary']['all_fk_ok']}")

    if not args.no_open:
        p = Path(html)
        if platform.system() == "Darwin":
            subprocess.run(["open", str(p)], check=False)
        else:
            webbrowser.open(p.as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
