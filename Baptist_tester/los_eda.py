#!/usr/bin/env python3
"""
Plotly EDA for length-of-stay regression on ``los_patient_frame.parquet``.

  python3 Baptist_tester/los_eda.py
  python3 Baptist_tester/los_eda.py --data-dir data --no-open

Outputs under ``<data-dir>/eda/``:
  - ``los_eda_report.html`` — target distribution, correlation heatmap, truncation summary
  - ``los_eda_binned_mean.html`` — binned mean LOS by numeric feature
  - ``los_eda_missingness.html`` — feature missingness bar chart
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_REPO = Path(__file__).resolve().parent.parent


def _resolve(p: Path) -> Path:
    return p.resolve() if p.is_absolute() else (_REPO / p).resolve()


def _load_prep():
    path = _REPO / "Baptist_tester" / "los_data_prep.py"
    spec = importlib.util.spec_from_file_location("los_data_prep", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _numeric_matrix(df: pd.DataFrame, cols: list[str], target: str) -> pd.DataFrame:
    use = [c for c in cols if c in df.columns]
    if target in df.columns:
        use = use + [target]
    num = df[use].apply(pd.to_numeric, errors="coerce")
    num = num.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
    std = num.std(numeric_only=True)
    return num.loc[:, std > 1e-12]


def _corr_heatmap(corr: pd.DataFrame, title: str) -> go.Figure:
    los_last = [c for c in corr.columns if c != "los_hours"] + (
        ["los_hours"] if "los_hours" in corr.columns else []
    )
    los_last = [c for c in los_last if c in corr.columns]
    c = corr.loc[los_last, los_last]
    return go.Figure(
        data=go.Heatmap(
            z=c.values,
            x=c.columns.tolist(),
            y=c.index.tolist(),
            zmin=-1,
            zmax=1,
            colorscale="RdBu",
            hovertemplate="%{y} vs %{x}<br>r=%{z:.3f}<extra></extra>",
        ),
        layout=dict(
            title=title,
            template="plotly_white",
            height=max(420, 28 * len(c) + 120),
            width=max(520, 22 * len(c) + 160),
            xaxis=dict(tickangle=-45),
            margin=dict(l=120, r=40, t=80, b=120),
        ),
    )


def _binned_mean_los(x: pd.Series, y: pd.Series, *, n_bins: int = 12, min_n: int = 8) -> pd.DataFrame | None:
    xv = pd.to_numeric(x, errors="coerce")
    yv = pd.to_numeric(y, errors="coerce")
    mask = xv.notna() & yv.notna()
    xv, yv = xv[mask], yv[mask]
    if len(xv) < 2 * min_n:
        return None
    try:
        bins = pd.qcut(xv, q=min(n_bins, max(2, len(xv) // min_n)), duplicates="drop")
    except ValueError:
        bins = pd.cut(xv, bins=min(n_bins, 8))
    g = pd.DataFrame({"bin": bins, "los": yv}).groupby("bin", observed=True)
    rows = []
    for b, sub in g:
        if len(sub) < min_n:
            continue
        rows.append(
            {
                "label": str(b),
                "mid": float(sub["los"].median()) if hasattr(b, "mid") else float(np.median(xv[bins == b])),
                "mean_los": float(sub["los"].mean()),
                "median_los": float(sub["los"].median()),
                "n": int(len(sub)),
                "x_lo": float(xv[bins == b].min()),
                "x_hi": float(xv[bins == b].max()),
            }
        )
    return pd.DataFrame(rows) if rows else None


def _target_distribution(df: pd.DataFrame) -> go.Figure:
    los = pd.to_numeric(df["los_hours"], errors="coerce").dropna()
    loglos = np.log1p(los)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("LOS (hours)", "log₁p(LOS hours)"))
    fig.add_trace(go.Histogram(x=los, nbinsx=30, name="raw"), row=1, col=1)
    fig.add_trace(go.Histogram(x=loglos, nbinsx=30, name="log1p"), row=1, col=2)
    fig.update_layout(
        title="Target distribution (included cohort)",
        template="plotly_white",
        showlegend=False,
        height=380,
    )
    return fig


def _truncation_summary(df: pd.DataFrame) -> go.Figure:
    flags = ["truncated_death", "truncated_hospice", "los_truncated"]
    rows = []
    for f in flags:
        if f not in df.columns:
            continue
        for v in (0, 1):
            sub = df[df[f].astype(int) == v]
            if sub.empty:
                continue
            rows.append(
                {
                    "flag": f,
                    "value": "yes" if v else "no",
                    "n": len(sub),
                    "mean_los_h": float(sub["los_hours"].mean()),
                }
            )
    d = pd.DataFrame(rows)
    fig = go.Figure(
        data=go.Bar(
            x=d["flag"] + " = " + d["value"],
            y=d["mean_los_h"],
            text=d["n"].astype(str) + " pts",
            textposition="outside",
        ),
        layout=dict(
            title="Mean LOS by truncation flag",
            yaxis_title="Mean los_hours",
            template="plotly_white",
            height=400,
        ),
    )
    return fig


def _binned_mean_figure(df: pd.DataFrame, features: list[str], target: str) -> go.Figure:
    n = len(features)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=features)
    y = df[target]
    for i, feat in enumerate(features):
        r, c = i // cols + 1, i % cols + 1
        bt = _binned_mean_los(df[feat], y)
        if bt is None or bt.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=bt["mid"],
                y=bt["mean_los"],
                mode="lines+markers",
                name=feat,
                showlegend=False,
                hovertext=bt["label"],
            ),
            row=r,
            col=c,
        )
        fig.update_xaxes(title_text=feat, row=r, col=c)
        fig.update_yaxes(title_text="mean LOS (h)", row=r, col=c)
    fig.update_layout(
        title="Binned mean LOS by feature (quantile bins)",
        template="plotly_white",
        height=280 * rows,
    )
    return fig


def _missingness_figure(df: pd.DataFrame, cols: list[str]) -> go.Figure:
    rates = {c: float(df[c].isna().mean()) for c in cols if c in df.columns}
    s = pd.Series(rates).sort_values(ascending=True)
    fig = go.Figure(
        go.Bar(x=s.values * 100, y=s.index, orientation="h"),
        layout=dict(
            title="Feature missingness (%)",
            xaxis_title="% missing",
            template="plotly_white",
            height=max(320, 24 * len(s) + 80),
        ),
    )
    return fig


def _html_page(figures: list[tuple[str, go.Figure]], title: str) -> str:
    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        f"<title>{title}</title></head><body>",
        f"<h1>{title}</h1>",
    ]
    for heading, fig in figures:
        parts.append(f"<h2>{heading}</h2>")
        parts.append(fig.to_html(full_html=False, include_plotlyjs="cdn"))
    parts.append("</body></html>")
    return "\n".join(parts)


def _open_files(paths: list[Path]) -> None:
    if platform.system() == "Darwin":
        for i, p in enumerate(paths):
            if i:
                time.sleep(0.4)
            subprocess.run(["open", str(p)], check=False)
    else:
        for p in paths:
            webbrowser.open(p.as_uri())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Parquet bundle (default data/ = 5000 patients)",
    )
    ap.add_argument("--force-prep", action="store_true", help="Re-run los_data_prep first")
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    data_dir = _resolve(args.data_dir)
    eda_dir = data_dir / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)

    prep = _load_prep()
    frame_path = data_dir / prep.PATIENT_FRAME_NAME
    if args.force_prep or not frame_path.is_file():
        prep.run_prep(data_dir, force=args.force_prep)

    df = pd.read_parquet(frame_path)
    feat_doc = json.loads((data_dir / prep.FEATURE_COLUMNS_NAME).read_text(encoding="utf-8"))
    features = feat_doc.get("feature_columns_v1_snapshot") or feat_doc.get("feature_columns", [])
    target = feat_doc.get("label_columns", ["los_hours_total"])[0]
    if target not in df.columns and "los_hours" in df.columns:
        target = "los_hours"

    num = _numeric_matrix(df, features, target)
    corr = num.corr(numeric_only=True)

    fig_target = _target_distribution(df)
    fig_trunc = _truncation_summary(df)
    fig_corr = _corr_heatmap(corr, "Pearson correlation (numeric features + los_hours)")
    fig_bins = _binned_mean_figure(df, features, target)
    fig_miss = _missingness_figure(df, features + ["los_hours", "log1p_los_hours"])

    report_path = eda_dir / "los_eda_report.html"
    report_path.write_text(
        _html_page(
            [
                ("Target", fig_target),
                ("Truncation flags", fig_trunc),
                ("Correlations", fig_corr),
            ],
            "LOS EDA report",
        ),
        encoding="utf-8",
    )
    bins_path = eda_dir / "los_eda_binned_mean.html"
    bins_path.write_text(
        _html_page([("Binned mean LOS", fig_bins)], "LOS binned mean"),
        encoding="utf-8",
    )
    miss_path = eda_dir / "los_eda_missingness.html"
    miss_path.write_text(
        _html_page([("Missingness", fig_miss)], "LOS missingness"),
        encoding="utf-8",
    )

    audit = {
        "n_patients": int(len(df)),
        "los_hours_mean": float(df["los_hours"].mean()),
        "los_hours_median": float(df["los_hours"].median()),
        "los_truncated_n": int(df["los_truncated"].sum()),
        "top_corr_with_los": (
            corr["los_hours"].drop("los_hours", errors="ignore").abs().sort_values(ascending=False).head(8).to_dict()
            if "los_hours" in corr.columns
            else {}
        ),
        "outputs": [
            str(report_path.relative_to(data_dir)),
            str(bins_path.relative_to(data_dir)),
            str(miss_path.relative_to(data_dir)),
        ],
    }
    (eda_dir / "los_eda_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    print(f"Patients: {len(df)} | truncated: {int(df['los_truncated'].sum())}")
    print(f"Report: {report_path}")
    print(f"Binned mean: {bins_path}")
    print(f"Missingness: {miss_path}")

    if not args.no_open:
        _open_files([report_path, bins_path, miss_path])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
