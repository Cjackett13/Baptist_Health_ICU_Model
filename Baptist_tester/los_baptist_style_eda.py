#!/usr/bin/env python3
"""
LOS EDA HTML suite mirroring ``baptist_sample_eda.py`` layout (regression target).

Outputs under ``<data-dir>/eda/``:
  - los_eda_report.html
  - los_eda_high_corr_dist.html
  - los_eda_feature_distributions_all.html
  - los_eda_focal_hist_qq.html
  - los_eda_raw_vs_log1p.html
  - los_eda_curated_dist.html
  - los_eda_index.html (updated)
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from los_model_config import LABEL_PRIMARY, LABEL_TRAIN_TRANSFORM, MODEL_FEATURE_COLUMNS, TRANSFORMS

_REPO = Path(__file__).resolve().parent.parent
_STD_NORMAL = NormalDist(0, 1)
CORR_THRESHOLD = 0.35

FOCAL_FEATURES = [
    "med_infusion_mean_12h",
    "med_distinct_12h",
    "proc_n_12h",
    "clinical_event_n_12h",
    "current_scai_12h",
    "scai_prop_ge3_12h",
]

LOG_FEATURES = [c for c, t in TRANSFORMS.items() if t == "log1p" and c in MODEL_FEATURE_COLUMNS]
SQRT_FEATURES = [c for c, t in TRANSFORMS.items() if t == "sqrt" and c in MODEL_FEATURE_COLUMNS]


def _resolve(data_dir: Path) -> Path:
    return data_dir.resolve() if data_dir.is_absolute() else (_REPO / data_dir).resolve()


def _html_page(figures: list[tuple[str, go.Figure]], title: str) -> str:
    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        f"<title>{title}</title></head><body><h1>{title}</h1>",
    ]
    for h, fig in figures:
        parts.append(f"<h2>{h}</h2>")
        parts.append(fig.to_html(full_html=False, include_plotlyjs="cdn"))
    parts.append("</body></html>")
    return "\n".join(parts)


def _binned_mean_los(x: pd.Series, y: pd.Series, *, n_bins: int = 12) -> pd.DataFrame | None:
    xv = pd.to_numeric(x, errors="coerce")
    yv = pd.to_numeric(y, errors="coerce")
    m = xv.notna() & yv.notna()
    xv, yv = xv[m], yv[m]
    if len(xv) < 30:
        return None
    try:
        bins = pd.qcut(xv, q=min(n_bins, 8), duplicates="drop")
    except ValueError:
        bins = pd.cut(xv, bins=min(n_bins, 8))
    rows = []
    for b, sub in pd.DataFrame({"bin": bins, "los": yv}).groupby("bin", observed=True):
        rows.append(
            {
                "mid": float(xv[bins == b].median()),
                "mean_los": float(sub["los"].mean()),
                "n": len(sub),
                "label": str(b),
            }
        )
    return pd.DataFrame(rows) if rows else None


def _corr_heatmap(corr: pd.DataFrame, title: str) -> go.Figure:
    return go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            zmin=-1,
            zmax=1,
            colorscale="RdBu",
        ),
        layout=dict(
            title=title,
            template="plotly_white",
            height=max(480, 26 * len(corr) + 120),
            xaxis=dict(tickangle=-45),
        ),
    )


def _qq_vs_normal(vals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    v = np.sort(vals[np.isfinite(vals)])
    if len(v) < 8:
        return np.array([]), np.array([])
    probs = (np.arange(1, len(v) + 1) - 0.5) / len(v)
    z = np.array([_STD_NORMAL.inv_cdf(p) for p in probs])
    return z, v


def _dist_page(df: pd.DataFrame, features: list[str], title: str) -> go.Figure:
    y = df[LABEL_PRIMARY]
    features = [f for f in features if f in df.columns]
    n = len(features)
    if n == 0:
        return go.Figure(layout=dict(title=f"{title} (no features)"))
    cols = 3
    rows = max(1, int(np.ceil(n / cols)))
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=features)
    for i, feat in enumerate(features):
        r, c = i // cols + 1, i % cols + 1
        bt = _binned_mean_los(df[feat], y)
        if bt is None:
            continue
        fig.add_trace(
            go.Bar(x=bt["mid"], y=bt["mean_los"], name=feat, showlegend=False),
            row=r,
            col=c,
        )
    fig.update_layout(title=title, template="plotly_white", height=280 * rows)
    return fig


def _focal_qq_page(df: pd.DataFrame) -> go.Figure:
    y = pd.to_numeric(df[LABEL_PRIMARY], errors="coerce")
    n = len(FOCAL_FEATURES)
    cols = 2
    rows = int(np.ceil(n / cols))
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=FOCAL_FEATURES)
    for i, feat in enumerate(FOCAL_FEATURES):
        r, c = i // cols + 1, i % cols + 1
        x = pd.to_numeric(df[feat], errors="coerce")
        try:
            tert = pd.qcut(y, 3, labels=["Low LOS", "Mid LOS", "High LOS"])
        except ValueError:
            tert = pd.Series(["All"] * len(y), index=y.index)
        for lab, sub in pd.DataFrame({"x": x, "t": tert}).groupby("t", observed=True):
            z, v = _qq_vs_normal(sub["x"].to_numpy())
            if len(v):
                fig.add_trace(go.Scatter(x=z, y=v, mode="markers", name=str(lab), showlegend=False), row=r, col=c)
    fig.update_layout(title="Focal features — Q–Q vs normal (colored by LOS tertile)", template="plotly_white", height=900)
    return fig


def _raw_vs_log_page(df: pd.DataFrame) -> go.Figure:
  features = [c for c in LOG_FEATURES if c in df.columns]
  if not features:
      return go.Figure(layout=dict(title="No log1p features"))
  n = len(features)
  fig = make_subplots(rows=n, cols=2, subplot_titles=sum([[f"{f} raw", f"{f} log1p"] for f in features], []))
  y = df[LABEL_PRIMARY]
  for i, feat in enumerate(features):
    raw = pd.to_numeric(df[feat], errors="coerce")
    for j, series in enumerate([raw, np.log1p(raw.clip(lower=0))]):
      bt = _binned_mean_los(series, y)
      if bt is None:
          continue
      fig.add_trace(go.Scatter(x=bt["mid"], y=bt["mean_los"], mode="lines+markers", showlegend=False), row=i + 1, col=j + 1)
  fig.update_layout(title="Raw vs log1p binning — mean LOS", template="plotly_white", height=320 * n)
  return fig


def _curated_page(df: pd.DataFrame) -> tuple[go.Figure, go.Figure]:
    guide_rows = [
        {"feature": c, "transform": TRANSFORMS.get(c, "none"), "use_in_model": c in MODEL_FEATURE_COLUMNS}
        for c in sorted(set(MODEL_FEATURE_COLUMNS) | set(TRANSFORMS.keys()))
        if c in df.columns or c in MODEL_FEATURE_COLUMNS
    ]
    tbl = pd.DataFrame(guide_rows)
    fig_table = go.Figure(
        data=[
            go.Table(
                header=dict(values=list(tbl.columns)),
                cells=dict(values=[tbl[c].astype(str).tolist() for c in tbl.columns]),
            )
        ],
        layout=dict(title="LOS feature transform guide (v1 model)"),
    )
    model_feats = [c for c in MODEL_FEATURE_COLUMNS if c in df.columns]
    fig_dist = _dist_page(df, model_feats, "Curated model features — binned mean LOS")
    return fig_table, fig_dist


def run_baptist_style_eda(data_dir: Path, *, df: pd.DataFrame | None = None) -> Path:
    data_dir = _resolve(data_dir)
    eda_dir = data_dir / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)
    if df is None:
        df = pd.read_parquet(data_dir / "los_modeling_frame.parquet")

    num = df[MODEL_FEATURE_COLUMNS + [LABEL_PRIMARY]].apply(pd.to_numeric, errors="coerce")
    corr = num.corr()
    los_corr = corr[LABEL_PRIMARY].drop(LABEL_PRIMARY, errors="ignore").abs().sort_values(ascending=False)

    report_path = eda_dir / "los_eda_report.html"
    report_path.write_text(
        _html_page(
            [
                ("Correlation heatmap", _corr_heatmap(corr, f"Feature correlation vs {LABEL_PRIMARY}")),
                (
                    "Target distribution",
                    go.Figure(
                        data=[
                            go.Histogram(x=df[LABEL_PRIMARY], nbinsx=40, name="raw"),
                            go.Histogram(x=df[LABEL_TRAIN_TRANSFORM], nbinsx=40, name="log1p"),
                        ],
                        layout=dict(title="LOS target distributions", barmode="overlay"),
                    ),
                ),
            ],
            "LOS EDA report",
        ),
        encoding="utf-8",
    )

    high = [f for f in los_corr.index if f in MODEL_FEATURE_COLUMNS and los_corr[f] >= CORR_THRESHOLD]
    if not high:
        high = los_corr.head(6).index.tolist()
    high_path = eda_dir / "los_eda_high_corr_dist.html"
    high_path.write_text(
        _html_page([("High |r| features", _dist_page(df, high, f"|ρ| ≥ {CORR_THRESHOLD}"))], "LOS high correlation"),
        encoding="utf-8",
    )

    all_path = eda_dir / "los_eda_feature_distributions_all.html"
    all_path.write_text(
        _html_page(
            [("All model features", _dist_page(df, MODEL_FEATURE_COLUMNS, "Binned mean LOS"))],
            "LOS all feature distributions",
        ),
        encoding="utf-8",
    )

    focal_path = eda_dir / "los_eda_focal_hist_qq.html"
    focal_path.write_text(
        _html_page(
            [
                ("Binned mean LOS", _dist_page(df, FOCAL_FEATURES, "Focal features")),
                ("Q–Q vs normal", _focal_qq_page(df)),
            ],
            "LOS focal hist & Q–Q",
        ),
        encoding="utf-8",
    )

    rawlog_path = eda_dir / "los_eda_raw_vs_log1p.html"
    rawlog_path.write_text(_html_page([("Raw vs log1p", _raw_vs_log_page(df))], "LOS raw vs log1p"), encoding="utf-8")

    fig_tbl, fig_cur = _curated_page(df)
    curated_path = eda_dir / "los_eda_curated_dist.html"
    curated_path.write_text(
        _html_page([("Transform guide", fig_tbl), ("Distributions", fig_cur)], "LOS curated distributions"),
        encoding="utf-8",
    )

    sections = [
        ("Main report", "los_eda_report.html"),
        ("High |r| distributions", "los_eda_high_corr_dist.html"),
        ("All feature distributions", "los_eda_feature_distributions_all.html"),
        ("Focal hist & Q–Q", "los_eda_focal_hist_qq.html"),
        ("Raw vs log1p", "los_eda_raw_vs_log1p.html"),
        ("Curated + transform guide", "los_eda_curated_dist.html"),
        ("Full EDA index (extended)", "los_eda_index.html"),
    ]
    index_path = eda_dir / "los_eda_index.html"
    lis = "".join(f'<li><a href="{href}">{name}</a></li>' for name, href in sections)
    index_path.write_text(
        f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>LOS EDA</title></head>"
        f"<body><h1>LOS EDA (baptist_sample_eda layout)</h1><ul>{lis}</ul></body></html>",
        encoding="utf-8",
    )
    audit = {
        "mirror_of": "baptist_sample_eda.py",
        "outputs": [f"eda/{h}" for _, h in sections],
        "corr_threshold_high": CORR_THRESHOLD,
    }
    (eda_dir / "los_eda_baptist_mirror_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return index_path
