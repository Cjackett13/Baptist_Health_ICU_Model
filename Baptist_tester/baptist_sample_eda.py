#!/usr/bin/env python3
"""
Baptist sample EDA — in-hospital mortality as the target.

- Per-table Pearson correlation matrices (numeric patient-level aggregates + ``died``).
- Summary charts (class balance, cross-table top correlations).
- **Recommended transformations** table (log / sqrt / identity / ordinal) aligned with focal diagnostics.
- Binned ``P(died)`` uses that guidance (e.g. **log₁p** or **√** scale for bin edges; hover shows **raw** value range).
- **Raw vs log₁p** page for log-recommended columns: equal-width bins on raw ``x`` vs ``log₁p(x)`` side by side.
- By default, **extreme values are masked** (1–99th percentile of raw ``x``) for mortality binning and focal Q–Q only; use ``--eda-outliers none`` to disable.

Data: clean parquet bundle under ``Baptist_tester/synth_cs_data`` (default).

  pip install plotly pandas pyarrow numpy
  python3 Baptist_tester/baptist_sample_eda.py
  python3 Baptist_tester/baptist_sample_eda.py --eda-outliers none
  python3 Baptist_tester/baptist_sample_eda.py --eda-outliers iqr --eda-iqr-k 1.5

Outputs (default ``<data-dir>/eda/``):
  - ``baptist_sample_eda_report.html`` — correlation heatmaps + embedded high-|r| distributions
  - ``baptist_sample_eda_high_corr_dist.html`` — distributions for |r|≥threshold only
  - ``baptist_sample_eda_feature_distributions_all.html`` — all numeric attributes (binned P(died), transforms when listed)
  - ``baptist_sample_eda_focal_hist_qq.html`` — focal vars: binned ``P(died)`` + stratified Q–Q (transformed)
  - ``baptist_sample_eda_raw_vs_log1p.html`` — raw vs log₁p binning for log-recommended features

Opens HTML files in the default browser (macOS: ``open`` per file with a short delay; report + standalone pages).
"""

from __future__ import annotations

import html
import argparse
import platform
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_REPO = Path(__file__).resolve().parent.parent
CORR_THRESHOLD = 0.65
_STD_NORMAL = NormalDist(0, 1)


@dataclass(frozen=True)
class EdaOutlierTrim:
    """Mask extreme feature values before binning / Q–Q (visualization only; parquet data unchanged)."""

    mode: str = "percentile"  # none | percentile | iqr
    pct_lo: float = 1.0
    pct_hi: float = 99.0
    iqr_k: float = 1.5


def _eda_trim_caption(t: EdaOutlierTrim) -> str:
    if t.mode == "none":
        return ""
    if t.mode == "percentile":
        return f" — viz trim: {t.pct_lo:g}–{t.pct_hi:g} %ile of raw x"
    return f" — viz trim: IQR fence ×{t.iqr_k:g}"


def _apply_eda_trim(s: pd.Series, t: EdaOutlierTrim) -> pd.Series:
    """Set values outside fences to NaN (dropped from bins and Q–Q for that feature)."""
    if t.mode == "none":
        return s
    s = pd.to_numeric(s, errors="coerce")
    v = s.dropna()
    if len(v) < 40:
        return s
    arr = v.to_numpy(dtype=float)
    if t.mode == "percentile":
        lo, hi = float(np.percentile(arr, t.pct_lo)), float(np.percentile(arr, t.pct_hi))
        if not (np.isfinite(lo) and np.isfinite(hi)) or lo > hi:
            return s
        return s.where((s >= lo) & (s <= hi))
    if t.mode == "iqr":
        q1, q3 = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
        iqr = q3 - q1
        if iqr <= 1e-12:
            return s
        lo, hi = q1 - t.iqr_k * iqr, q3 + t.iqr_k * iqr
        return s.where((s >= lo) & (s <= hi))
    return s

# Pooled ``wide`` columns (``table__feature``) for focal histograms + Q–Q plots.
FOCAL_TABLE_FEATURES: tuple[tuple[str, str], ...] = (
    ("medication_admin", "med_distinct"),
    ("medication_admin", "med_infusion_mean"),
    ("procedure_event", "proc_n"),
    ("procedure_event", "proc_distinct_nom"),
    ("scai_stage_hourly", "scai_std"),
    ("scai_stage_hourly", "current_scai"),
    ("scai_stage_hourly", "scai_prop_ge3"),
)

# Pooled-column guidance: EDA copy + ``plot`` mode for binning / QQ (except ordinal QQ stays raw).
FEATURE_TRANSFORM_GUIDE: tuple[dict[str, str], ...] = (
    {
        "key": "medication_admin__med_infusion_mean",
        "emoji": "🟠",
        "tier": "Sqrt or Box–Cox",
        "detail": "Moderate skew; values often in a modest positive range (e.g. 0–2.5).",
        "plot": "sqrt",
        "plot_note": "Bins on √x; hover shows raw interval",
    },
    {
        "key": "procedure_event__proc_n",
        "emoji": "🟠",
        "tier": "Sqrt or Box–Cox",
        "detail": "Discrete count; square root is a simple stabilizer.",
        "plot": "sqrt",
        "plot_note": "Bins on √x; hover shows raw interval",
    },
    {
        "key": "medication_admin__med_distinct",
        "emoji": "🟠",
        "tier": "Sqrt or Box–Cox",
        "detail": "Small discrete counts (e.g. 2–6); sqrt often helps.",
        "plot": "sqrt",
        "plot_note": "Bins on √x; hover shows raw interval",
    },
    {
        "key": "scai_stage_hourly__scai_std",
        "emoji": "🟡",
        "tier": "Likely fine",
        "detail": "Small positive range (e.g. 0–1.5); check tails but often OK raw.",
        "plot": "identity",
        "plot_note": "Equal-width bins on raw x",
    },
    {
        "key": "scai_stage_hourly__current_scai",
        "emoji": "🟡",
        "tier": "Likely fine",
        "detail": "Latest hourly SCAI_STAGE_NUM (0–4); lower is less severe in this bundle.",
        "plot": "identity",
        "plot_note": "Equal-width bins on raw x",
    },
    {
        "key": "scai_stage_hourly__scai_prop_ge3",
        "emoji": "🟡",
        "tier": "Likely fine (bounded proportion)",
        "detail": "Already on (0–1). For linear models consider a logit link / logit-transformed feature.",
        "plot": "identity",
        "plot_note": "Raw bins; consider logit if using linear/additive models",
    },
    {
        "key": "procedure_event__proc_distinct_nom",
        "emoji": "⚪",
        "tier": "Investigate further",
        "detail": "Very small integer range (e.g. 1–4); treat as ordinal/categorical rather than continuous.",
        "plot": "ordinal",
        "plot_note": "One P(died) per distinct value (ordinal)",
    },
)

TRANSFORM_PLOT: dict[str, str] = {row["key"]: row["plot"] for row in FEATURE_TRANSFORM_GUIDE}


def _transform_recommendations_html() -> str:
    """Static HTML table from ``FEATURE_TRANSFORM_GUIDE``."""
    parts = [
        "<hr><h2>Recommended transformations by feature</h2>",
        "<table border='1' cellpadding='6' cellspacing='0' "
        "style='border-collapse:collapse;font-size:14px;max-width:1100px'>",
        "<thead><tr><th>Tier</th><th>Feature</th><th>Notes</th><th>Applied in binned P(died) figures</th></tr></thead><tbody>",
    ]
    for row in FEATURE_TRANSFORM_GUIDE:
        parts.append(
            "<tr>"
            f"<td>{row['emoji']} {html.escape(row['tier'])}</td>"
            f"<td><code>{html.escape(row['key'])}</code></td>"
            f"<td>{html.escape(row['detail'])}</td>"
            f"<td>{html.escape(row['plot_note'])}</td>"
            "</tr>"
        )
    parts.append("</tbody></table>")
    return "\n".join(parts)


def _binning_subtitle(key: str) -> str:
    m = TRANSFORM_PLOT.get(key, "identity")
    if m == "log1p":
        return "bins: log₁p(x)"
    if m == "sqrt":
        return "bins: √x"
    if m == "ordinal":
        return "ordinal (by value)"
    return ""


def _binned_mortality_ordinal(x: pd.Series, died: pd.Series, *, min_bin_n: int = 5) -> pd.DataFrame | None:
    """One row per rounded integer level with enough patients."""
    died_f = died.astype(float)
    mask = x.notna() & died_f.notna()
    xv = pd.to_numeric(x[mask], errors="coerce")
    yv = died_f[mask].astype(int)
    if len(xv) < max(2 * min_bin_n, 10):
        return None
    kk = np.round(xv.to_numpy(dtype=float)).astype(int)
    df = pd.DataFrame({"k": kk, "d": yv.to_numpy(dtype=int)})
    g = df.groupby("k", sort=True).agg(n=("d", "size"), deaths=("d", "sum"))
    g = g[g["n"] >= min_bin_n].copy()
    if g.empty:
        return None
    rows = []
    for k, row in g.iterrows():
        n_i = int(row["n"])
        d_i = int(row["deaths"])
        p = d_i / n_i if n_i else 0.0
        fk = float(k)
        rows.append(
            {
                "left": fk - 0.42,
                "right": fk + 0.42,
                "mid": fk,
                "n": n_i,
                "p_died": float(p),
                "raw_left": fk,
                "raw_right": fk,
            }
        )
    return pd.DataFrame(rows)


def _binned_mortality_for_key(
    key: str,
    raw: pd.Series,
    died: pd.Series,
    *,
    n_bins: int = 16,
    min_bin_n: int = 5,
    force_mode: str | None = None,
) -> tuple[pd.DataFrame | None, str]:
    """Binned mortality; ``force_mode`` overrides ``TRANSFORM_PLOT`` for raw-vs-transform comparisons."""
    mode = force_mode if force_mode is not None else TRANSFORM_PLOT.get(key, "identity")
    raw = pd.to_numeric(raw, errors="coerce").replace([np.inf, -np.inf], np.nan)

    if mode == "ordinal":
        bt = _binned_mortality_ordinal(raw, died, min_bin_n=min_bin_n)
        return bt, "distinct value (rounded)"

    if mode == "log1p":
        xb = pd.Series(np.log1p(np.clip(raw.to_numpy(dtype=float), 0.0, None)), index=raw.index)
        xlab = "log₁p(feature) — hover shows raw range"
    elif mode == "sqrt":
        xb = pd.Series(np.sqrt(np.clip(raw.to_numpy(dtype=float), 0.0, None)), index=raw.index)
        xlab = "√feature — hover shows raw range"
    else:
        xb = raw
        xlab = "raw feature value"

    bt = _binned_mortality_table(xb, died, n_bins=n_bins, min_bin_n=min_bin_n)
    if bt is None or bt.empty:
        return None, xlab
    if mode == "log1p":
        bt = bt.assign(
            raw_left=np.expm1(bt["left"].to_numpy(dtype=float)),
            raw_right=np.expm1(bt["right"].to_numpy(dtype=float)),
        )
    elif mode == "sqrt":
        bt = bt.assign(
            raw_left=np.square(bt["left"].to_numpy(dtype=float)),
            raw_right=np.square(bt["right"].to_numpy(dtype=float)),
        )
    else:
        bt = bt.assign(raw_left=bt["left"], raw_right=bt["right"])
    return bt, xlab


def _series_for_qq(key: str, raw: pd.Series) -> np.ndarray:
    """Match focal binning transform for QQ (ordinal / identity uses raw)."""
    mode = TRANSFORM_PLOT.get(key, "identity")
    raw = pd.to_numeric(raw, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if raw.empty:
        return np.array([])
    xv = raw.to_numpy(dtype=float)
    if mode == "log1p":
        return np.log1p(np.clip(xv, 0.0, None))
    if mode == "sqrt":
        return np.sqrt(np.clip(xv, 0.0, None))
    return xv


def _resolve(p: Path) -> Path:
    return p.resolve() if p.is_absolute() else (_REPO / p).resolve()


def _numeric_matrix(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include=[np.number]).copy()
    for c in num.columns:
        num[c] = pd.to_numeric(num[c], errors="coerce")
    num = num.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
    std = num.std(numeric_only=True)
    num = num.loc[:, std > 1e-12]
    return num


def _corr_heatmap(corr: pd.DataFrame, title: str) -> go.Figure:
    died_last = [c for c in corr.columns if c != "died"] + (["died"] if "died" in corr.columns else [])
    died_last = [c for c in died_last if c in corr.columns]
    C = corr.loc[died_last, died_last]
    return go.Figure(
        data=go.Heatmap(
            z=C.values,
            x=C.columns.tolist(),
            y=C.index.tolist(),
            zmin=-1,
            zmax=1,
            colorscale="RdBu",
            hovertemplate="%{y} vs %{x}<br>r=%{z:.3f}<extra></extra>",
        ),
        layout=dict(
            title=title,
            template="plotly_white",
            height=max(420, 28 * len(C) + 120),
            width=max(520, 22 * len(C) + 160),
            xaxis=dict(tickangle=-45),
            margin=dict(l=120, r=40, t=80, b=120),
        ),
    )


def _load_base(data_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    person = pd.read_parquet(data_dir / "person.parquet")
    enc = pd.read_parquet(data_dir / "encounter.parquet")
    base = person.merge(enc, on="PERSON_ID", how="left", suffixes=("_person", ""))
    died = base["DECEASED_DT_TM"].notna().astype(int)
    died.name = "died"
    return base, died


def build_table_frames(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Patient-level numeric feature sets per source table (+ died column)."""
    base, died = _load_base(data_dir)
    died_s = died.copy()
    died_s.index = base["PERSON_ID"].values
    died_s = died_s.groupby(level=0).first()

    out: dict[str, pd.DataFrame] = {}

    # --- person + encounter (one row per patient in bundle) ---
    pe = base.set_index("PERSON_ID").copy()
    pe["died"] = died_s.reindex(pe.index).fillna(0).astype(int)
    drop_cols = {
        "NAME_LAST_TXT",
        "NAME_FIRST_TXT",
        "BIRTH_DT_TM",
        "DECEASED_DT_TM",
        "REG_DT_TM",
        "DISCH_DT_TM",
    }
    for c in list(pe.columns):
        if c in drop_cols or pe[c].dtype == "datetime64[ns]" or pe[c].dtype == "datetimetz":
            pe = pe.drop(columns=[c], errors="ignore")
    out["person_encounter"] = _numeric_matrix(pe.reset_index())

    # --- medication_admin ---
    med = pd.read_parquet(data_dir / "medication_admin.parquet")
    med["INFUSION_RATE"] = pd.to_numeric(med.get("INFUSION_RATE"), errors="coerce")
    g = med.groupby("PERSON_ID").agg(
        med_distinct=("MEDICATION_CD", pd.Series.nunique),
        med_infusion_mean=("INFUSION_RATE", "mean"),
    )
    g = g.join(died_s, how="left").fillna({"died": 0})
    g["died"] = g["died"].astype(int)
    out["medication_admin"] = _numeric_matrix(g.reset_index())

    # --- procedure_event ---
    proc = pd.read_parquet(data_dir / "procedure_event.parquet")
    g = proc.groupby("PERSON_ID").agg(
        proc_n=("PROCEDURE_ID", "count"),
        proc_distinct_nom=("NOMENCLATURE_CD", pd.Series.nunique),
    )
    g = g.join(died_s, how="left").fillna({"died": 0})
    g["died"] = g["died"].astype(int)
    out["procedure_event"] = _numeric_matrix(g.reset_index())

    # --- diagnosis ---
    dx = pd.read_parquet(data_dir / "diagnosis.parquet")
    dx["DIAG_PRIORITY"] = pd.to_numeric(dx.get("DIAG_PRIORITY"), errors="coerce")
    g = dx.groupby("PERSON_ID").agg(
        dx_n=("DIAGNOSIS_ID", "count"),
        dx_distinct_nom=("NOMENCLATURE_CD", pd.Series.nunique),
        dx_priority_mean=("DIAG_PRIORITY", "mean"),
    )
    g = g.join(died_s, how="left").fillna({"died": 0})
    g["died"] = g["died"].astype(int)
    out["diagnosis"] = _numeric_matrix(g.reset_index())

    # --- scai_stage_hourly (via encounter → person) ---
    sc = pd.read_parquet(data_dir / "scai_stage_hourly.parquet")
    enc = pd.read_parquet(data_dir / "encounter.parquet")[["ENCOUNTER_ID", "PERSON_ID"]]
    sc = sc.merge(enc, on="ENCOUNTER_ID", how="left")
    sc["_ge3"] = (pd.to_numeric(sc["SCAI_STAGE_NUM"], errors="coerce") >= 3).astype(float)
    sc_sorted = sc.sort_values(
        ["PERSON_ID", "EVENT_DT_TM", "HOUR_FROM_ADMIT"],
        ascending=[True, True, True],
        na_position="last",
    )
    current_scai = sc_sorted.groupby("PERSON_ID", sort=False)["SCAI_STAGE_NUM"].last()
    g = sc.groupby("PERSON_ID").agg(
        scai_std=("SCAI_STAGE_NUM", "std"),
        scai_prop_ge3=("_ge3", "mean"),
    )
    g["current_scai"] = pd.to_numeric(current_scai.reindex(g.index), errors="coerce")
    g = g.join(died_s, how="left").fillna({"died": 0})
    g["died"] = g["died"].astype(int)
    out["scai_stage_hourly"] = _numeric_matrix(g.reset_index())

    # code_value: reference only — skip
    return {k: v for k, v in out.items() if v.shape[1] >= 2}


def pooled_patient_frame(base: pd.DataFrame, table_frames: dict[str, pd.DataFrame], died_s: pd.Series) -> pd.DataFrame:
    """One row per patient; columns ``table__feature`` merged on ``PERSON_ID``."""
    wide = base[["PERSON_ID"]].copy()
    for name, df in table_frames.items():
        dfc = df.drop(columns=["died"], errors="ignore").copy()
        if "PERSON_ID" not in dfc.columns:
            continue
        rename = {c: f"{name}__{c}" for c in dfc.columns if c != "PERSON_ID"}
        dfc = dfc.rename(columns=rename)
        wide = wide.merge(dfc, on="PERSON_ID", how="left")
    wide["died"] = wide["PERSON_ID"].map(died_s).fillna(0).astype(int)
    return wide


def _norm_ppf(p: np.ndarray) -> np.ndarray:
    """Standard normal inverse CDF (stdlib only; works without ``numpy.erfinv`` / ``math.erfinv``)."""
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1.0 - 1e-12)
    flat = p.ravel()
    out = np.fromiter((_STD_NORMAL.inv_cdf(float(x)) for x in flat), dtype=np.float64, count=flat.size)
    return out.reshape(p.shape)


def _wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    den = 1.0 + z * z / n
    c = (p + z * z / (2.0 * n)) / den
    rad = z * np.sqrt(max(0.0, p * (1.0 - p) / n + z * z / (4.0 * n * n))) / den
    return max(0.0, c - rad), min(1.0, c + rad)


def _binned_mortality_table(
    x: pd.Series,
    died: pd.Series,
    *,
    n_bins: int = 16,
    min_bin_n: int = 5,
) -> pd.DataFrame | None:
    """Equal-width bins on ``x``; columns ``left``, ``right``, ``mid``, ``n``, ``p_died``."""
    died = died.astype(float)
    mask = x.notna() & died.notna()
    xv = x[mask].astype(float)
    yv = died[mask].astype(int)
    if len(xv) < max(2 * min_bin_n, 20):
        return None
    lo, hi = float(xv.min()), float(xv.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None
    nb = int(max(5, min(n_bins, len(xv) // max(min_bin_n, 1))))
    try:
        cat = pd.cut(xv, bins=nb, include_lowest=True, duplicates="drop")
    except (ValueError, TypeError):
        return None
    g = yv.groupby(cat, observed=False).agg(n="size", deaths="sum")
    g = g[g["n"] >= min_bin_n].copy()
    if g.empty:
        return None
    rows = []
    for iv, row in g.iterrows():
        n_i = int(row["n"])
        d_i = int(row["deaths"])
        p = d_i / n_i if n_i else 0.0
        rows.append(
            {
                "left": float(iv.left),
                "right": float(iv.right),
                "mid": float((iv.left + iv.right) / 2.0),
                "n": n_i,
                "p_died": float(p),
            }
        )
    return pd.DataFrame(rows).sort_values("mid")


def _death_likelihood_bar_trace(
    bt: pd.DataFrame,
    *,
    overall_p: float,
    show_colorbar: bool,
) -> tuple[go.Bar, go.Scatter]:
    """Bar heights = ``P(died)`` in bin; error bars = Wilson 95%% CI; dashed line = cohort death rate."""
    lo_ci, hi_ci = zip(*(_wilson_ci(float(p), int(n)) for p, n in zip(bt["p_died"], bt["n"])))
    err_hi = [h - p for h, p in zip(hi_ci, bt["p_died"])]
    err_lo = [p - l for l, p in zip(lo_ci, bt["p_died"])]
    widths = (bt["right"] - bt["left"]).to_numpy(dtype=float) * 0.88
    if "raw_left" in bt.columns and "raw_right" in bt.columns:
        hover = [
            (
                f"raw ≈ {rl:.4g}" if abs(float(rr) - float(rl)) < 1e-6 else f"raw ∈ [{rl:.4g}, {rr:.4g}]"
            )
            + f"<br>P(died)={p:.1%} &nbsp; n={n}<extra></extra>"
            for rl, rr, p, n in zip(bt["raw_left"], bt["raw_right"], bt["p_died"], bt["n"])
        ]
    else:
        hover = [
            f"[{L:.4g}, {R:.4g}]<br>P(died)={p:.1%} &nbsp; n={n}<extra></extra>"
            for L, R, p, n in zip(bt["left"], bt["right"], bt["p_died"], bt["n"])
        ]
    bar = go.Bar(
        x=bt["mid"],
        y=bt["p_died"],
        width=widths,
        marker=dict(
            color=bt["p_died"],
            colorscale="Reds",
            cmin=0.0,
            cmax=1.0,
            line=dict(color="rgba(0,0,0,0.35)", width=0.5),
            showscale=show_colorbar,
            **(
                {"colorbar": dict(title="P(died)", tickformat=".0%")}
                if show_colorbar
                else {}
            ),
        ),
        error_y=dict(
            type="data",
            symmetric=False,
            array=err_hi,
            arrayminus=err_lo,
            thickness=1.2,
            width=4,
            color="rgba(30,30,30,0.55)",
        ),
        hovertext=hover,
        hoverinfo="text",
        showlegend=False,
    )
    x_line = np.array([float(bt["left"].min()), float(bt["right"].max())], dtype=float)
    ref = go.Scatter(
        x=x_line,
        y=np.full(2, overall_p),
        mode="lines",
        line=dict(color="#64748b", width=1.5, dash="dash"),
        hovertemplate=f"cohort P(died)={overall_p:.1%}<extra></extra>",
        showlegend=False,
    )
    return bar, ref


def _qq_sorted_vs_theoretical(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Blom plotting positions vs N(0,1) quantiles; ``y`` = sorted raw ``x``."""
    x = x[np.isfinite(x)]
    if x.size < 3:
        return np.array([]), np.array([])
    ys = np.sort(x.astype(float))
    n = ys.size
    i = np.arange(1, n + 1, dtype=float)
    p = (i - 0.375) / (n + 0.25)
    theo = _norm_ppf(p)
    return theo, ys


def focal_histogram_figure(
    wide: pd.DataFrame,
    keys: list[str],
    ncols: int = 3,
    *,
    trim: EdaOutlierTrim = EdaOutlierTrim(),
) -> go.Figure | None:
    """Binned empirical ``P(died)`` on the scale from ``TRANSFORM_PLOT`` (focal columns)."""
    if not keys or wide.empty or "died" not in wide.columns:
        return None
    died = wide["died"].astype(int)
    overall = float(died.mean())
    nrows = int(np.ceil(len(keys) / ncols))
    n_slots = nrows * ncols
    titles = []
    for k in keys:
        sub = _binning_subtitle(k)
        titles.append(f"{k}<br><sup style='font-size:10px'>{sub}</sup>" if sub else k)
    titles.extend([""] * (n_slots - len(keys)))
    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=titles,
        vertical_spacing=0.10,
        horizontal_spacing=0.06,
    )
    colorbar_done = False
    for idx, key in enumerate(keys):
        if key not in wide.columns:
            continue
        row = idx // ncols + 1
        cc = idx % ncols + 1
        s = _apply_eda_trim(
            pd.to_numeric(wide[key], errors="coerce").replace([np.inf, -np.inf], np.nan),
            trim,
        )
        bt, xlab = _binned_mortality_for_key(key, s, died)
        if bt is None or bt.empty:
            continue
        bar, ref = _death_likelihood_bar_trace(
            bt,
            overall_p=overall,
            show_colorbar=not colorbar_done,
        )
        colorbar_done = True
        fig.add_trace(bar, row=row, col=cc)
        fig.add_trace(ref, row=row, col=cc)
        fig.update_xaxes(title_text=xlab, row=row, col=cc)
    if not fig.data:
        return None
    fig.update_layout(
        title_text=(
            "Focal features — empirical P(in-hospital death) "
            "(binning follows recommended transforms where set; Wilson 95% CI; dashed = cohort rate)"
            + _eda_trim_caption(trim)
        ),
        template="plotly_white",
        barmode="overlay",
        height=min(3600, 200 + nrows * 240),
        width=min(1500, 100 + ncols * 320),
        showlegend=False,
        margin=dict(t=100, l=50, r=60, b=40),
    )
    fig.update_yaxes(title_text="P(died)", tickformat=".0%")
    return fig


def _qq_plot_suffix(key: str) -> str:
    m = TRANSFORM_PLOT.get(key, "identity")
    if m == "log1p":
        return "log₁p(x)"
    if m == "sqrt":
        return "√x"
    if m == "ordinal":
        return "raw (ordinal)"
    return "raw x"


def focal_qq_figure(
    wide: pd.DataFrame,
    keys: list[str],
    ncols: int = 3,
    *,
    trim: EdaOutlierTrim = EdaOutlierTrim(),
) -> go.Figure | None:
    """Stratified normal Q–Q on the **same transformed** scale as binned P(died) plots.

    Survived vs died appear as **two line traces** (ordered transform vs N(0,1) quantiles),
    highlighting how each outcome’s distribution departs from a normal reference (risk context).
    """
    if not keys or wide.empty or "died" not in wide.columns:
        return None
    y = wide["died"].astype(int)
    nrows = int(np.ceil(len(keys) / ncols))
    n_slots = nrows * ncols
    titles = [
        f"{k}<br><sup style='font-size:10px'>Q–Q (transformed): {_qq_plot_suffix(k)}</sup>" for k in keys
    ] + [""] * (n_slots - len(keys))
    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=titles,
        vertical_spacing=0.10,
        horizontal_spacing=0.06,
    )
    for idx, key in enumerate(keys):
        if key not in wide.columns:
            continue
        row = idx // ncols + 1
        cc = idx % ncols + 1
        s = _apply_eda_trim(
            pd.to_numeric(wide[key], errors="coerce").replace([np.inf, -np.inf], np.nan),
            trim,
        )
        x0 = _series_for_qq(key, s.loc[y == 0])
        x1 = _series_for_qq(key, s.loc[y == 1])
        t0, z0 = _qq_sorted_vs_theoretical(x0)
        t1, z1 = _qq_sorted_vs_theoretical(x1)
        if t0.size:
            fig.add_trace(
                go.Scatter(
                    x=t0,
                    y=z0,
                    mode="lines+markers",
                    name="Survived",
                    legendgroup="s",
                    showlegend=idx == 0,
                    line=dict(color="#16a34a", width=2.2),
                    marker=dict(color="#16a34a", size=5, symbol="circle"),
                    hovertemplate="survived<br>theoretical q=%{x:.3f}<br>ordered transform=%{y:.4g}<extra></extra>",
                ),
                row=row,
                col=cc,
            )
        if t1.size:
            fig.add_trace(
                go.Scatter(
                    x=t1,
                    y=z1,
                    mode="lines+markers",
                    name="Died",
                    legendgroup="d",
                    showlegend=idx == 0,
                    line=dict(color="#dc2626", width=2.2),
                    marker=dict(color="#dc2626", size=5, symbol="diamond"),
                    hovertemplate="died<br>theoretical q=%{x:.3f}<br>ordered transform=%{y:.4g}<extra></extra>",
                ),
                row=row,
                col=cc,
            )
        vals = _series_for_qq(key, s)
        if vals.size >= 3:
            mu, sigma = float(np.mean(vals)), float(np.std(vals, ddof=0)) + 1e-12
            tx = np.linspace(-3.5, 3.5, 50)
            fig.add_trace(
                go.Scatter(
                    x=tx,
                    y=mu + sigma * tx,
                    mode="lines",
                    line=dict(color="#64748b", width=1.5, dash="dash"),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row,
                col=cc,
            )
    fig.update_layout(
        title_text=(
            "Focal features — Q–Q vs N(0,1) on the recommended transform (same as binned mortality), "
            "stratified by outcome: green line = survived, red line = died (order statistics connected); "
            "dashed = N(μ,σ) on pooled transformed data"
            + _eda_trim_caption(trim)
        ),
        template="plotly_white",
        height=min(3600, 200 + nrows * 240),
        width=min(1500, 100 + ncols * 320),
        legend=dict(orientation="h", y=1.03, x=0.5, xanchor="center"),
        margin=dict(t=120, l=50, r=40, b=40),
    )
    fig.update_xaxes(title_text="Theoretical N(0,1) quantile")
    fig.update_yaxes(title_text="Ordered transformed value")
    return fig


def _log1p_recommended_keys_in_wide(wide: pd.DataFrame) -> list[str]:
    """Pooled columns flagged for log₁p in the guide and present in ``wide``."""
    return [row["key"] for row in FEATURE_TRANSFORM_GUIDE if row["plot"] == "log1p" and row["key"] in wide.columns]


def raw_vs_log1p_mortality_figure(
    wide: pd.DataFrame,
    *,
    trim: EdaOutlierTrim = EdaOutlierTrim(),
) -> go.Figure | None:
    """Side-by-side binned P(died): equal-width bins on raw ``x`` vs on ``log₁p(x)`` for log-recommended features."""
    keys = _log1p_recommended_keys_in_wide(wide)
    if not keys or "died" not in wide.columns:
        return None
    died = wide["died"].astype(int)
    overall = float(died.mean())
    nrows = len(keys)
    st: list[str] = []
    for k in keys:
        st.extend([f"{k}<br><sub>raw bins</sub>", f"{k}<br><sub>log₁p bins</sub>"])
    fig = make_subplots(
        rows=nrows,
        cols=2,
        subplot_titles=st,
        vertical_spacing=0.08,
        horizontal_spacing=0.08,
    )
    colorbar_done = False
    any_trace = False
    for i, key in enumerate(keys):
        row = i + 1
        s = _apply_eda_trim(
            pd.to_numeric(wide[key], errors="coerce").replace([np.inf, -np.inf], np.nan),
            trim,
        )
        for col, fm, xtitle in (
            (1, "identity", "raw x (equal-width)"),
            (2, "log1p", "log₁p(x); hover shows raw range"),
        ):
            bt, _ = _binned_mortality_for_key(key, s, died, force_mode=fm)
            if bt is None or bt.empty:
                continue
            any_trace = True
            bar, ref = _death_likelihood_bar_trace(
                bt,
                overall_p=overall,
                show_colorbar=not colorbar_done,
            )
            colorbar_done = True
            fig.add_trace(bar, row=row, col=col)
            fig.add_trace(ref, row=row, col=col)
            fig.update_xaxes(title_text=xtitle, row=row, col=col)
    if not any_trace:
        return None
    fig.update_layout(
        title_text=(
            "Log-recommended features — raw x vs log₁p(x)=log(1+x): "
            "empirical P(in-hospital death) per bin (Wilson 95% CI; dashed = cohort death rate)"
            + _eda_trim_caption(trim)
        ),
        template="plotly_white",
        barmode="overlay",
        height=min(4400, 140 + nrows * 300),
        width=1100,
        showlegend=False,
        margin=dict(t=110, l=55, r=60, b=50),
    )
    fig.update_yaxes(title_text="P(died)", tickformat=".0%")
    return fig


def all_features_dist_figure(
    wide: pd.DataFrame,
    ncols: int = 3,
    *,
    trim: EdaOutlierTrim = EdaOutlierTrim(),
) -> go.Figure | None:
    """Binned empirical ``P(died)`` for every numeric column (single risk scale per panel)."""
    if wide.empty or "died" not in wide.columns:
        return None
    died = wide["died"].astype(int)
    overall = float(died.mean())
    cols_drop = [c for c in ("PERSON_ID",) if c in wide.columns]
    num = wide.drop(columns=cols_drop + ["died"], errors="ignore").select_dtypes(include=[np.number]).copy()
    for c in num.columns:
        num[c] = pd.to_numeric(num[c], errors="coerce")
    num = num.replace([np.inf, -np.inf], np.nan)
    std = num.std(numeric_only=True)
    num = num.loc[:, std > 1e-12]
    if num.shape[1] == 0:
        return None

    feats = list(num.columns)
    nrows = int(np.ceil(len(feats) / ncols))
    subplot_titles = []
    for c in feats:
        s_raw = num[c]
        s_plot = _apply_eda_trim(s_raw, trim)
        r = s_plot.corr(died) if s_plot.notna().sum() > 2 else float("nan")
        line = f"{c}<br>r={r:.3f}" if pd.notna(r) else f"{c}<br>r=n/a"
        sub = _binning_subtitle(c)
        if sub:
            line += f"<br><sup style='font-size:10px'>{sub}</sup>"
        subplot_titles.append(line)

    n_slots = nrows * ncols
    if len(subplot_titles) < n_slots:
        subplot_titles = subplot_titles + [""] * (n_slots - len(subplot_titles))

    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=subplot_titles,
        vertical_spacing=0.09,
        horizontal_spacing=0.06,
    )
    colorbar_done = False
    for idx, col in enumerate(feats):
        row = idx // ncols + 1
        cc = idx % ncols + 1
        s = _apply_eda_trim(num[col], trim)
        bt, xlab = _binned_mortality_for_key(col, s, died)
        if bt is None or bt.empty:
            continue
        bar, ref = _death_likelihood_bar_trace(
            bt,
            overall_p=overall,
            show_colorbar=not colorbar_done,
        )
        colorbar_done = True
        fig.add_trace(bar, row=row, col=cc)
        fig.add_trace(ref, row=row, col=cc)
        fig.update_xaxes(title_text=xlab, row=row, col=cc)

    if not fig.data:
        return None

    fig.update_layout(
        title_text=(
            "All numeric attributes — empirical P(died) by bin "
            "(transforms from recommendation table when present; Wilson 95% CI; dashed = cohort rate)"
            + _eda_trim_caption(trim)
        ),
        template="plotly_white",
        barmode="overlay",
        height=min(4200, 240 + nrows * 200),
        width=min(1500, 120 + ncols * 320),
        showlegend=False,
        margin=dict(t=100, l=50, r=60, b=40),
    )
    fig.update_yaxes(title_text="P(died)", tickformat=".0%")
    return fig


def _launch_html_files(paths: list[Path]) -> None:
    """Open each HTML in the default browser (macOS ``open`` is more reliable than double ``webbrowser``)."""
    for i, p in enumerate(paths):
        if not p.is_file():
            continue
        if i:
            time.sleep(0.55)
        p = p.resolve()
        if platform.system() == "Darwin":
            subprocess.run(["open", str(p)], check=False)
        else:
            webbrowser.open(p.as_uri())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("Baptist_tester/synth_cs_data"))
    ap.add_argument("--out-dir", type=Path, default=None, help="Default: <data-dir>/eda")
    ap.add_argument("--no-open", action="store_true")
    ap.add_argument("--threshold", type=float, default=CORR_THRESHOLD, help="|Pearson r| vs died (default 0.65)")
    ap.add_argument(
        "--eda-outliers",
        choices=("none", "percentile", "iqr"),
        default="percentile",
        help="Mask extreme raw feature values before mortality bins and focal Q–Q (viz only; default 1–99%%ile).",
    )
    ap.add_argument(
        "--eda-pct-lo",
        type=float,
        default=1.0,
        help="Lower percentile when --eda-outliers percentile (default 1).",
    )
    ap.add_argument(
        "--eda-pct-hi",
        type=float,
        default=99.0,
        help="Upper percentile when --eda-outliers percentile (default 99).",
    )
    ap.add_argument(
        "--eda-iqr-k",
        type=float,
        default=1.5,
        help="IQR multiplier when --eda-outliers iqr (Tukey fence).",
    )
    args = ap.parse_args()

    data_dir = _resolve(args.data_dir)
    out_dir = _resolve(args.out_dir) if args.out_dir else data_dir / "eda"
    out_dir.mkdir(parents=True, exist_ok=True)
    thr = float(args.threshold)
    trim = EdaOutlierTrim(
        mode=args.eda_outliers,
        pct_lo=float(args.eda_pct_lo),
        pct_hi=float(args.eda_pct_hi),
        iqr_k=float(args.eda_iqr_k),
    )

    if not (data_dir / "person.parquet").is_file():
        print(f"Missing parquet bundle under {data_dir}", file=sys.stderr)
        return 1

    base, died = _load_base(data_dir)
    died_s = died.copy()
    died_s.index = base["PERSON_ID"].values
    died_s = died_s.groupby(level=0).first()

    table_frames = build_table_frames(data_dir)
    n = len(died)
    n1, n0 = int(died.sum()), int((1 - died).sum())

    # --- Report HTML pieces ---
    html_chunks: list[str] = []
    first_js = True

    def add_fig(fig: go.Figure) -> None:
        nonlocal first_js
        html_chunks.append(
            fig.to_html(
                include_plotlyjs="cdn" if first_js else False,
                full_html=False,
                default_height=fig.layout.height,
                default_width=fig.layout.width,
            )
        )
        first_js = False

    # Class balance
    add_fig(
        go.Figure(
            data=[
                go.Pie(
                    labels=["Survived", "Died"],
                    values=[n0, n1],
                    marker=dict(colors=["#16a34a", "#dc2626"]),
                    hole=0.35,
                )
            ],
            layout=dict(
                title=f"In-hospital mortality (target) — n={n}, death rate={died.mean():.1%}",
                template="plotly_white",
                height=420,
                width=520,
            ),
        )
    )

    html_chunks.append(_transform_recommendations_html())

    corr_tables: dict[str, pd.DataFrame] = {}
    for tname, tdf in table_frames.items():
        if "died" not in tdf.columns or tdf.shape[1] < 2:
            continue
        c = tdf.corr(numeric_only=True, min_periods=3)
        if c.empty or "died" not in c.index:
            continue
        corr_tables[tname] = c
        add_fig(_corr_heatmap(c, f"Correlation matrix — {tname} (includes died)"))

    # Top |r| vs died across tables (bar)
    rows = []
    for tname, tdf in table_frames.items():
        if "died" not in tdf.columns:
            continue
        for col in tdf.columns:
            if col == "died":
                continue
            s = pd.to_numeric(tdf[col], errors="coerce")
            if s.notna().sum() < 3:
                continue
            r = s.corr(tdf["died"])
            if pd.isna(r):
                continue
            rows.append({"table": tname, "feature": col, "r": float(r), "abs_r": abs(float(r))})
    top_df = pd.DataFrame(rows).sort_values("abs_r", ascending=False) if rows else pd.DataFrame()

    if not top_df.empty:
        head = top_df.head(25)
        add_fig(
            go.Figure(
                data=[
                    go.Bar(
                        y=[f"{a}::{b}" for a, b in zip(head["table"], head["feature"])],
                        x=head["r"],
                        orientation="h",
                        marker_color=np.where(head["r"] >= 0, "#b91c1c", "#1d4ed8"),
                        hovertemplate="%{y}<br>r=%{x:.3f}<extra></extra>",
                    )
                ],
                layout=dict(
                    title="Top 25 |Pearson r| with died (patient-level aggregates)",
                    template="plotly_white",
                    height=640,
                    width=720,
                    xaxis_title="correlation with died",
                    margin=dict(l=220, t=60),
                ),
            )
        )

    wide = pooled_patient_frame(base, table_frames, died_s)
    pooled_num = _numeric_matrix(wide.drop(columns=["PERSON_ID"], errors="ignore"))
    if "died" in pooled_num.columns and pooled_num.shape[1] >= 2:
        c_all = pooled_num.corr(numeric_only=True, min_periods=3)
        add_fig(_corr_heatmap(c_all, "Pooled patient-level features — full correlation matrix (includes died)"))

    # --- High-|r| feature distributions (embed in main HTML so one browser tab shows all plots) ---
    high_cols: list[tuple[str, str, float]] = []
    if not top_df.empty:
        for _, r in top_df.iterrows():
            if r["abs_r"] >= thr:
                high_cols.append((str(r["table"]), str(r["feature"]), float(r["r"])))

    dist_path = out_dir / "baptist_sample_eda_high_corr_dist.html"
    all_dist_path = out_dir / "baptist_sample_eda_feature_distributions_all.html"
    focal_path = out_dir / "baptist_sample_eda_focal_hist_qq.html"
    html_chunks.append(
        "<hr><h2>Mortality likelihood vs high-|correlation| features</h2>"
        "<p>Each panel: <b>empirical P(in-hospital death)</b> within a feature range. "
        "Bin edges follow the <b>recommended transform</b> for that column when it appears in the table above "
        "(otherwise equal-width bins on raw <i>x</i>). Whiskers = Wilson 95% CI; dashed = cohort death rate. "
        "Hover shows <b>raw</b> value range when a transform is used. "
        f"See <code>{all_dist_path.name}</code> for all numeric attributes and "
        f"<code>{focal_path.name}</code> for focal variables (same logic + Q–Q).</p>"
    )

    if high_cols:
        ncols = 3
        nrows = int(np.ceil(len(high_cols) / ncols))
        stitles = []
        for t, c, rv in high_cols:
            key = f"{t}__{c}"
            sub = _binning_subtitle(key)
            line = f"{t}::{c}<br>r={rv:.2f}"
            if sub:
                line += f"<br><sup style='font-size:10px'>{sub}</sup>"
            stitles.append(line)
        fig = make_subplots(
            rows=nrows,
            cols=ncols,
            subplot_titles=stitles,
            vertical_spacing=0.12,
            horizontal_spacing=0.07,
        )
        died = wide["died"].astype(int)
        overall_p = float(died.mean())
        colorbar_done = False
        for idx, (tname, col, rv) in enumerate(high_cols):
            row = idx // ncols + 1
            ccol = idx % ncols + 1
            key = f"{tname}__{col}"
            if key not in wide.columns:
                continue
            s = _apply_eda_trim(
                pd.to_numeric(wide[key], errors="coerce").replace([np.inf, -np.inf], np.nan),
                trim,
            )
            bt, xlab = _binned_mortality_for_key(key, s, died)
            if bt is None or bt.empty:
                continue
            bar, ref = _death_likelihood_bar_trace(
                bt,
                overall_p=overall_p,
                show_colorbar=not colorbar_done,
            )
            colorbar_done = True
            fig.add_trace(bar, row=row, col=ccol)
            fig.add_trace(ref, row=row, col=ccol)
            fig.update_xaxes(title_text=xlab, row=row, col=ccol)
        if not fig.data:
            html_chunks.append(
                f"<p><i>No binned mortality curve could be built for features with |r| ≥ {thr:g}. "
                "Try lowering <code>--threshold</code> or check data sparsity.</i></p>"
            )
            dist_path.write_text(
                "<!DOCTYPE html><html><body><h1>No panels</h1>"
                "<p>Binned P(died) could not be computed for the high-|r| feature set.</p></body></html>",
                encoding="utf-8",
            )
            print(f"Wrote {dist_path} (empty notice)")
        else:
            fig.update_layout(
                title_text=(
                    f"Empirical P(died) by feature range (|r| ≥ {thr:g} vs died) — "
                    "recommended transforms where listed; Wilson 95% CI; dashed = cohort rate"
                    + _eda_trim_caption(trim)
                ),
                template="plotly_white",
                barmode="overlay",
                height=min(2600, 200 + nrows * 260),
                width=min(1400, 100 + ncols * 320),
                showlegend=False,
                margin=dict(t=100, l=50, r=60, b=40),
            )
            fig.update_yaxes(title_text="P(died)", tickformat=".0%")
            add_fig(fig)
            fig.write_html(str(dist_path), include_plotlyjs="cdn", full_html=True)
            print(f"Wrote {dist_path}")
    else:
        html_chunks.append(
            f"<p><i>No patient-level feature had |r| ≥ {thr:g} vs died. "
            "Lower <code>--threshold</code> to see distribution panels.</i></p>"
        )
        dist_path.write_text(
            "<!DOCTYPE html><html><body><h1>No features ≥ threshold</h1>"
            f"<p>No patient-level feature had |r| ≥ {thr:g} vs died.</p></body></html>",
            encoding="utf-8",
        )
        print(f"Wrote {dist_path} (empty notice)")

    fig_all = all_features_dist_figure(wide, trim=trim)
    if fig_all is not None:
        fig_all.write_html(str(all_dist_path), include_plotlyjs="cdn", full_html=True)
        print(f"Wrote {all_dist_path}")
    else:
        all_dist_path.write_text(
            "<!DOCTYPE html><html><body><h1>No numeric features</h1>"
            "<p>No patient-level numeric columns remained after filtering for the pooled frame.</p></body></html>",
            encoding="utf-8",
        )
        print(f"Wrote {all_dist_path} (empty notice)")

    focal_keys = [f"{t}__{c}" for t, c in FOCAL_TABLE_FEATURES if f"{t}__{c}" in wide.columns]
    f_hist = focal_histogram_figure(wide, focal_keys, trim=trim) if focal_keys else None
    f_qq = focal_qq_figure(wide, focal_keys, trim=trim) if focal_keys else None
    if f_hist is not None or f_qq is not None:
        parts = [
            "<!DOCTYPE html><html><head><meta charset='utf-8'><title>Focal feature distributions & Q–Q</title></head><body>",
            "<h1>Focal variables (clinical / medication / procedure / hourly SCAI)</h1>",
        ]
        first_js = True
        if f_hist is not None:
            parts.append("<h2>Binned P(in-hospital death) by feature range</h2>")
            parts.append(
                f_hist.to_html(
                    include_plotlyjs="cdn" if first_js else False,
                    full_html=False,
                    default_height=f_hist.layout.height,
                    default_width=f_hist.layout.width,
                )
            )
            first_js = False
        if f_qq is not None:
            parts.append(
                "<h2>Q–Q plots (transformed scale, survived vs died)</h2>"
                "<p>Same <code>log₁p</code> / <code>√</code> / raw transform as the binned P(died) charts. "
                "Each outcome is a <b>line</b> through ordered transform vs N(0,1) quantiles; separation "
                "between lines relates to how mortality shifts the feature distribution.</p>"
            )
            parts.append(
                f_qq.to_html(
                    include_plotlyjs="cdn" if first_js else False,
                    full_html=False,
                    default_height=f_qq.layout.height,
                    default_width=f_qq.layout.width,
                )
            )
        parts.append("</body></html>")
        focal_path.write_text("\n".join(parts), encoding="utf-8")
        print(f"Wrote {focal_path}")
    else:
        focal_path.write_text(
            "<!DOCTYPE html><html><body><h1>No focal columns</h1>"
            "<p>None of the expected focal pooled columns were present in the wide frame.</p></body></html>",
            encoding="utf-8",
        )
        print(f"Wrote {focal_path} (empty notice)")

    raw_vs_log_path = out_dir / "baptist_sample_eda_raw_vs_log1p.html"
    fig_rl = raw_vs_log1p_mortality_figure(wide, trim=trim)
    if fig_rl is not None:
        fig_rl.write_html(str(raw_vs_log_path), include_plotlyjs="cdn", full_html=True)
        print(f"Wrote {raw_vs_log_path}")
        html_chunks.append(
            "<hr><h2>Raw vs log₁p binning (log-recommended columns)</h2>"
            "<p>For each feature flagged for a log-type transform: <b>left</b> — equal-width bins on <b>raw</b> "
            "<i>x</i>; <b>right</b> — equal-width bins on <code>log₁p(x)=log(1+x)</code> (zeros safe). "
            "Same Wilson CIs and cohort reference line in both columns. Standalone: "
            f"<code>{raw_vs_log_path.name}</code>.</p>"
        )
        add_fig(fig_rl)
    else:
        raw_vs_log_path.write_text(
            "<!DOCTYPE html><html><body><h1>Raw vs log₁p</h1>"
            "<p>No log-recommended pooled columns were present, or binning failed for all.</p></body></html>",
            encoding="utf-8",
        )
        print(f"Wrote {raw_vs_log_path} (empty notice)")

    report_path = out_dir / "baptist_sample_eda_report.html"
    report_path.write_text(
        "<!DOCTYPE html><html><head><meta charset='utf-8'><title>Baptist sample EDA</title></head><body>"
        "<h1>Baptist sample EDA — mortality target</h1>"
        "<p>Patient-level aggregates per source table; Pearson correlation. "
        "The report includes a <b>recommended transformations</b> table; binned mortality plots use that "
        "guidance (log₁p, √, raw, or ordinal) where the column is listed. "
        "By default, extreme raw feature values outside the <b>1st–99th percentile</b> are masked for those "
        "mortality panels and focal Q–Q only (parquet data unchanged); use <code>--eda-outliers none</code> to show all points. "
        f"High-|r| threshold for the multi-panel figure: <b>≥ {thr:g}</b> (absolute value vs died). "
        f"<b>Scroll down</b> for correlation heatmaps and high-|r| panels. Standalone: "
        f"<code>baptist_sample_eda_high_corr_dist.html</code>, "
        f"<code>baptist_sample_eda_feature_distributions_all.html</code>, "
        f"<code>baptist_sample_eda_focal_hist_qq.html</code> (focal: binned P(died) + Q–Q), "
        f"<code>baptist_sample_eda_raw_vs_log1p.html</code> (raw vs log₁p bins for log-recommended features).</p>"
        + "".join(html_chunks)
        + "</body></html>",
        encoding="utf-8",
    )
    print(f"Wrote {report_path}")

    if not args.no_open:
        _launch_html_files([report_path, dist_path, all_dist_path, focal_path, raw_vs_log_path])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
