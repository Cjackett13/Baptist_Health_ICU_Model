#!/usr/bin/env python3
"""
Comprehensive Plotly LOS EDA (regression target).

Writes HTML reports + JSON audits under ``<data-dir>/eda/``.
Invoked by ``los_eda.py`` or directly:

  python3 Baptist_tester/los_eda_full.py --data-dir data --no-open
"""

from __future__ import annotations

import json
import platform
import subprocess
import time
import webbrowser
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

_REPO = Path(__file__).resolve().parent.parent
TARGET_DEFAULT = "los_hours_total"
COUNT_FEATURES = [
    "proc_n_12h",
    "med_distinct_12h",
    "clinical_event_n_12h",
    "med_admin_rows_12h",
    "n_diagnoses_12h",
]
CATEGORICAL_FEATURES = [
    "admit_type_cd",
    "admit_src_cd",
    "unit_cd",
    "race_cd",
    "ethnicity_cd",
    "icd_prefix",
]
EARLY_FULL_PAIRS = [
    ("med_distinct_12h", "med_distinct"),
    ("med_infusion_mean_12h", "med_infusion_mean"),
    ("clinical_event_n_12h", "clinical_event_n"),
    ("proc_n_12h", "proc_n"),
    ("scai_mean_12h", "current_scai"),
    ("scai_last_12h", "current_scai"),
]


def _resolve(p: Path) -> Path:
    return p.resolve() if p.is_absolute() else (_REPO / p).resolve()


def load_eda_frame(data_dir: Path) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    data_dir = _resolve(data_dir)
    groups = json.loads((data_dir / "los_modeling_column_groups.json").read_text(encoding="utf-8"))
    features = groups["feature_columns"]
    labels = pd.read_parquet(data_dir / "los_encounter_labels.parquet")
    feat = pd.read_parquet(data_dir / "los_modeling_features.parquet")
    df = labels.merge(feat, on=["ENCOUNTER_ID", "PERSON_ID"], how="inner", suffixes=("", "_feat"))

    pf_path = data_dir / "los_patient_frame.parquet"
    if pf_path.is_file():
        import pyarrow.parquet as pq

        audit_cols = [
            "ENCOUNTER_ID",
            "DISCH_DISPOSITION_CD",
            "truncated_death",
            "truncated_hospice",
            "los_truncated",
            "admit_type_cd",
            "admit_src_cd",
            "unit_cd",
        ]
        names = pq.read_schema(pf_path).names
        use = [c for c in audit_cols if c in names]
        pf = pd.read_parquet(pf_path, columns=use)
        df = df.merge(pf, on="ENCOUNTER_ID", how="left", suffixes=("", "_pf"))

    return df, features, groups


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


def _index_page(sections: list[tuple[str, str]], title: str) -> str:
    lis = "".join(f'<li><a href="{href}">{name}</a></li>' for name, href in sections)
    return f"""<!DOCTYPE html><html><head><meta charset='utf-8'><title>{title}</title></head>
<body><h1>{title}</h1><ul>{lis}</ul></body></html>"""


def fig_target_hist_kde(df: pd.DataFrame, target: str) -> go.Figure:
    y = pd.to_numeric(df[target], errors="coerce").dropna()
    logy = np.log1p(y)
    fig = make_subplots(rows=1, cols=2, subplot_titles=(f"{target} (raw)", f"log1p({target})"))
    for col, series, name in [(1, y, "raw"), (2, logy, "log1p")]:
        fig.add_trace(go.Histogram(x=series, nbinsx=40, name=f"{name} hist", histnorm="probability density"), row=1, col=col)
        if len(series) > 10:
            kde_x = np.linspace(series.min(), series.max(), 200)
            kde = stats.gaussian_kde(series.to_numpy())
            fig.add_trace(
                go.Scatter(x=kde_x, y=kde(kde_x), mode="lines", name=f"{name} KDE", line=dict(width=2)),
                row=1,
                col=col,
            )
    fig.update_layout(title="Target distribution: histogram + KDE", template="plotly_white", height=420, barmode="overlay")
    return fig


def fig_target_qq(df: pd.DataFrame, target: str) -> go.Figure:
    y = pd.to_numeric(df[target], errors="coerce").dropna().to_numpy()
    logy = np.log1p(y)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Raw vs normal", "log1p vs normal"))
    for col, vals in [(1, y), (2, logy)]:
        if len(vals) < 8:
            continue
        osm, osr = stats.probplot(vals, dist="norm")[0]
        fig.add_trace(go.Scatter(x=osm, y=osr, mode="markers", name=f"col{col}"), row=1, col=col)
        lims = [min(osm.min(), osr.min()), max(osm.max(), osr.max())]
        fig.add_trace(
            go.Scatter(x=lims, y=lims, mode="lines", line=dict(dash="dash", color="black"), showlegend=False),
            row=1,
            col=col,
        )
    fig.update_layout(title="Target Q–Q plots (raw vs log1p)", template="plotly_white", height=420)
    return fig


def fig_target_by_flags(df: pd.DataFrame, target: str) -> go.Figure:
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Disposition", "Death flag", "Truncated LOS"))
    if "DISCH_DISPOSITION_CD" in df.columns:
        for disp, sub in df.groupby("DISCH_DISPOSITION_CD", dropna=False):
            fig.add_trace(
                go.Box(y=pd.to_numeric(sub[target], errors="coerce"), name=str(disp), showlegend=False),
                row=1,
                col=1,
            )
    if "truncated_death" in df.columns:
        for v, lab in [(0, "alive"), (1, "death flag")]:
            sub = df[df["truncated_death"].astype(int) == v]
            fig.add_trace(go.Box(y=pd.to_numeric(sub[target], errors="coerce"), name=lab), row=1, col=2)
    if "los_truncated" in df.columns:
        for v, lab in [(0, "not truncated"), (1, "truncated")]:
            sub = df[df["los_truncated"].astype(int) == v]
            fig.add_trace(go.Box(y=pd.to_numeric(sub[target], errors="coerce"), name=lab), row=1, col=3)
    fig.update_layout(title=f"{target} by disposition / death / truncation", template="plotly_white", height=440)
    return fig


def fig_target_by_admit(df: pd.DataFrame, target: str) -> go.Figure:
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Admit type", "Admit source", "Unit"))
    pairs = [("admit_type_cd", 1), ("admit_src_cd", 2), ("unit_cd", 3)]
    for col_name, c in pairs:
        if col_name not in df.columns:
            continue
        for val, sub in df.groupby(col_name, dropna=False):
            if sub.empty:
                continue
            fig.add_trace(
                go.Box(y=pd.to_numeric(sub[target], errors="coerce"), name=str(val)[:24], showlegend=False),
                row=1,
                col=c,
            )
    fig.update_layout(title=f"{target} by admit context", template="plotly_white", height=440)
    return fig


def fig_target_vs_scai(df: pd.DataFrame, target: str) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, subplot_titles=("vs current_scai_12h", "vs scai_slope_12h"))
    for col, feat in [(1, "current_scai_12h"), (2, "scai_slope_12h")]:
        if feat not in df.columns:
            continue
        x = pd.to_numeric(df[feat], errors="coerce")
        y = pd.to_numeric(df[target], errors="coerce")
        mask = x.notna() & y.notna()
        fig.add_trace(
            go.Scatter(x=x[mask], y=y[mask], mode="markers", marker=dict(size=4, opacity=0.35), showlegend=False),
            row=1,
            col=col,
        )
        if mask.sum() > 20:
            r, _ = stats.spearmanr(x[mask], y[mask], nan_policy="omit")
            fig.add_annotation(
                text=f"Spearman ρ={r:.3f}",
                xref="x domain",
                yref="y domain",
                x=0.05,
                y=0.95,
                showarrow=False,
                row=1,
                col=col,
            )
    fig.update_layout(title=f"{target} vs early SCAI", template="plotly_white", height=420)
    return fig


def fig_missingness_heatmap(df: pd.DataFrame, features: list[str], *, max_rows: int = 400) -> go.Figure:
    use = [c for c in features if c in df.columns]
    sub = df[use]
    if len(sub) > max_rows:
        sub = sub.sample(n=max_rows, random_state=0)
    miss = sub.apply(lambda s: s.isna() | (s.astype("string").str.strip() == ""))
    z = miss.astype(int).to_numpy()
    fig = go.Figure(
        data=go.Heatmap(
            z=z.T,
            x=[f"row{i}" for i in range(len(sub))],
            y=use,
            colorscale=[[0, "#f7fbff"], [1, "#08306b"]],
            showscale=True,
        ),
        layout=dict(
            title=f"Missingness heatmap (sample {len(sub)} encounters × features)",
            template="plotly_white",
            height=max(400, 12 * len(use) + 120),
            xaxis=dict(showticklabels=False, title="encounters (sampled)"),
            yaxis=dict(title="feature"),
        ),
    )
    return fig


def fig_numeric_scatter_grid(df: pd.DataFrame, features: list[str], target: str, *, top_n: int = 9) -> go.Figure:
    num_feats = [c for c in features if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    corrs = {}
    y = pd.to_numeric(df[target], errors="coerce")
    for c in num_feats:
        x = pd.to_numeric(df[c], errors="coerce")
        mask = x.notna() & y.notna()
        if mask.sum() < 10:
            continue
        corrs[c] = abs(stats.spearmanr(x[mask], y[mask], nan_policy="omit").statistic)
    top = sorted(corrs, key=corrs.get, reverse=True)[:top_n]
    cols = 3
    rows = int(np.ceil(len(top) / cols)) or 1
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[f"{f} (|ρ|={corrs[f]:.2f})" for f in top])
    for i, feat in enumerate(top):
        r, c = i // cols + 1, i % cols + 1
        x = pd.to_numeric(df[feat], errors="coerce")
        mask = x.notna() & y.notna()
        fig.add_trace(
            go.Scatter(x=x[mask], y=y[mask], mode="markers", marker=dict(size=3, opacity=0.35), showlegend=False),
            row=r,
            col=c,
        )
    fig.update_layout(title=f"Numeric features vs {target} (top |Spearman|)", template="plotly_white", height=280 * rows)
    return fig


def fig_categorical_violin(df: pd.DataFrame, target: str) -> go.Figure:
    cats = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    cols = min(3, len(cats)) or 1
    rows = int(np.ceil(len(cats) / cols)) or 1
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=cats)
    y = pd.to_numeric(df[target], errors="coerce")
    for i, feat in enumerate(cats):
        r, c = i // cols + 1, i % cols + 1
        for val, sub in df.groupby(feat, dropna=False):
            fig.add_trace(
                go.Violin(y=y.loc[sub.index], name=str(val)[:20], showlegend=False, box_visible=True),
                row=r,
                col=c,
            )
    fig.update_layout(title=f"{target} by categorical features", template="plotly_white", height=300 * rows)
    return fig


def _corr_with_target_block(
    full_corr: pd.DataFrame, spearman_df: pd.DataFrame, target: str, *, top_n: int = 25
) -> pd.DataFrame:
    top_feats = spearman_df.head(top_n)["feature"].tolist()
    cols = list(dict.fromkeys([c for c in top_feats if c in full_corr.columns] + [target]))
    return full_corr.loc[cols, cols]


def _feature_only_corr(df: pd.DataFrame, features: list[str], target: str, *, top_k: int = 20) -> pd.DataFrame:
    feats = [c for c in features if c in df.columns and c != target]
    num = df[feats].apply(pd.to_numeric, errors="coerce")
    std = num.std()
    num = num.loc[:, std > 1e-12]
    corr = num.corr()
    los_r = (
        corr[target].drop(target, errors="ignore").abs().sort_values(ascending=False)
        if target in corr.columns
        else pd.Series(dtype=float)
    )
    keep = los_r.head(top_k).index.tolist() if len(los_r) else num.columns.tolist()[:top_k]
    keep = [c for c in keep if c in corr.columns]
    return corr.loc[keep, keep]


def fig_corr_heatmap(corr: pd.DataFrame, title: str) -> go.Figure:
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
            height=max(480, 26 * len(corr) + 140),
            width=max(560, 22 * len(corr) + 160),
            xaxis=dict(tickangle=-45),
            margin=dict(l=120, b=120),
        ),
    )


def scan_multicollinearity(corr: pd.DataFrame, threshold: float = 0.85) -> list[dict[str, Any]]:
    pairs = []
    cols = corr.columns.tolist()
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            r = float(corr.loc[a, b])
            if np.isfinite(r) and abs(r) >= threshold:
                pairs.append({"feature_a": a, "feature_b": b, "pearson_r": round(r, 4)})
    return sorted(pairs, key=lambda x: -abs(x["pearson_r"]))


def outlier_trim_report(df: pd.DataFrame, features: list[str]) -> list[dict[str, Any]]:
    rows = []
    for col in features:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue
        p01, p99 = float(s.quantile(0.01)), float(s.quantile(0.99))
        q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        rows.append(
            {
                "feature": col,
                "n": int(len(s)),
                "p01": p01,
                "p99": p99,
                "n_below_p01": int((s < p01).sum()),
                "n_above_p99": int((s > p99).sum()),
                "iqr_lo": lo,
                "iqr_hi": hi,
                "n_below_iqr": int((s < lo).sum()),
                "n_above_iqr": int((s > hi).sum()),
                "pct_trimmed_1_99": round(100 * ((s < p01) | (s > p99)).mean(), 3),
            }
        )
    return rows


def fig_count_transforms(df: pd.DataFrame) -> go.Figure:
    feats = [c for c in COUNT_FEATURES if c in df.columns]
    cols = 3
    rows = int(np.ceil(len(feats) / cols)) or 1
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=feats)
    for i, feat in enumerate(feats):
        r, c = i // cols + 1, i % cols + 1
        x = pd.to_numeric(df[feat], errors="coerce").dropna()
        if x.empty:
            continue
        fig.add_trace(go.Histogram(x=x, name="raw", opacity=0.5, nbinsx=30), row=r, col=c)
        fig.add_trace(go.Histogram(x=np.sqrt(x.clip(lower=0)), name="sqrt", opacity=0.5, nbinsx=30), row=r, col=c)
        fig.add_trace(go.Histogram(x=np.log1p(x.clip(lower=0)), name="log1p", opacity=0.5, nbinsx=30), row=r, col=c)
    fig.update_layout(title="Count features: raw vs sqrt vs log1p", template="plotly_white", height=280 * rows, barmode="overlay")
    return fig


def leakage_early_vs_full(data_dir: Path, df: pd.DataFrame, target: str) -> tuple[go.Figure, list[dict[str, Any]]]:
    data_dir = _resolve(data_dir)
    mort_path = data_dir / "duckdb_patient_features.parquet"
    rows = []
    fig = go.Figure()
    if not mort_path.is_file():
        fig.update_layout(
            title="Mortality full-stay sidecar not found — skip leakage compare (LOS uses ≤12h features only)"
        )
        return fig, rows

    full = pd.read_parquet(mort_path)
    merged = df[["ENCOUNTER_ID", "PERSON_ID", target]].merge(full, on="PERSON_ID", how="inner", suffixes=("", "_full"))
    for early, late in EARLY_FULL_PAIRS:
        if early not in merged.columns or late not in merged.columns:
            continue
        e = pd.to_numeric(merged[early], errors="coerce")
        l = pd.to_numeric(merged[late], errors="coerce")
        y = pd.to_numeric(merged[target], errors="coerce")
        mask = e.notna() & l.notna() & y.notna()
        if mask.sum() < 20:
            continue
        r_early, _ = stats.spearmanr(e[mask], y[mask], nan_policy="omit")
        r_full, _ = stats.spearmanr(l[mask], y[mask], nan_policy="omit")
        r_el, _ = stats.spearmanr(e[mask], l[mask], nan_policy="omit")
        rows.append(
            {
                "early_12h": early,
                "full_stay": late,
                "spearman_early_vs_los": round(float(r_early), 4),
                "spearman_full_vs_los": round(float(r_full), 4),
                "delta_full_minus_early": round(float(r_full - r_early), 4),
                "spearman_early_vs_full": round(float(r_el), 4),
                "leakage_risk": "high" if abs(r_full) > abs(r_early) + 0.05 else "ok",
            }
        )
        fig.add_trace(
            go.Scatter(
                x=[early],
                y=[abs(r_full) - abs(r_early)],
                mode="markers+text",
                text=[f"Δ|r|={abs(r_full) - abs(r_early):.3f}"],
                textposition="top center",
                name=f"{early} vs {late}",
            )
        )
    fig.update_layout(
        title="Early-window vs full-stay |ρ| with LOS (positive Δ ⇒ full-stay more aligned)",
        yaxis_title="|ρ(full)| − |ρ(early)|",
        template="plotly_white",
        height=420,
    )
    return fig, rows


def duplicate_patient_report(df: pd.DataFrame, features: list[str]) -> dict[str, Any]:
    dup_pid = df.groupby("PERSON_ID").size()
    multi = dup_pid[dup_pid > 1]
    enc_dup = int(df["ENCOUNTER_ID"].duplicated().sum())
    near_dup_pairs = 0
    if len(multi) > 0 and len(features) > 0:
        num = df.groupby("PERSON_ID")[features].apply(
            lambda g: g.apply(pd.to_numeric, errors="coerce").fillna(0).values.flatten()
        )
        # skip expensive all-pairs; report count only
    return {
        "n_encounters": int(len(df)),
        "n_unique_person_id": int(df["PERSON_ID"].nunique()),
        "n_person_id_with_multiple_encounters": int(len(multi)),
        "person_ids_with_multiple_encounters": multi.head(20).to_dict() if len(multi) else {},
        "duplicate_encounter_id_count": enc_dup,
        "note": "1:1 PERSON_ID:ENCOUNTER_ID expected for Dataset B ICU cohort.",
    }


def target_skew_outlier_note(df: pd.DataFrame, target: str) -> dict[str, Any]:
    y = pd.to_numeric(df[target], errors="coerce").dropna()
    return {
        "classification_imbalance": "N/A — continuous regression target",
        "skewness": round(float(stats.skew(y)), 4),
        "kurtosis": round(float(stats.kurtosis(y)), 4),
        "mean": round(float(y.mean()), 3),
        "median": round(float(y.median()), 3),
        "std": round(float(y.std()), 3),
        "p01": round(float(y.quantile(0.01)), 3),
        "p99": round(float(y.quantile(0.99)), 3),
        "note": "Right-skew common; consider log1p transform for modeling; inspect high-LOS outliers.",
    }


def spearman_table(df: pd.DataFrame, features: list[str], target: str) -> pd.DataFrame:
    rows = []
    y = pd.to_numeric(df[target], errors="coerce")
    for col in features:
        if col not in df.columns:
            continue
        x = pd.to_numeric(df[col], errors="coerce")
        mask = x.notna() & y.notna()
        if mask.sum() < 5:
            continue
        try:
            r, p = stats.spearmanr(x[mask], y[mask], nan_policy="omit")
        except (ValueError, FloatingPointError):
            continue
        if not np.isfinite(r):
            continue
        rows.append({"feature": col, "spearman_rho": float(r), "p_value": float(p), "n": int(mask.sum())})
    return pd.DataFrame(rows).sort_values("spearman_rho", key=lambda s: s.abs(), ascending=False)


def run_full_eda(
    data_dir: Path,
    *,
    target: str = TARGET_DEFAULT,
    artifact_dir: Path | None = None,
) -> dict[str, Any]:
    """LOS-focused EDA. Reports go to ``artifact_dir/eda`` (default ``<data-dir>/los_model/eda``)."""
    data_dir = _resolve(data_dir)
    root = _resolve(artifact_dir) if artifact_dir is not None else data_dir / "los_model"
    eda_dir = root / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)

    df, features, groups = load_eda_frame(data_dir)
    if target not in df.columns and "los_hours" in df.columns:
        target = "los_hours"

    num_feats = [c for c in features if c in df.columns and pd.api.types.is_numeric_dtype(pd.to_numeric(df[c], errors="coerce"))]
    full_corr_num = df[num_feats + [target]].apply(pd.to_numeric, errors="coerce")
    full_corr = full_corr_num.corr()
    feat_corr = _feature_only_corr(df, num_feats, target, top_k=22)

    spearman_df = spearman_table(df, features, target)
    multi_pairs = scan_multicollinearity(feat_corr, threshold=0.85)
    outlier_rep = outlier_trim_report(df, num_feats)
    dup_rep = duplicate_patient_report(df, num_feats[:20])
    skew_note = target_skew_outlier_note(df, target)
    fig_leak, leak_rows = leakage_early_vs_full(data_dir, df, target)

    outputs: list[tuple[str, str]] = []

    def write(name: str, filename: str, figs: list[tuple[str, go.Figure]], page_title: str) -> Path:
        path = eda_dir / filename
        path.write_text(_html_page(figs, page_title), encoding="utf-8")
        outputs.append((name, f"eda/{filename}"))
        return path

    write(
        "Target (hist/KDE, Q–Q, skew)",
        "los_eda_target.html",
        [
            ("Histogram + KDE", fig_target_hist_kde(df, target)),
            ("Q–Q raw vs log1p", fig_target_qq(df, target)),
        ],
        "Length of stay (hours) — target distribution",
    )
    write(
        "Target by cohort flags",
        "los_eda_target_cohort.html",
        [
            ("LOS by disposition / truncation flags (not mortality model)", fig_target_by_flags(df, target)),
            ("Admit type / source / unit", fig_target_by_admit(df, target)),
            ("SCAI stage / slope", fig_target_vs_scai(df, target)),
        ],
        "Length of stay vs admit / SCAI cohort",
    )
    write(
        "Missingness",
        "los_eda_missingness.html",
        [("Feature × row heatmap (sample)", fig_missingness_heatmap(df, features))],
        "LOS missingness",
    )
    write(
        "Numeric vs LOS",
        "los_eda_numeric_scatter.html",
        [("Scatter grid (top |Spearman|)", fig_numeric_scatter_grid(df, features, target))],
        "Numeric features vs LOS",
    )
    write(
        "Categorical vs LOS",
        "los_eda_categorical.html",
        [("Violin / box by category", fig_categorical_violin(df, target))],
        "Categorical features vs LOS",
    )
    write(
        "Feature correlations",
        "los_eda_feature_corr.html",
        [
            ("High-|r| feature heatmap (no target)", fig_corr_heatmap(feat_corr, "Feature–feature Pearson (top |r| with LOS)")),
            (
                "All features + LOS",
                fig_corr_heatmap(_corr_with_target_block(full_corr, spearman_df, target, top_n=25), f"Correlation map: top features + {target}"),
            ),
        ],
        "LOS feature correlations",
    )
    write(
        "Count transforms",
        "los_eda_count_transforms.html",
        [("raw / sqrt / log1p", fig_count_transforms(df))],
        "Count feature distributions",
    )
    write(
        "Leakage sanity",
        "los_eda_leakage.html",
        [("Early ≤12h vs full-stay", fig_leak)],
        "LOS leakage: ≤12h features vs full-stay mortality sidecar (sanity only)",
    )

    index_path = eda_dir / "los_eda_index.html"
    index_path.write_text(
        _index_page(outputs, "Length of stay (LOS) model — EDA index"),
        encoding="utf-8",
    )

    audit = {
        "model": "length_of_stay_regression",
        "not_mortality_model": True,
        "artifact_dir": str(root),
        "n_encounters": int(len(df)),
        "target": target,
        "target_skew_outliers": skew_note,
        "duplicate_patient_check": dup_rep,
        "multicollinearity_pairs_abs_r_ge_0.85": multi_pairs,
        "outlier_trim_report": outlier_rep,
        "leakage_early_vs_full": leak_rows,
        "top_spearman_with_los": spearman_df.head(15).to_dict(orient="records"),
        "top_pearson_with_los": (
            full_corr[target].drop(target, errors="ignore").abs().sort_values(ascending=False).head(15).to_dict()
            if target in full_corr.columns
            else {}
        ),
        "outputs": [f"{root.name}/eda/los_eda_index.html"] + [href for _, href in outputs],
    }
    audit_path = eda_dir / "los_eda_full_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")

    return {"index": index_path, "audit": audit_path, "outputs": outputs}


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
    import argparse

    ap = argparse.ArgumentParser(description="Full LOS EDA report suite")
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--target", default=TARGET_DEFAULT)
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    result = run_full_eda(args.data_dir, target=args.target)
    print(f"Index: {result['index']}")
    print(f"Audit: {result['audit']}")
    if not args.no_open:
        _open_files([result["index"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
