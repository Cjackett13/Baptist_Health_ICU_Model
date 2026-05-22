#!/usr/bin/env python3
"""
Streamlit LOS EDA: correlation matrix vs length of stay + Q–Q plots for top features.

  pip install streamlit plotly pandas pyarrow numpy scipy
  streamlit run Baptist_tester/los_streamlit_eda.py

Default data: ``data/`` (5000 patients). Does not touch mortality model code.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO / "Baptist_tester") not in sys.path:
    sys.path.insert(0, str(_REPO / "Baptist_tester"))

from los_feature_policy import (  # noqa: E402
    FORBIDDEN_AS_FEATURES,
    LABEL_COLUMNS,
    validate_modeling_columns,
)

_STD_NORMAL = NormalDist(0, 1)
DEFAULT_DATA_DIR = _REPO / "data"


@st.cache_data(show_spinner="Loading LOS frames…")
def load_frames(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    root = Path(data_dir)
    ckpt = pd.read_parquet(root / "los_checkpoint_frame.parquet")
    patient = pd.read_parquet(root / "los_patient_frame.parquet")
    feat_path = root / "los_feature_columns.json"
    feat_doc = json.loads(feat_path.read_text(encoding="utf-8")) if feat_path.is_file() else {}
    return ckpt, patient, feat_doc


def _resolve_data_dir() -> Path:
    default = str(DEFAULT_DATA_DIR)
    user = st.sidebar.text_input("Data directory", value=default)
    return Path(user).expanduser().resolve()


def _pick_frame(
    ckpt: pd.DataFrame,
    patient: pd.DataFrame,
    grain: str,
) -> pd.DataFrame:
    if grain == "All checkpoint rows":
        return ckpt.copy()
    if grain == "Last checkpoint per patient":
        return (
            ckpt.sort_values(["PERSON_ID", "prediction_hour"])
            .groupby("PERSON_ID", as_index=False)
            .tail(1)
        )
    return patient.copy()


def _numeric_for_corr(df: pd.DataFrame, target: str) -> pd.DataFrame:
    block = FORBIDDEN_AS_FEATURES | LABEL_COLUMNS | {
        "PERSON_ID",
        "ENCOUNTER_ID",
        "log1p_los_hours_total",
        "log1p_remaining_los_hours",
        "los_hours",
    }
    num = df.select_dtypes(include=[np.number]).copy()
    drop = [c for c in num.columns if c in block or c == target]
    num = num.drop(columns=[c for c in drop if c in num.columns], errors="ignore")
    if target in df.columns:
        num[target] = pd.to_numeric(df[target], errors="coerce")
    num = num.replace([np.inf, -np.inf], np.nan)
    std = num.std(numeric_only=True)
    num = num.loc[:, std > 1e-12]
    return num.dropna(how="all", axis=1)


def _corr_figure(corr: pd.DataFrame, target: str) -> go.Figure:
    cols = [c for c in corr.columns if c != target] + ([target] if target in corr.columns else [])
    c = corr.loc[cols, cols]
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
            title=f"Pearson correlation (target: {target})",
            template="plotly_white",
            height=max(520, 26 * len(c) + 140),
            width=max(640, 22 * len(c) + 180),
            xaxis=dict(tickangle=-45),
            margin=dict(l=120, r=40, t=72, b=140),
        ),
    )


def _qq_vs_normal(vals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 8:
        return np.array([]), np.array([])
    v = np.sort(v)
    n = len(v)
    probs = (np.arange(1, n + 1) - 0.5) / n
    z = np.array([_STD_NORMAL.inv_cdf(p) for p in probs])
    return v, z


def _qq_figure(
    feature: str,
    series: pd.Series,
    los: pd.Series,
    *,
    nq: int = 200,
) -> go.Figure:
    """Q–Q vs normal; color by LOS tertile (low / mid / high)."""
    df = pd.DataFrame({"x": pd.to_numeric(series, errors="coerce"), "los": pd.to_numeric(los, errors="coerce")})
    df = df.dropna()
    fig = go.Figure()
    if len(df) < 20:
        fig.update_layout(title=f"{feature} — insufficient data")
        return fig

    try:
        df["los_tertile"] = pd.qcut(df["los"], q=3, labels=["Low LOS", "Mid LOS", "High LOS"])
    except ValueError:
        df["los_tertile"] = "All"

    colors = {"Low LOS": "#2ecc71", "Mid LOS": "#f39c12", "High LOS": "#e74c3c", "All": "#3498db"}
    for label, sub in df.groupby("los_tertile", observed=True):
        xv, zv = _qq_vs_normal(sub["x"].to_numpy())
        if len(xv) == 0:
            continue
        fig.add_trace(
            go.Scatter(
                x=zv,
                y=xv,
                mode="markers",
                name=str(label),
                marker=dict(size=5, opacity=0.65, color=colors.get(str(label), "#333")),
                hovertemplate="theoretical z=%{x:.2f}<br>value=%{y:.3g}<extra></extra>",
            )
        )

    zline = np.linspace(-3, 3, 50)
    mu, sigma = float(df["x"].mean()), float(df["x"].std(ddof=1))
    if sigma > 1e-12:
        fig.add_trace(
            go.Scatter(
                x=zline,
                y=mu + sigma * zline,
                mode="lines",
                name="Normal ref.",
                line=dict(color="black", dash="dash", width=1.5),
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        title=f"Q–Q vs normal — {feature}<br><sup>colored by LOS tertile</sup>",
        xaxis_title="Theoretical normal quantile (z)",
        yaxis_title="Feature value",
        template="plotly_white",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def main() -> None:
    st.set_page_config(page_title="LOS correlation & Q–Q", layout="wide")
    st.title("Length of stay — correlation & Q–Q")
    st.caption("LOS-only EDA on the 5000-patient `data/` bundle. Mortality pipeline not used.")

    data_dir = _resolve_data_dir()
    if not (data_dir / "los_checkpoint_frame.parquet").is_file():
        st.error(f"Missing checkpoint frame. Run: `python3 Baptist_tester/los_data_prep.py --data-dir {data_dir} --force`")
        st.stop()

    ckpt, patient, feat_doc = load_frames(str(data_dir))

    st.sidebar.header("Options")
    grain = st.sidebar.radio(
        "Row grain",
        ["Last checkpoint per patient", "All checkpoint rows", "One row per patient (12h snapshot)"],
        index=0,
    )
    target = st.sidebar.selectbox(
        "LOS target",
        ["los_hours_total", "remaining_los_hours"],
        index=0,
    )
    top_n = st.sidebar.slider("Top features for Q–Q", min_value=3, max_value=12, value=6)

    df = _pick_frame(ckpt, patient, grain)
    st.sidebar.metric("Rows in analysis", f"{len(df):,}")
    st.sidebar.metric("Patients", f"{df['PERSON_ID'].nunique():,}")

    num = _numeric_for_corr(df, target)
    if target not in num.columns:
        st.error(f"Target column {target!r} not found.")
        st.stop()

    corr = num.corr(numeric_only=True)
    los_corr = corr[target].drop(target, errors="ignore").sort_values(key=lambda s: s.abs(), ascending=False)

    st.subheader("Correlation with length of stay")
    c1, c2 = st.columns([2, 1])
    with c1:
        st.plotly_chart(_corr_figure(corr, target), use_container_width=True)
    with c2:
        st.markdown("**Top |r| with target**")
        tbl = pd.DataFrame(
            {"feature": los_corr.head(15).index, "pearson_r": los_corr.head(15).values}
        )
        st.dataframe(tbl, hide_index=True, use_container_width=True)
        st.markdown(
            f"**Spearman ρ** (top 5): "
            + ", ".join(
                f"`{f}`={stats.spearmanr(num[f], num[target], nan_policy='omit').statistic:.3f}"
                for f in los_corr.head(5).index
            )
        )

    top_features = [str(f) for f in los_corr.head(top_n).index if f in df.columns]
    try:
        validate_modeling_columns(top_features)
    except ValueError as exc:
        st.warning(f"Feature guard: {exc}")

    st.subheader(f"Q–Q plots — top {len(top_features)} features by |r|")
    st.caption("Empirical quantiles vs normal; points colored by LOS tertile (low / mid / high).")

    if not top_features:
        st.info("No features selected for Q–Q.")
        return

    cols = 2
    rows = int(np.ceil(len(top_features) / cols))
    for r in range(rows):
        row_cols = st.columns(cols)
        for c in range(cols):
            i = r * cols + c
            if i >= len(top_features):
                break
            feat = top_features[i]
            with row_cols[c]:
                st.plotly_chart(
                    _qq_figure(feat, df[feat], df[target]),
                    use_container_width=True,
                )

    with st.expander("Feature manifest (from los_feature_columns.json)"):
        st.json(feat_doc.get("primary_features", feat_doc.get("feature_columns_v2_checkpoint", [])))


if __name__ == "__main__":
    main()
