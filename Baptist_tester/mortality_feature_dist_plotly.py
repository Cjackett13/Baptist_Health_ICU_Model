#!/usr/bin/env python3
"""
Plotly feature distributions by mortality (survived vs died) for each model covariate.

Uses the same patient frame as the mortality XGBoost model and `mortality_qq_plotly.py`
(`build_patient_frame` on the synthetic parquet bundle).

  pip install plotly pandas pyarrow numpy
  python3 Baptist_tester/mortality_feature_dist_plotly.py
  python3 Baptist_tester/mortality_feature_dist_plotly.py --no-open

No trained XGBoost file required.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_REPO = Path(__file__).resolve().parent.parent

SUBPLOT_TITLE = {
    "scai_std": "scai_std — SCAI variability",
    "current_scai": "current_scai — most recent hourly SCAI (0–4)",
    "scai_prop_ge3": "scai_prop_ge3 — share of hours SCAI stage ≥ 3 (D–E)",
    "prop_hours_stage_ge4": "prop_hours_stage_ge4 — share hrs SCAI≥4",
    "mean_HR_all_hours": "mean HR",
    "mean_RR_all_hours": "mean RR",
    "mean_MAP_all_hours": "mean MAP",
    "mean_SBP_all_hours": "mean SBP",
    "mean_DBP_all_hours": "mean DBP",
    "mean_SPO2_all_hours": "mean SpO₂",
    "mean_TEMP_all_hours": "mean temp",
    "mean_LACTATE_all_hours": "mean lactate",
    "mean_NT_PROBNP_all_hours": "mean NT‑proBNP",
    "mean_TROPONIN_I_all_hours": "mean troponin I",
    "mean_CREATININE_all_hours": "mean creatinine",
    "mean_BUN_all_hours": "mean BUN",
    "mean_PH_all_hours": "mean pH",
    "mean_VIS_all_hours": "mean VIS",
    "mean_PCWP_all_hours": "mean PCWP",
    "mean_CO_all_hours": "mean CO",
    "mean_CI_all_hours": "mean CI",
    "mean_SCAI_STAGE_all_hours": "mean SCAI (events)",
    "n_procedures": "n procedures (rows)",
    "n_medications": "n distinct medications",
    "n_med_admin": "n med administrations",
    "LOS_HOURS": "LOS (hours)",
    "age_at_admit": "age at admit",
}


def _load_mortality_module():
    path = _REPO / "Frontend" / "lib" / "models" / "mortality_model.py"
    spec = importlib.util.spec_from_file_location("mortality_model", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _resolve_repo_path(p: Path) -> Path:
    if p.is_absolute():
        return p.resolve()
    return (_REPO / p).resolve()


def _nbins(s: pd.Series, d: pd.Series) -> int:
    n = max(len(s), len(d), 1)
    return int(max(8, min(45, round(np.sqrt(n)))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("Baptist_tester/synth_cs_data"))
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output HTML (default: <data-dir>/mortality_feature_dist_plotly.html)",
    )
    ap.add_argument("--no-open", action="store_true", help="Do not open a browser tab")
    ap.add_argument("--cols", type=int, default=3, help="Subplot columns (default 3)")
    args = ap.parse_args()

    data_dir = _resolve_repo_path(args.data_dir)
    out_path = (
        _resolve_repo_path(args.out) if args.out else data_dir / "mortality_feature_dist_plotly.html"
    )

    mm = _load_mortality_module()
    df = mm.build_patient_frame(data_dir)
    y = df["died"].to_numpy(dtype=int)
    n0, n1 = int((y == 0).sum()), int((y == 1).sum())

    features = [c for c in mm.FEATURE_COLUMNS if c in df.columns]
    if not features:
        print("No feature columns found in patient frame.", file=sys.stderr)
        return 1

    ncols = max(1, min(args.cols, len(features)))
    nrows = int(np.ceil(len(features) / ncols))
    n_slots = nrows * ncols

    subplot_titles = [SUBPLOT_TITLE.get(f, f) for f in features] + [""] * (n_slots - len(features))

    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=subplot_titles,
        vertical_spacing=0.05,
        horizontal_spacing=0.06,
    )

    for idx, feat in enumerate(features):
        row = idx // ncols + 1
        col = idx % ncols + 1
        surv = pd.to_numeric(df.loc[y == 0, feat], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        died = pd.to_numeric(df.loc[y == 1, feat], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        bins = _nbins(surv, died)
        show_leg = idx == 0

        if len(surv) == 0 and len(died) == 0:
            fig.add_trace(
                go.Scatter(
                    x=[0.5],
                    y=[0.5],
                    mode="text",
                    text=["insufficient<br>non-null data"],
                    textposition="middle center",
                    showlegend=False,
                ),
                row=row,
                col=col,
            )
            continue

        if len(surv) >= 1:
            fig.add_trace(
                go.Histogram(
                    x=surv,
                    name="Survived",
                    legendgroup="survived",
                    showlegend=show_leg,
                    marker_color="#16a34a",
                    opacity=0.55,
                    nbinsx=bins,
                    histnorm="percent",
                    hovertemplate=f"{feat}<br>survived %{{x:.4g}}<br>% of survived %{{y:.2f}}%<extra></extra>",
                ),
                row=row,
                col=col,
            )
        if len(died) >= 1:
            fig.add_trace(
                go.Histogram(
                    x=died,
                    name="Died",
                    legendgroup="died",
                    showlegend=show_leg,
                    marker_color="#dc2626",
                    opacity=0.55,
                    nbinsx=bins,
                    histnorm="percent",
                    hovertemplate=f"{feat}<br>died %{{x:.4g}}<br>% of died %{{y:.2f}}%<extra></extra>",
                ),
                row=row,
                col=col,
            )
        fig.update_xaxes(title_text="value", row=row, col=col, title_font_size=10)
        fig.update_yaxes(
            title_text="% of group (sums to 100% per color)",
            row=row,
            col=col,
            title_font_size=9,
        )

    for idx in range(len(features), n_slots):
        r = idx // ncols + 1
        c = idx % ncols + 1
        fig.update_xaxes(visible=False, row=r, col=c)
        fig.update_yaxes(visible=False, row=r, col=c)

    fig.update_layout(
        title_text=(
            f"Feature distributions by mortality (% of patients within each outcome): survived vs died "
            f"(n={len(y)}, died={n1}, survived={n0}, death rate={y.mean():.1%})"
        ),
        template="plotly_white",
        barmode="overlay",
        height=min(3200, 220 + nrows * 240),
        width=min(1400, 120 + ncols * 300),
        margin=dict(t=100, l=50, r=40, b=40),
        font=dict(size=11),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn", full_html=True)
    print(f"Wrote {out_path}")
    if not args.no_open:
        webbrowser.open(out_path.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
