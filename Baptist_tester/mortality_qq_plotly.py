#!/usr/bin/env python3
"""
Plotly empirical Q–Q plots by mortality (died vs survived) for each model covariate.

For each continuous feature, x = quantiles among **survived**, y = quantiles among **died**
(same probability grid). Points above the y=x line mean decedents take **higher** values at
that rank (for features where higher = worse, that matches higher mortality risk).

Feature order matches the mortality model covariates from ``FEATURE_COLUMNS``.

  pip install plotly pandas pyarrow numpy
  python3 Baptist_tester/mortality_qq_plotly.py
  python3 Baptist_tester/mortality_qq_plotly.py --no-open

No trained XGBoost file required — uses only the synthetic parquet bundle.
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

# Short titles for subplots (full clinical wording lives in docs / model README)
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


def empirical_qq(x: np.ndarray, y: np.ndarray, nq: int = 99) -> tuple[np.ndarray, np.ndarray]:
    if len(x) < 2 or len(y) < 2:
        return np.array([]), np.array([])
    qs = np.linspace(0.01, 0.99, nq)
    return np.quantile(x, qs), np.quantile(y, qs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("Baptist_tester/synth_cs_data"))
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output HTML (default: <data-dir>/mortality_qq_plotly.html)",
    )
    ap.add_argument("--no-open", action="store_true", help="Do not open a browser tab")
    ap.add_argument("--cols", type=int, default=3, help="Subplot columns (default 3)")
    args = ap.parse_args()

    data_dir = _resolve_repo_path(args.data_dir)
    out_path = _resolve_repo_path(args.out) if args.out else data_dir / "mortality_qq_plotly.html"

    mm = _load_mortality_module()
    df = mm.build_patient_frame(data_dir)
    y = df["died"].to_numpy(dtype=int)
    n0, n1 = int((y == 0).sum()), int((y == 1).sum())

    # Same order as mortality model; only columns present in the frame
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
        qx, qy = empirical_qq(surv.to_numpy(dtype=float), died.to_numpy(dtype=float))

        if len(qx) == 0:
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

        fig.add_trace(
            go.Scatter(
                x=qx,
                y=qy,
                mode="markers",
                marker=dict(size=5, color="#2563eb"),
                showlegend=False,
                hovertemplate=f"{feat}<br>survived q: %{{x:.4g}}<br>died q: %{{y:.4g}}<extra></extra>",
            ),
            row=row,
            col=col,
        )
        lo = float(min(np.min(qx), np.min(qy)))
        hi = float(max(np.max(qx), np.max(qy)))
        if lo == hi:
            hi = lo + 1e-9
        fig.add_trace(
            go.Scatter(
                x=[lo, hi],
                y=[lo, hi],
                mode="lines",
                line=dict(dash="dash", color="#94a3b8", width=1),
                showlegend=False,
            ),
            row=row,
            col=col,
        )
        fig.update_xaxes(title_text="survived", row=row, col=col, title_font_size=10)
        fig.update_yaxes(title_text="died", row=row, col=col, title_font_size=10)

    # Blank unused subplots in last row
    for idx in range(len(features), n_slots):
        row = idx // ncols + 1
        col = idx % ncols + 1
        fig.update_xaxes(visible=False, row=row, col=col)
        fig.update_yaxes(visible=False, row=row, col=col)

    fig.update_layout(
        title_text=(
            f"Q–Q by mortality: survived (x) vs died (y) per feature "
            f"(n={len(y)}, died={n1}, survived={n0}, death rate={y.mean():.1%})"
        ),
        template="plotly_white",
        height=min(3200, 220 + nrows * 240),
        width=min(1400, 120 + ncols * 300),
        margin=dict(t=100, l=50, r=40, b=40),
        font=dict(size=11),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn", full_html=True)
    print(f"Wrote {out_path}")
    if not args.no_open:
        webbrowser.open(out_path.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
