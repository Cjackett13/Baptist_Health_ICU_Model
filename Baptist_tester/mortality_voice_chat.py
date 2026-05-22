#!/usr/bin/env python3
"""
Simple browser UI for the synthetic ICU mortality demo.

  pip install gradio pandas pyarrow numpy scikit-learn xgboost joblib

  python Baptist_tester/mortality_voice_chat.py
  python Baptist_tester/mortality_voice_chat.py --port 7861 --no-browser

**Current SCAI (A–E)** anchors in-hospital mortality to registry bands (A &lt;1%, B ~5%, C ~25%,
D ~45%, E ~70%), blended with the trained XGBoost pipeline; other filled fields tilt within the band.
Blank optional fields stay missing (no imputation).

Synthetic demo only — **not medical advice**.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import socket
import sys
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent


def _load_mortality_module():
    models_dir = _REPO / "Frontend" / "lib" / "models"
    if str(models_dir) not in sys.path:
        sys.path.insert(0, str(models_dir))
    path = models_dir / "mortality_model.py"
    spec = importlib.util.spec_from_file_location("mortality_model", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_mm = _load_mortality_module()
_resolve_repo_path = _mm._resolve_repo_path


def _register_mortality_pickles() -> None:
    """Bundles trained as ``python mortality_model.py`` pickle nested estimators under
    ``__main__.*``. Register the real classes on ``sys.modules['__main__']`` so
    ``joblib.load`` succeeds when this script is ``__main__``."""
    main = sys.modules.get("__main__")
    if main is None:
        return
    for name in ("TemperatureScaledBinaryCalibrator", "AveragedBinaryCalibrators"):
        if hasattr(main, name):
            continue
        if hasattr(_mm, name):
            setattr(main, name, getattr(_mm, name))


_register_mortality_pickles()

# (column, label, lo, hi, step) — clip bounds + UI hints for the SCAI-band demo (synthetic-cohort scale).
SLIDER_FEATURES: list[tuple[str, str, float, float, float]] = [
    ("med_distinct", "Distinct medications", 0.0, 10.0, 1.0),
    ("med_infusion_mean", "Mean infusion rate", 0.0, 5.0, 0.1),
    ("proc_n", "Number of procedures", 0.0, 10.0, 1.0),
    ("proc_distinct_nom", "Different Types of Procedures", 0.0, 10.0, 1.0),
    ("current_scai", "Current SCAI (most recent hourly)", 0.0, 4.0, 1.0),
    ("scai_prop_ge3", "Percentage of time SCAI was ≥ D", 0.0, 100.0, 1.0),
    ("scai_std", "SCAI hourly variability (std dev)", 0.0, 2.0, 0.1),
    ("scai_slope_12h", "SCAI change first 12h (stage units)", -5.0, 5.0, 0.1),
    ("infusion_early_over_all_ratio", "Early infusion vs. total infusion", 0.0, 5.0, 0.05),
    ("clinical_event_n", "Clinical event count (approx)", 0.0, 10000.0, 1.0),
    ("med_per_clinical_event", "Med administrations per clinical event", 0.0, 0.1, 0.01),
    ("proc_n_bucket", "Bucketed number of procedures", 0.0, 2.0, 1.0),
    ("med_distinct_bucket", "Bucketed number of medications", 0.0, 2.0, 1.0),
    ("med_residual_within_scai", "Distinct meds relative to SCAI", -5.0, 5.0, 0.1),
    ("infusion_residual_within_scai", "Infusion relative to SCAI", -5.0, 5.0, 0.05),
    ("proc_residual_within_scai", "Procedures relative to SCAI", -5.0, 5.0, 0.1),
    ("demo_profile_bucket", "Demographics bucket (age × weight × sex)", 0.0, 23.0, 1.0),
]

# ≤10 words each; shown under the range hint in the Gradio form.
_FEATURE_EXPLAIN: dict[str, str] = {
    "med_distinct": "Count of distinct medication codes for this encounter.",
    "med_infusion_mean": "Mean infusion rate across medication administration rows.",
    "proc_n": "Total procedure rows documented for the patient.",
    "proc_distinct_nom": "Count of distinct procedure nomenclature codes.",
    "current_scai": "Latest shock stage A–E from hourly SCAI (numeric 0–4).",
    "scai_prop_ge3": "Fraction of hourly SCAI rows at D or worse.",
    "scai_std": "Standard deviation of hourly SCAI stage within the encounter.",
    "scai_slope_12h": "Stage change from first to last read within twelve hours.",
    "infusion_early_over_all_ratio": "Mean infusion first 12h divided by encounter-wide mean.",
    "clinical_event_n": "Count of rows in the clinical_event table for the patient.",
    "med_per_clinical_event": "Med admin rows divided by clinical events plus one.",
    "proc_n_bucket": "Buckets: none; one to three procedures; over three.",
    "med_distinct_bucket": "Buckets: up to two meds; three to five; over five.",
    "med_residual_within_scai": "Distinct meds minus average for same rounded stage.",
    "infusion_residual_within_scai": "Mean infusion minus average for same rounded stage.",
    "proc_residual_within_scai": "Procedure count minus average for same rounded stage.",
    "demo_profile_bucket": "Ordinal bucket 0–23 from age, weight, and sex bands (low model weight).",
}

_SLIDER_BY_COL: dict[str, tuple[str, str, float, float, float]] = {r[0]: r for r in SLIDER_FEATURES}


def aligned_slider_max(lo: float, hi: float, step: float) -> float:
    if step <= 0 or hi <= lo:
        return float(hi)
    n = int(round((hi - lo) / step))
    if n < 0:
        n = 0
    out = lo + n * step
    tol = 1e-9 * max(1.0, abs(hi))
    while out > hi + tol and n > 0:
        n -= 1
        out = lo + n * step
    return float(out)


def load_artifacts(data_dir: Path, model_dir: Path) -> tuple[object, list[str], dict]:
    bundle = model_dir / "xgb_mortality_pipeline.joblib"
    meta_path = model_dir / "xgb_mortality_model_meta.json"
    if not bundle.is_file():
        raise FileNotFoundError(
            f"Missing {bundle}. Train first:\n"
            f"  python3 Frontend/lib/models/mortality_model.py "
            f"--data-dir Baptist_tester/synth_cs_data"
        )
    models_dir = _REPO / "Frontend" / "lib" / "models"
    if str(models_dir) not in sys.path:
        sys.path.insert(0, str(models_dir))
    blob = joblib.load(bundle)
    pipe = blob["pipeline"]
    feature_names = list(blob["feature_names"])
    meta: dict = {}
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return pipe, feature_names, meta


def _spec_for(col: str) -> tuple[str, str, float, float, float]:
    if col in _SLIDER_BY_COL:
        return _SLIDER_BY_COL[col]
    return (col, col.replace("_", " ").title(), 0.0, 1e12, 1.0)


def _hint_range_html(col: str, lo: float, hi: float) -> str:
    """Compact green–red range (e.g. ``0-100%``, ``0-20``); en dash when lower bound is negative."""
    g = "#15803d"
    r = "#b91c1c"
    gs = f"color:{g};font-weight:700"
    rs = f"color:{r};font-weight:700"
    ndash = "\u2013"

    if col == "scai_prop_ge3":
        return f"<span style='{gs}'>0</span>-<span style='{rs}'>100%</span>"
    if col == "current_scai":
        return f"<span style='{gs}'>A</span>{ndash}<span style='{rs}'>E</span>"
    if col in ("proc_n_bucket", "med_distinct_bucket"):
        loi, hii = int(round(lo)), int(round(hi))
        return f"<span style='{gs}'>{loi}</span>-<span style='{rs}'>{hii}</span>"

    def fmt_num(v: float) -> str:
        iv = int(round(v))
        if abs(v - iv) < 1e-6:
            return str(iv)
        return f"{v:g}"

    lo_s, hi_s = fmt_num(lo), fmt_num(hi)
    if lo_s.startswith("-"):
        return f"<span style='{gs}'>{lo_s}</span>{ndash}<span style='{rs}'>{hi_s}</span>"
    return f"<span style='{gs}'>{lo_s}</span>-<span style='{rs}'>{hi_s}</span>"


def _hint_markdown(col: str, label: str, lo: float, hi: float) -> str:
    """Green = survival-favorable end of range; red = mortality-favorable end (demo UI only)."""
    range_html = _hint_range_html(col, lo, hi)
    head = f"**{label}** — {range_html}"
    sub = _FEATURE_EXPLAIN.get(col)
    if sub:
        head += f"<br><small style='opacity:0.88'>{sub}</small>"
    tbl = "border-collapse:collapse;font-size:0.88em;margin-top:0.25em"
    th = "text-align:left;padding:1px 12px 1px 0"
    td = "padding:1px 12px 1px 0"
    if col == "proc_n_bucket":
        return (
            head
            + f"<br><small><table style='{tbl}'><tr>"
            + f"<th style='{th}'>Procedure rows</th><th style='{th}'>Input (0–2)</th></tr>"
            + f"<tr><td style='{td}'>0</td><td style='{td}'>0</td></tr>"
            + f"<tr><td style='{td}'>1–3</td><td style='{td}'>1</td></tr>"
            + f"<tr><td style='{td}'>&gt;3</td><td style='{td}'>2</td></tr>"
            + "</table></small>"
        )
    if col == "med_distinct_bucket":
        return (
            head
            + f"<br><small><table style='{tbl}'><tr>"
            + f"<th style='{th}'>Distinct meds</th><th style='{th}'>Input (0–2)</th></tr>"
            + f"<tr><td style='{td}'>≤2</td><td style='{td}'>0</td></tr>"
            + f"<tr><td style='{td}'>3–5</td><td style='{td}'>1</td></tr>"
            + f"<tr><td style='{td}'>&gt;5</td><td style='{td}'>2</td></tr>"
            + "</table></small>"
        )
    return head


def _clinical_event_stats_markdown(data_dir: Path) -> str:
    """Max clinical_event_n in DuckDB sidecar and how often that max appears."""
    p = data_dir / "duckdb_patient_features.parquet"
    if not p.is_file():
        return ""
    try:
        s = pd.read_parquet(p, columns=["clinical_event_n"])["clinical_event_n"]
        mx = float(s.max())
        n_at = int((s == mx).sum())
        return (
            "<small><strong>Clinical event count (dataset)</strong> in "
            "<code>duckdb_patient_features.parquet</code>: "
            f"highest value <strong>{mx:g}</strong>; "
            f"that value occurs <strong>{n_at}</strong> time(s) in <strong>{len(s)}</strong> patients.</small>"
        )
    except Exception:  # noqa: BLE001
        return ""


_SCAI_LETTER_TO_STAGE: dict[str, float] = {"A": 0.0, "B": 1.0, "C": 2.0, "D": 3.0, "E": 4.0}

_SCAI_STAGE_TO_P_DEATH_RANGE: dict[int, tuple[float, float]] = _mm.SCAI_STAGE_MORTALITY_RANGE


def _current_scai_from_ui(raw: object) -> float:
    """Map UI value to model stage 0–4: A=0 … E=4; empty = missing. Numeric strings still accepted."""
    if raw is None:
        return float("nan")
    if isinstance(raw, str):
        s = raw.strip().upper()
        if not s:
            return float("nan")
        if s in _SCAI_LETTER_TO_STAGE:
            return _SCAI_LETTER_TO_STAGE[s]
        try:
            return float(s)
        except ValueError:
            return float("nan")
    if isinstance(raw, (float, int, np.floating, np.integer)):
        if isinstance(raw, float) and np.isnan(raw):
            return float("nan")
        return float(raw)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float("nan")


# Within-SCAI-band tilt: weights from ``mortality_threshold_audit.json`` permutation-importance
# (mean drop in AUROC when shuffled; absolute value). Renormalized over modifier features only.
_PERM_IMPORTANCE_ABS: dict[str, float] = {
    "med_distinct": abs(0.021451612903225854),
    "med_infusion_mean": abs(-0.017806451612903194),
    "proc_n": abs(0.0012258064516129374),
    "proc_distinct_nom": abs(0.0029677419354838972),
    "scai_prop_ge3": abs(0.009129032258064579),
    "scai_std": abs(0.0151935483870968),
    "scai_slope_12h": 0.02,
    "infusion_early_over_all_ratio": 0.015,
    "clinical_event_n": 0.025,
    "med_per_clinical_event": 0.018,
    "proc_n_bucket": 0.008,
    "med_distinct_bucket": 0.008,
    "med_residual_within_scai": 0.02,
    "infusion_residual_within_scai": 0.018,
    "proc_residual_within_scai": 0.006,
    "current_scai": 0.42,
}
_pi_sum = sum(_PERM_IMPORTANCE_ABS.values()) or 1.0
_BAND_MODIFIER_WEIGHT: dict[str, float] = {k: v / _pi_sum for k, v in _PERM_IMPORTANCE_ABS.items()}


def _sort_feature_names_by_train_mi(cols: list[str], meta: dict) -> list[str]:
    """Stable descending sort by ``mutual_information_train_only`` (training label association)."""
    mi = meta.get("mutual_information_train_only")
    if not isinstance(mi, dict) or not mi:
        return list(cols)
    order = {c: i for i, c in enumerate(cols)}

    def sort_key(c: str) -> tuple[float, int]:
        return (-float(mi.get(c, 0.0)), order[c])

    return sorted(cols, key=sort_key)


def _display_feature_order(feature_names: list[str], meta: dict | None = None) -> list[str]:
    """``current_scai`` first (required); other fields by training MI (strongest association first)."""
    meta = meta or {}
    if "current_scai" not in feature_names:
        return _sort_feature_names_by_train_mi(list(feature_names), meta)
    rest = [c for c in feature_names if c != "current_scai"]
    return ["current_scai"] + _sort_feature_names_by_train_mi(rest, meta)


def _band_position_from_other_features(row: dict[str, float], feature_names: list[str]) -> float:
    """
    Map optional covariates to ``t`` in [0, 1]: 0 = healthier end of the SCAI band, 1 = sicker end.
    Each filled feature uses its UI range; higher values = higher mortality tilt. Weights follow
    permutation-importance magnitudes from the trained XGBoost audit (synthetic cohort).
    """
    num = 0.0
    den = 0.0
    for col in feature_names:
        if col == "current_scai":
            continue
        v = row.get(col)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        _c, _lab, lo, hi, step = _spec_for(col)
        hi_eff = aligned_slider_max(lo, hi, step)
        span = max(float(hi_eff - lo), 1e-12)
        x = float(np.clip(float(v), lo, hi_eff))
        r = float(np.clip((x - lo) / span, 0.0, 1.0))
        w = float(_BAND_MODIFIER_WEIGHT.get(col, 0.0))
        if w <= 0.0:
            w = 1.0 / max(len(feature_names) - 1, 1)
        num += w * r
        den += w
    if den <= 0.0:
        return 0.5
    return float(np.clip(num / den, 0.0, 1.0))


def _p_death_from_scai_bands(row: dict[str, float], feature_names: list[str]) -> tuple[float, int]:
    """Return (p_death, stage_int) when ``current_scai`` is finite; caller must check."""
    stage = int(round(float(row["current_scai"])))
    stage = max(0, min(4, stage))
    lo, hi = _SCAI_STAGE_TO_P_DEATH_RANGE[stage]
    t = _band_position_from_other_features(row, feature_names)
    p = lo + (hi - lo) * t
    return float(np.clip(p, 0.0, 1.0)), stage


def _blank_to_nan(v: object) -> float:
    if v is None:
        return float("nan")
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return float("nan")
        try:
            return float(s)
        except ValueError:
            return float("nan")
    if isinstance(v, (float, int, np.floating, np.integer)):
        if isinstance(v, float) and np.isnan(v):
            return float("nan")
        return float(v)
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def run_calculate(
    pipe: object,
    feature_names: list[str],
    display_order: list[str],
    meta: dict,
    *field_values: object,
) -> str:
    row: dict[str, float] = {}
    vals = list(field_values)
    if len(vals) != len(display_order):
        return (
            "Input mismatch — please refresh the page.\n"
            f"(Expected {len(display_order)} fields, got {len(vals)}.)"
        )
    by_disp = dict(zip(display_order, vals, strict=True))
    vals_model = [by_disp[c] for c in feature_names]

    for col, raw in zip(feature_names, vals_model, strict=True):
        v = _current_scai_from_ui(raw) if col == "current_scai" else _blank_to_nan(raw)
        if np.isnan(v):
            row[col] = float("nan")
            continue
        _c, _lab, lo, hi, step = _spec_for(col)
        hi_eff = aligned_slider_max(lo, hi, step)
        clipped = float(np.clip(v, lo, hi_eff))
        # Model was trained on fraction 0–1; UI collects 0–100 for this feature.
        row[col] = clipped / 100.0 if col == "scai_prop_ge3" else clipped

    if all(np.isnan(row.get(c, float("nan"))) for c in feature_names):
        return (
            "Enter at least one value in any field, then click Calculate Mortality Rate again.\n"
            "(Blank fields count as missing; XGBoost handles NaNs.)"
        )

    try:
        X = pd.DataFrame([{c: row.get(c, np.nan) for c in feature_names}])
        p_ml = float(pipe.predict_proba(X[feature_names])[:, 1][0])  # type: ignore[union-attr]
        p_ml = float(np.clip(p_ml, 0.0, 1.0))
    except Exception as exc:  # noqa: BLE001
        return f"Model prediction error: {exc}"

    p_death = p_ml
    if not np.isnan(row.get("current_scai", float("nan"))):
        t = _band_position_from_other_features(row, feature_names)
        p_death = _mm.apply_scai_registry_mortality_blend(
            p_ml,
            current_scai=float(row["current_scai"]),
            band_position_t=t,
        )

    p_surv = max(0.0, min(1.0, 1.0 - p_death))
    auroc = meta.get("test_auroc")
    auprc = meta.get("test_auprc")
    brier = meta.get("test_brier")
    metrics = ""
    if auroc is not None and auprc is not None and brier is not None:
        metrics = (
            f"\n\n*Saved bundle test metrics (n={meta.get('n_test', '?')}): "
            f"AUROC={float(auroc):.3f}, AUPRC={float(auprc):.3f}, Brier={float(brier):.3f}*"
        )

    stage_note = ""
    if not np.isnan(row.get("current_scai", float("nan"))):
        stage = int(round(float(row["current_scai"])))
        stage = max(0, min(4, stage))
        letter = "ABCDE"[stage]
        lo, hi = _SCAI_STAGE_TO_P_DEATH_RANGE[stage]
        stage_note = (
            f"\n\nRegistry SCAI band (max stage, synthetic cohort): "
            f"**{letter}** **{lo:.0%}–{hi:.0%}**; blend weight on band = "
            f"{_mm.SCAI_REGISTRY_BLEND_WEIGHT:.0%}."
        )

    blend_note = ""
    if not np.isnan(row.get("current_scai", float("nan"))):
        blend_note = (
            f"\nXGBoost raw: **{p_ml:.1%}** → registry-blended: **{p_death:.1%}** "
            f"(current SCAI drives the band ceiling)."
        )

    return (
        "Result (synthetic demo — not medical advice)\n\n"
        f"**In-hospital death probability:** **{p_death:.1%}**{blend_note}\n"
        f"Implied survival to discharge (1 − mortality): **{p_surv:.1%}**"
        f"{metrics}"
        f"{stage_note}\n\n"
        "Blank optional fields are left missing (not imputed). "
        f"Active features: {len(feature_names)} (includes `demo_profile_bucket` at low weight)."
    )


def find_listen_port(host: str, preferred: int, n: int = 30) -> int:
    for port in range(preferred, preferred + n):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind((host, port))
            except OSError:
                continue
            return port
    raise OSError(
        f"No free TCP port in {preferred}..{preferred + n - 1} on {host}. "
        "Stop other Gradio apps or pass --port with a free value."
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("Baptist_tester/synth_cs_data"))
    ap.add_argument("--model-dir", type=Path, default=None, help="Defaults to --data-dir")
    ap.add_argument("--host", default="127.0.0.1", help="Bind address")
    ap.add_argument("--port", type=int, default=7860, help="Preferred port")
    ap.add_argument("--share", action="store_true", help="Create a temporary public Gradio link")
    ap.add_argument("--listen-all", action="store_true", help="Bind 0.0.0.0")
    ap.add_argument("--no-browser", action="store_true", help="Do not open a browser tab")
    args = ap.parse_args()
    listen_host = "0.0.0.0" if args.listen_all else args.host

    data_dir = _resolve_repo_path(args.data_dir)
    model_dir = _resolve_repo_path(args.model_dir) if args.model_dir else data_dir

    try:
        pipe, feature_names, meta = load_artifacts(data_dir, model_dir)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 1

    field_inputs: list[gr.Textbox | gr.Dropdown] = []

    with gr.Blocks(title="Mortality model (demo)") as demo:
        gr.Markdown(
            "## In-hospital mortality — **synthetic** demo\n\n"
            "Uses the saved **XGBoost mortality pipeline** (`predict_proba`) from "
            f"`{model_dir.name}/xgb_mortality_pipeline.joblib`.\n\n"
            "Fill **Current SCAI (A–E)** and any optional clinical fields, then click "
            "**Calculate Mortality Rate**. Leave boxes empty for missing values.\n\n"
            "**Not medical advice.**"
        )
        ce_line = _clinical_event_stats_markdown(data_dir)
        if ce_line:
            gr.Markdown(ce_line)
        if meta:
            cm = meta.get("test_confusion_at_alert_threshold") or {}
            thr = meta.get("death_alert_threshold", 0.5)
            gr.Markdown(
                f"*Held-out test: AUROC={float(meta.get('test_auroc', 0)):.3f}, "
                f"AUPRC={float(meta.get('test_auprc', 0)):.3f}, Brier={float(meta.get('test_brier', 0)):.3f} "
                f"({meta.get('n_test', '?')} patients). "
                f"Alert threshold **{float(thr):.2f}**: accuracy **{float(cm.get('accuracy', 0)):.1%}**, "
                f"FN={cm.get('fn', '?')}, FP={cm.get('fp', '?')} "
                f"(TN={cm.get('tn', '?')}, TP={cm.get('tp', '?')}).*"
            )

        display_order = _display_feature_order(feature_names, meta)

        gr.Markdown(
            "### Current SCAI (required), then optional modifiers\n\n"
            "*Optional rows are ordered by **training mutual information** with death (higher ≈ stronger linear association on the train split; not causal.)*"
        )
        for col in display_order:
            _c, label, lo, hi, step = _spec_for(col)
            hi_eff = aligned_slider_max(lo, hi, step)
            hint = _hint_markdown(col, label, lo, hi_eff)
            with gr.Row():
                gr.Markdown(hint, elem_id=None)
                if col == "current_scai":
                    dd = gr.Dropdown(
                        choices=["", "A", "B", "C", "D", "E"],
                        value="",
                        label="",
                        info="Sets mortality range (A–E); other fields refine inside the range",
                        container=True,
                    )
                    field_inputs.append(dd)
                else:
                    tb = gr.Textbox(
                        label="",
                        value="",
                        placeholder="optional",
                        lines=1,
                        max_lines=1,
                        container=True,
                    )
                    field_inputs.append(tb)

        def _on_calculate(*vals: object) -> str:
            return run_calculate(pipe, feature_names, display_order, meta, *vals)

        calc_btn = gr.Button("Calculate Mortality Rate", variant="primary", size="lg")
        pred_out = gr.Textbox(
            label="Mortality estimate",
            value="Click Calculate Mortality Rate when you are ready. The result appears here.",
            lines=10,
            max_lines=20,
            interactive=False,
        )

        calc_btn.click(
            fn=_on_calculate,
            inputs=field_inputs,
            outputs=pred_out,
        )

    listen_port = find_listen_port(listen_host, args.port)
    if listen_port != args.port:
        print(
            f"\n  Port {args.port} was busy — using {listen_port} instead.\n",
            file=sys.stderr,
        )
    local_url = f"http://127.0.0.1:{listen_port}/"
    print(
        "\n  Starting server — **leave this terminal open** while you use the app.\n"
        f"  Local URL: {local_url}\n"
        "  Press Ctrl+C when finished.\n",
        file=sys.stderr,
    )
    demo.launch(
        server_name=listen_host,
        server_port=listen_port,
        share=args.share,
        inbrowser=not args.no_browser,
        show_error=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
