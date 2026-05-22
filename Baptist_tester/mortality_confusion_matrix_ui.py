#!/usr/bin/env python3
"""
Interactive held-out test confusion matrix for the mortality XGBoost bundle.

  python3 Baptist_tester/mortality_confusion_matrix_ui.py --data-dir Baptist_tester/synth_cs_data

Default port **7862** (mortality_voice_chat often uses 7860/7861).
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
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

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


def _register_mortality_pickles(mm: object) -> None:
    main = sys.modules.get("__main__")
    if main is None:
        return
    for name in ("TemperatureScaledBinaryCalibrator", "AveragedBinaryCalibrators"):
        if hasattr(main, name):
            continue
        if hasattr(mm, name):
            setattr(main, name, getattr(mm, name))


def _resolve_repo_path(p: Path) -> Path:
    return p.resolve() if p.is_absolute() else (_REPO / p).resolve()


def find_listen_port(host: str, preferred: int, n: int = 30) -> int:
    for port in range(preferred, preferred + n):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind((host, port))
            except OSError:
                continue
            return port
    raise OSError(f"No free port in {preferred}..{preferred + n - 1} on {host}")


def load_test_predictions(
    data_dir: Path,
    model_dir: Path,
) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    mm = _load_mortality_module()
    _register_mortality_pickles(mm)

    meta_path = model_dir / "xgb_mortality_model_meta.json"
    bundle_path = model_dir / "xgb_mortality_pipeline.joblib"
    if not bundle_path.is_file():
        raise FileNotFoundError(f"Missing {bundle_path}")

    meta: dict = {}
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    blob = joblib.load(bundle_path)
    pipe = blob["pipeline"]
    feat_saved: list[str] = list(blob.get("feature_names", meta.get("feature_columns", [])))

    _, X_test, _, y_test, _, _, _ = mm.prepare_train_test_features(
        data_dir,
        pct_lo=float(meta.get("outlier_pct_lo", 1.0)),
        pct_hi=float(meta.get("outlier_pct_hi", 99.0)),
        random_state=int(meta.get("split_seed", 42)),
        feature_cols=meta.get("feature_columns"),
    )

    proba = pipe.predict_proba(X_test[feat_saved])[:, 1]
    y = np.asarray(y_test).astype(int)
    return y, proba, feat_saved, meta


def metrics_at_threshold(y: np.ndarray, proba: np.ndarray, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    ppv = float(precision_score(y, pred, zero_division=0))
    rec = float(recall_score(y, pred, zero_division=0))
    f1 = float(f1_score(y, pred, zero_division=0))
    acc = float((tp + tn) / max(len(y), 1))
    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "ppv": ppv,
        "recall": rec,
        "f1": f1,
        "accuracy": acc,
        "threshold": float(threshold),
        "n_test": int(len(y)),
        "n_deaths": int(y.sum()),
    }


def confusion_heatmap(m: dict) -> go.Figure:
    z = [[m["tn"], m["fp"]], [m["fn"], m["tp"]]]
    labels = [
        ["TN<br>survived ✓", "FP<br>false alarm"],
        ["FN<br>missed death", "TP<br>death caught"],
    ]
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=["Pred: survived", "Pred: death"],
            y=["True: survived", "True: death"],
            text=[[f"{v}<br>{labels[i][j]}" for j, v in enumerate(row)] for i, row in enumerate(z)],
            texttemplate="%{text}",
            textfont={"size": 14},
            colorscale=[[0, "#e8f5e9"], [0.5, "#fff9c4"], [1, "#ffcdd2"]],
            showscale=True,
            colorbar={"title": "Count"},
        )
    )
    fig.update_layout(
        title=(
            f"Test confusion matrix (n={m['n_test']}, deaths={m['n_deaths']}) "
            f"@ threshold {m['threshold']:.2f}"
        ),
        template="plotly_white",
        height=420,
        margin=dict(t=60, b=40, l=80, r=40),
    )
    return fig


def summary_markdown(m: dict, meta: dict) -> str:
    cm_bundle = meta.get("test_confusion_at_alert_threshold") or {}
    return (
        f"## Classification @ **{m['threshold']:.2f}**\n\n"
        f"| | |\n|---|---|\n"
        f"| **Accuracy** | **{m['accuracy']:.1%}** |\n"
        f"| **False negatives** | **{m['fn']}** |\n"
        f"| False positives | {m['fp']} |\n"
        f"| True positives | {m['tp']} |\n"
        f"| True negatives | {m['tn']} |\n"
        f"| PPV | {m['ppv']:.1%} |\n"
        f"| Recall | {m['recall']:.1%} |\n"
        f"| F1 | {m['f1']:.3f} |\n\n"
        f"*Bundle recommended alert threshold: **{float(meta.get('death_alert_threshold', 0.4)):.2f}** "
        f"(saved FN={cm_bundle.get('fn', '?')}, FP={cm_bundle.get('fp', '?')}).*"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("Baptist_tester/synth_cs_data"))
    ap.add_argument("--model-dir", type=Path, default=None)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7862)
    ap.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Initial slider value (default: death_alert_threshold from meta)",
    )
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--listen-all", action="store_true")
    args = ap.parse_args()

    data_dir = _resolve_repo_path(args.data_dir)
    model_dir = _resolve_repo_path(args.model_dir) if args.model_dir else data_dir
    listen_host = "0.0.0.0" if args.listen_all else args.host

    try:
        y, proba, _, meta = load_test_predictions(data_dir, model_dir)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    alert_thr = float(
        args.threshold if args.threshold is not None else meta.get("death_alert_threshold", 0.4)
    )
    m0 = metrics_at_threshold(y, proba, alert_thr)
    print(
        f"\n  Test n={m0['n_test']}, threshold={m0['threshold']:.2f}\n"
        f"  FN={m0['fn']}  FP={m0['fp']}  accuracy={m0['accuracy']:.1%}\n",
        file=sys.stderr,
    )

    auroc = meta.get("test_auroc")
    auprc = meta.get("test_auprc")
    brier = meta.get("test_brier")
    header = (
        "## Mortality model — test confusion matrix\n\n"
        "Held-out patients from the saved bundle. Drag the slider to explore the "
        "false-negative / false-positive tradeoff.\n\n"
    )
    if auroc is not None:
        header += (
            f"*Probabilistic metrics: AUROC={float(auroc):.3f}, "
            f"AUPRC={float(auprc):.3f}, Brier={float(brier):.3f}*\n\n"
        )

    def update(threshold: float) -> tuple[go.Figure, str]:
        m = metrics_at_threshold(y, proba, threshold)
        return confusion_heatmap(m), summary_markdown(m, meta)

    with gr.Blocks(title="Mortality confusion matrix") as demo:
        gr.Markdown(header)
        thresh = gr.Slider(
            minimum=0.05,
            maximum=0.95,
            value=alert_thr,
            step=0.01,
            label="Death probability threshold",
        )
        plot = gr.Plot(label="Confusion matrix")
        stats = gr.Markdown()
        demo.load(fn=update, inputs=[thresh], outputs=[plot, stats])
        thresh.change(fn=update, inputs=[thresh], outputs=[plot, stats])

    listen_port = find_listen_port(listen_host, args.port)
    if listen_port != args.port:
        print(f"  Port {args.port} busy — using {listen_port}\n", file=sys.stderr)
    url = f"http://127.0.0.1:{listen_port}/"
    print(
        f"\n  Confusion matrix UI: {url}\n"
        f"  FN={m0['fn']}  FP={m0['fp']}  @ threshold {alert_thr:.2f}\n"
        "  Press Ctrl+C to stop.\n",
        file=sys.stderr,
    )
    demo.launch(
        server_name=listen_host,
        server_port=listen_port,
        inbrowser=not args.no_browser,
        show_error=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
