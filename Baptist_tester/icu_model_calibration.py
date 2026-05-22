"""Pickle-safe helpers for probability calibration (used by model_development)."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def logit_temperature_probs(p: np.ndarray, temperature: float) -> np.ndarray:
    """Strictly monotone in p for T>0; preserves AUROC / AUPRC ranking."""
    T = max(float(temperature), 1e-6)
    eps = 1e-7
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    z = np.log(p / (1.0 - p)) / T
    out = 1.0 / (1.0 + np.exp(-z))
    return out


class TemperatureScaledClassifier:
    """Wraps a fitted classifier and applies logit temperature scaling on predict_proba."""

    def __init__(self, wrapped: Any, temperature: float = 1.0):
        self.wrapped = wrapped
        self.temperature = float(temperature)
        self.classes_ = np.asarray(getattr(wrapped, "classes_", [0, 1]))

    def predict_proba(self, X) -> np.ndarray:
        p = self.wrapped.predict_proba(X)[:, 1]
        p2 = logit_temperature_probs(p, self.temperature)
        return np.column_stack([1.0 - p2, p2])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def maybe_refine_logit_temperature(
    model: Any,
    X_val,
    y_val: np.ndarray,
) -> Tuple[Any, Dict[str, float]]:
    """Pick T>=0 that minimizes validation Brier; ranking unchanged so AUROC/AUPRC stable."""
    p0 = model.predict_proba(X_val)[:, 1]
    base = dict(
        auroc=float(roc_auc_score(y_val, p0)),
        auprc=float(average_precision_score(y_val, p0)),
        brier=float(brier_score_loss(y_val, p0)),
    )
    Ts = np.exp(np.linspace(np.log(0.42), np.log(2.05), 36))
    best_T, best_b, best_p = 1.0, base["brier"], p0
    for T in Ts:
        p = logit_temperature_probs(p0, float(T))
        b = float(brier_score_loss(y_val, p))
        if b < best_b:
            best_b = b
            best_T = float(T)
            best_p = p
    if best_T != 1.0 and best_b < base["brier"] - 1e-7:
        return TemperatureScaledClassifier(model, best_T), dict(
            used_temperature=True,
            temperature=best_T,
            auroc=float(roc_auc_score(y_val, best_p)),
            auprc=float(average_precision_score(y_val, best_p)),
            brier=best_b,
            brier_before_temperature=base["brier"],
        )
    return model, dict(used_temperature=False, **base, brier_before_temperature=base["brier"])
