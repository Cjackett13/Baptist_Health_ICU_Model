"""
SHAP computation for any XGBoost-backed model.

Always run SHAP on the raw XGBoost model (before calibration wrappers)
so values reflect true feature attribution, not the calibration transform.
"""

import shap
import pandas as pd
from app.schemas.output import ShapValueOut

_explainer_cache: dict = {}


def compute_shap(
    model_key: str,
    xgb_model,
    X: pd.DataFrame,
    top_k: int = 5,
) -> list[ShapValueOut]:
    """
    Compute SHAP values for a single-row feature DataFrame.

    Args:
        model_key:  Stable string key used to cache the TreeExplainer.
        xgb_model:  The raw XGBoost model (not a wrapper).
        X:          Feature DataFrame, shape (1, n_features).
        top_k:      Number of top features to return by |SHAP|.
    """
    if model_key not in _explainer_cache:
        _explainer_cache[model_key] = shap.TreeExplainer(xgb_model)
    explainer = _explainer_cache[model_key]

    raw = explainer.shap_values(X)
    # For binary classifiers shap_values() may return a list [neg_class, pos_class]
    if isinstance(raw, list):
        raw = raw[1]

    row = raw[0]
    pairs = sorted(
        zip(X.columns.tolist(), row),
        key=lambda t: abs(t[1]),
        reverse=True,
    )[:top_k]

    result: list[ShapValueOut] = []
    for feat, val in pairs:
        is_pos = float(val) >= 0
        result.append(ShapValueOut(
            feature=feat,
            value=round(float(val), 4),
            is_positive=is_pos,
            direction="positive" if is_pos else "negative",
            description=feat.replace("_", " ").title(),
        ))
    return result
