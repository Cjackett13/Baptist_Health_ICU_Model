"""
Custom model wrapper classes for Christie's vasopressor models.

These must be defined (importable) before pickle.load() is called on
binary_alert_model.pkl and high_severity_classifier.pkl.
Class names and __init__ signatures must exactly match vasopressor_model/src/inference.py.
"""

import numpy as np


class IsotonicCalibratedModel:
    """XGBoost + isotonic regression calibration. Used by binary_alert_model.pkl."""

    def __init__(self, base_model, iso_reg):
        self.base_model = base_model
        self.iso_reg = iso_reg

    def predict_proba(self, X):
        raw = self.base_model.predict_proba(X)[:, 1]
        cal = np.clip(self.iso_reg.predict(raw), 0.0, 1.0)
        return np.column_stack([1.0 - cal, cal])


class PlattXGB:
    """XGBoost + Platt sigmoid calibration. Used by high_severity_classifier.pkl."""

    def __init__(self, xgb_model, platt_lr, threshold):
        self.xgb_model = xgb_model
        self.platt_lr = platt_lr
        self.threshold = threshold

    def predict_proba(self, X):
        raw = self.xgb_model.predict_proba(X)[:, 1].reshape(-1, 1)
        cal = self.platt_lr.predict_proba(raw)[:, 1]
        return np.column_stack([1.0 - cal, cal])
