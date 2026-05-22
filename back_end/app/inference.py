"""Backward-compatible re-exports (use ``mortality_inference`` for full bundle)."""

from __future__ import annotations

from app.mortality_inference import (
    predict_mortality_from_features,
    predict_mortality_legacy_eight as predict_mortality,
)

__all__ = ["predict_mortality", "predict_mortality_from_features"]
