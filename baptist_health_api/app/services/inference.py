"""
Central inference service — runs all six models for one patient encounter
and assembles a PredictionsOut response.

Christie's models (SCAI + vasopressor) are fully wired; stubs exist for
Johnathan's (MCS/VA-ECMO) and Lilly's (mortality/LOS) until they add their
feature extractors and model files.
"""

import numpy as np
from datetime import datetime, timezone

from app.models import loader
from app.models.features import (
    scai_features,
    vasopressor_features,
    mcs_features,
    va_ecmo_features,
    mortality_features,
    los_features,
)
from app.schemas.output import PredictionsOut
from app.services import shap_service, derive

# Thresholds from deployment_config_clean.json (approximate; load from file if available)
_SCAI_THRESHOLDS: dict[int, float] = {0: 0.15, 1: 0.15, 2: 0.15, 3: 0.02}
_VASOPRESSOR_THRESHOLD = 0.1882
_HIGH_SEVERITY_THRESHOLD = 0.50

_SCAI_STAGE_LABEL = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}


def _unwrap_xgb(model):
    """Strip calibration/ensemble wrappers to reach the first raw XGBClassifier for SHAP."""
    if hasattr(model, "base"):                 # SigmoidCalibratedModel / BetaCalibratedModel
        model = model.base
    if hasattr(model, "models"):               # EnsembleXGBClassifier
        model = model.models[0]
    if hasattr(model, "calibrated_estimator"): # TemperatureScaledBinaryCalibrator
        model = model.calibrated_estimator
    if hasattr(model, "calibrators"):          # AveragedBinaryCalibrators
        model = model.calibrators[0]
    if hasattr(model, "estimator"):            # CalibratedClassifierCV
        model = model.estimator
    elif hasattr(model, "base_estimator"):
        model = model.base_estimator
    if hasattr(model, "steps"):               # sklearn Pipeline — pick final step
        model = model.steps[-1][1]
    return model


async def run_all_models(
    encounter_id: int,
    hour_from_admit: int,
    raw_data: dict | None = None,
) -> PredictionsOut:
    """
    Run all available models and return a complete PredictionsOut.

    Args:
        encounter_id:    Patient encounter identifier.
        hour_from_admit: ICU hour aligned to the scoring row.
        raw_data:        Pre-fetched patient clinical data dict. If None,
                         fetches via data_access.get_patient_data().
    """
    if raw_data is None:
        from app.services.data_access import get_patient_data
        raw_data = await get_patient_data(encounter_id, hour_from_admit)

    # ── Christie — SCAI deterioration ─────────────────────────────────────────
    scai_bundle = loader.get("scai")
    scai_feature_cols: list[str] = scai_bundle["feature_cols"]
    scai_df = scai_features.extract(raw_data, feature_cols=scai_feature_cols)

    scai_raw = scai_bundle["xgboost_model"].predict_proba(scai_df)[:, 1]
    scai_prob = float(
        scai_bundle["platt_scaler"].predict_proba(scai_raw.reshape(-1, 1))[:, 1][0]
    )
    scai_stage_num = int(scai_df["SCAI_STAGE_NUM"].iloc[0])
    scai_threshold = _SCAI_THRESHOLDS.get(scai_stage_num, 0.15)
    current_scai_stage = _SCAI_STAGE_LABEL.get(scai_stage_num, "B")
    scai_label = "Likely to worsen" if scai_prob >= scai_threshold else "Unlikely to worsen"

    # ── Christie — vasopressor ────────────────────────────────────────────────
    vaso_df = vasopressor_features.extract(raw_data)

    vaso_prob = float(
        loader.get("vasopressor_binary").predict_proba(vaso_df)[:, 1][0]
    )
    count_probs = loader.get("vasopressor_count").predict_proba(vaso_df)
    vaso_count = int(np.argmax(count_probs, axis=1)[0])

    high_sev_prob = float(
        loader.get("vasopressor_severity").predict_proba(vaso_df)[:, 1][0]
    )
    critical_alert = high_sev_prob >= _HIGH_SEVERITY_THRESHOLD

    # ── Johnathan — MCS ──────────────────────────────────────────────────────
    mcs_prob, mcs_needed, shap_mcs = 0.15, False, []
    if loader.get("mcs") is not None:
        try:
            mcs_df        = mcs_features.extract(raw_data)
            mcs_bundle    = loader.get("mcs")
            mcs_model     = mcs_bundle["model"]
            mcs_threshold = mcs_bundle.get("default_threshold", 0.10)
            mcs_prob      = float(mcs_model.predict_proba(mcs_df)[:, 1][0])
            mcs_needed    = mcs_prob >= mcs_threshold
            shap_mcs      = shap_service.compute_shap("mcs", _unwrap_xgb(mcs_model), mcs_df)
        except Exception:
            pass

    # ── Johnathan — VA-ECMO ───────────────────────────────────────────────────
    ecmo_prob, ecmo_needed, shap_ecmo = 0.10, False, []
    if loader.get("va_ecmo") is not None:
        try:
            ecmo_df        = va_ecmo_features.extract(raw_data)
            ecmo_bundle    = loader.get("va_ecmo")
            ecmo_model     = ecmo_bundle["model"]
            ecmo_threshold = ecmo_bundle.get("default_threshold", 0.10)
            ecmo_prob      = float(ecmo_model.predict_proba(ecmo_df)[:, 1][0])
            ecmo_needed    = ecmo_prob >= ecmo_threshold
            shap_ecmo      = shap_service.compute_shap("va_ecmo", _unwrap_xgb(ecmo_model), ecmo_df)
        except Exception:
            pass

    # ── Lilly — mortality ─────────────────────────────────────────────────────
    hospital_mortality, shap_mort = 0.0, []
    if loader.get("mortality") is not None:
        try:
            mort_df        = mortality_features.extract(raw_data)
            mort_bundle    = loader.get("mortality")
            mort_model     = mort_bundle["pipeline"] if isinstance(mort_bundle, dict) else mort_bundle
            hospital_mortality = float(mort_model.predict_proba(mort_df)[:, 1][0])
            shap_mort      = shap_service.compute_shap("mortality", _unwrap_xgb(mort_model), mort_df)
        except Exception:
            pass

    # ── Lilly — length of stay ────────────────────────────────────────────────
    # Pipeline predicts log1p(los_hours); invert then convert to days.
    hospital_los = 0.0
    if loader.get("los") is not None:
        try:
            los_df    = los_features.extract(raw_data)
            los_model = loader.get("los")
            hospital_los = float(np.expm1(los_model.predict(los_df)[0]) / 24)
        except Exception:
            pass

    # ── Derived fields ─────────────────────────────────────────────────────────
    return PredictionsOut(
        readmission_risk=derive.readmission_risk(hospital_mortality, hospital_los),
        icu_transfer_risk=derive.icu_transfer_risk(scai_prob),
        in_hospital_expiry=round(hospital_mortality, 4),
        hospital_mortality=round(hospital_mortality, 4),
        icu_mortality=derive.icu_mortality(hospital_mortality),
        mortality_risk=round(hospital_mortality, 4),
        shap_mortality=shap_mort,
        hospital_los_days=round(hospital_los, 2),
        icu_los_days=derive.icu_los_days(hospital_los),
        scai_deterioration_6h_prob=round(scai_prob, 4),
        scai_deterioration_6h_label=scai_label,
        current_scai_stage=current_scai_stage,
        vasopressor_probability=round(vaso_prob, 4),
        predicted_vasopressor_count=vaso_count,
        critical_alert=critical_alert,
        mcs_12h_probability=round(mcs_prob, 4),
        mcs_12h_needed=mcs_needed,
        va_ecmo_12h_probability=round(ecmo_prob, 4),
        va_ecmo_12h_needed=ecmo_needed,
        shap_mcs_12h=shap_mcs,
        shap_va_ecmo_12h=shap_ecmo,
        shap_transfer=[],
        shap_readmission=[],
        home_care_suggestions=[],
        last_updated=datetime.now(timezone.utc).isoformat(),
    )
