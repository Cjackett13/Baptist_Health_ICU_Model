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

# SCAI thresholds: A/B/C from scai_deterioration_model; D from scai_stage_d_model bundle
_SCAI_ABC_THRESHOLDS: dict[int, float] = {0: 0.15, 1: 0.15, 2: 0.15}
_SCAI_D_THRESHOLD = 0.14          # bundle threshold is None; use tuned default
_VASOPRESSOR_THRESHOLD = 0.19     # from binary_alert_model bundle["threshold"]
_HIGH_SEVERITY_THRESHOLD = 0.01   # from high_severity_classifier bundle["threshold"]

_SCAI_STAGE_LABEL = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}


def _unwrap_xgb(model):
    """Iteratively strip calibration/ensemble wrappers to reach the raw XGBClassifier for SHAP."""
    from xgboost import XGBClassifier, XGBRegressor
    for _ in range(10):
        if isinstance(model, (XGBClassifier, XGBRegressor)):
            return model
        if hasattr(model, "base_model"):              # IsotonicCalibratedModel
            model = model.base_model
        elif hasattr(model, "xgb_model"):             # PlattXGB
            model = model.xgb_model
        elif hasattr(model, "calibrated_estimator"):  # TemperatureScaledBinaryCalibrator
            model = model.calibrated_estimator
        elif hasattr(model, "calibrators"):           # AveragedBinaryCalibrators
            model = model.calibrators[0]
        elif hasattr(model, "models"):                # EnsembleXGBClassifier
            model = model.models[0]
        elif hasattr(model, "calibrated_classifiers_"):  # CalibratedClassifierCV — use fitted copy
            model = model.calibrated_classifiers_[0].estimator
        elif hasattr(model, "estimator"):
            model = model.estimator
        elif hasattr(model, "base_estimator"):
            model = model.base_estimator
        elif hasattr(model, "base"):
            model = model.base
        elif hasattr(model, "steps"):                 # sklearn Pipeline
            model = model.steps[-1][1]
        else:
            break
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

    # ── Christie — SCAI deterioration (split model: ABC vs D) ─────────────────
    # extract() picks the right bundle based on the patient's current stage
    scai_stage_num = int(
        raw_data.get("df", raw_data.get("scai_stage_hourly", {}))
        if False else  # placeholder — stage read below after extract
        0
    )
    # Build features first (extract reads stage internally to choose bundle)
    scai_df = scai_features.extract(raw_data)

    # Determine stage from the feature frame (SCAI_STAGE_NUM passes through)
    if "SCAI_STAGE_NUM" in scai_df.columns:
        scai_stage_num = int(scai_df["SCAI_STAGE_NUM"].iloc[0])
    else:
        df_raw = raw_data.get("df")
        scai_stage_num = int(df_raw["SCAI_STAGE_NUM"].iloc[-1]) if df_raw is not None else 1

    bundle_key = "scai_d" if scai_stage_num == 3 else "scai_abc"
    scai_bundle = loader.get(bundle_key)

    scai_raw = scai_bundle["xgboost_model"].predict_proba(scai_df)[:, 1]
    scai_prob = float(
        scai_bundle["platt_scaler"].predict_proba(scai_raw.reshape(-1, 1))[:, 1][0]
    )
    if scai_stage_num == 3:
        scai_threshold = _SCAI_D_THRESHOLD
    else:
        scai_threshold = _SCAI_ABC_THRESHOLDS.get(scai_stage_num, 0.15)
    current_scai_stage = _SCAI_STAGE_LABEL.get(scai_stage_num, "B")
    scai_label = "Likely to worsen" if scai_prob >= scai_threshold else "Unlikely to worsen"
    shap_scai = shap_service.compute_shap("scai", _unwrap_xgb(scai_bundle["xgboost_model"]), scai_df)

    # ── Christie — vasopressor ────────────────────────────────────────────────
    vaso_df = vasopressor_features.extract(raw_data)

    # binary_alert: dict with "model" (XGBClassifier) + "platt" (LogisticRegression)
    vaso_bin_bundle = loader.get("vasopressor_binary")
    vaso_raw = vaso_bin_bundle["model"].predict_proba(vaso_df)[:, 1]
    vaso_prob = float(
        vaso_bin_bundle["platt"].predict_proba(vaso_raw.reshape(-1, 1))[:, 1][0]
    )

    # ordinal_count: dict with "model" + "label_map" {0:1, 1:2, 2:3}
    vaso_cnt_bundle = loader.get("vasopressor_count")
    count_probs = vaso_cnt_bundle["model"].predict_proba(vaso_df)
    count_raw = int(np.argmax(count_probs, axis=1)[0])
    label_map = vaso_cnt_bundle.get("label_map", {0: 1, 1: 2, 2: 3})
    vaso_count = label_map.get(count_raw, count_raw)

    # high_severity: dict with "model" (PlattXGB) — call predict_proba directly
    vaso_sev_bundle = loader.get("vasopressor_severity")
    high_sev_prob = float(vaso_sev_bundle["model"].predict_proba(vaso_df)[:, 1][0])
    critical_alert = high_sev_prob >= _HIGH_SEVERITY_THRESHOLD
    shap_vasopressor = shap_service.compute_shap(
        "vasopressor", _unwrap_xgb(vaso_bin_bundle["model"]), vaso_df
    )

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
        shap_scai=shap_scai,
        vasopressor_probability=round(vaso_prob, 4),
        predicted_vasopressor_count=vaso_count,
        critical_alert=critical_alert,
        shap_vasopressor=shap_vasopressor,
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
