"""
Baptist Health ICU prediction API — mortality + LOS + patient aggregator.

Run locally (port 8001 matches Flutter ``ApiConfig.baseUrl`` when ``useLiveApi=true``):

  cd back_end
  pip install -r requirements.txt
  python back_end/scripts/sync_all_artifacts.py
  uvicorn app.main:app --reload --host 0.0.0.0 --port 8001

Flutter: set ``ApiConfig.useLiveApi = true`` in ``Frontend/lib/config/api_config.dart``.
"""

from __future__ import annotations

import logging
import os

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.feature_store import lookup_los_features, lookup_mortality_features
from app.inference import predict_mortality
from app.los_inference import predict_los_from_features
from app.mortality_bundle import load_shap_bundle_dict, predictions_page_as_json
from app.mortality_inference import predict_mortality_from_features
from app.mortality_meta import load_mortality_meta, meta_path
from app.patient_predictions import (
    get_patient,
    list_patients,
    predict_for_patient_dict,
    reload_seed_patients,
)
from app.schemas import (
    CombinedPredictResponse,
    LosPredictResponse,
    MortalityFeaturesRequest,
    MortalityModelMetaResponse,
    MortalityPredictResponse,
    PatientPredictRequest,
    PatientsListResponse,
    PredictionsPageResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_origins = os.environ.get("CORS_ALLOW_ORIGINS", "*")
_allow = [o.strip() for o in _origins.split(",") if o.strip()]

app = FastAPI(
    title="Baptist ICU Prediction API",
    version="2.0.0",
    description="Mortality classifier + LOS regressor + live patient feed for Flutter.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allow if _allow != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "baptist-icu-predictions"}


@app.post("/admin/reload-seed")
def admin_reload_seed() -> dict[str, int]:
    """Reload patients_seed.json from disk (run after align/export scripts)."""
    patients = reload_seed_patients()
    return {"reloaded": len(patients)}


# ── Patient aggregator (Flutter PatientRepository when useLiveApi=true) ──


@app.get("/patients", response_model=PatientsListResponse)
def get_patients(live: bool = Query(True, description="Run ML models on each patient")) -> PatientsListResponse:
    try:
        payload = list_patients(live=live)
        return PatientsListResponse(**payload)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/patients/{patient_key}")
def get_patient_by_key(
    patient_key: str,
    live: bool = Query(True),
) -> dict:
    row = get_patient(patient_key, live=live)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Patient not found: {patient_key}")
    return row


@app.post("/v1/predict/patient", response_model=CombinedPredictResponse)
def predict_patient_combined(body: PatientPredictRequest) -> CombinedPredictResponse:
    """Run mortality + LOS on one patient (parquet lookup + optional overrides)."""
    if not body.person_id and not body.encounter_id:
        raise HTTPException(status_code=400, detail="Provide person_id and/or encounter_id")

    features: dict = {}
    if body.person_id:
        row = lookup_mortality_features(body.person_id)
        if row:
            features.update(row)
    if body.encounter_id:
        row = lookup_los_features(body.encounter_id)
        if row:
            features.update(row)
    if body.feature_overrides:
        features.update(body.feature_overrides)

    if not features:
        raise HTTPException(
            status_code=404,
            detail="No features found for ids (check data/ parquets and seed ids)",
        )

    mort_scores, mort_ver, mort_src, alert_thr, alert = predict_mortality_from_features(features)
    los_h, icu_h, los_ver, los_src = predict_los_from_features(features)

    mortality = MortalityPredictResponse(
        hospital_mortality=mort_scores.hospital_mortality,
        icu_mortality=mort_scores.icu_mortality,
        in_hospital_expiry=mort_scores.in_hospital_expiry,
        death_alert_threshold=alert_thr,
        death_alert=alert,
        model_version=mort_ver,
        source=mort_src,
    )
    los = LosPredictResponse(
        hospital_los_hours=los_h,
        icu_los_hours=icu_h,
        hospital_los_days=round(los_h / 24.0, 2),
        icu_los_days=round(icu_h / 24.0, 2),
        model_version=los_ver,
        source=los_src,
    )
    return CombinedPredictResponse(
        person_id=body.person_id,
        encounter_id=body.encounter_id,
        mortality=mortality,
        los=los,
    )


# ── Mortality (legacy 8-field + bundle) ──


@app.get("/v1/mortality/model-meta", response_model=MortalityModelMetaResponse)
def get_mortality_model_meta() -> MortalityModelMetaResponse:
    try:
        meta = load_mortality_meta()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail="Run: python back_end/scripts/sync_all_artifacts.py",
        ) from exc
    return MortalityModelMetaResponse(
        death_alert_threshold=float(meta.get("death_alert_threshold", 0.5)),
        death_alert_threshold_oof_selected=meta.get("death_alert_threshold_oof_selected"),
        death_alert_threshold_adjustment=meta.get("death_alert_threshold_adjustment"),
        test_meets_fp_cap_90_at_alert=meta.get("test_meets_fp_cap_90_at_alert"),
        meta_path=str(meta_path()),
    )


@app.post("/v1/predict/mortality", response_model=MortalityPredictResponse)
def predict_mortality_endpoint(body: MortalityFeaturesRequest) -> MortalityPredictResponse:
    scores, version, source, alert_thr, alert = predict_mortality(
        num_medications=body.num_medications,
        number_inpatient=body.number_inpatient,
        num_lab_procedures=body.num_lab_procedures,
        time_in_hospital=body.time_in_hospital,
        number_diagnoses=body.number_diagnoses,
        number_emergency=body.number_emergency,
        number_outpatient=body.number_outpatient,
        age_mid=body.age_mid,
    )
    return MortalityPredictResponse(
        hospital_mortality=scores.hospital_mortality,
        icu_mortality=scores.icu_mortality,
        in_hospital_expiry=scores.in_hospital_expiry,
        death_alert_threshold=alert_thr,
        death_alert=alert,
        model_version=version,
        source=source,
    )


@app.post("/v1/predict/los", response_model=LosPredictResponse)
def predict_los_endpoint(body: PatientPredictRequest) -> LosPredictResponse:
    features: dict = {}
    if body.encounter_id:
        row = lookup_los_features(body.encounter_id)
        if row:
            features.update(row)
    if body.person_id and not features:
        row = lookup_mortality_features(body.person_id)
        if row:
            features.update(row)
    if body.feature_overrides:
        features.update(body.feature_overrides)
    if not features:
        raise HTTPException(status_code=404, detail="No LOS features for encounter_id")
    los_h, icu_h, ver, src = predict_los_from_features(features)
    return LosPredictResponse(
        hospital_los_hours=los_h,
        icu_los_hours=icu_h,
        hospital_los_days=round(los_h / 24.0, 2),
        icu_los_days=round(icu_h / 24.0, 2),
        model_version=ver,
        source=src,
    )


@app.get("/v1/mortality/shap-bundle")
def get_mortality_shap_bundle() -> dict:
    try:
        return load_shap_bundle_dict()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/v1/mortality/predictions", response_model=PredictionsPageResponse)
def get_mortality_predictions(
    page: int = Query(0, ge=0),
    page_size: int = Query(50, ge=1, le=500),
) -> PredictionsPageResponse:
    try:
        data = predictions_page_as_json(page=page, page_size=page_size)
        return PredictionsPageResponse(**data)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
