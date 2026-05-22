from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class MortalityFeaturesRequest(BaseModel):
    """Same field names as Flutter `PatientFeatures.toJson`."""

    num_medications: int = Field(ge=0, le=200)
    number_inpatient: int = Field(ge=0, le=50)
    num_lab_procedures: int = Field(ge=0, le=200)
    time_in_hospital: int = Field(ge=1, le=60)
    number_diagnoses: int = Field(ge=1, le=50)
    number_emergency: int = Field(ge=0, le=50)
    number_outpatient: int = Field(ge=0, le=50)
    age_mid: float = Field(ge=0, le=120)


class MortalityPredictResponse(BaseModel):
    hospital_mortality: float
    icu_mortality: float
    in_hospital_expiry: float
    death_alert_threshold: float
    death_alert: bool
    model_version: str
    source: str  # "trained" | "fallback"


class MortalityModelMetaResponse(BaseModel):
    """Subset of ``xgb_mortality_model_meta.json`` for clients and demos."""

    death_alert_threshold: float
    death_alert_threshold_oof_selected: float | None = None
    death_alert_threshold_adjustment: str | None = None
    test_meets_fp_cap_90_at_alert: bool | None = None
    meta_path: str


class PredictionsPageResponse(BaseModel):
    model_version: str
    total: int
    page: int
    page_size: int
    columns: list[str]
    rows: list[dict[str, Any]]


class PatientPredictRequest(BaseModel):
    """Identify a cohort patient and optional feature overrides."""

    person_id: str | None = None
    encounter_id: str | None = None
    feature_overrides: dict[str, float | int | str] | None = None


class LosPredictResponse(BaseModel):
    hospital_los_hours: float
    icu_los_hours: float
    hospital_los_days: float
    icu_los_days: float
    model_version: str
    source: str


class CombinedPredictResponse(BaseModel):
    person_id: str | None = None
    encounter_id: str | None = None
    mortality: MortalityPredictResponse
    los: LosPredictResponse


class PatientsListResponse(BaseModel):
    patients: list[dict[str, Any]]
    count: int
    live: bool
