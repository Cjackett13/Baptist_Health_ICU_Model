from __future__ import annotations
from pydantic import BaseModel


class ShapValueOut(BaseModel):
    """Maps 1:1 to Flutter ShapValue.fromJson."""
    feature: str
    value: float
    is_positive: bool
    direction: str   # 'positive' | 'negative'
    description: str


class HomeCareSuggestionOut(BaseModel):
    """Maps 1:1 to Flutter HomeCareSuggestion.fromJson."""
    category: str    # 'diet' | 'exercise' | 'medication' | 'monitoring' | 'lifestyle'
    title: str
    description: str
    impact: str


class PredictionsOut(BaseModel):
    """
    Maps 1:1 to Flutter PatientPredictions.fromJson.
    Every field name here must match the JSON key Flutter reads.

    Derived fields (no backing model):
      readmission_risk   = weighted blend of hospital_mortality + hospital_los_days
      icu_transfer_risk  = derived from scai_deterioration_6h_prob
      in_hospital_expiry = hospital_mortality
      icu_mortality      = clamp(hospital_mortality * 1.1, 0, 1)
      icu_los_days       = clamp(hospital_los_days * 0.40, 0.5, hospital_los_days)
    """

    # ── Derived ───────────────────────────────────────────────────────────────
    readmission_risk: float
    icu_transfer_risk: float
    in_hospital_expiry: float

    # ── Lilly — mortality ─────────────────────────────────────────────────────
    hospital_mortality: float
    icu_mortality: float
    mortality_risk: float            # == hospital_mortality
    shap_mortality: list[ShapValueOut] = []

    # ── Lilly — length of stay ────────────────────────────────────────────────
    hospital_los_days: float
    icu_los_days: float              # derived from hospital_los_days

    # ── Christie — SCAI deterioration ─────────────────────────────────────────
    scai_deterioration_6h_prob: float
    scai_deterioration_6h_label: str  # 'Likely to worsen' | 'Unlikely to worsen'
    current_scai_stage: str           # 'A' | 'B' | 'C' | 'D' | 'E'
    shap_scai: list[ShapValueOut] = []

    # ── Christie — vasopressor ────────────────────────────────────────────────
    vasopressor_probability: float
    predicted_vasopressor_count: int
    critical_alert: bool             # high_severity_classifier, P(≥2); internal only
    shap_vasopressor: list[ShapValueOut] = []

    # ── Johnathan — MCS / VA-ECMO ────────────────────────────────────────────
    mcs_12h_probability: float
    mcs_12h_needed: bool
    va_ecmo_12h_probability: float
    va_ecmo_12h_needed: bool
    shap_mcs_12h: list[ShapValueOut] = []
    shap_va_ecmo_12h: list[ShapValueOut] = []

    # ── Shared ────────────────────────────────────────────────────────────────
    shap_transfer: list[ShapValueOut] = []
    shap_readmission: list[ShapValueOut] = []
    home_care_suggestions: list[HomeCareSuggestionOut] = []
    last_updated: str                # ISO-8601 UTC


# ── Endpoint response wrappers ────────────────────────────────────────────────

class PatientsResponse(BaseModel):
    """GET /patients — matches PredictionApiClient.fetchPatients()."""
    patients: list[dict]   # PatientRecord-shaped dicts; typed loosely to avoid nesting all sub-models


class ModelResult(BaseModel):
    """
    Single model result within a cohort batch item.
    Backward-compatible with Flutter ShockEscalationApiClient._mergeResults().
    """
    model: str           # 'mcs' | 'ecmo' | 'scai' | 'vasopressor' | 'mortality' | 'los'
    probability: float
    alert: bool
    reasons: list[ShapValueOut] = []


class CohortPredictionItem(BaseModel):
    encounter_id: int
    results: list[ModelResult]    # backward-compat slice for ShockEscalationApiClient
    predictions: PredictionsOut   # full output for new callers


class CohortBatchResponse(BaseModel):
    """POST /predict/cohort/batch — backward-compatible with ShockEscalationApiClient."""
    predictions: list[CohortPredictionItem]
