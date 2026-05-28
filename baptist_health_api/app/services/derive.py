"""
Derived field calculations for the three PatientPredictions fields that have
no backing model. Logic agreed in planning:

  readmission_risk   = weighted blend of hospital_mortality (60%) + LOS proxy (40%)
  icu_transfer_risk  = scai_deterioration_6h_prob scaled up slightly (clinical: SCAI
                       worsening is the primary driver of ICU escalation)
  in_hospital_expiry = hospital_mortality (same value, different label in UI)
  icu_mortality      = clamp(hospital_mortality * 1.1, 0, 1)
  icu_los_days       = clamp(hospital_los_days * 0.40, 0.5, hospital_los_days)
"""


def readmission_risk(hospital_mortality: float, hospital_los_days: float) -> float:
    los_proxy = min(hospital_los_days / 30.0, 1.0)
    return round(min(hospital_mortality * 0.60 + los_proxy * 0.40, 1.0), 4)


def icu_transfer_risk(scai_deterioration_prob: float) -> float:
    return round(min(scai_deterioration_prob * 1.15, 1.0), 4)


def icu_mortality(hospital_mortality: float) -> float:
    return round(min(hospital_mortality * 1.10, 1.0), 4)


def icu_los_days(hospital_los_days: float) -> float:
    return round(max(min(hospital_los_days * 0.40, hospital_los_days), 0.5), 2)
