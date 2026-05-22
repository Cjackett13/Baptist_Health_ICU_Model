"""Deterministic mortality estimates when no trained artifact is present.

Mirrors `Frontend/lib/models/mortality_model.dart` so local and server stay aligned.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class MortalityScores:
    hospital_mortality: float
    icu_mortality: float
    in_hospital_expiry: float

    @property
    def peak(self) -> float:
        return max(self.hospital_mortality, self.icu_mortality, self.in_hospital_expiry)


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def predict_fallback(
    *,
    num_medications: int,
    number_inpatient: int,
    num_lab_procedures: int,
    time_in_hospital: int,
    number_diagnoses: int,
    number_emergency: int,
    number_outpatient: int,
    age_mid: float,
) -> MortalityScores:
    z = (
        -5.15
        + 0.042 * (age_mid - 55.0)
        + 0.20 * max(-2, min(11, time_in_hospital - 3))
        + 0.017 * max(-7, min(32, num_medications - 8))
        + 0.11 * max(0, min(8, number_inpatient))
        + 0.048 * max(-3, min(12, number_diagnoses - 4))
        + 0.075 * max(0, min(6, number_emergency))
        + 0.0055 * max(-34, min(85, num_lab_procedures - 35))
        - 0.014 * max(0, min(15, number_outpatient))
    )
    hospital = max(0.01, min(0.85, _sigmoid(z)))
    icu = max(0.01, min(0.92, hospital * 1.12 + 0.018))
    expiry = max(0.01, min(0.88, hospital * 0.90 + 0.01))
    return MortalityScores(
        hospital_mortality=round(hospital, 4),
        icu_mortality=round(icu, 4),
        in_hospital_expiry=round(expiry, 4),
    )
