from pydantic import BaseModel


class CohortPatientIn(BaseModel):
    """One patient in a batch predict request. Matches Flutter ShockEscalationApiClient payload."""
    encounter_id: int
    hour_from_admit: int = 24


class CohortBatchRequest(BaseModel):
    patients: list[CohortPatientIn]
