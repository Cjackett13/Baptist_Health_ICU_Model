from fastapi import APIRouter, HTTPException
from app.schemas.output import PatientsResponse

router = APIRouter()


@router.get("/patients", response_model=PatientsResponse)
async def get_patients(live: bool = False):
    """
    Return all active ICU patients with full predictions embedded.
    Used by Flutter PredictionApiClient.fetchPatients().

    TODO: implement with real patient roster + run_all_models() per patient.
    For now, returns 503 until the data layer is wired.
    """
    raise HTTPException(
        status_code=503,
        detail="Patient roster endpoint not yet implemented. Wire data_access.py first.",
    )


@router.get("/patients/{encounter_id}")
async def get_patient(encounter_id: str):
    """
    Return one patient record with full predictions.
    Used by Flutter PredictionApiClient.fetchPatient().

    TODO: implement with real data layer.
    """
    raise HTTPException(
        status_code=503,
        detail=f"Single-patient endpoint not yet implemented. encounter_id={encounter_id}",
    )
