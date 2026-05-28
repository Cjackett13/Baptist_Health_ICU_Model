from fastapi import APIRouter, HTTPException

from app.schemas.input import CohortBatchRequest
from app.schemas.output import (
    CohortBatchResponse,
    CohortPredictionItem,
    ModelResult,
)
from app.services.inference import run_all_models

router = APIRouter()


@router.post("/predict/cohort/batch", response_model=CohortBatchResponse)
async def predict_cohort_batch(body: CohortBatchRequest):
    """
    Run all 6 models for every patient in the batch.

    Backward-compatible with Flutter ShockEscalationApiClient:
      - response.predictions[i].results still contains 'mcs' and 'ecmo' entries
        so the existing Flutter merge logic works unchanged.
      - response.predictions[i].predictions carries the full PredictionsOut for
        new callers that want all model outputs in one shot.
    """
    items: list[CohortPredictionItem] = []

    for patient in body.patients:
        try:
            preds = await run_all_models(patient.encounter_id, patient.hour_from_admit)
        except NotImplementedError as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Feature engineering not yet wired for encounter {patient.encounter_id}: {exc}",
            )
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Inference error for encounter {patient.encounter_id}: {exc}",
            )

        # Build the backward-compat results slice
        results = [
            ModelResult(
                model="mcs",
                probability=preds.mcs_12h_probability,
                alert=preds.mcs_12h_needed,
                reasons=preds.shap_mcs_12h,
            ),
            ModelResult(
                model="ecmo",
                probability=preds.va_ecmo_12h_probability,
                alert=preds.va_ecmo_12h_needed,
                reasons=preds.shap_va_ecmo_12h,
            ),
            ModelResult(
                model="scai",
                probability=preds.scai_deterioration_6h_prob,
                alert=preds.scai_deterioration_6h_label == "Likely to worsen",
                reasons=[],
            ),
            ModelResult(
                model="vasopressor",
                probability=preds.vasopressor_probability,
                alert=preds.vasopressor_probability >= 0.1882,
                reasons=[],
            ),
        ]

        items.append(CohortPredictionItem(
            encounter_id=patient.encounter_id,
            results=results,
            predictions=preds,
        ))

    return CohortBatchResponse(predictions=items)
