from fastapi import APIRouter
from app.models import loader

router = APIRouter()


@router.get("/health")
def health():
    model_status = loader.status()
    return {
        "status": "ok" if loader.is_healthy() else "degraded",
        "models": model_status,
    }
