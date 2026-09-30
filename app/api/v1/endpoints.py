from fastapi import APIRouter, HTTPException, status
from app.schemas.request import SystemOneRequest
from app.schemas.response import SystemOneResponse
from app.services.inference_service import inference_service
from app.services.scheduler import scheduler
from app.core.config import settings


router = APIRouter()

@router.post("/systemone", response_model=SystemOneResponse, status_code = status.HTTP_200_OK,
             summary= "Evalaute state against questions",
             description= "Take a state and evaluate choice, score and Noul questions in parallel.")
async def evaluate_system_one(request: SystemOneRequest) -> SystemOneResponse:
    """
    Evaluate a state against a set of questions using the System One model.

    This endpoint accepts a request containing a state and a list of questions. It evaluates the state against each question in parallel and returns the results.

    - **request**: A `SystemOneRequest` object containing the state and questions to evaluate.
    - **response**: A `SystemOneResponse` object containing the evaluation results for each question.

    Raises:
        HTTPException: If the model engine is not initialized or if there is an error during evaluation.
    """
    try:
        response = await scheduler.submit(request)
        return response
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.get("/models", summary="List available models")
async def list_models():
    """ Returns available models, matching the typesafe specifications."""
    return {
        "models": [{
            "name": settings.model_version,
            "description": f"Fast system one decision engine powered by {settings.model_name}",
            "release_date": "2026-09-30"
        },
        {
            "name": "jev-latest",
            "description" : "Alias pointing to the latest stable release",
            "release_date": "2026-09-30"
        }
        ]
    }

@router.get("/health", summary="Health check")
async def health_check():
    is_ready = inference_service.engine is not None
    return {
        "status": "healthy" if is_ready else "intializing",
        "device": settings.device,
        "model_loaded": is_ready,
    }
