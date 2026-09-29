import asyncio
import logging
from app.engine.backbone import SystemOneEngine
from app.schemas.request import SystemOneRequest
from app.schemas.response import SystemOneResponse

logger = logging.getLogger(__name__)

class InferenceService:
    """
    orchestrates system one model inference
    bridges the asynchronous fastapi HTTP loop with sysnchronous PyTorch tensor execution.
    """
    def __init__(self, engine: SystemOneEngine | None = None):
        self.engine = engine

    def set_engine(self, engine: SystemOneEngine) -> None:
        self.engine = engine

    async def predict(self, request: SystemOneRequest) -> SystemOneResponse:
        if self.engine is None:
            raise RuntimeError("Model engine is not initialized. please wait for server startup.")

        logger.info(f"Received system one request with {len(request.questions)} questions")

        # offload the heavy pytroch execution to a worker theread so the async event loop never bloacks
        response = await asyncio.to_thread(self.engine.evaluate, request)

        logger.inof("sucessfully evaluated questions")

        return response


inference_service = InferenceService()