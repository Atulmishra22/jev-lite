import logging
from contextlib import asynccontextmanager
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.engine.backbone import SystemOneEngine
from app.services.inference_service import inference_service
from app.api.v1.router import api_router

# configure structured logging

logging.basicConfig(
    level= logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("jev-lite")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP PHASE ---
    logger.info(f"Initializing {settings.api_title} on device: {settings.device}...")

    # 1. Load the QWen Backbone & Decision Heads into memory ONCE
    engine = SystemOneEngine(
        model_name= settings.model_name,
        device= settings.device,
    )

    # 2. inject into the inference service
    inference_service.set_engine(engine)
    logger.info("system one engine loaded successfully and ready for traffic!")

    yield  # app is live and serving requests here

    # --- SHUTDOWN PHASE ---
    logger.info("shutting down jev-lite service...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("cleanup complete. Goodbye!")


app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description= "production grade system one decision engine with Qwen & parallel question evaluation.",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],

)

# Include the API router
app.include_router(api_router)

@app.get("/", tags=["Root"])
async def root():
    return {"message": "welcome to jev-lite system one engine.",
            "docs": "/docs",
            "health":"/v1/health",
            "models": settings.model_version,
            }