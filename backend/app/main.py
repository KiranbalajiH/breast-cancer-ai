from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.core.config import settings
from app.api.prediction import router as prediction_router
from app.api.model import router as model_router
from app.api.image_analysis import router as image_analysis_router
from app.services.model_service import model_service

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup event
    print("Starting up server, loading ML model...")
    model_service.load_model()
    yield
    # Shutdown event
    print("Shutting down server...")

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(prediction_router, prefix=settings.API_V1_STR)
app.include_router(model_router, prefix=f"{settings.API_V1_STR}/model")
app.include_router(image_analysis_router, prefix=f"{settings.API_V1_STR}/image-analysis")

