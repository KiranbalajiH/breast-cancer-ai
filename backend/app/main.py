from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.core.config import settings
from app.api.prediction import router as prediction_router
from app.api.model import router as model_router
from app.image_model import image_classifier

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup event
    print("Starting up server, loading Image AI models...")
    try:
        image_classifier.load_model(load_v14=True)
    except Exception as e:
        print(f"Image classifier loading failed at startup: {e}")
    yield
    # Shutdown event
    print("Shutting down server...")

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan
)

# CORS
origins = [origin.strip() for origin in settings.BACKEND_CORS_ORIGINS.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(prediction_router, prefix=settings.API_V1_STR)
app.include_router(model_router, prefix=f"{settings.API_V1_STR}/model")

