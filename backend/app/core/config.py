import os
from pydantic_settings import BaseSettings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class Settings(BaseSettings):
    PROJECT_NAME: str = "Breast Cancer Detection Image Application"
    API_V1_STR: str = "/api"
    IMAGE_MODEL_PATH: str = os.path.join(BASE_DIR, "models", "breast_image_classifier.keras")
    IMAGE_METADATA_PATH: str = os.path.join(BASE_DIR, "models", "breast_image_classifier_metadata.json")

    # Thresholds and Safety Configs
    CONFIDENCE_HIGH_THRESHOLD: float = 0.75
    CONFIDENCE_MODERATE_THRESHOLD: float = 0.60
    CONFIDENCE_MARGIN_THRESHOLD: float = 0.15
    CONFIDENCE_REVIEW_THRESHOLD: float = 0.50

    # Image Quality Thresholds
    QUALITY_BLUR_THRESHOLD: float = 20.0
    QUALITY_DARK_THRESHOLD: float = 15.0
    QUALITY_BRIGHT_THRESHOLD: float = 240.0
    QUALITY_LOW_INFO_THRESHOLD: float = 5.0

    # Modality / Safeguards
    SAFEGUARD_SATURATION_THRESHOLD: float = 12.0
    SAFEGUARD_COLOR_DIFF_THRESHOLD: float = 8.0
    SAFEGUARD_MIN_ASPECT_RATIO: float = 0.4
    SAFEGUARD_MAX_ASPECT_RATIO: float = 2.5

    # CORS Configurations
    BACKEND_CORS_ORIGINS: str = "http://localhost:3000,http://127.0.0.1:3000,https://breast-cancer-ai-ten.vercel.app,*"

    class Config:
        case_sensitive = True

settings = Settings()
