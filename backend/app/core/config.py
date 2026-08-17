import os
from pydantic_settings import BaseSettings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class Settings(BaseSettings):
    PROJECT_NAME: str = "Breast Cancer Detection ML Application"
    API_V1_STR: str = "/api"
    MODEL_PATH: str = os.path.join(BASE_DIR, "models", "breast_cancer_model.joblib")
    METADATA_PATH: str = os.path.join(BASE_DIR, "models", "metadata.json")

    class Config:
        case_sensitive = True

settings = Settings()
