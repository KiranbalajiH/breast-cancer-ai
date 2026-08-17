import pytest
from fastapi.testclient import TestClient
import numpy as np

from app.main import app

client = TestClient(app)

@pytest.fixture(scope="module")
def sample_features():
    # Provide a dummy sample that matches the schema with all 30 features
    return {
        "mean radius": 14.2,
        "mean texture": 19.2,
        "mean perimeter": 90.0,
        "mean area": 600.0,
        "mean smoothness": 0.1,
        "mean compactness": 0.1,
        "mean concavity": 0.05,
        "mean concave points": 0.05,
        "mean symmetry": 0.15,
        "mean fractal dimension": 0.06,
        "radius error": 0.3,
        "texture error": 1.0,
        "perimeter error": 2.5,
        "area error": 30.0,
        "smoothness error": 0.005,
        "compactness error": 0.01,
        "concavity error": 0.01,
        "concave points error": 0.01,
        "symmetry error": 0.01,
        "fractal dimension error": 0.003,
        "worst radius": 16.0,
        "worst texture": 25.0,
        "worst perimeter": 105.0,
        "worst area": 800.0,
        "worst smoothness": 0.13,
        "worst compactness": 0.2,
        "worst concavity": 0.2,
        "worst concave points": 0.1,
        "worst symmetry": 0.25,
        "worst fractal dimension": 0.08
    }

def test_health_check():
    # Make sure app startup loads the model, but since TestClient does not run lifespan
    # we might need to manually trigger model load if not using 'with TestClient'
    with TestClient(app) as c:
        response = c.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert "model_name" in data

def test_predict_success(sample_features):
    with TestClient(app) as c:
        response = c.post("/api/predict", json=sample_features)
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "confidence" in data
        assert "probabilities" in data

def test_predict_missing_features(sample_features):
    with TestClient(app) as c:
        # Remove a required field
        del sample_features["mean radius"]
        response = c.post("/api/predict", json=sample_features)
        assert response.status_code == 422 # Unprocessable Entity from FastAPI Validation
