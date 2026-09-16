import pytest
import io
import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.main import app

@pytest.fixture(scope="module")
def sample_ultrasound_png():
    # Create a synthetic 224x224 grayscale ultrasound image for testing
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    cv2.circle(img, (112, 112), 40, (120, 120, 120), -1)
    is_success, buffer = cv2.imencode(".png", img)
    assert is_success
    return io.BytesIO(buffer.tobytes())

def test_health_check():
    with TestClient(app) as c:
        response = c.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
        assert data["model_loaded"] is True

def test_image_model_status():
    with TestClient(app) as c:
        response = c.get("/api/image-model/status")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
        assert data["model_loaded"] is True
        assert "classes" in data

def test_image_predict_success(sample_ultrasound_png):
    with TestClient(app) as c:
        response = c.post(
            "/api/image-predict",
            files={"file": ("test.png", sample_ultrasound_png, "image/png")}
        )
        assert response.status_code == 200
        data = response.json()
        assert "predicted_class" in data
        assert "confidence" in data
        assert "probabilities" in data
        assert "explanation" in data
