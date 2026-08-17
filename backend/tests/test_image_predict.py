import io
import pytest
import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

def _create_dummy_image(format=".png"):
    # Generate a dummy RGB image array using numpy and encode it with OpenCV
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    _, buffer = cv2.imencode(format, img)
    return buffer.tobytes()

def test_image_model_status():
    with TestClient(app) as c:
        response = c.get("/api/image-model/status")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data

def test_image_predict_success():
    with TestClient(app) as c:
        image_bytes = _create_dummy_image()
        # Upload using multipart/form-data
        response = c.post(
            "/api/image-predict",
            files={"file": ("test.png", image_bytes, "image/png")}
        )
        assert response.status_code == 200
        data = response.json()
        assert "predicted_class" in data
        assert "confidence" in data
        assert "probabilities" in data
        assert data["predicted_class"] in ["benign", "malignant", "normal"]
        probs = data["probabilities"]
        assert "benign" in probs
        assert "malignant" in probs
        assert "normal" in probs
        assert abs(sum(probs.values()) - 1.0) < 1e-4

def test_image_predict_invalid_file_type():
    with TestClient(app) as c:
        response = c.post(
            "/api/image-predict",
            files={"file": ("test.txt", b"not-an-image", "text/plain")}
        )
        assert response.status_code == 400
        assert "invalid file type" in response.json()["detail"].lower()

def test_image_predict_empty_file():
    with TestClient(app) as c:
        response = c.post(
            "/api/image-predict",
            files={"file": ("test.png", b"", "image/png")}
        )
        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()
