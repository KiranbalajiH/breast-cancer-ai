import io
import cv2
import numpy as np
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

from app.main import app
from app.image_model import image_classifier

client = TestClient(app)

def _create_image(mean=120, std=30, blur_kernel=None, color=False, aspect_ratio=(224, 224)):
    # Create image of specific size
    w, h = aspect_ratio
    if color:
        # Create colored image (red block)
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:, :, 2] = 255 # Red channel set to 255
    else:
        # Create grayscale-like 3 channel image
        if std == 0:
            base = np.full((h, w), mean, dtype=np.uint8)
        else:
            base = np.random.normal(mean, std, (h, w)).clip(0, 255).astype(np.uint8)
        if blur_kernel:
            base = cv2.GaussianBlur(base, blur_kernel, 0)
        img = cv2.merge([base, base, base])
        
    _, buffer = cv2.imencode(".png", img)
    return buffer.tobytes()

@pytest.fixture
def mock_predict():
    with patch.object(image_classifier, 'model') as mock_model:
        # Mock predict method to return a MagicMock or numpy array
        yield mock_model

def test_high_confidence(mock_predict):
    # Mock return probabilities [0.85, 0.10, 0.05] (benign, malignant, normal)
    mock_predict.predict.return_value = np.array([[0.85, 0.10, 0.05]])
    
    img_bytes = _create_image()
    response = client.post(
        "/api/image-predict",
        files={"file": ("test.png", img_bytes, "image/png")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "high_confidence"
    assert data["prediction"] == "benign"
    assert data["confidence"] == 0.85
    assert data["image_quality"] == "acceptable"

def test_moderate_confidence(mock_predict):
    # Mock return probabilities [0.20, 0.70, 0.10] -> malignant
    mock_predict.predict.return_value = np.array([[0.20, 0.70, 0.10]])
    
    img_bytes = _create_image()
    response = client.post(
        "/api/image-predict",
        files={"file": ("test.png", img_bytes, "image/png")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "moderate_confidence"
    assert data["prediction"] == "malignant"

def test_low_confidence(mock_predict):
    # Mock return probabilities [0.58, 0.38, 0.04] (margin 0.20 >= 0.15)
    mock_predict.predict.return_value = np.array([[0.58, 0.38, 0.04]])
    
    img_bytes = _create_image()
    response = client.post(
        "/api/image-predict",
        files={"file": ("test.png", img_bytes, "image/png")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "low_confidence"
    assert data["prediction"] == "benign"

def test_review_required_low_confidence(mock_predict):
    # Mock return probabilities [0.45, 0.40, 0.15] (max 0.45 < 0.50)
    mock_predict.predict.return_value = np.array([[0.45, 0.40, 0.15]])
    
    img_bytes = _create_image()
    response = client.post(
        "/api/image-predict",
        files={"file": ("test.png", img_bytes, "image/png")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "review_required"
    assert "reviewed by a qualified clinician" in data["message"]

def test_review_required_close_competition(mock_predict):
    # Mock return probabilities [0.52, 0.48, 0.00] (margin 0.04 < 0.15)
    mock_predict.predict.return_value = np.array([[0.52, 0.48, 0.00]])
    
    img_bytes = _create_image()
    response = client.post(
        "/api/image-predict",
        files={"file": ("test.png", img_bytes, "image/png")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "review_required"

def test_invalid_image_corruption():
    # Corrupted / invalid image bytes
    response = client.post(
        "/api/image-predict",
        files={"file": ("test.png", b"corrupted-non-image-data", "image/png")}
    )
    assert response.status_code == 400
    assert "could not decode image" in response.json()["detail"].lower()

def test_empty_file():
    response = client.post(
        "/api/image-predict",
        files={"file": ("test.png", b"", "image/png")}
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()

def test_extremely_dark_image(mock_predict):
    mock_predict.predict.return_value = np.array([[0.80, 0.15, 0.05]])
    img_bytes = _create_image(mean=5, std=2) # mean < 15
    response = client.post(
        "/api/image-predict",
        files={"file": ("test.png", img_bytes, "image/png")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["image_quality"] == "poor"
    assert any("dark" in warning.lower() for warning in data["quality_warnings"])

def test_extremely_bright_image(mock_predict):
    mock_predict.predict.return_value = np.array([[0.80, 0.15, 0.05]])
    img_bytes = _create_image(mean=245, std=2) # mean > 240
    response = client.post(
        "/api/image-predict",
        files={"file": ("test.png", img_bytes, "image/png")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["image_quality"] == "poor"
    assert any("bright" in warning.lower() for warning in data["quality_warnings"])

def test_extremely_blurry_image(mock_predict):
    mock_predict.predict.return_value = np.array([[0.80, 0.15, 0.05]])
    img_bytes = _create_image(mean=120, std=5, blur_kernel=(21, 21)) # heavily blurred
    response = client.post(
        "/api/image-predict",
        files={"file": ("test.png", img_bytes, "image/png")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["image_quality"] == "poor"
    assert any("blurry" in warning.lower() for warning in data["quality_warnings"])

def test_unsupported_colored_image(mock_predict):
    mock_predict.predict.return_value = np.array([[0.80, 0.15, 0.05]])
    img_bytes = _create_image(color=True) # Red block
    response = client.post(
        "/api/image-predict",
        files={"file": ("test.png", img_bytes, "image/png")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "unsupported_or_review_required"
    assert "color content" in data["message"]

def test_unsupported_aspect_ratio(mock_predict):
    mock_predict.predict.return_value = np.array([[0.80, 0.15, 0.05]])
    img_bytes = _create_image(aspect_ratio=(500, 100)) # 5:1 ratio (unsupported)
    response = client.post(
        "/api/image-predict",
        files={"file": ("test.png", img_bytes, "image/png")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "unsupported_or_review_required"
    assert "aspect ratio" in data["message"]

def test_normal_valid_flow(mock_predict):
    mock_predict.predict.return_value = np.array([[0.80, 0.15, 0.05]])
    img_bytes = _create_image(mean=120, std=30)
    response = client.post(
        "/api/image-predict",
        files={"file": ("test.png", img_bytes, "image/png")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "high_confidence"
    assert data["image_quality"] == "acceptable"
    assert len(data["quality_warnings"]) == 0

def test_gradcam_successful_generation():
    # Test prediction flow WITH actual model (no mocking) to verify Grad-CAM output
    img_bytes = _create_image(mean=120, std=30)
    response = client.post(
        "/api/image-predict",
        files={"file": ("test.png", img_bytes, "image/png")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "explanation" in data
    expl = data["explanation"]
    assert expl["available"] is True
    assert expl["type"] == "grad_cam"
    assert expl["heatmap"].startswith("data:image/png;base64,")
    assert expl["overlay"].startswith("data:image/png;base64,")
    assert "disclaimer" in expl
