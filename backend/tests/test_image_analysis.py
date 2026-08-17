import pytest
from fastapi.testclient import TestClient
import numpy as np
import cv2
import io
import os

from app.main import app
from image_processing.preprocessing import preprocess_image
from image_processing.segmentation import segment_nuclei
from image_processing.feature_extraction import extract_all_nuclei_features
from image_processing.aggregation import aggregate_features, FEATURE_NAMES_ORDERED
from image_processing.compatibility import validate_compatibility

client = TestClient(app)


def _generate_test_image(width=100, height=100, num_nuclei=4):
    """Generate a simple synthetic image containing dark ellipses (simulated nuclei) on a light background."""
    img = np.ones((height, width, 3), dtype=np.uint8) * 220
    # Draw simple dark ellipses
    for i in range(num_nuclei):
        cx = 20 + i * 20
        cy = 20 + i * 20
        cv2.ellipse(img, (cx, cy), (8, 6), 15 * i, 0, 360, (50, 50, 50), -1)
    return img


def test_preprocessing():
    # Generate BGR image
    test_img = _generate_test_image()
    gray, preprocessed = preprocess_image(test_img)
    
    assert gray.shape == (100, 100)
    assert preprocessed.shape == (100, 100)
    assert gray.dtype == np.uint8
    assert preprocessed.dtype == np.uint8


def test_segmentation():
    test_img = _generate_test_image()
    gray, preprocessed = preprocess_image(test_img)
    seg_result = segment_nuclei(gray, preprocessed, min_nucleus_area=10)
    
    assert "contours" in seg_result
    assert "binary_mask" in seg_result
    assert "overlay" in seg_result
    assert "labeled_mask" in seg_result
    assert seg_result["num_nuclei"] >= 3
    assert len(seg_result["contours"]) == seg_result["num_nuclei"]


def test_feature_extraction_and_aggregation():
    test_img = _generate_test_image()
    gray, preprocessed = preprocess_image(test_img)
    seg_result = segment_nuclei(gray, preprocessed, min_nucleus_area=10)
    
    # Feature extraction
    nuclei_features = extract_all_nuclei_features(seg_result["contours"], gray)
    assert len(nuclei_features) == seg_result["num_nuclei"]
    
    # Verify the 10 features exist on a single nucleus
    single_nucleus = nuclei_features[0]
    expected_keys = [
        "radius", "texture", "perimeter", "area", "smoothness",
        "compactness", "concavity", "concave_points", "symmetry", "fractal_dimension"
    ]
    for key in expected_keys:
        assert key in single_nucleus
        assert isinstance(single_nucleus[key], float)
        
    # Feature aggregation
    agg_result = aggregate_features(nuclei_features, min_nuclei=2)
    assert agg_result is not None
    features_30, metadata = agg_result
    
    # Check that we have exactly 30 features in the expected order
    assert len(features_30) == 30
    assert list(features_30.keys()) == FEATURE_NAMES_ORDERED


def test_compatibility_validation():
    # Create a dummy compatible features dict
    compatible_features = {
        "mean radius": 14.0, "mean texture": 19.0, "mean perimeter": 90.0, "mean area": 600.0,
        "mean smoothness": 0.09, "mean compactness": 0.1, "mean concavity": 0.08, "mean concave points": 0.05,
        "mean symmetry": 0.18, "mean fractal dimension": 0.06,
        "radius error": 0.4, "texture error": 1.2, "perimeter error": 2.8, "area error": 40.0,
        "smoothness error": 0.007, "compactness error": 0.02, "concavity error": 0.03, "concave points error": 0.01,
        "symmetry error": 0.02, "fractal dimension error": 0.003,
        "worst radius": 16.0, "worst texture": 25.0, "worst perimeter": 107.0, "worst area": 880.0,
        "worst smoothness": 0.13, "worst compactness": 0.25, "worst concavity": 0.27, "worst concave points": 0.11,
        "worst symmetry": 0.29, "worst fractal dimension": 0.08
    }
    
    report = validate_compatibility(compatible_features)
    assert report["overall_verdict"] == "Compatible"
    assert report["prediction_allowed"] is True
    
    # Create an incompatible features dict (e.g. extremely large area)
    incompatible_features = compatible_features.copy()
    incompatible_features["mean area"] = 99999.0
    incompatible_features["worst area"] = 999999.0
    incompatible_features["mean compactness"] = 55.0
    
    report_incompat = validate_compatibility(incompatible_features)
    assert report_incompat["overall_verdict"] in ["Potentially Incompatible", "Incompatible"]
    assert report_incompat["prediction_allowed"] is False
    assert "outside the validated feature distribution" in report_incompat["message"] or "blocked" in report_incompat["message"]


def test_api_extract_endpoint():
    test_img = _generate_test_image()
    _, buffer = cv2.imencode(".png", test_img)
    img_bytes = io.BytesIO(buffer.tobytes())
    
    with TestClient(app) as c:
        response = c.post(
            "/api/image-analysis/extract",
            files={"file": ("test.png", img_bytes, "image/png")}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["num_nuclei"] >= 3
        assert "features" in data
        assert len(data["features"]) == 30
        assert "compatibility" in data
        assert "diagnostic_images" in data


def test_api_predict_endpoint_blocking():
    # Uploading synthetic image should extract features but block prediction
    # because synthetic features differ significantly from human tissue training data distribution
    test_img = _generate_test_image(width=500, height=500, num_nuclei=12)
    _, buffer = cv2.imencode(".png", test_img)
    img_bytes = io.BytesIO(buffer.tobytes())
    
    with TestClient(app) as c:
        response = c.post(
            "/api/image-analysis/predict",
            files={"file": ("test.png", img_bytes, "image/png")}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["prediction"] is None
        assert data["prediction_blocked"] is True
        assert "block_reason" in data


def test_api_validation_rejections():
    with TestClient(app) as c:
        # 1. Invalid file format
        response = c.post(
            "/api/image-analysis/extract",
            files={"file": ("test.txt", io.BytesIO(b"not an image"), "text/plain")}
        )
        assert response.status_code == 400
        assert "Unsupported file type" in response.json()["detail"]
        
        # 2. Empty file
        response = c.post(
            "/api/image-analysis/extract",
            files={"file": ("test.png", io.BytesIO(b""), "image/png")}
        )
        assert response.status_code == 400
        assert "Empty file" in response.json()["detail"]
        
        # 3. Too small image
        small_img = np.zeros((10, 10, 3), dtype=np.uint8)
        _, buffer = cv2.imencode(".png", small_img)
        response = c.post(
            "/api/image-analysis/extract",
            files={"file": ("test.png", io.BytesIO(buffer.tobytes()), "image/png")}
        )
        assert response.status_code == 400
        assert "Image too small" in response.json()["detail"]


def test_existing_manual_predict_unaffected():
    benign_data = {
        "mean radius": 13.54, "mean texture": 14.36, "mean perimeter": 87.46,
        "mean area": 566.3, "mean smoothness": 0.09779, "mean compactness": 0.08129,
        "mean concavity": 0.06664, "mean concave points": 0.04781,
        "mean symmetry": 0.1885, "mean fractal dimension": 0.05766,
        "radius error": 0.2699, "texture error": 0.7886, "perimeter error": 2.058,
        "area error": 23.56, "smoothness error": 0.008462, "compactness error": 0.0146,
        "concavity error": 0.02387, "concave points error": 0.01315,
        "symmetry error": 0.0198, "fractal dimension error": 0.0023,
        "worst radius": 15.11, "worst texture": 19.26, "worst perimeter": 99.7,
        "worst area": 711.2, "worst smoothness": 0.144, "worst compactness": 0.1773,
        "worst concavity": 0.239, "worst concave points": 0.1288,
        "worst symmetry": 0.2977, "worst fractal dimension": 0.07259
    }
    
    with TestClient(app) as c:
        response = c.post("/api/predict", json=benign_data)
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert data["prediction"].lower() == "benign"
        assert "confidence" in data
        assert "probabilities" in data
