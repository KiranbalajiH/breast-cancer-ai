"""
Experimental Image Analysis API Endpoints
==========================================
Provides endpoints for uploading cell microscopy images and extracting
the 30 Wisconsin-style features via computer vision.

Endpoints:
    POST /api/image-analysis/extract  — Extract features + compatibility report
    POST /api/image-analysis/predict  — Extract + predict (only if compatible)

These endpoints are SEPARATE from the production prediction pipeline.
The existing POST /api/predict remains unchanged.
"""

import io
import base64
import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Any, Dict

from image_processing.preprocessing import preprocess_image
from image_processing.segmentation import segment_nuclei
from image_processing.feature_extraction import extract_all_nuclei_features
from image_processing.aggregation import aggregate_features, FEATURE_NAMES_ORDERED
from image_processing.compatibility import validate_compatibility
from app.schemas.prediction import BreastCancerFeatures
from app.services.model_service import model_service

router = APIRouter(tags=["Image Analysis (Experimental)"])

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/jpg"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


def _encode_image_base64(image: np.ndarray) -> str:
    """Encode a numpy image to base64 PNG string."""
    _, buffer = cv2.imencode(".png", image)
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


async def _read_and_validate_image(file: UploadFile) -> np.ndarray:
    """Read and validate an uploaded image file."""
    # Validate content type
    if file.content_type and file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Only JPEG and PNG images are accepted."
        )
    
    # Read file bytes
    contents = await file.read()
    
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({len(contents) / 1024 / 1024:.1f} MB). Maximum size is 10 MB."
        )
    
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")
    
    # Decode image
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Could not decode the uploaded file as an image. Please provide a valid JPEG or PNG image."
        )
    
    # Basic sanity check on dimensions
    h, w = image.shape[:2]
    if h < 50 or w < 50:
        raise HTTPException(
            status_code=400,
            detail=f"Image too small ({w}x{h}). Minimum 50x50 pixels required."
        )
    
    return image


def _run_pipeline(image: np.ndarray) -> Dict[str, Any]:
    """
    Run the full image analysis pipeline.
    
    Returns a dict with all intermediate results, features, and compatibility.
    """
    # Step 1: Preprocess
    gray_original, preprocessed = preprocess_image(image)
    
    # Step 2: Segment
    seg_result = segment_nuclei(gray_original, preprocessed)
    
    num_nuclei = seg_result["num_nuclei"]
    
    # Generate diagnostic images
    diagnostic_images = {
        "original": _encode_image_base64(image),
        "preprocessed": _encode_image_base64(preprocessed),
        "binary_mask": _encode_image_base64(seg_result["binary_mask"]),
        "nuclei_overlay": _encode_image_base64(seg_result["overlay"]),
    }
    
    if num_nuclei == 0:
        return {
            "success": False,
            "num_nuclei": 0,
            "message": "No cell nuclei could be detected in this image. "
                       "This may indicate the image is not a suitable microscopy image, "
                       "or the image quality/contrast is insufficient for segmentation.",
            "diagnostic_images": diagnostic_images,
            "features": None,
            "compatibility": None,
        }
    
    # Step 3: Extract per-nucleus features
    nuclei_features = extract_all_nuclei_features(seg_result["contours"], gray_original)
    
    if len(nuclei_features) < 3:
        return {
            "success": False,
            "num_nuclei": len(nuclei_features),
            "message": f"Only {len(nuclei_features)} nucleus/nuclei could be reliably measured. "
                       f"At least 3 are required for meaningful statistical aggregation.",
            "diagnostic_images": diagnostic_images,
            "features": None,
            "compatibility": None,
        }
    
    # Step 4: Aggregate
    agg_result = aggregate_features(nuclei_features)
    
    if agg_result is None:
        return {
            "success": False,
            "num_nuclei": len(nuclei_features),
            "message": "Feature aggregation failed. Insufficient data for meaningful statistics.",
            "diagnostic_images": diagnostic_images,
            "features": None,
            "compatibility": None,
        }
    
    features_30, aggregation_metadata = agg_result
    
    # Step 5: Compatibility validation
    compatibility = validate_compatibility(features_30)
    
    # Order the features to match the model exactly
    ordered_features = {name: features_30[name] for name in FEATURE_NAMES_ORDERED}
    
    return {
        "success": True,
        "num_nuclei": seg_result["num_nuclei"],
        "num_measured": len(nuclei_features),
        "message": f"Successfully extracted features from {len(nuclei_features)} nuclei "
                   f"(out of {seg_result['num_nuclei']} detected).",
        "diagnostic_images": diagnostic_images,
        "features": ordered_features,
        "compatibility": compatibility,
        "aggregation_metadata": aggregation_metadata,
    }


@router.post("/extract")
async def extract_features(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Upload a cell microscopy image and extract the 30 candidate features.
    
    Returns extracted features, compatibility report, and diagnostic images.
    Does NOT run prediction.
    """
    image = await _read_and_validate_image(file)
    result = _run_pipeline(image)
    return result


@router.post("/predict")
async def predict_from_image(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Upload a cell microscopy image, extract features, validate compatibility,
    and run prediction through the existing SVM model if compatible.
    
    Returns extraction results + prediction (or rejection reason).
    """
    image = await _read_and_validate_image(file)
    result = _run_pipeline(image)
    
    if not result["success"]:
        return {
            **result,
            "prediction": None,
            "prediction_blocked": True,
            "block_reason": result["message"],
        }
    
    compatibility = result["compatibility"]
    
    if not compatibility["prediction_allowed"]:
        return {
            **result,
            "prediction": None,
            "prediction_blocked": True,
            "block_reason": compatibility["message"],
        }
    
    # Run prediction through existing model
    try:
        features_input = BreastCancerFeatures(**result["features"])
        prediction = model_service.predict(features_input)
        
        return {
            **result,
            "prediction": {
                "prediction": prediction.prediction,
                "prediction_code": prediction.prediction_code,
                "confidence": prediction.confidence,
                "probabilities": prediction.probabilities,
                "model": {
                    "name": prediction.model.name,
                    "version": prediction.model.version,
                },
            },
            "prediction_blocked": False,
            "block_reason": None,
        }
    except Exception as e:
        return {
            **result,
            "prediction": None,
            "prediction_blocked": True,
            "block_reason": f"Model prediction failed: {str(e)}",
        }
