import os
from fastapi import APIRouter, HTTPException, File, UploadFile
from typing import Dict, Any

from app.schemas.prediction import BreastCancerFeatures, PredictionResponse
from app.services.model_service import model_service
from app.image_model import image_classifier

router = APIRouter()

@router.get("/health", response_model=Dict[str, Any])
def health_check():
    status = model_service.get_health_status()
    if not status["model_loaded"]:
        raise HTTPException(status_code=503, detail="Model not loaded or unavailable")
    return status

@router.post("/predict", response_model=PredictionResponse)
def predict(features: BreastCancerFeatures):
    try:
        prediction = model_service.predict(features)
        return prediction
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing prediction: {str(e)}")

@router.get("/image-model/status", response_model=Dict[str, Any])
def get_image_model_status():
    return image_classifier.get_status()

@router.post("/image-predict", response_model=Dict[str, Any])
async def image_predict(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
        
    allowed_types = {"image/jpeg", "image/png", "image/jpg"}
    content_type = file.content_type
    ext = os.path.splitext(file.filename.lower())[1]
    
    # Validate type
    if (content_type and content_type not in allowed_types) and (ext not in {".jpg", ".jpeg", ".png"}):
        raise HTTPException(status_code=400, detail="Invalid file type. Only JPEG and PNG are supported.")
        
    try:
        image_bytes = await file.read()
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
            
        prediction = image_classifier.predict_image(image_bytes)
        return prediction
    except HTTPException as e:
        raise e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected prediction failure: {str(e)}")

