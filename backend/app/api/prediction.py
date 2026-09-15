import os
from fastapi import APIRouter, HTTPException, File, UploadFile
from typing import Dict, Any

from app.image_model import image_classifier

router = APIRouter()

@router.get("/health", response_model=Dict[str, Any])
def health_check():
    status = image_classifier.get_status()
    if not status["model_loaded"]:
        raise HTTPException(status_code=503, detail="Image classifier model not loaded or unavailable")
    return status

@router.get("/image-model/status", response_model=Dict[str, Any])
def get_image_model_status():
    return image_classifier.get_status()

@router.post("/image-predict", response_model=Dict[str, Any])
async def image_predict(file: UploadFile = File(...), model: str = "v5b"):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
        
    allowed_types = {"image/jpeg", "image/png", "image/jpg"}
    content_type = file.content_type
    ext = os.path.splitext(file.filename.lower())[1]
    
    if (content_type and content_type not in allowed_types) and (ext not in {".jpg", ".jpeg", ".png"}):
        raise HTTPException(status_code=400, detail="Invalid file type. Only JPEG and PNG are supported.")
        
    try:
        image_bytes = await file.read()
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
            
        prediction = image_classifier.predict_image(image_bytes, model_version=model)
        return prediction
    except HTTPException as e:
        raise e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected prediction failure: {str(e)}")

@router.post("/image-predict-v14", response_model=Dict[str, Any])
async def image_predict_v14(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
        
    allowed_types = {"image/jpeg", "image/png", "image/jpg"}
    content_type = file.content_type
    ext = os.path.splitext(file.filename.lower())[1]
    
    if (content_type and content_type not in allowed_types) and (ext not in {".jpg", ".jpeg", ".png"}):
        raise HTTPException(status_code=400, detail="Invalid file type. Only JPEG and PNG are supported.")
        
    try:
        image_bytes = await file.read()
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
            
        prediction = image_classifier.predict_image_v14(image_bytes)
        return prediction
    except HTTPException as e:
        raise e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected prediction failure: {str(e)}")
