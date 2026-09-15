import os
import json
from fastapi import APIRouter, HTTPException
from typing import Dict, Any

from app.image_model import image_classifier

router = APIRouter()

@router.get("/metadata", response_model=Dict[str, Any])
def get_metadata():
    if image_classifier.metadata is None:
        raise HTTPException(status_code=503, detail="Image model metadata not loaded")
    return image_classifier.metadata
