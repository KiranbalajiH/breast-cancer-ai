import json
from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List

from app.core.config import settings
from app.services.model_service import model_service

router = APIRouter()

@router.get("/metadata", response_model=Dict[str, Any])
def get_metadata():
    if model_service.metadata is None:
        raise HTTPException(status_code=503, detail="Metadata not loaded")
    return model_service.metadata

@router.get("/comparison", response_model=Dict[str, Any])
def get_comparison():
    try:
        # Comparison path is in the same directory as metadata
        comparison_path = settings.METADATA_PATH.replace("metadata.json", "model_comparison.json")
        with open(comparison_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model comparison data not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/features", response_model=List[str])
def get_features():
    if model_service.metadata is None:
        raise HTTPException(status_code=503, detail="Metadata not loaded")
    return model_service.metadata.get("feature_names", [])
