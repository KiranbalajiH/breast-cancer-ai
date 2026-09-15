from pydantic import BaseModel
from typing import Dict, Optional, List

class ModelMetadata(BaseModel):
    name: str
    version: str

class ImageExplanation(BaseModel):
    available: bool
    type: Optional[str] = None
    heatmap: Optional[str] = None
    overlay: Optional[str] = None
    disclaimer: Optional[str] = None

class ImagePredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]
    status: Optional[str] = None
    message: Optional[str] = None
    image_quality: Optional[str] = None
    quality_warnings: Optional[List[str]] = None
    model_version: Optional[str] = None
    explanation: Optional[ImageExplanation] = None
