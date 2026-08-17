import json
import joblib
import pandas as pd
from typing import Dict, Any

from app.core.config import settings
from app.schemas.prediction import BreastCancerFeatures, PredictionResponse, ModelMetadata

class ModelService:
    def __init__(self):
        self.model = None
        self.metadata = None

    def load_model(self):
        try:
            self.model = joblib.load(settings.MODEL_PATH)
            with open(settings.METADATA_PATH, 'r') as f:
                self.metadata = json.load(f)
            print(f"Model loaded: {self.metadata.get('model_name', 'Unknown')}")
        except FileNotFoundError as e:
            print(f"Error loading model: {e}")
            self.model = None
            self.metadata = None

    def get_health_status(self) -> Dict[str, Any]:
        if self.model is not None and self.metadata is not None:
            return {
                "status": "healthy",
                "model_loaded": True,
                "model_name": self.metadata.get("model_name"),
                "model_version": self.metadata.get("model_version")
            }
        return {
            "status": "unhealthy",
            "model_loaded": False,
            "model_name": None,
            "model_version": None
        }

    def predict(self, features: BreastCancerFeatures) -> PredictionResponse:
        if self.model is None or self.metadata is None:
            raise ValueError("Model is not loaded.")

        # Convert input features to pandas DataFrame ensuring correct order
        feature_names = self.metadata.get("feature_names", [])
        
        # Pydantic model dump with by_alias=True returns keys matching the feature names
        input_data = features.model_dump(by_alias=True)
        
        # Build DataFrame with the exact column order expected by the model
        df = pd.DataFrame([input_data], columns=feature_names)

        # Get prediction and probabilities
        pred = self.model.predict(df)[0]
        probs = self.model.predict_proba(df)[0]
        
        # Sklearn target_names are usually 0: malignant, 1: benign
        target_names = self.metadata.get("target_names", ["malignant", "benign"])
        class_name = target_names[pred].capitalize()
        class_code = "M" if "malignant" in class_name.lower() else "B"

        # Construct probabilities dict
        probabilities = {
            target_names[i].lower(): float(probs[i]) for i in range(len(target_names))
        }

        # Confidence is the probability of the predicted class
        confidence = float(probs[pred])

        return PredictionResponse(
            prediction=class_name,
            prediction_code=class_code,
            confidence=confidence,
            probabilities=probabilities,
            model=ModelMetadata(
                name=self.metadata.get("model_name"),
                version=self.metadata.get("model_version")
            )
        )

# Global singleton
model_service = ModelService()
