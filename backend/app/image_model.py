import os
import cv2
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

class ImageClassifier:
    def __init__(self, model_path=None, metadata_path=None):
        backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if model_path is None:
            model_path = os.path.join(backend_dir, "models", "breast_image_classifier.keras")
        if metadata_path is None:
            metadata_path = os.path.join(backend_dir, "models", "breast_image_classifier_metadata.json")
            
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.model = None
        self.metadata = None
        self.classes = ["benign", "malignant", "normal"]
        self.image_size = (224, 224)
        
    def load_model(self):
        if self.model is None:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model not found at {self.model_path}")
            self.model = tf.keras.models.load_model(self.model_path)
            
        if self.metadata is None:
            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                if "classes" in self.metadata:
                    self.classes = self.metadata["classes"]
                if "image_size" in self.metadata:
                    self.image_size = tuple(self.metadata["image_size"])

    def get_status(self):
        try:
            self.load_model()
            return {
                "status": "ready",
                "model_loaded": True,
                "classes": self.classes
            }
        except Exception:
            return {
                "status": "unavailable",
                "model_loaded": False
            }

    def predict_image(self, image_bytes: bytes):
        self.load_model()
        
        # Read image from bytes
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image file or format")
            
        # Preprocess: convert to RGB and resize
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, self.image_size)
        
        # Expand dimensions and apply MobileNetV2 preprocessing
        x = np.expand_dims(img_resized, axis=0).astype(np.float32)
        x = preprocess_input(x)
        
        # Run inference
        preds = self.model.predict(x)[0]
        pred_idx = np.argmax(preds)
        
        return {
            "predicted_class": self.classes[pred_idx],
            "confidence": float(preds[pred_idx]),
            "probabilities": {
                self.classes[i]: float(preds[i]) for i in range(len(self.classes))
            }
        }

# Global singleton
image_classifier = ImageClassifier()
