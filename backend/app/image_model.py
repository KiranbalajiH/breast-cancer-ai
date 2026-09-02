import os
import cv2
import json
import base64
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_mnv2
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_effnet
from app.core.config import settings

def letterbox_preprocess(img, target_size=(224, 224), pad_color=(0, 0, 0)):
    h, w = img.shape[:2]
    target_w, target_h = target_size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
    
    pad_w = target_w - new_w
    pad_h = target_h - new_h
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color)
    return padded

class ImageClassifier:
    def __init__(self, model_path=None, metadata_path=None):
        backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if model_path is None:
            model_path = os.path.join(backend_dir, "models", "breast_image_classifier.keras")
        if metadata_path is None:
            metadata_path = os.path.join(backend_dir, "models", "breast_image_classifier_metadata.json")
            
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.v11_model_path = os.path.join(backend_dir, "models", "candidates", "v11_mobilenetv2_b.keras")
        self.v13_model_path = os.path.join(backend_dir, "models", "candidates", "v13_efficientnet_b0.keras")
        
        self.model = None
        self.v11_model = None
        self.v13_model = None
        self.metadata = None
        self.classes = ["benign", "malignant", "normal"]
        self.image_size = (224, 224)
        
    def load_model(self, load_v14=False):
        if self.model is None:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Production model not found at {self.model_path}")
            self.model = tf.keras.models.load_model(self.model_path)
            
        if load_v14:
            if self.v11_model is None and os.path.exists(self.v11_model_path):
                self.v11_model = tf.keras.models.load_model(self.v11_model_path)
            if self.v13_model is None and os.path.exists(self.v13_model_path):
                self.v13_model = tf.keras.models.load_model(self.v13_model_path)
                
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

    def generate_gradcam(self, x, original_img, pred_idx):
        # Validate tensor shapes
        if len(x.shape) != 4 or x.shape[0] != 1 or x.shape[1:3] != self.image_size:
            raise ValueError(f"Invalid input tensor shape for Grad-CAM: {x.shape}")
            
        # Detect if base model is nested (like having 'densenet121', 'efficientnetb0', or 'mobilenetv2')
        nested_base_layer = None
        for layer in self.model.layers:
            if any(name in layer.name.lower() for name in ['mobilenet', 'densenet', 'efficientnet', 'base_model']):
                nested_base_layer = layer
                break
            elif hasattr(layer, 'layers') and any(any(kw in sub_l.name.lower() for kw in ('conv', 'relu')) for sub_l in getattr(layer, 'layers', [])):
                nested_base_layer = layer
                break
                
        if nested_base_layer is not None:
            # Nested model case (V2 candidate models)
            base_model = nested_base_layer
            found = False
            for layer in reversed(base_model.layers):
                try:
                    shape = layer.output.shape
                except AttributeError:
                    try:
                        shape = layer.output_shape
                    except AttributeError:
                        continue
                if len(shape) == 4 and any(kw in layer.name.lower() for kw in ('conv', 'concat', 'relu', 'pool')):
                    target_layer = layer
                    found = True
                    break
            if not found:
                raise ValueError("No suitable convolutional layer found in base model for Grad-CAM.")
                
            base_submodel = tf.keras.models.Model(inputs=base_model.input, outputs=[target_layer.output, base_model.output])
            
            # Find the dense layer from the model
            dense_layer = None
            for layer in self.model.layers:
                if isinstance(layer, tf.keras.layers.Dense):
                    dense_layer = layer
                    break
            if dense_layer is None:
                raise ValueError("Dense layer not found in model.")
                
            with tf.GradientTape() as tape:
                conv_outputs, base_outputs = base_submodel(x)
                tape.watch(conv_outputs)
                # Mathematical operations to bypass Keras layer tracing bugs
                pooling_output = tf.reduce_mean(base_outputs, axis=[1, 2])
                logits = tf.matmul(pooling_output, dense_layer.kernel) + dense_layer.bias
                predictions = tf.nn.softmax(logits)
                class_channel = predictions[:, pred_idx]
                
            grads = tape.gradient(class_channel, conv_outputs)
            
        else:
            # Flattened model case (like V1 baseline model)
            last_conv_layer_name = "Conv_1"
            try:
                self.model.get_layer(last_conv_layer_name)
            except ValueError:
                found = False
                for layer in reversed(self.model.layers):
                    try:
                        shape = layer.output.shape
                    except AttributeError:
                        try:
                            shape = layer.output_shape
                        except AttributeError:
                            continue
                    if len(shape) == 4 and any(kw in layer.name.lower() for kw in ('conv', 'concat', 'relu', 'pool')):
                        last_conv_layer_name = layer.name
                        found = True
                        break
                if not found:
                    raise ValueError("No suitable convolutional layer found in model for Grad-CAM.")
                
            grad_model = tf.keras.models.Model(
                inputs=[self.model.inputs],
                outputs=[self.model.get_layer(last_conv_layer_name).output, self.model.output]
            )
            
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(x)
                class_channel = predictions[:, pred_idx]
                
            grads = tape.gradient(class_channel, conv_outputs)
            
        # Pool gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weighted combination
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Relu and norm
        heatmap = tf.maximum(heatmap, 0.0)
        max_val = tf.math.reduce_max(heatmap)
        if max_val == 0.0:
            max_val = 1e-10
        heatmap = heatmap / max_val
        heatmap_numpy = heatmap.numpy()
        
        # Resize to match original image dimensions
        h_orig, w_orig = original_img.shape[:2]
        heatmap_resized = cv2.resize(heatmap_numpy, (w_orig, h_orig))
        heatmap_scaled = np.uint8(255 * heatmap_resized)
        
        # Colormap
        heatmap_color = cv2.applyColorMap(heatmap_scaled, cv2.COLORMAP_JET)
        
        # Blend
        overlay = cv2.addWeighted(original_img, 0.65, heatmap_color, 0.35, 0)
        
        # Encode as base64 png
        success_heat, heat_buf = cv2.imencode('.png', heatmap_color)
        success_over, over_buf = cv2.imencode('.png', overlay)
        
        if not success_heat or not success_over:
            raise ValueError("Failed to encode Grad-CAM images to PNG.")
            
        heat_b64 = base64.b64encode(heat_buf.tobytes()).decode('utf-8')
        over_b64 = base64.b64encode(over_buf.tobytes()).decode('utf-8')
        
        return {
            "heatmap": f"data:image/png;base64,{heat_b64}",
            "overlay": f"data:image/png;base64,{over_b64}"
        }

    def predict_image(self, image_bytes: bytes, model_version: str = "v5b"):
        if model_version.lower() in ["v14", "v14-ensemble"]:
            return self.predict_image_v14(image_bytes)
            
        self.load_model()
        
        # Step 5: Validate uploaded file (Blocking validations)
        if not image_bytes or len(image_bytes) == 0:
            raise ValueError("Uploaded file is empty.")
            
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image file or format. Could not decode image.")
            
        h, w = img.shape[:2]
        if h < 50 or w < 50:
            raise ValueError(f"Image dimensions too small ({w}x{h}). Minimum 50x50 pixels required.")
            
        # Step 6: Basic Image Quality Check (Non-blocking)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_intensity = float(np.mean(gray))
        std_intensity = float(np.std(gray))
        
        # Blurry check (variance of Laplacian)
        blur_val = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        
        quality_warnings = []
        if blur_val < settings.QUALITY_BLUR_THRESHOLD:
            quality_warnings.append("Image appears excessively blurry.")
        if mean_intensity < settings.QUALITY_DARK_THRESHOLD:
            quality_warnings.append("Image appears excessively dark.")
        if mean_intensity > settings.QUALITY_BRIGHT_THRESHOLD:
            quality_warnings.append("Image appears excessively bright.")
        if std_intensity < settings.QUALITY_LOW_INFO_THRESHOLD:
            quality_warnings.append("Image has extremely low contrast or details (low information).")
            
        image_quality = "poor" if len(quality_warnings) > 0 else "acceptable"
        
        # Step 7: Unsupported Image Modality Check (Non-blocking)
        b, g, r = cv2.split(img)
        channel_diff = float(np.mean(np.abs(b.astype(float) - g.astype(float)) + np.abs(g.astype(float) - r.astype(float))))
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mean_saturation = float(np.mean(hsv[:, :, 1]))
        
        aspect_ratio = float(w / h)
        
        is_suspicious = False
        suspicious_reason = ""
        if channel_diff > settings.SAFEGUARD_COLOR_DIFF_THRESHOLD or mean_saturation > settings.SAFEGUARD_SATURATION_THRESHOLD:
            is_suspicious = True
            suspicious_reason = "Image contains significant color content, which is atypical for grayscale breast ultrasound scans."
        elif aspect_ratio < settings.SAFEGUARD_MIN_ASPECT_RATIO or aspect_ratio > settings.SAFEGUARD_MAX_ASPECT_RATIO:
            is_suspicious = True
            suspicious_reason = f"Image aspect ratio ({aspect_ratio:.2f}) is outside the typical range for clinical ultrasound scans."
            
        # Preprocess: convert to RGB and letterbox resize
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = letterbox_preprocess(img_rgb, self.image_size)
        
        # Expand dimensions and apply MobileNetV2 preprocessing
        x = np.expand_dims(img_resized, axis=0).astype(np.float32)
        x = preprocess_mnv2(x)
        
        # Run inference
        preds = self.model.predict(x, verbose=0)[0]
        pred_idx = np.argmax(preds)
        predicted_class = self.classes[pred_idx]
        confidence = float(preds[pred_idx])
        probabilities = {self.classes[i]: float(preds[i]) for i in range(len(self.classes))}
        
        sorted_probs = sorted(preds, reverse=True)
        margin = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else 0.0
        
        # Step 8: Apply prediction status threshold logic
        if is_suspicious:
            status = "unsupported_or_review_required"
            message = f"Unsupported input detected: {suspicious_reason} Prediction may be unreliable and should be verified."
        elif confidence < settings.CONFIDENCE_REVIEW_THRESHOLD or margin < settings.CONFIDENCE_MARGIN_THRESHOLD:
            status = "review_required"
            message = "The model prediction has low confidence or close competition and should be reviewed by a qualified clinician."
        elif confidence >= settings.CONFIDENCE_HIGH_THRESHOLD:
            status = "high_confidence"
            message = "AI prediction generated successfully with high confidence."
        elif confidence >= settings.CONFIDENCE_MODERATE_THRESHOLD:
            status = "moderate_confidence"
            message = "AI prediction generated successfully with moderate confidence."
        else:
            status = "low_confidence"
            message = "AI prediction generated successfully but has lower confidence."
            
        version = self.metadata.get("model_version", "v5.0.0-b") if self.metadata else "v5.0.0-b"
        
        explanation_result = {
            "available": False,
            "type": "grad_cam",
            "heatmap": "",
            "overlay": "",
            "disclaimer": "Highlighted regions indicate areas that influenced the AI model prediction. They do not represent a confirmed tumour boundary or medical diagnosis."
        }
        
        try:
            gradcam_data = self.generate_gradcam(x, img, pred_idx)
            explanation_result.update({
                "available": True,
                "heatmap": gradcam_data["heatmap"],
                "overlay": gradcam_data["overlay"]
            })
        except Exception as e:
            print(f"Grad-CAM generation failed: {str(e)}")
            explanation_result.update({
                "available": False,
                "error": "Failed to generate visual explanation."
            })
            
        return {
            "predicted_class": predicted_class,
            "prediction": predicted_class,
            "confidence": confidence,
            "probabilities": probabilities,
            "status": status,
            "message": message,
            "image_quality": image_quality,
            "quality_warnings": quality_warnings,
            "model_version": version,
            "explanation": explanation_result
        }

    def predict_image_v14(self, image_bytes: bytes):
        self.load_model(load_v14=True)
        if self.v11_model is None or self.v13_model is None:
            raise FileNotFoundError("V14 ensemble candidate models (V11-B / V13-B0) are missing.")
            
        if not image_bytes or len(image_bytes) == 0:
            raise ValueError("Uploaded file is empty.")
            
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image file or format. Could not decode image.")
            
        h, w = img.shape[:2]
        if h < 50 or w < 50:
            raise ValueError(f"Image dimensions too small ({w}x{h}). Minimum 50x50 pixels required.")
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_intensity = float(np.mean(gray))
        std_intensity = float(np.std(gray))
        blur_val = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        
        quality_warnings = []
        if blur_val < settings.QUALITY_BLUR_THRESHOLD:
            quality_warnings.append("Image appears excessively blurry.")
        if mean_intensity < settings.QUALITY_DARK_THRESHOLD:
            quality_warnings.append("Image appears excessively dark.")
        if mean_intensity > settings.QUALITY_BRIGHT_THRESHOLD:
            quality_warnings.append("Image appears excessively bright.")
        if std_intensity < settings.QUALITY_LOW_INFO_THRESHOLD:
            quality_warnings.append("Image has extremely low contrast or details (low information).")
            
        image_quality = "poor" if len(quality_warnings) > 0 else "acceptable"
        
        b, g, r = cv2.split(img)
        channel_diff = float(np.mean(np.abs(b.astype(float) - g.astype(float)) + np.abs(g.astype(float) - r.astype(float))))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mean_saturation = float(np.mean(hsv[:, :, 1]))
        aspect_ratio = float(w / h)
        
        is_suspicious = False
        suspicious_reason = ""
        if channel_diff > settings.SAFEGUARD_COLOR_DIFF_THRESHOLD or mean_saturation > settings.SAFEGUARD_SATURATION_THRESHOLD:
            is_suspicious = True
            suspicious_reason = "Image contains significant color content, which is atypical for grayscale breast ultrasound scans."
        elif aspect_ratio < settings.SAFEGUARD_MIN_ASPECT_RATIO or aspect_ratio > settings.SAFEGUARD_MAX_ASPECT_RATIO:
            is_suspicious = True
            suspicious_reason = f"Image aspect ratio ({aspect_ratio:.2f}) is outside the typical range for clinical ultrasound scans."
            
        # Letterbox preprocess
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = letterbox_preprocess(img_rgb, self.image_size)
        
        # Parallel model inputs
        x11 = preprocess_mnv2(np.expand_dims(img_resized, axis=0).astype(np.float32))
        x13 = preprocess_effnet(np.expand_dims(img_resized, axis=0).astype(np.float32))
        
        p11 = self.v11_model.predict(x11, verbose=0)[0]
        p13 = self.v13_model.predict(x13, verbose=0)[0]
        
        # V14 50/50 Probability Ensemble
        preds = 0.5 * p11 + 0.5 * p13
        
        # V14 Malignant Threshold Policy (t* = 0.58)
        if preds[1] >= 0.58:
            pred_idx = 1
        else:
            pred_idx = 0 if preds[0] >= preds[2] else 2
            
        predicted_class = self.classes[pred_idx]
        confidence = float(preds[pred_idx])
        probabilities = {self.classes[i]: float(preds[i]) for i in range(len(self.classes))}
        
        sorted_probs = sorted(preds, reverse=True)
        margin = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else 0.0
        
        if is_suspicious:
            status = "unsupported_or_review_required"
            message = f"Unsupported input detected: {suspicious_reason} Prediction may be unreliable and should be verified."
        elif confidence < settings.CONFIDENCE_REVIEW_THRESHOLD or margin < settings.CONFIDENCE_MARGIN_THRESHOLD:
            status = "review_required"
            message = "The model prediction has low confidence or close competition and should be reviewed by a qualified clinician."
        elif confidence >= settings.CONFIDENCE_HIGH_THRESHOLD:
            status = "high_confidence"
            message = "AI prediction generated successfully with high confidence."
        elif confidence >= settings.CONFIDENCE_MODERATE_THRESHOLD:
            status = "moderate_confidence"
            message = "AI prediction generated successfully with moderate confidence."
        else:
            status = "low_confidence"
            message = "AI prediction generated successfully but has lower confidence."
            
        explanation_result = {
            "available": False,
            "type": "grad_cam",
            "heatmap": "",
            "overlay": "",
            "disclaimer": "Highlighted regions indicate areas that influenced the AI model prediction. They do not represent a confirmed tumour boundary or medical diagnosis."
        }
        
        try:
            gradcam_data = self.generate_gradcam(x11, img, pred_idx)
            explanation_result.update({
                "available": True,
                "heatmap": gradcam_data["heatmap"],
                "overlay": gradcam_data["overlay"]
            })
        except Exception as e:
            print(f"Grad-CAM generation failed: {str(e)}")
            explanation_result.update({
                "available": False,
                "error": "Failed to generate visual explanation."
            })
            
        return {
            "predicted_class": predicted_class,
            "prediction": predicted_class,
            "confidence": confidence,
            "probabilities": probabilities,
            "status": status,
            "message": message,
            "image_quality": image_quality,
            "quality_warnings": quality_warnings,
            "model_version": "v14-ensemble-50/50",
            "explanation": explanation_result
        }

# Global singleton
image_classifier = ImageClassifier()
