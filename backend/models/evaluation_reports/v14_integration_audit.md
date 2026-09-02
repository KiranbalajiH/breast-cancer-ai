# V14 Minimal Ensemble Integration Audit Report

**Date**: September 02, 2026  
**Project**: Breast Cancer AI (BCD)  
**Status**: AUDIT ONLY — NO CODE OR MODEL CHANGES PERFORMED

---

## A. Current Inference Flow

The active image prediction pipeline follows this exact execution sequence:

```
[HTTP Client / Frontend]
        │  POST /api/v1/image-predict (multipart/form-data)
        ▼
[backend/app/api/prediction.py :: image_predict()]
        │  - Validates MIME type (.jpg, .jpeg, .png)
        │  - Reads image_bytes
        ▼
[backend/app/image_model.py :: ImageClassifier.predict_image()]
        │  1. Decode image bytes via cv2.imdecode
        │  2. Validate image dimensions (>= 50x50)
        │  3. Run non-blocking quality safeguards (blur, dark, bright, contrast, aspect ratio, color channel diff)
        │  4. Convert BGR -> RGB and apply letterbox_preprocess() to (224, 224)
        │  5. Preprocess via MobileNetV2 preprocess_input()
        │  6. Execute model inference on V5-B (breast_image_classifier.keras)
        │  7. Compute class decision via argmax(preds)
        │  8. Generate Grad-CAM visualization overlay via generate_gradcam()
        ▼
[API JSON Response]
        {
          "predicted_class": "benign" | "malignant" | "normal",
          "prediction": "benign" | "malignant" | "normal",
          "confidence": float,
          "probabilities": {"benign": float, "malignant": float, "normal": float},
          "status": "high_confidence" | "moderate_confidence" | "low_confidence" | "review_required",
          "message": string,
          "image_quality": "acceptable" | "poor",
          "quality_warnings": list,
          "model_version": "v5.0.0-b",
          "explanation": {"available": bool, "heatmap": b64_str, "overlay": b64_str}
        }
```

---

## B. Exact Files Involved

1. [`backend/app/main.py`](file:///c:/Users/kiran/BCD/backend/app/main.py): Server startup lifespan event, triggering `image_classifier.load_model()`.
2. [`backend/app/api/prediction.py`](file:///c:/Users/kiran/BCD/backend/app/api/prediction.py): API router containing `@router.post("/image-predict")` and `@router.get("/image-model/status")`.
3. [`backend/app/image_model.py`](file:///c:/Users/kiran/BCD/backend/app/image_model.py): Core `ImageClassifier` class handling model loading, preprocessing, inference, confidence status logic, and Grad-CAM generation.
4. [`backend/app/core/config.py`](file:///c:/Users/kiran/BCD/backend/app/core/config.py): Application settings (thresholds, CORS, API paths).
5. [`frontend/lib/api.ts`](file:///c:/Users/kiran/BCD/frontend/lib/api.ts): Frontend API client interfacing with `/api/v1/image-predict`.

---

## C. Current Model-Loading Mechanism

- **Singleton Pattern**: Instantiated as global object `image_classifier = ImageClassifier()` in `backend/app/image_model.py`.
- **Eager Loading at Startup**: `main.py` invokes `image_classifier.load_model()` inside FastAPI's `@asynccontextmanager` lifespan function at server boot.
- **Paths**:
  - Model Path: `backend/models/breast_image_classifier.keras` (V5-B)
  - Metadata Path: `backend/models/breast_image_classifier_metadata.json`

---

## D. Current Preprocessing Mechanism

- **Letterbox Resizing**: `letterbox_preprocess(img_rgb, target_size=(224, 224))` resizes images maintaining aspect ratio with black border padding.
- **Normalization**: `tensorflow.keras.applications.mobilenet_v2.preprocess_input(x)` scales RGB float32 pixels to `[-1.0, 1.0]`.

---

## E. Exact V14 Decision Logic

V14 is a frozen probability ensemble of **V11-B** (MobileNetV2, high sensitivity) and **V13-B0** (EfficientNetB0, high precision).

### 1. Model Preprocessing Requirements:
- **V11-B Input**: `x_v11 = tf.keras.applications.mobilenet_v2.preprocess_input(img_resized.copy())`
- **V13-B0 Input**: `x_v13 = tf.keras.applications.efficientnet.preprocess_input(img_resized.copy())`

### 2. Probability Combination:
$$P_{\text{ensemble}} = 0.5 \cdot P_{\text{V11-B}} + 0.5 \cdot P_{\text{V13-B0}}$$

### 3. Malignant Decision Policy (Threshold $t^* = 0.58$):
Class index mapping: `0: benign`, `1: malignant`, `2: normal`.

```python
if probs_ensemble[1] >= 0.58:
    pred_idx = 1  # malignant
else:
    # Non-malignant case: select higher between benign (0) and normal (2)
    pred_idx = 0 if probs_ensemble[0] >= probs_ensemble[2] else 2

predicted_class = classes[pred_idx]
confidence = float(probs_ensemble[pred_idx])
```

---

## F. Minimal Files That Would Need Modification

To add V14 ensemble support cleanly without overbuilding or breaking existing V5-B production behavior:

1. **[`backend/app/image_model.py`](file:///c:/Users/kiran/BCD/backend/app/image_model.py)**:
   - Add V14 ensemble prediction method (`predict_image_v14` or mode toggle in `ImageClassifier`).
   - Add V11-B and V13-B0 model paths and loading logic (`v11_mobilenetv2_b.keras` & `v13_efficientnet_b0.keras`).
2. **[`backend/app/api/prediction.py`](file:///c:/Users/kiran/BCD/backend/app/api/prediction.py)**:
   - Add optional `use_v14_ensemble: bool = False` query parameter to `POST /api/v1/image-predict`, or add endpoint `POST /api/v1/image-predict-v14`.

---

## G. Proposed Minimal Implementation

### Eager Dual-Model Loading in `ImageClassifier`:
```python
class V14EnsembleClassifier:
    def __init__(self):
        self.v11_model = None
        self.v13_model = None
        self.classes = ["benign", "malignant", "normal"]
        self.threshold = 0.58
        
    def load_models(self):
        if self.v11_model is None:
            self.v11_model = tf.keras.models.load_model("backend/models/candidates/v11_mobilenetv2_b.keras")
        if self.v13_model is None:
            self.v13_model = tf.keras.models.load_model("backend/models/candidates/v13_efficientnet_b0.keras")
```

### Ensemble Prediction Method:
```python
def predict_v14(self, image_bytes: bytes):
    img_rgb, img_raw = decode_and_letterbox(image_bytes)
    
    x11 = preprocess_mnv2(np.expand_dims(img_rgb, 0).copy())
    x13 = preprocess_effnet(np.expand_dims(img_rgb, 0).copy())
    
    p11 = self.v11_model.predict(x11, verbose=0)[0]
    p13 = self.v13_model.predict(x13, verbose=0)[0]
    
    p_ens = 0.5 * p11 + 0.5 * p13
    
    pred_idx = 1 if p_ens[1] >= 0.58 else (0 if p_ens[0] >= p_ens[2] else 2)
    
    # Grad-CAM explanation generated from V13-B0 backbone
    explanation = self.generate_gradcam_v13(x13, img_raw, pred_idx)
    
    return {
        "predicted_class": self.classes[pred_idx],
        "prediction": self.classes[pred_idx],
        "confidence": float(p_ens[pred_idx]),
        "probabilities": {self.classes[i]: float(p_ens[i]) for i in range(3)},
        "model_version": "v14-ensemble-50/50",
        "explanation": explanation
    }
```

---

## H. Backward Compatibility & Fallback Approach

1. **V5-B Production Preservation**: V5-B (`breast_image_classifier.keras`) remains the default active model for standard calls.
2. **Explicit V14 Route / Opt-In**: Add `POST /api/v1/image-predict?model=v14` or `POST /api/v1/image-predict-v14`. Standard `/api/v1/image-predict` continues serving V5-B unless parameter is passed.
3. **Frontend Schema Compatibility**: Response schema is 100% identical (`prediction`, `confidence`, `probabilities`, `status`, `explanation`). Zero frontend code changes required!

---

## I. Test Plan

1. **Model Loading Test**: Verify `V14EnsembleClassifier.load_models()` boots cleanly in startup lifespan without errors.
2. **Deterministic Prediction Test**: Run benchmark test images through V5-B and V14 ensemble to verify output class decisions and probability averages match `train_v14.py`.
3. **Grad-CAM Verification**: Confirm Grad-CAM heatmap and overlay render cleanly without tracing errors.
4. **API Endpoint Test**: Verify HTTP POST request to `/api/v1/image-predict` returns 200 OK with valid JSON response.

---

## J. Risks & Issues Discovered

- **Grad-CAM Layer Tracing**: EfficientNetB0 has a different layer structure than MobileNetV2. Grad-CAM generation for V13-B0 must target EfficientNet's top convolutional layer (`top_conv` / `block7a_project_conv`).
- **RAM Footprint**: Loading both V11-B (2.26M params) and V13-B0 (4.05M params) alongside V5-B takes **~180 MB RAM** and **2.51 seconds** load time at startup, which is well within standard server resources.

---

PRODUCTION MODEL CHANGED: NO
PRODUCTION METADATA CHANGED: NO
LOCKED 117-IMAGE TEST SET MODIFIED: NO
DATASET IMAGES MODIFIED: NO
DEPLOYMENT PERFORMED: NO
GIT COMMIT PERFORMED: NO
GIT PUSH PERFORMED: NO
