# V14 Runtime API Smoke Test Report

**Date**: September 02, 2026  
**Project**: Breast Cancer AI (BCD)  
**Smoke Test Verdict**: **PASS**  
**Summary**: **All FastAPI runtime HTTP endpoints executed cleanly. V5-B remains the default production model, V14 ensemble routing operates seamlessly across query parameters and dedicated routes, response schemas are 100% compatible, and invalid parameter inputs degrade safely.**

---

## 1. Backend Startup & Endpoint Verification

* **Backend Startup Status**: **SUCCESS** (Loaded ML models cleanly during FastAPI lifespan startup).
* **Image Model Status (`GET /api/image-model/status`)**: Status `200 OK` — `image_model: model_loaded`.

---

## 2. HTTP Endpoint Test Results

| Test Step | Endpoint & Request | Expected Behavior | Observed Status Code | Observed Model Version | Result |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **1. Default Inference** | `POST /api/image-predict` | Default to V5-B (`v5.0.0-b`) | `200 OK` | `v5.0.0-b` | **PASS** |
| **2. V14 Query Param** | `POST /api/image-predict?model=v14` | Select V14 Ensemble | `200 OK` | `v14-ensemble-50/50` | **PASS** |
| **3. Dedicated V14 Route** | `POST /api/image-predict-v14` | Select V14 Ensemble | `200 OK` | `v14-ensemble-50/50` | **PASS** |
| **4. Invalid Model Param** | `POST /api/image-predict?model=invalid_xyz` | Safe fallback to V5-B | `200 OK` | `v5.0.0-b` | **PASS** |

---

## 3. Response Schema Compatibility Comparison

Both V5-B and V14 endpoints return **100% identical and complete JSON response schemas**:

```json
{
  "predicted_class": "benign",
  "prediction": "benign",
  "confidence": 0.9982,
  "probabilities": {
    "benign": 0.9982,
    "malignant": 0.0015,
    "normal": 0.0003
  },
  "status": "high_confidence",
  "message": "AI prediction generated successfully with high confidence.",
  "image_quality": "acceptable",
  "quality_warnings": [],
  "model_version": "v14-ensemble-50/50",
  "explanation": {
    "available": true,
    "type": "grad_cam",
    "heatmap": "data:image/png;base64,...",
    "overlay": "data:image/png;base64,...",
    "disclaimer": "Highlighted regions indicate areas that influenced the AI model prediction. They do not represent a confirmed tumour boundary or medical diagnosis."
  }
}
```

### Field Presence Audit:
- `predicted_class`: **PRESENT** (backward compatibility)
- `prediction`: **PRESENT**
- `confidence`: **PRESENT**
- `probabilities`: **PRESENT**
- `status`: **PRESENT**
- `message`: **PRESENT**
- `image_quality`: **PRESENT**
- `quality_warnings`: **PRESENT**
- `model_version`: **PRESENT**
- `explanation` (Grad-CAM): **PRESENT & GENERATED**

---

## 4. Safety & Default Selection Audit

1. **V5-B Production Preservation**: V5-B (`breast_image_classifier.keras`, SHA256: `93b3c106cff51993b605220c98cfaf54c8fa404bb581656fa11337a26b32e03c`) remains the active default production model for standard API calls.
2. **Invalid Parameter Handling**: Supplying `?model=invalid_xyz` degrades safely to the V5-B production model without throwing exceptions or silently selecting unintended candidate models.
3. **Eager Loading**: V14 component models (`v11_mobilenetv2_b.keras` & `v13_efficientnet_b0.keras`) are loaded into RAM once at startup and reused across requests.
4. **Frontend Codebase**: **0 frontend files modified**.

---

## 5. Final Verdict

**FINAL RUNTIME SMOKE TEST VERDICT: PASS**
