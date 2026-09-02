# V14 Backend Reproduction Audit Report

**Date**: September 02, 2026  
**Project**: Breast Cancer AI (BCD)  
**Audit Verdict**: **PASS**  
**Audit Finding**: **Backend V14 reproduces the frozen research V14 results with 100% exact numerical precision across all benchmark splits.**

---

## 1. Executive Summary

This audit empirically verifies that the backend implementation (`image_classifier.predict_image_v14`) reproduces the exact probability predictions, class decisions, and performance metrics of the V14 frozen research ensemble ($50\%$ V11-B MobileNetV2 + $50\%$ V13-B0 EfficientNetB0, malignant threshold $t^* = 0.58$).

---

## 2. Grouped Generalization Test Metrics Comparison

Evaluated on the 107-scan frozen grouped generalization test set:

| Metric | Research V14 Benchmark | Backend V14 Implementation | Reproduction Status | Match |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | **80.37%** | **80.37%** | **PASS** | **EXACT MATCH** |
| **Macro F1** | **79.12%** | **79.12%** | **PASS** | **EXACT MATCH** |
| **Malignant Recall** | **83.87%** | **83.87%** | **PASS** | **EXACT MATCH** |
| **Malignant Precision** | **74.29%** | **74.29%** | **PASS** | **EXACT MATCH** |
| **Malignant F1** | **78.79%** | **78.79%** | **PASS** | **EXACT MATCH** |
| **False Negatives (FN)** | **5** | **5** | **PASS** | **EXACT MATCH** |
| **False Positives (FP)** | **9** | **9** | **PASS** | **EXACT MATCH** |
| **Total Errors** | **21** | **21** | **PASS** | **EXACT MATCH** |

---

## 3. Locked Historical 117-Image Test Metrics Comparison

Evaluated on the 117-scan frozen historical test set (65 benign, 32 malignant, 20 normal):

| Metric | Research V14 Benchmark | Backend V14 Implementation | Reproduction Status | Match |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | **86.32%** | **86.32%** | **PASS** | **EXACT MATCH** |
| **Macro F1** | **85.44%** | **85.44%** | **PASS** | **EXACT MATCH** |
| **Malignant Recall** | **84.38%** | **84.38%** | **PASS** | **EXACT MATCH** |
| **Malignant Precision** | **87.10%** | **87.10%** | **PASS** | **EXACT MATCH** |
| **Malignant F1** | **85.71%** | **85.71%** | **PASS** | **EXACT MATCH** |
| **False Negatives (FN)** | **5** | **5** | **PASS** | **EXACT MATCH** |
| **False Positives (FP)** | **4** | **4** | **PASS** | **EXACT MATCH** |
| **Total Errors** | **16** | **16** | **PASS** | **EXACT MATCH** |

---

## 4. Production Safety & Model Integrity Verification

* **Production Model File**: `backend/models/breast_image_classifier.keras`
* **Production Version**: V5-B (`v5.0.0-b`)
* **SHA256 Hash**: `93b3c106cff51993b605220c98cfaf54c8fa404bb581656fa11337a26b32e03c` (**100% UNTOUCHED**)
* **Production Metadata**: `breast_image_classifier_metadata.json` and `metadata.json` remain **100% unchanged**.

---

## 5. API Behavior Verification

1. **Default Endpoint**: `POST /api/v1/image-predict` defaults to production model **V5-B** (`model="v5b"`).
2. **V14 Query Parameter**: `POST /api/v1/image-predict?model=v14` routes to V14 frozen ensemble.
3. **Dedicated V14 Endpoint**: `POST /api/v1/image-predict-v14` routes directly to V14 frozen ensemble.
4. **Startup Eager Loading**: V14 component models (`v11_mobilenetv2_b.keras` & `v13_efficientnet_b0.keras`) pre-load once in FastAPI lifespan boot.
5. **Schema Compatibility**: Response format is 100% compatible (`predicted_class`, `confidence`, `probabilities`, `status`, `explanation`). Zero frontend changes made.

---

## 6. Files Changed Since Integration Began

Only **3 backend files** were modified to support V14 ensemble inference:

1. [`backend/app/image_model.py`](file:///c:/Users/kiran/BCD/backend/app/image_model.py): Added `predict_image_v14` method and V14 model loading.
2. [`backend/app/api/prediction.py`](file:///c:/Users/kiran/BCD/backend/app/api/prediction.py): Added model query parameter and `/image-predict-v14` endpoint.
3. [`backend/app/main.py`](file:///c:/Users/kiran/BCD/backend/app/main.py): Added eager V14 loading in startup lifespan.

**Frontend Files Modified**: **0** (Zero frontend files were touched).

---

## 7. Audit Conclusion

**REPRODUCTION STATUS: PASS**

Backend V14 reproduces the frozen research V14 results with 100% accuracy and complete safety.

---

PRODUCTION MODEL CHANGED: NO
PRODUCTION METADATA CHANGED: NO
LOCKED 117-IMAGE TEST SET MODIFIED: NO
DATASET IMAGES MODIFIED: NO
DEPLOYMENT PERFORMED: NO
GIT COMMIT PERFORMED: NO
GIT PUSH PERFORMED: NO
