# Tabular SVM Backend / API Integration Audit Report

**Audit Date:** 2026-09-08  
**Target Endpoint:** `POST /api/predict`  
**Model Loaded:** [`backend/models/breast_cancer_model.joblib`](file:///c:/Users/kiran/BCD/backend/models/breast_cancer_model.joblib)  
**Metadata:** [`backend/models/metadata.json`](file:///c:/Users/kiran/BCD/backend/models/metadata.json)  
**Scope:** Verification of ModelService loading, Pydantic schema mapping, feature ordering, direct vs. API response alignment, and backend contract compliance.

---

## 1. Endpoint Tested

* **Route:** `POST /api/predict` (defined in [`backend/app/api/prediction.py`](file:///c:/Users/kiran/BCD/backend/app/api/prediction.py#L18))
* **Service:** [`ModelService`](file:///c:/Users/kiran/BCD/backend/app/services/model_service.py) singleton (`app.services.model_service.model_service`)
* **Request Schema:** [`BreastCancerFeatures`](file:///c:/Users/kiran/BCD/backend/app/schemas/prediction.py#L3-L37)
* **Response Schema:** [`PredictionResponse`](file:///c:/Users/kiran/BCD/backend/app/schemas/prediction.py#L43-L49)

---

## 2. Model Loading Result

* **Status:** Success (`status: "healthy"`)
* **Loaded Pipeline:** `StandardScaler` + `SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)`
* **Model Name in Metadata:** `"SVM (RBF) + StandardScaler Pipeline"`
* **Model Version:** `"1.2.0"`
* **Loading Behavior:** Cleanly loaded via `joblib.load()` inside `ModelService.load_model()` during FastAPI lifespan initialization.

---

## 3. Feature Mapping Verification

* **Schema Validation:** The 30 numerical input fields defined in `BreastCancerFeatures` map 1-to-1 with the `feature_names` array in `metadata.json`:
  1. `mean radius`
  2. `mean texture`
  3. `mean perimeter`
  4. `mean area`
  5. `mean smoothness`
  6. `mean compactness`
  7. `mean concavity`
  8. `mean concave points`
  9. `mean symmetry`
  10. `mean fractal dimension`
  11. `radius error`
  12. `texture error`
  13. `perimeter error`
  14. `area error`
  15. `smoothness error`
  16. `compactness error`
  17. `concavity error`
  18. `concave points error`
  19. `symmetry error`
  20. `fractal dimension error`
  21. `worst radius`
  22. `worst texture`
  23. `worst perimeter`
  24. `worst area`
  25. `worst smoothness`
  26. `worst compactness`
  27. `worst concavity`
  28. `worst concave points`
  29. `worst symmetry`
  30. `worst fractal dimension`

* **Ordering Verification:** `ModelService` constructs a 1-row `pandas.DataFrame` explicitly re-ordering keys to match `metadata.get("feature_names")`. Zero column order mismatches occurred.

---

## 4. Test Sample Evaluation Results

Real samples were extracted from `data.csv` (`C:\Users\kiran\Downloads\data.csv`) and tested against `POST /api/predict`:

### A. Known Benign Sample (Patient ID: 8510426)
* **Expected Class:** `B` (Benign)
* **HTTP Response Code:** `200 OK`
* **Returned Prediction:** `"Benign"`
* **Returned Prediction Code:** `"B"`
* **Confidence Score:** `0.9924` (99.24%)
* **API Probabilities:** `{"benign": 0.992435, "malignant": 0.007565}`
* **Verdict:** **PASS**

### B. Known Malignant Sample (Patient ID: 842302)
* **Expected Class:** `M` (Malignant)
* **HTTP Response Code:** `200 OK`
* **Returned Prediction:** `"Malignant"`
* **Returned Prediction Code:** `"M"`
* **Confidence Score:** `0.9811` (98.11%)
* **API Probabilities:** `{"benign": 0.018864, "malignant": 0.981136}`
* **Verdict:** **PASS**

---

## 5. Direct Model vs API Comparison

| Test Case | Direct Pipeline Probabilities | FastAPI `/api/predict` Response | Agreement |
| :--- | :--- | :--- | :--- |
| **Benign (ID: 8510426)** | `[B: 0.992435, M: 0.007565]` | `{"benign": 0.992435, "malignant": 0.007565}` | **Exact (100%)** |
| **Malignant (ID: 842302)** | `[B: 0.018864, M: 0.981136]` | `{"benign": 0.018864, "malignant": 0.981136}` | **Exact (100%)** |

---

## 6. Response Schema Verification

The API response payload returned by `POST /api/predict` complies 100% with `PredictionResponse`:
```json
{
  "prediction": "Benign",
  "prediction_code": "B",
  "confidence": 0.9924352572188573,
  "probabilities": {
    "benign": 0.9924352572188573,
    "malignant": 0.007564742781142653
  },
  "model": {
    "name": "SVM (RBF) + StandardScaler Pipeline",
    "version": "1.2.0"
  }
}
```

---

## 7. Files Modified

* **Backend API Code Modified:** **0 files** (Existing `ModelService` and FastAPI router handle the new saved pipeline without any modification).
* **Test Artifact Added:** [`backend/tests/test_tabular_api_audit.py`](file:///c:/Users/kiran/BCD/backend/tests/test_tabular_api_audit.py) (Integration test suite).

---

## 8. Final Audit Verdict

**PASS**
