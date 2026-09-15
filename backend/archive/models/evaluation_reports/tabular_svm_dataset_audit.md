# Tabular SVM Dataset & Backend Audit Report

**Audit Date:** 2026-09-06  
**Dataset Analyzed:** Wisconsin Breast Cancer Diagnostic Dataset (`data.csv`)  
**Target Component:** Tabular SVM Cancer Risk Prediction Pipeline  
**Scope:** Dataset Verification, Backend Audit, Frontend Compatibility, & Pipeline Architecture (No code/model modifications made).

---

## 1. Dataset Summary

* **File Name:** `data.csv` (located at `C:\Users\kiran\Downloads\data.csv`)
* **Total Samples (Rows):** 569
* **Total Columns:** 33
* **Target Column:** `diagnosis`
* **Data Format:** CSV with header row
* **Data Integrity:** 0 duplicate rows, 0 duplicate ID entries

---

## 2. Class Distribution

| Target Code | Diagnosis Class | Sample Count | Percentage |
| :--- | :--- | :--- | :--- |
| **B** | Benign (Non-cancerous) | 357 | 62.74% |
| **M** | Malignant (Cancerous) | 212 | 37.26% |
| **Total** | | **569** | **100.00%** |

* **Class Ratio:** ~1.68:1 (Benign to Malignant) — Moderately balanced binary classification problem.

---

## 3. Feature List (30 Numerical ML Features)

The dataset contains 30 continuous numerical features computed from digitized images of fine needle aspirates (FNA) of breast masses, divided into three feature measurement groups (Mean, Standard Error, Worst/Largest):

### Group 1: Mean Measurements (10 features)
1. `radius_mean` (float64, range: 6.981 – 28.110)
2. `texture_mean` (float64, range: 9.710 – 39.280)
3. `perimeter_mean` (float64, range: 43.790 – 188.500)
4. `area_mean` (float64, range: 143.500 – 2501.000)
5. `smoothness_mean` (float64, range: 0.05263 – 0.16340)
6. `compactness_mean` (float64, range: 0.01938 – 0.34540)
7. `concavity_mean` (float64, range: 0.00000 – 0.42680)
8. `concave points_mean` (float64, range: 0.00000 – 0.20120)
9. `symmetry_mean` (float64, range: 0.10600 – 0.30400)
10. `fractal_dimension_mean` (float64, range: 0.04996 – 0.09744)

### Group 2: Standard Error (SE) Measurements (10 features)
11. `radius_se` (float64, range: 0.11150 – 2.87300)
12. `texture_se` (float64, range: 0.36020 – 4.88500)
13. `perimeter_se` (float64, range: 0.75700 – 21.98000)
14. `area_se` (float64, range: 6.80200 – 542.20000)
15. `smoothness_se` (float64, range: 0.00171 – 0.03113)
16. `compactness_se` (float64, range: 0.00225 – 0.13540)
17. `concavity_se` (float64, range: 0.00000 – 0.39600)
18. `concave points_se` (float64, range: 0.00000 – 0.05279)
19. `symmetry_se` (float64, range: 0.00788 – 0.07895)
20. `fractal_dimension_se` (float64, range: 0.00090 – 0.02984)

### Group 3: Worst / Largest Measurements (10 features)
21. `radius_worst` (float64, range: 7.930 – 36.040)
22. `texture_worst` (float64, range: 12.020 – 49.540)
23. `perimeter_worst` (float64, range: 50.410 – 251.200)
24. `area_worst` (float64, range: 185.200 – 4254.000)
25. `smoothness_worst` (float64, range: 0.07117 – 0.22260)
26. `compactness_worst` (float64, range: 0.02729 – 1.05800)
27. `concavity_worst` (float64, range: 0.00000 – 1.25200)
28. `concave points_worst` (float64, range: 0.00000 – 0.29100)
29. `symmetry_worst` (float64, range: 0.15650 – 0.66380)
30. `fractal_dimension_worst` (float64, range: 0.05504 – 0.20750)

---

## 4. Columns Excluded

1. `id` (int64): Unique patient identifier column. Must be excluded to prevent data leakage or arbitrary ID memorization by the model.
2. `Unnamed: 32` (float64): Unnamed trailing column resulting from an extra comma in the original CSV file. Contains 569 missing values (100% NaN). Must be dropped during preprocessing.

---

## 5. Missing / Duplicate Analysis

* **Missing Feature Values:** 0 across all 30 numerical ML features.
* **Missing Target Values:** 0 across the `diagnosis` column.
* **Missing Value Column:** `Unnamed: 32` contains 569 NaN values (to be dropped).
* **Duplicate Sample Rows:** 0 (0.00%).
* **Duplicate Patient IDs:** 0 (0.00%).
* **Data Cleanliness Verdict:** Excellent. Zero imputations required for ML features.

---

## 6. Existing Tabular Backend Flow

The existing backend handles tabular predictions via the following components:

* **API Endpoint:** `POST /api/predict` defined in [`backend/app/api/prediction.py`](file:///c:/Users/kiran/BCD/backend/app/api/prediction.py#L18)
* **Model Service Singleton:** `ModelService` in [`backend/app/services/model_service.py`](file:///c:/Users/kiran/BCD/backend/app/services/model_service.py)
* **Model Artifact:** [`backend/models/breast_cancer_model.joblib`](file:///c:/Users/kiran/BCD/backend/models/breast_cancer_model.joblib)
* **Model Metadata:** [`backend/models/metadata.json`](file:///c:/Users/kiran/BCD/backend/models/metadata.json)
* **Request Schema:** `BreastCancerFeatures` Pydantic model in [`backend/app/schemas/prediction.py`](file:///c:/Users/kiran/BCD/backend/app/schemas/prediction.py#L3-L37)
* **Response Schema:** `PredictionResponse` Pydantic model in [`backend/app/schemas/prediction.py`](file:///c:/Users/kiran/BCD/backend/app/schemas/prediction.py#L43-L49)

### Backend Execution Flow:
1. `POST /api/predict` receives JSON payload.
2. FastAPI validates request against `BreastCancerFeatures` Pydantic model (using field aliases).
3. `model_service.predict(features)` converts model payload to a 1-row `pandas.DataFrame` matching `feature_names` in `metadata.json`.
4. `self.model.predict(df)` and `self.model.predict_proba(df)` execute inference on the scikit-learn Pipeline (StandardScaler + SVM).
5. Returns `PredictionResponse` JSON containing predicted class ("Benign" / "Malignant"), code ("B" / "M"), confidence score, and probability distribution.

---

## 7. Existing Frontend / API Flow

* **User Interface:** Tabular SVM tab on `/predict` page ([`frontend/app/predict/page.tsx`](file:///c:/Users/kiran/BCD/frontend/app/predict/page.tsx))
* **API Client:** `predictCancer(features)` function in [`frontend/lib/api.ts`](file:///c:/Users/kiran/BCD/frontend/lib/api.ts#L64-L77)
* **Frontend Interface:** `BreastCancerFeatures` in [`frontend/lib/api.ts`](file:///c:/Users/kiran/BCD/frontend/lib/api.ts#L7-L38)

### End-to-End Execution Trace:
```
[Frontend UI: Tabular SVM Form]
        │
        ▼ (Form Submission)
[frontend/lib/api.ts: predictCancer()]
        │
        ▼ (HTTP POST /api/predict)
[FastAPI Router: backend/app/api/prediction.py]
        │
        ▼ (Pydantic Schema Validation)
[backend/app/schemas/prediction.py: BreastCancerFeatures]
        │
        ▼ (Model Service Singleton)
[backend/app/services/model_service.py: ModelService.predict()]
        │
        ▼ (Scikit-Learn Pipeline Inference)
[backend/models/breast_cancer_model.joblib]
        │
        ▼ (Formatted Prediction Response)
[Frontend Results Display: Confidence, Probabilities, Class Verdict]
```

---

## 8. Dataset ↔ Frontend Feature Compatibility

The 30 numerical ML features in `data.csv` correspond **1-to-1** with the 30 fields expected by the frontend interface `BreastCancerFeatures` and the backend Pydantic model. 

### Header Name Mapping:
| `data.csv` Feature Header | Backend Pydantic Alias | Frontend `BreastCancerFeatures` Key |
| :--- | :--- | :--- |
| `radius_mean` | `mean radius` | `mean radius` |
| `texture_mean` | `mean texture` | `mean texture` |
| `perimeter_mean` | `mean perimeter` | `mean perimeter` |
| `area_mean` | `mean area` | `mean area` |
| `smoothness_mean` | `mean smoothness` | `mean smoothness` |
| `compactness_mean` | `mean compactness` | `mean compactness` |
| `concavity_mean` | `mean concavity` | `mean concavity` |
| `concave points_mean` | `mean concave points` | `mean concave points` |
| `symmetry_mean` | `mean symmetry` | `mean symmetry` |
| `fractal_dimension_mean` | `mean fractal dimension` | `mean fractal dimension` |
| `radius_se` | `radius error` | `radius error` |
| `texture_se` | `texture error` | `texture error` |
| `perimeter_se` | `perimeter error` | `perimeter error` |
| `area_se` | `area error` | `area error` |
| `smoothness_se` | `smoothness error` | `smoothness error` |
| `compactness_se` | `compactness error` | `compactness error` |
| `concavity_se` | `concavity error` | `concavity error` |
| `concave points_se` | `concave points error` | `concave points error` |
| `symmetry_se` | `symmetry error` | `symmetry error` |
| `fractal_dimension_se` | `fractal dimension error` | `fractal dimension error` |
| `radius_worst` | `worst radius` | `worst radius` |
| `texture_worst` | `worst texture` | `worst texture` |
| `perimeter_worst` | `worst perimeter` | `worst perimeter` |
| `area_worst` | `worst area` | `worst area` |
| `smoothness_worst` | `worst smoothness` | `worst smoothness` |
| `compactness_worst` | `worst compactness` | `worst compactness` |
| `concavity_worst` | `worst concavity` | `worst concavity` |
| `concave points_worst` | `worst concave points` | `worst concave points` |
| `symmetry_worst` | `worst symmetry` | `worst symmetry` |
| `fractal_dimension_worst` | `worst fractal dimension` | `worst fractal dimension` |

* **Compatibility Verdict:** **100% Compatible**. No schema modifications required.

---

## 9. Recommended SVM Pipeline

When training is initiated in future steps, the following lightweight, robust pipeline is recommended:

1. **Preprocessing Pipeline:**
   - Drop `id` and `Unnamed: 32`.
   - Map `diagnosis`: `'B' -> 0` (Benign), `'M' -> 1` (Malignant).
   - Rename column headers to match backend `feature_names` standard (`mean radius`, `radius error`, etc.).
2. **Train/Test Split:**
   - Stratified 80/20 split (`train_test_split(..., test_size=0.20, stratify=y, random_state=42)`).
3. **Model Pipeline Construction:**
   - `scikit-learn` `Pipeline`:
     ```python
     Pipeline([
         ('scaler', StandardScaler()),
         ('svm', SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42))
     ])
     ```
4. **Evaluation Protocol:**
   - Report test accuracy, precision, recall (sensitivity), specificity, F1-score, ROC-AUC score, and confusion matrix (`tn`, `fp`, `fn`, `tp`).

---

## 10. Data Leakage Safeguards

To ensure absolute scientific rigor and avoid target/data leakage:
1. `id` and `Unnamed: 32` columns are explicitly dropped before feature extraction.
2. `StandardScaler` is encapsulated inside the `scikit-learn` `Pipeline` so scaling parameters ($\mu$, $\sigma$) are computed **strictly from `X_train`**.
3. `X_test` remains completely unseen during scaler fitting and parameter tuning.
4. Feature name validation is enforced before serialization into `breast_cancer_model.joblib`.

---

## 11. Exact Files That Would Need Modification (During Future Training Phase)

When training is authorized:
* [`backend/data/breast_cancer.csv`](file:///c:/Users/kiran/BCD/backend/data/breast_cancer.csv) (Copy `data.csv` into project data directory)
* `backend/training/train_tabular_svm.py` (Script to run pipeline training)
* [`backend/models/breast_cancer_model.joblib`](file:///c:/Users/kiran/BCD/backend/models/breast_cancer_model.joblib) (Saved trained model binary)
* [`backend/models/metadata.json`](file:///c:/Users/kiran/BCD/backend/models/metadata.json) (Updated training metrics and feature list)

* **Files to remain untouched:** Frontend files, image models (`v5b`, `v14`), FastAPI prediction endpoints.

---

## 12. Final Recommendation

**READY TO TRAIN**
