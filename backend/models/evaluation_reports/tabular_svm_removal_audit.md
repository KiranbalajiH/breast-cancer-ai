# Tabular SVM Removal Audit

**Audit Date**: September 15, 2026  
**Project**: Breast Cancer Detection (BCD) Platform  
**Repository Path**: `C:\Users\kiran\BCD`  
**Status**: COMPLETE (Tabular SVM Halted & Removed)

---

## Reason for Removal

The Tabular SVM component (Wisconsin Breast Cancer 30-feature tabular risk prediction pipeline) has been officially **HALTED** and is no longer part of the Breast Cancer Detection (BCD) platform. BCD is now exclusively focused on deep learning classification of breast ultrasound images (BUSI dataset).

---

## Removed From Active System

1. **Frontend (`frontend/`)**:
   - Tabular SVM mode/tab removed from `/predict` workspace (`frontend/app/predict/page.tsx`).
   - 30-feature numerical input form, sample loader buttons (`Load Benign`, `Load Malignant`), and tabular state removed.
   - Tabular API client functions (`predictCancer()`) and types (`BreastCancerFeatures`, `PredictionResponse`) removed from `frontend/lib/api.ts`.
   - References removed from `frontend/app/about/page.tsx`, `frontend/app/analytics/page.tsx`, `frontend/app/page.tsx`, and `README.md`.

2. **Backend API (`backend/app/`)**:
   - `POST /api/predict` tabular endpoint removed from `backend/app/api/prediction.py`.
   - Tabular Pydantic models (`BreastCancerFeatures`, `PredictionResponse`) removed from `backend/app/schemas/prediction.py`.
   - `ModelService` tabular singleton (`backend/app/services/model_service.py`) deleted.
   - Startup loading of tabular SVM model removed from `backend/app/main.py`.
   - Tabular configuration paths removed from `backend/app/core/config.py`.

3. **Active Model Artifacts**:
   - `backend/models/breast_cancer_model.joblib` (deleted).
   - `backend/models/metadata.json` (deleted tabular metadata).
   - `backend/models/model_comparison.json` (deleted).

4. **Tabular Datasets & Scripts**:
   - `backend/data/breast_cancer.csv` (deleted).
   - `backend/run_diagnostic_test.py` (deleted).
   - `backend/tests/test_tabular_api_audit.py` (deleted).

---

## Archived Historical Artifacts

To preserve historical research provenance, non-runtime tabular artifacts were moved to the archive directory structure (`backend/archive/`):

1. **Training Script**:
   - Archived: `backend/archive/training/train_tabular_svm.py`

2. **Active Evaluation Reports**:
   - Archived: `backend/archive/models/evaluation_reports/tabular_svm_dataset_audit.md`
   - Archived: `backend/archive/models/evaluation_reports/tabular_svm_training_report.md`
   - Archived: `backend/archive/models/evaluation_reports/tabular_svm_api_audit.md`

---

## Image System Verification

1. **Production Image Model (V5-B)**:
   - File: `backend/models/breast_image_classifier.keras`
   - SHA256 Hash: `93b3c106cff51993b605220c98cfaf54c8fa404bb581656fa11337a26b32e03c` (**INTACT & UNCHANGED**)
   - Status: Active Production Default. Loads cleanly via `ImageClassifier`.

2. **Research Image Model (V14 Ensemble)**:
   - Candidate Files: `backend/models/candidates/v11_mobilenetv2_b.keras` & `backend/models/candidates/v13_efficientnet_b0.keras` (**INTACT**)
   - Status: Active Research Champion (accessible via UI toggle and `model=v14` query param).

3. **Grad-CAM Visual Explanations**:
   - Generates JET colormap heatmaps for all image predictions (**INTACT & UNCHANGED**).

4. **Datasets & Test Sets**:
   - BUSI ultrasound image dataset (`dataset/BUSI/`): 776 clean scans (**INTACT**).
   - Historical 117-image test set & Grouped 107-image test set (**FROZEN & UNCHANGED**).

---

## Search Verification

A full repository-wide search was conducted for legacy tabular terms:

| Search Query | Occurrences in Active Code | Occurrences in Archive / Reports | Classification |
| :--- | :---: | :---: | :--- |
| `"Tabular SVM"` | 0 | Archived reports & removal audits | Allowed Historical Record |
| `"tabular"` | 0 | Historical docs / archive files | Allowed Historical Record |
| `"breast_cancer_model"` | 0 | Archived scripts (`train.py`, `train_tabular_svm.py`) | Allowed Historical Record |
| `"StandardScaler"` | 0 | `backend/archive/` training scripts & reports | Allowed Historical Record |
| `"SVC("` | 0 | `backend/archive/` scripts & reports | Allowed Historical Record |
| `"predictCancer"` | 0 | Archived API audit report | Allowed Historical Record |
| `"BreastCancerFeatures"`| 0 | Archived API audit report | Allowed Historical Record |
| `"breast_cancer.csv"` | 0 | Docstring comment in `compatibility.py` & archive | Allowed Historical Record |
| `"Wisconsin"` | 0 | RMD reference & removal audit reports | Allowed Historical Record |

**Summary**: Active frontend (`frontend/`) and active backend (`backend/app/`) contain **ZERO** functional references to Tabular SVM.

---

## Final Status

BCD is now an **IMAGE-ONLY** breast ultrasound detection platform.
