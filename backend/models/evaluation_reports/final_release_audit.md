# BCD Final Release Audit

**Audit Date**: September 15, 2026  
**Project**: Breast Cancer Detection (BCD) Platform  
**Repository Path**: `C:\Users\kiran\BCD`  
**Audit Scope**: Final Release Verification, Model Hierarchy & Governance Audit (Image-Only System)

---

## Production Models

### 1. Image Classification Model (Production)
- **Model Version**: V5-B (`v5.0.0-b`)
- **Architecture**: MobileNetV2 with Aspect-Ratio Letterboxed Preprocessing ($224 \times 224 \times 3$)
- **File Location**: `backend/models/breast_image_classifier.keras`
- **Metadata Location**: `backend/models/breast_image_classifier_metadata.json`
- **SHA256 Hash**: `93b3c106cff51993b605220c98cfaf54c8fa404bb581656fa11337a26b32e03c`
- **Performance Summary**:
  - Historical 117-Image Test Set: **80.34% Accuracy**, **87.50% Malignant Recall** (4 False Negatives)
  - Grouped 107-Image Test Set: **70.09% Accuracy**, **83.87% Malignant Recall**
- **Status**: **PRODUCTION DEFAULT**. Loads cleanly as the default image model in backend `ImageClassifier` and frontend UI.

---

## Research Models

### V14 Ensemble (Research Champion)
- **Composition**: Frozen 50/50 probability ensemble combining:
  - **V11-B** (`backend/models/candidates/v11_mobilenetv2_b.keras`): MobileNetV2 with 50% Augmentation (96.88% Historical Recall)
  - **V13-B0** (`backend/models/candidates/v13_efficientnet_b0.keras`): EfficientNetB0 Transfer Learning (89.29% Historical Precision)
- **Decision Threshold**: $t^* = 0.58$
- **Performance Summary**:
  - Historical 117-Image Test Set: **86.32% Accuracy**, **84.38% Malignant Recall**, **87.10% Malignant Precision**
  - Grouped 107-Image Test Set: **80.37% Accuracy**, **77.42% Malignant Recall**, **82.76% Malignant Precision**
- **Status**: **RESEARCH CHAMPION (OPT-IN)**. Accessible in backend API (`model=v14`) and frontend UI toggle. NOT promoted to production default.

---

## Halted & Archived Components

### Tabular SVM (Halted & Removed)
- **Status**: **HALTED & REMOVED**. The Tabular SVM implementation (Wisconsin Breast Cancer tabular prediction) has been completely removed from the active project.
- **Historical Reports**: Archived to `backend/archive/models/evaluation_reports/` (`tabular_svm_dataset_audit.md`, `tabular_svm_training_report.md`, `tabular_svm_api_audit.md`).
- **Training Script**: Archived to `backend/archive/training/train_tabular_svm.py`.

---

## Rejected Experiments

### Final 2-Epoch MobileNetV2 Candidate
- **Architecture**: MobileNetV2 trained for 2 epochs on the clean BUSI dataset
- **Candidate Path (Archived)**: `backend/archive/models/candidates/final_2epoch_mobilenetv2.keras`
- **Metadata Path (Archived)**: `backend/archive/models/candidates/final_2epoch_mobilenetv2_metadata.json`
- **SHA256 Hash**: `3b31161e4311d657e07bfed8139eec7e58017afd87bc9690CAF781DB86E03C10`
- **Evaluation Report**: `backend/models/evaluation_reports/final_2epoch_training_report.md` (preserved for experiment provenance)
- **Performance Summary**:
  - Historical 117-Image Test Set: **52.14% Accuracy**, **45.16% Malignant Recall**, **63.64% Malignant Precision**, **52.83% Malignant F1** (17 FN, 8 FP, 56 Total Errors)
  - Grouped 107-Image Test Set: **53.39% Accuracy**, **41.94% Malignant Recall**, **50.00% Malignant Precision**, **45.61% Malignant F1** (18 FN, 13 FP, 55 Total Errors)
- **Status**: **REJECTED**. Failed to outperform production V5-B (80.34% Hist / 70.09% Grouped) or research champion V14 (86.32% Hist / 80.37% Grouped). Moved to archive directory. Must NOT be promoted.

---

## Dataset / Frozen Test Integrity

1. **Historical Test Set**: 117 images (`dataset/BUSI/`, seed 42) — **INTACT & FROZEN**. No training leakage detected.
2. **Grouped Test Set**: 107 images across 102 unique patient/lesion clusters (pHash MAD < 12.0) — **INTACT & FROZEN**. Zero cluster overlap with training/validation sets.

---

## Backend Verification

- **Module Imports & Loading**: Verified clean loading of `ImageClassifier`.
- **Image API Default**: `ImageClassifier.predict_image(..., model_version='v5b')` executes inference against V5-B (`breast_image_classifier.keras`) with full Grad-CAM generation.
- **V14 Opt-in Endpoint**: `ImageClassifier.predict_image(..., model_version='v14')` correctly routes to V11-B + V13-B0 probability averaging.

---

## Frontend Verification

- **Next.js TypeScript Compiler (`tsc`)**: Executed `npx tsc --noEmit` across `frontend/` codebase. Result: **0 errors**.
- **Model Selector UI**: Verified default state is V5-B Production model, with V14 Ensemble accessible via explicit user opt-in toggle.
- **Grad-CAM Overlay Viewer**: Component verified structurally intact.

---

## Reports

Core audit and evaluation reports in `backend/models/evaluation_reports/`:

1. `project_cleanup_audit.md`
2. `v5_final_evaluation_report.md`
3. `v14_final_evaluation_report.md`
4. `v14_integration_audit.md`
5. `v14_reproduction_audit.md`
6. `v14_runtime_smoke_test.md`
7. `final_end_to_end_ui_test.md`
8. `final_2epoch_training_report.md`
9. `BCD_PROJECT_MASTER_SUMMARY.md`
10. `tabular_svm_removal_audit.md`

---

## Model Hashes

| Model Role | Model File Path | SHA256 Hash | Status |
| :--- | :--- | :--- | :--- |
| **Production Image Model (V5-B)** | `backend/models/breast_image_classifier.keras` | `93b3c106cff51993b605220c98cfaf54c8fa404bb581656fa11337a26b32e03c` | Active Production |
| **Research Model Component (V11-B)** | `backend/models/candidates/v11_mobilenetv2_b.keras` | `97d6fb567dfaa5bcf56be0edbaebed5394be4dc2bda165b4c1aa55ebc08dd6ca` | Research Opt-in |
| **Research Model Component (V13-B0)**| `backend/models/candidates/v13_efficientnet_b0.keras` | `b9f3458efbf6443c5bdfbbceee6ad0cfb8d5aee5ad5733fcd98ca0f9b691d1e4` | Research Opt-in |
| **Rejected Candidate (2-Epoch)** | `backend/archive/models/candidates/final_2epoch_mobilenetv2.keras` | `3b31161e4311d657e07bfed8139eec7e58017afd87bc9690caf781db86e03c10` | Rejected & Archived |

---

## Git State

- **Active Branch**: `main`
- **Head Commit**: `58641c7 Integrate V14 ensemble image prediction`
- **Git Actions**: **NO COMMITS OR PUSHES PERFORMED** during audit per instructions.

---

## Final Release Decision

V5-B remains the production image model.  
V14 remains the research model.  
The 2-epoch candidate is rejected.  
Tabular SVM branch of development was halted and archived.  
BCD is now an **IMAGE-ONLY** breast ultrasound detection platform.  
No model was automatically promoted.  
No production image model was overwritten.
itten.  
