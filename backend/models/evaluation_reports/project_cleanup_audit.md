# BCD Project Cleanup & Consolidation Audit Report

**Date**: September 02, 2026  
**Project**: Breast Cancer AI (BCD)  
**Status**: PHASE 1 — AUDIT & PROPOSAL ONLY (NO DELETIONS PERFORMED)

---

## 1. Executive Summary

This audit assesses all files across the `BCD` repository to prepare for a minimal, production-ready consolidation.

### Final Model & System State:
- **Active Production Model**: `backend/models/breast_image_classifier.keras` (**V5-B**, SHA256: `93b3c106cff51993b605220c98cfaf54c8fa404bb581656fa11337a26b32e03c`)
- **V14 Research Champion**: Frozen probability ensemble of **V11-B** (50%) + **V13-B0** (50%) at threshold `0.58`.
- **Ensemble Component Models**: `backend/models/candidates/v11_mobilenetv2_b.keras` and `backend/models/candidates/v13_efficientnet_b0.keras`.

---

## 2. Classification of Files and Space Impact

| Category | Description | File Count | Size (MB) | Action |
| :--- | :--- | :---: | :---: | :--- |
| **A. KEEP (Runtime App)** | API backend, Next.js frontend, production V5-B model & metadata, config files | 117 | 24.13 MB | **KEEP IN PLACE** |
| **B. KEEP (V14 Ensemble)** | Candidate models V11-B and V13-B0, train_v14.py, V14 evaluation reports | 6 | 44.10 MB | **KEEP IN PLACE** |
| **C. KEEP (Reproducibility/Data)** | BUSI dataset (776 scans), train_v11.py, train_v13.py, V11 & V13 reports | 1,577 | 253.24 MB | **KEEP IN PLACE** |
| **D. ARCHIVE (Historical)** | Historical V1–V10/V12 candidate models (.keras), old evaluation reports, old training scripts | 100 | 567.49 MB | **MOVE TO ARCHIVE** |
| **E. SAFE TO DELETE (Temp)** | Old backup models (v1_backup, v2), duplicate comparison reports, __pycache__ | 33 | 38.00 MB | **SAFE TO DELETE** |
| **F. UNKNOWN (Requires Approval)** | `backend/inspect_dataset.py` | 1 | 0.001 MB | **REQUIRES APPROVAL** |
| **TOTAL RECLAIMABLE SPACE** | **ARCHIVE (Category D) + DELETE (Category E)** | **133 files** | **605.49 MB** | **~0.59 GB Saved** |

---

## 3. Detailed Categorization

### Category A: KEEP — Required by Application / Runtime (117 Files, 24.13 MB)
- `backend/models/breast_image_classifier.keras` (Production V5-B model, 18.01 MB)
- `backend/models/breast_image_classifier_metadata.json` (Production V5-B metadata)
- `backend/models/breast_cancer_model.joblib` (Production tabular SVM risk model)
- `backend/models/breast_ultrasound_model.joblib` (Production tabular ultrasound model)
- `backend/models/metadata.json` & `ultrasound_metadata.json` (Tabular metadata)
- `backend/app/` (All 13 FastAPI backend application source files)
- `frontend/` (All Next.js React frontend source files, components, styles, configs)
- `README.md`, `requirements.txt`, `pyrefly.toml`, `package.json`

### Category B: KEEP — Required for V14 Ensemble (6 Files, 44.10 MB)
- `backend/models/candidates/v11_mobilenetv2_b.keras` (18.01 MB)
- `backend/models/candidates/v13_efficientnet_b0.keras` (18.01 MB)
- `backend/training/train_v14.py` (22.0 KB)
- `backend/models/evaluation_reports/v14_final_evaluation_report.md` (4.0 KB)
- `backend/models/evaluation_reports/v14_evaluation_results.json` (5.9 KB)
- `backend/models/evaluation_reports/v14_validation_summary.json` (259.5 KB)

### Category C: KEEP — Required for Evaluation / Reproducibility (1,577 Files, 253.24 MB)
- `dataset/BUSI/` (776 clean images + masks, 1,573 files, 252.79 MB)
- `backend/training/train_v11.py` & `backend/training/train_v13.py`
- `backend/models/evaluation_reports/v11_final_evaluation_report.md` & `v13_final_evaluation_report.md`
- `backend/models/evaluation_reports/v11_evaluation_results.json` & `v13_evaluation_results.json`
- `backend/models/evaluation_reports/v11_validation_summary.json` & `v13_validation_summary.json`

### Category D: ARCHIVE — Historical Experiment Artifacts (100 Files, 567.49 MB)
- 32 Obsolete Candidate Models in `backend/models/candidates/`:
  - `mobilenetv2_v3_mildweights.keras`, `mobilenetv2_v3_noweights.keras` (57.8 MB)
  - `v4_efficientnet_d.keras`, `v4_mobilenetv2_a/b/c/headonly.keras` (85.1 MB)
  - `v5_mobilenetv2_a/b/c/d.keras` (68.7 MB)
  - `v6_mobilenetv2_a/b/c/d.keras` (68.7 MB)
  - `v7_mobilenetv2_a/b/c/d.keras` (68.9 MB)
  - `v9_mobilenetv2_auxloss/baseline/cropaug/dualview.keras` (68.9 MB)
  - `v10_mobilenetv2_a/b/c/d/e.keras` (85.9 MB)
  - `v11_mobilenetv2_a/c/d/e.keras` (68.7 MB)
- Historical Candidate Metadata JSON files (`v4_*_metadata.json`, `v11_*_metadata.json`, etc.)
- Historical Training Scripts in `backend/training/`: `train_v2.py` through `train_v10.py`, `train_v12.py`, `evaluate_v3.py`, `evaluate_v4.py`, `calibrate_v7_d.py`, `analyze_misclassifications.py`, `audit_busi.py`, `audit_v5b.py`, etc.
- Historical Evaluation Reports in `backend/models/evaluation_reports/` for V1, V4, V5, V6, V7, V8, V9, V10, V12.

### Category E: SAFE TO DELETE — Temporary & Unnecessary Files (33 Files, 38.00 MB)
- `backend/models/breast_image_classifier_v1_backup.keras` (9.66 MB)
- `backend/models/breast_image_classifier_v2.keras` (27.91 MB)
- `backend/models/breast_image_classifier_v2_metadata.json` (8.66 KB)
- `backend/models/candidates/comparison_report.json`, `comparison_report.md`, `comparison_report_20260824_090046.json`, `comparison_report_20260824_090046.md`
- `backend/training/__pycache__/` (Python bytecode cache)
- `backend/app/__pycache__/` (Python bytecode cache)

### Category F: UNKNOWN — Requires Manual Approval (1 File, 0.001 MB)
- `backend/inspect_dataset.py` (Ad-hoc dataset inspection script)

---

## 4. Proposed Final Concise Project Tree

```
BCD/
├── backend/
│   ├── app/                                   # FastAPI Application API
│   │   ├── main.py
│   │   ├── image_model.py
│   │   ├── model.py
│   │   ├── api/
│   │   └── core/
│   ├── models/
│   │   ├── breast_image_classifier.keras        # V5-B Active Production Model
│   │   ├── breast_image_classifier_metadata.json # Production Metadata
│   │   ├── breast_cancer_model.joblib            # Tabular Model
│   │   ├── breast_ultrasound_model.joblib        # Tabular Model
│   │   ├── metadata.json                         # Tabular Metadata
│   │   ├── ultrasound_metadata.json              # Tabular Metadata
│   │   ├── candidates/
│   │   │   ├── v11_mobilenetv2_b.keras           # V14 Ensemble Component 1
│   │   │   └── v13_efficientnet_b0.keras         # V14 Ensemble Component 2
│   │   └── evaluation_reports/
│   │       ├── v11_final_evaluation_report.md
│   │       ├── v13_final_evaluation_report.md
│   │       ├── v14_final_evaluation_report.md
│   │       ├── v14_evaluation_results.json
│   │       └── v14_validation_summary.json
│   └── training/
│       ├── train_v11.py                       # V11-B Reproducibility
│       ├── train_v13.py                       # V13-B0 Reproducibility
│       └── train_v14.py                       # V14 Ensemble Script
├── dataset/
│   └── BUSI/                                  # 776 Clean Trainable Ultrasound Scans
├── frontend/                                  # Next.js React Frontend UI
├── README.md                                  # Project Documentation
├── requirements.txt                           # Python Dependencies
├── pyrefly.toml                               # Environment Configuration
└── package.json                               # Node Dependencies
```

---

PRODUCTION MODEL CHANGED: NO
PRODUCTION METADATA CHANGED: NO
LOCKED 117-IMAGE TEST SET MODIFIED: NO
DATASET IMAGES MODIFIED: NO
DEPLOYMENT PERFORMED: NO
GIT COMMIT PERFORMED: NO
GIT PUSH PERFORMED: NO
