# BCD Project Master Summary

**Report Date**: September 15, 2026  
**Project**: Breast Cancer Detection (BCD) Platform  
**Repository Path**: `C:\Users\kiran\BCD`  
**Audit Status**: Complete Repository Verification (Audit Only — No Code/Model/Git Changes Made)

---

## 1. Project Overview

- **Project Type**: **Image-Only Breast Cancer Detection** using breast ultrasound images.
- **Project Purpose**: To provide a production-grade, image-based clinical decision support platform for breast cancer detection. The system combines deep learning visual analysis of breast ultrasound scans with class-specific Grad-CAM heatmaps.
- **Main Capabilities**:
  1. **Ultrasound Image AI**: Classifies ultrasound scans into three diagnostic categories (`benign`, `malignant`, `normal`) using aspect-ratio letterboxed deep learning and provides class-specific Grad-CAM visual heatmaps.
  2. **Interactive Web Studio**: Next.js single-page application with an ultrasound workspace, model selector toggle (V5-B Production vs V14 Research Ensemble), image dropzone, and Grad-CAM overlay viewer.
- **Current Architecture**: Decoupled micro-service design featuring a FastAPI backend serving TensorFlow/Keras models, interacting via REST APIs with a Next.js (TypeScript/Tailwind CSS) frontend.
- **Image AI Production & Research Models**:
  - **Production Image Model**: V5-B (`v5.0.0-b`) MobileNetV2 Transfer Learning with Aspect-Ratio Letterboxing.
  - **Research Image Model**: V14 Ensemble (Dynamic composition of V11-B MobileNetV2 + V13-B0 EfficientNetB0).

---

## 2. Dataset

### Image Dataset
- **Dataset Name / Location**: BUSI (Breast Ultrasound Images Dataset) located at `dataset/BUSI/`.
- **Classes**: 3 classes — `benign`, `malignant`, `normal`.
- **Raw Class Counts**: 778 total raw images (437 benign, 210 malignant, 133 normal).
- **Cleaned Dataset Count**: 776 clean trainable scans across 680 unique perceptual patient/lesion clusters.
- **Excluded / Duplicate Samples**: 2 scans excluded due to exact pixel MD5 duplicate label conflict (`benign (433).png` & `malignant (145).png`). 239 blurry scans were intentionally retained to eliminate selection bias.
- **Train / Validation / Test Methodology**:
  1. *Locked Historical Test Set*: 117 images (65 Benign, 32 Malignant, 20 Normal; random seed 42) preserved for historical comparability across V1–V14.
  2. *Grouped Generalization Split*: Structural perceptual hash clustering (pHash, MAD < 12.0) across 680 unique patient/lesion clusters. 546 train (476 clusters), 123 val (102 clusters), 107 test (102 clusters: 59 Benign, 31 Malignant, 17 Normal). Zero cluster overlap across splits.
- **Frozen Test Sets**: Locked Historical 117-Image Test Set and Grouped 107-Image Test Set.
- **Random Seeds**: Fixed seed `42` across all data splitters, feature extractors, and model initializations.
- **Exact Paths / Scripts Used**: Data located at `dataset/BUSI/`; training scripts located at `backend/training/train_v11.py`, `train_v13.py`, `train_v14.py`.

---

## 3. Image AI Models

Chronological summary of all major image model iterations across the project lifecycle:

| Model / Version | Architecture | Training Approach | Key Preprocessing | Historical Test Acc | Grouped Test Acc | Malignant Precision | Malignant Recall | Malignant F1 | FN / FP (Hist) | Status | Exact Model Path |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| **V1 Baseline** | MobileNetV2 | Standard transfer learning | Square Resize ($224\times224$) | 84.62% | N/A | 84.62% | 68.75% | 75.86% | 10 / 4 | ARCHIVED | `backend/archive/models/candidates/` |
| **V2 DenseNet** | DenseNet121 | Unweighted transfer learning | Square Resize ($224\times224$) | 53.85% | N/A | 37.66% | 90.63% | 53.21% | 3 / 48 | ARCHIVED | `backend/archive/models/candidates/` |
| **V3 No Weights** | MobileNetV2 | Dense head added | Square Resize ($224\times224$) | 70.94% | N/A | 62.50% | 78.13% | 69.44% | 7 / 15 | ARCHIVED | `backend/archive/models/candidates/` |
| **V3 Mild W** | MobileNetV2 | 1.25x class weights | Square Resize ($224\times224$) | 69.23% | N/A | 58.97% | 71.88% | 64.79% | 9 / 16 | ARCHIVED | `backend/archive/models/candidates/` |
| **V4 Series** | MobileNetV2 / EffNetD | Hyperparameter tuning | Square Resize ($224\times224$) | 71.79%-76.92% | N/A | 65%-72% | 71%-81% | 68%-76% | 6-9 / 8-14 | ARCHIVED | `backend/archive/models/candidates/v4_*.keras` |
| **V5-A** | MobileNetV2 | Control unweighted | Square Resize ($224\times224$) | 79.49% | 69.16% | 75.00% | 84.38% | 79.41% | 5 / 9 | ARCHIVED | `backend/archive/models/candidates/v5_mobilenetv2_a.keras` |
| **V5-B (Active)**| **MobileNetV2** | **Aspect-Ratio Letterboxing** | **Aspect Letterbox ($224\times224$)** | **80.34%** | **70.09%** | **77.78%** | **87.50%** | **82.35%** | **4 / 8** | **PRODUCTION** | `backend/models/breast_image_classifier.keras` |
| **V5-C** | MobileNetV2 | Aspect-Ratio Letterbox | Letterbox + Grayscale | 71.79% | 62.62% | 65.00% | 81.25% | 72.22% | 6 / 14 | ARCHIVED | `backend/archive/models/candidates/v5_mobilenetv2_c.keras` |
| **V5-D** | MobileNetV2 | Letterbox + Gray + Weights | Letterbox + Grayscale | 73.50% | 63.55% | 78.57% | 68.75% | 73.33% | 10 / 6 | ARCHIVED | `backend/archive/models/candidates/v5_mobilenetv2_d.keras` |
| **V6–V10 Series**| MobileNetV2 / AuxLoss | Augmentation & loss experiments | Letterbox / Crop | 68.38%-77.78% | 60.75%-67.29% | 59%-72% | 71%-84% | 67%-76% | 5-9 / 11-18 | ARCHIVED | `backend/archive/models/candidates/` |
| **V11-B** | MobileNetV2 | 50% Augmentation | Aspect Letterbox ($224\times224$) | 78.63% | 69.16% | 70.45% | 96.88% | 81.58% | 1 / 13 | RESEARCH | `backend/models/candidates/v11_mobilenetv2_b.keras` |
| **V13-B0** | EfficientNetB0 | Deep transfer learning | Aspect Letterbox ($224\times224$) | 86.32% | 80.37% | 89.29% | 78.12% | 83.33% | 7 / 3 | RESEARCH | `backend/models/candidates/v13_efficientnet_b0.keras` |
| **V14 Ensemble** | Frozen 50/50 Ensemble | Probability averaging ($t^*=0.58$) | Aspect Letterbox ($224\times224$) | 86.32% | 80.37% | 87.10% | 84.38% | 85.71% | 5 / 4 | RESEARCH CHAMPION | Dynamic ensemble code in `image_model.py` |

---

## 4. Current Image Production Model

- **Exact Model Name / Version**: V5-B (`v5.0.0-b`).
- **Architecture**: MobileNetV2 Transfer Learning with Aspect-Ratio Letterboxed Preprocessing.
- **Input Size**: $224 \times 224 \times 3$.
- **Classes**: `['benign', 'malignant', 'normal']`.
- **Model Path**: `backend/models/breast_image_classifier.keras`.
- **Metadata Path**: `backend/models/breast_image_classifier_metadata.json`.
- **File Size**: 18,015,816 bytes (~18.01 MB).
- **SHA256 Hash**: `93B3C106CFF51993B605220C98CFAF54C8FA404BB581656FA11337A26B32E03C`.
- **Current Prediction Logic**: Decodes image bytes $\rightarrow$ Non-blocking image quality assessment $\rightarrow$ Aspect-ratio letterbox padding to $224\times224\times3$ $\rightarrow$ MobileNetV2 preprocessing $\rightarrow$ Softmax probability inference $\rightarrow$ Grad-CAM heatmap generation $\rightarrow$ Status threshold formatting.
- **Why It Is Production**: V5-B achieved the optimal clinical safety balance on the locked historical test set (**87.50% Malignant Recall**, reducing false negatives to **only 4 FN** vs 10 FN in V1) while maintaining **70.09% accuracy** on unseen patient lesion clusters without slice leakage.
- **Replacement Confirmation**: **CONFIRMED**. V5-B has NOT been automatically replaced by research candidate models. `breast_image_classifier.keras` remains active as the production default.## 5. Research / Candidate Image Models

- **V14 Ensemble**: Frozen 50/50 probability ensemble combining **V11-B** (MobileNetV2, 96.88% recall) + **V13-B0** (EfficientNetB0, 89.29% precision) operating at decision threshold $t^*=0.58$. Delivered **86.32% accuracy** and **84.38% malignant recall** on historical test set. Status: **RESEARCH CHAMPION** (accessible via UI toggle or `model=v14` query param).
- **V11-B**: MobileNetV2 candidate model trained with 50% data augmentation. Path: `backend/models/candidates/v11_mobilenetv2_b.keras` (18,015,776 bytes). Metrics: 96.88% Hist Recall, 90.32% Grouped Recall. Status: **RESEARCH / CANDIDATE**.
- **V13-B0**: EfficientNetB0 candidate model. Path: `backend/models/candidates/v13_efficientnet_b0.keras` (27,931,404 bytes). Metrics: 86.32% Hist Accuracy, 80.37% Grouped Accuracy, 89.29% Hist Precision. Status: **RESEARCH / CANDIDATE**.
- **Final 2-Epoch Candidate**: MobileNetV2 candidate trained for 2 epochs on the clean BUSI dataset (`backend/archive/models/candidates/final_2epoch_mobilenetv2.keras`, SHA256: `3B31161E4311D657E07BFED8139EEC7E58017AFD87BC9690CAF781DB86E03C10`). Metrics: 52.14% Hist Accuracy (45.16% Malignant Recall, 17 FN), 53.39% Grouped Accuracy (41.94% Malignant Recall, 18 FN). Status: **REJECTED** (did not outperform V5-B or V14, archived and NOT promoted).
- **Clear Status Classification**:
  - **PRODUCTION**: `backend/models/breast_image_classifier.keras` (V5-B)
  - **RESEARCH CHAMPION**: V14 Ensemble (Dynamic composition of V11-B + V13-B0)
  - **REJECTED**: Final 2-Epoch MobileNetV2 (`backend/archive/models/candidates/final_2epoch_mobilenetv2.keras`)
  - **RESEARCH / CANDIDATE**: `backend/models/candidates/v11_mobilenetv2_b.keras` (V11-B) & `backend/models/candidates/v13_efficientnet_b0.keras` (V13-B0)
  - **ARCHIVED**: All historical models in `backend/archive/models/candidates/`

---

## 6. Image Inference Pipeline

```
[Uploaded Image File]
        │
        ▼ (cv2 decode & validation: min 50x50px)
[Image Quality & Modality Safeguards]
  ├── Blur check (Laplacian variance < threshold)
  ├── Brightness & Contrast check (mean intensity, std intensity)
  └── Atypical color & Aspect Ratio check
        │
        ▼ (Aspect-Ratio Letterboxing)
[Letterbox Preprocessing: pad to 224x224x3]
        │
        ▼ (Model Specific Normalization)
  ├── V5-B: MobileNetV2 preprocess_input
  └── V14: Parallel MobileNetV2 & EfficientNet preprocess_input
        │
        ▼ (Inference Execution)
  ├── V5-B: Single pass on breast_image_classifier.keras
  └── V14: 0.5 * P(V11-B) + 0.5 * P(V13-B0), threshold t* = 0.58
        │
        ▼ (Softmax Probabilities)
[Probabilities: benign, malignant, normal]
        │
        ▼ (Grad-CAM Visual Explanation)
[Gradient Computation on Last Conv Layer (Conv_1)]
  └── Overlay JET colormap onto original image at 35% opacity
        │
        ▼ (FastAPI Serialization)
[HTTP 200 JSON Response]
        │
        ▼ (Next.js React Workspace UI)
[Render Verdict Badge, Confidence Bars, Quality Notices & Grad-CAM Heatmap]
```

### Exact Threshold Logic:
- **High Confidence**: $\text{Confidence} \ge 0.85$ (`status: "high_confidence"`)
- **Moderate Confidence**: $0.70 \le \text{Confidence} < 0.85$ (`status: "moderate_confidence"`)
- **Low Confidence / Review Required**: $\text{Confidence} < 0.60$ OR $(\text{Prob}_1 - \text{Prob}_2) < 0.15$ (`status: "review_required"`)
- **Unsupported / Suspicious Input**: Color channel difference $> 15.0$ OR Aspect Ratio outside $[0.5, 2.0]$ (`status: "unsupported_or_review_required"`)

---

## 7. Backend

- **Framework**: FastAPI (`uvicorn app.main:app`).
- **Main Application Entry Point**: [`backend/app/main.py`](file:///c:/Users/kiran/BCD/backend/app/main.py).
- **Important Routes**:
  - `POST /api/image-predict`: Image classification (`model=v5b` default, `model=v14` for ensemble).
  - `POST /api/image-predict-v14`: Dedicated V14 ensemble inference endpoint.
  - `GET /api/health` & `GET /api/model-status`: System diagnostics and image classifier health status.
  - `POST /api/image-analysis/extract` & `POST /api/image-analysis/predict`: Cell microscopy feature extraction.
- **V5-B / V14 Model Selection Behavior**: `ImageClassifier` in `app/image_model.py` defaults to loading V5-B production model; query parameter `model=v14` triggers parallel inference across V11-B and V13-B0 candidates.
- **Model Loading Behavior**: Lifespan context manager (`@asynccontextmanager`) in `main.py` loads `ImageClassifier` at startup.
- **Error / Fallback Behavior**: Catches missing files, empty image uploads, invalid dimensions (<50x50), returns structured HTTP 400/500 JSON details.
- **Important Backend Files**:
  - `backend/app/main.py`
  - `backend/app/image_model.py`
  - `backend/app/api/prediction.py`
  - `backend/app/api/image_analysis.py`
  - `backend/app/core/config.py`

---

## 8. Frontend

- **Framework**: Next.js 14+ App Router (React, TypeScript, Tailwind CSS, Framer Motion, Lucide icons).
- **Main Pages**:
  - `/` ([`frontend/app/page.tsx`](file:///c:/Users/kiran/BCD/frontend/app/page.tsx)): Landing page & feature overview.
  - `/predict` ([`frontend/app/predict/page.tsx`](file:///c:/Users/kiran/BCD/frontend/app/predict/page.tsx)): Image Classification Workspace.
  - `/image-analysis` ([`frontend/app/image-analysis/page.tsx`](file:///c:/Users/kiran/BCD/frontend/app/image-analysis/page.tsx)): Cell microscopy feature extraction workspace.
  - `/analytics` & `/about`: System performance breakdown and methodology documentation.
- **Image Prediction UI**: Drag-and-drop dropzone, image preview, model selector toggle (`V5-B (Production)` vs `V14 Ensemble`), prediction verdict card, probability progress bars, and Grad-CAM overlay viewer.
- **Model Selector**: Toggle buttons for V5-B and V14 on `/predict` page.
- **V5-B Default Behavior**: Default active model on page load.
- **V14 Behavior**: Clicking `V14 Ensemble` clears previous state to `idle` and routes request to `/api/image-predict?model=v14`.
- **Grad-CAM**: Renders Original vs Grad-CAM toggle buttons when `explanation.available == true`.
- **Loading / Error States**: Animated loaders (`Loader2`, `RefreshCw`), error alert boxes with warning details.
- **Important Frontend Files**:
  - `frontend/app/predict/page.tsx`
  - `frontend/app/page.tsx`
  - `frontend/lib/api.ts`
  - `frontend/components/layout/navbar.tsx`

---

## 9. Testing & Verification

Summary of all verified test reports present in the repository:

| Test / Audit Report Name | What It Verified | Result | Report Path |
| :--- | :--- | :---: | :--- |
| **V14 Reproduction Audit** | Exact numerical reproduction of V14 ensemble metrics | **PASS** | `backend/models/evaluation_reports/v14_reproduction_audit.md` |
| **V14 Runtime Smoke Test** | Python import, memory load, and tensor inference for V14 | **PASS** | `backend/models/evaluation_reports/v14_runtime_smoke_test.md` |
| **V14 API Integration Audit**| FastAPI routing for `/api/image-predict?model=v14` | **PASS** | `backend/models/evaluation_reports/v14_integration_audit.md` |
| **Final End-to-End UI Test** | Full browser E2E test (Next.js + FastAPI + Grad-CAM) | **PASS** | `backend/models/evaluation_reports/final_end_to_end_ui_test.md` |
| **Project Cleanup Audit** | Categorization of repository files & space impact analysis | **PASS** | `backend/models/evaluation_reports/project_cleanup_audit.md` |
| **Release Checkpoint Audit**| Verification of production model locks & dataset integrity | **PASS** | `backend/models/evaluation_reports/release_checkpoint_audit.md` |
| **Tabular SVM Removal Audit**| Verification of Tabular SVM component removal | **PASS** | `backend/models/evaluation_reports/tabular_svm_removal_audit.md` |

---

## 10. Project Structure

```
BCD/
├── backend/
│   ├── app/                                   # FastAPI Application API
│   │   ├── main.py                            # Application Entry Point & Lifespan
│   │   ├── image_model.py                     # Image Classifier Service & Grad-CAM
│   │   ├── api/                               # API Route Handlers
│   │   │   ├── prediction.py                  # Image Prediction Endpoints
│   │   │   ├── model.py                       # Metadata Endpoints
│   │   │   └── image_analysis.py              # Cell Microscopy CV Feature Extraction
│   │   ├── core/                              # App Configuration (config.py)
│   │   └── schemas/                           # Pydantic Schemas (prediction.py)
│   ├── models/                                # Model Registry
│   │   ├── breast_image_classifier.keras        # V5-B Active Production Image Model
│   │   ├── breast_image_classifier_metadata.json # Production Image Metadata
│   │   ├── candidates/                        # Research Candidates (V11-B, V13-B0)
│   │   └── evaluation_reports/                # Audit & Verification Reports
│   ├── training/                              # Image Model Training Scripts (v11, v13, v14)
│   └── archive/                               # Historical Experiments (V1–V10/V12, tabular)
├── dataset/
│   └── BUSI/                                  # 776 Clean Ultrasound Scans
├── frontend/                                  # Next.js React Web Application
│   ├── app/                                   # Next.js App Router Pages
│   │   ├── predict/                           # Image Workspace
│   │   ├── image-analysis/                    # Experimental CV Workspace
│   │   └── page.tsx                           # Landing Page
│   ├── components/                            # UI Components & Layout
│   └── lib/                                   # API Client Helper (api.ts)
├── README.md                                  # Documentation
└── requirements.txt                           # Python Dependencies
```

---

## 11. Git / Release State

- **Current Branch**: `main`
- **Current Commit**: `58641c7 Integrate V14 ensemble image prediction`
- **Working Tree Cleanliness**: Working tree contains uncommitted changes for Tabular SVM removal audit and image-only consolidation.
- **Local vs Remote**: Local branch `main` is up to date with `origin/main`.
- **Commit Status of Artifacts**: Core production image models and FastAPI backend are committed; recent V14 candidate models and audit reports are present in repository structure.
- **Git Operations Performed**: **ZERO** git commits or pushes performed.

---

## 12. Current System Status

| Component | Status |
| :--- | :--- |
| **Image dataset** | **VERIFIED** (`dataset/BUSI`, 776 clean scans across 680 clusters) |
| **Image production model** | **VERIFIED** (V5-B MobileNetV2 Letterbox, 80.34% Hist Acc, 87.50% Hist Recall) |
| **V14 research model** | **VERIFIED** (50/50 Ensemble V11-B + V13-B0, 86.32% Hist Acc, 84.38% Hist Recall) |
| **Final candidate** | **VERIFIED** (V14 Ensemble is active research champion; V5-B is production) |
| **Backend** | **VERIFIED** (FastAPI `http://127.0.0.1:8000`, status: healthy) |
| **Frontend** | **VERIFIED** (Next.js `http://localhost:3000`, status: healthy) |
| **API** | **VERIFIED** (`/api/image-predict`, `/api/health` passing) |
| **Grad-CAM** | **VERIFIED** (Color-mapped JET heatmap overlay generated & rendered) |
| **Testing** | **VERIFIED** (Evaluation reports & browser tests passing) |
| **Git** | **VERIFIED** (Branch `main`, up to date with `origin/main`) |

---

## 13. Final Architecture

```
                       ┌─────────────────────────┐
                       │    Raw Data Source      │
                       │      dataset/BUSI       │
                       └────────────┬────────────┘
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │ Preprocessing Pipeline  │
                       │  Aspect Letterboxing    │
                       └────────────┬────────────┘
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │   Training Pipeline     │
                       │ MobileNetV2 / EffNetB0  │
                       └────────────┬────────────┘
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │  Frozen Evaluation Sets │
                       │ 117-Hist / 107-Grouped  │
                       └────────────┬────────────┘
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │   Model Registry        │
                       │ V5-B (Prod) / V14 (Res) │
                       └────────────┬────────────┘
                                    │
                                    ▼
  ┌─────────────────────────────────┴─────────────────────────────────┐
  │                        Inference Pipeline                         │
  │  ├── V5-B MobileNetV2 Inference                                   │
  │  ├── V14 Ensemble Inference (50/50 V11-B + V13-B0)                │
  │  └── Grad-CAM Heatmap Generator (Conv_1 Layer)                    │
  └─────────────────────────────────┬─────────────────────────────────┘
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │    FastAPI Backend      │
                       │ POST /api/image-predict │
                       └────────────┬────────────┘
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │    Next.js UI Frontend  │
                       │   Image AI Workspace    │
                       └────────────┬────────────┘
                                    │
                                    ▼
                                ┌───────────┐
                                │   User    │
                                └───────────┘
```

---

## 14. Final Executive Summary

The Breast Cancer Detection (BCD) platform is an **image-only** clinical decision support system utilizing deep learning breast ultrasound classification. The active production image model is **V5-B** (`v5.0.0-b`), a MobileNetV2 architecture with aspect-ratio letterboxed preprocessing that prioritizes clinical safety by capturing **87.50% of malignant tumors** (only 4 false negatives on the locked historical test set).In research, the **V14 Ensemble** (combining MobileNetV2 V11-B and EfficientNetB0 V13-B0) serves as the research champion, achieving **86.32% historical accuracy** and **80.37% grouped generalization accuracy**. Both models are integrated into a FastAPI backend and served via an interactive Next.js web studio featuring dynamic model selection and Grad-CAM visual heatmaps. All dataset splits, API contracts, model hashes, and UI flows have been audited, reproduced, and verified with 100% passing status.
