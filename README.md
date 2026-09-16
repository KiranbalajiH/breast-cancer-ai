# Breast Cancer AI Prediction System

> **Clinical Disclaimer**
> This project is an experimental research and educational prototype. It is **not** a clinically approved or medically certified diagnostic system and must **not** be used as a substitute for professional medical advice, diagnosis, or treatment. All predictions are probabilistic estimates and carry inherent uncertainty. Always consult a qualified healthcare professional for clinical decisions.

<img width="1881" height="907" alt="image" src="https://github.com/user-attachments/assets/4fbbd7da-a8fa-460a-95b5-c226953f04dd" />


---

## Overview

The Breast Cancer AI Prediction System is a deep learning platform that classifies grayscale breast ultrasound scans into three diagnostic categories:

| Class | Description |
| :--- | :--- |
| **Benign** | Non-cancerous tumor or normal-appearing benign lesion |
| **Malignant** | High-risk or cancerous lesion requiring clinical evaluation |
| **Normal** | Healthy breast tissue with no focal lesion identified |

The system is built as a decoupled microservice architecture with a **FastAPI** backend serving TensorFlow/Keras models and a **Next.js** frontend providing an interactive visual workspace with Grad-CAM explainability overlays.

---

## Models

### V5-B — Production Model
- **Architecture:** MobileNetV2 transfer learning with ImageNet backbone
- **Preprocessing:** Aspect-ratio letterbox resize to 224 × 224 × 3
- **Head:** GlobalAveragePooling2D → Dropout(0.2) → Dense(3, softmax)
- **Status:** Active production model used for default inference

### V14 — Research Ensemble
- **Architecture:** Frozen probability ensemble of V11-B (MobileNetV2) + V13-B0 (EfficientNetB0)
- **Strategy:** 50/50 weighted probability averaging with fitted malignant threshold (t = 0.58)
- **Status:** Opt-in research model available via dedicated endpoint. **Not automatically promoted to production.**

---

## Core Features

1. **Deep Learning Image Classification** — Classifies breast ultrasound scans using the production V5-B MobileNetV2 model, with an opt-in V14 research ensemble (MobileNetV2 + EfficientNetB0).
2. **Probability Distribution Reporting** — Returns softmax probability scores across all three target classes alongside the primary class confidence score.
3. **Uncertainty & Low-Margin Safeguards** — Flags low-confidence predictions (max probability < 0.50) or close class competition (top-two margin < 0.15), transitioning status to `review_required` with a clinician alert.
4. **Image Quality Validation** — Analyzes uploads for extreme blurriness (Laplacian variance), underexposure, overexposure, and low contrast, marking low-fidelity inputs as `"poor"` quality.
5. **Modality & Annotation Safeguards** — Blocks non-ultrasound photographs, documents, color Doppler sweeps, burned-in calipers, or extreme aspect-ratio images, flagging status as `"unsupported_or_review_required"`.
6. **Visual Explainability (Grad-CAM)** — Generates class-activation heatmaps from the final convolutional layer (`Conv_1`) showing image regions that most influenced the prediction.
7. **Built-in Sample Ultrasound Scans** — Quick-load buttons (Sample Benign, Sample Malignant, Sample Normal) for immediate end-to-end platform testing.

---

## Dataset

The system is trained and evaluated on the **BUSI (Breast Ultrasound Images)** dataset.

| Class | Cleaned Count |
| :--- | :---: |
| Benign | 434 |
| Malignant | 209 |
| Normal | 133 |
| **Total** | **776** |

- **Excluded:** 2 scans (`benign (433).png` and `malignant (145).png`) removed due to exact pixel-duplicate cross-label conflict.
- **Source:** `dataset/BUSI/`

---

## Repository Architecture

```
breast-cancer-ai/
├── backend/                           # FastAPI backend microservice
│   ├── app/                           # Application source (lifespan, config, API endpoints, image pipeline)
│   │   └── api/                       # Route definitions (prediction, image analysis)
│   ├── image_processing/              # Image quality validation and preprocessing utilities
│   ├── data/                          # Dataset manifest metadata
│   ├── models/                        # Production and candidate model weights (.keras)
│   │   ├── candidates/                # Research candidate model artifacts
│   │   └── evaluation_reports/        # Independent validation, audit, and evaluation reports
│   ├── training/                      # Training scripts (v11, v13, v14, full-dataset experiments)
│   ├── tests/                         # Automated unit and safety tests
│   ├── archive/                       # Historical experiment iterations, legacy reports, and artifacts
│   └── requirements.txt              # Python dependencies
├── frontend/                          # Next.js frontend workspace
│   ├── app/                           # Next.js App Router pages (workspace, analytics)
│   ├── components/                    # Reusable UI widgets (Grad-CAM viewer, dropzone, model selector)
│   ├── public/                        # Static assets and sample ultrasound scans
│   └── lib/                           # API client layer, type interfaces, and contract definitions
├── dataset/                           # BUSI breast ultrasound image dataset
│   └── BUSI/                          # Class directories (benign/, malignant/, normal/)
├── RMD/                               # R Markdown exploratory dataset analysis reports
└── README.md                          # Project documentation
```

---

## Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **Backend** | Python, FastAPI, TensorFlow / Keras, NumPy, OpenCV, Uvicorn |
| **Frontend** | Next.js, React, TypeScript, Tailwind CSS |
| **ML Models** | MobileNetV2, EfficientNetB0, Softmax classification |
| **Explainability** | Grad-CAM heatmap generation |
| **Safeguards** | Image quality validation, modality verification, uncertainty alerts |

---

## Local Setup

### Prerequisites
- Python 3.9+
- Node.js 18+

### 1. Clone the Repository
```bash
git clone https://github.com/KiranbalajiH/breast-cancer-ai.git
cd breast-cancer-ai
```

### 2. Backend Setup (FastAPI + TensorFlow)
```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate — Windows PowerShell:
.\venv\Scripts\Activate.ps1

# Activate — macOS / Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the backend server
python -m uvicorn app.main:app --reload
```
- **API Root:** `http://localhost:8000`
- **OpenAPI Docs:** `http://localhost:8000/docs`

### 3. Frontend Setup (Next.js + Tailwind CSS)
```bash
cd frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```
- **Workspace UI:** `http://localhost:3000`

---

## Environment Variables

### Backend (`backend/.env`)
| Variable | Description | Example |
| :--- | :--- | :--- |
| `BACKEND_CORS_ORIGINS` | Comma-separated allowed frontend origins | `http://localhost:3000` |

### Frontend (`frontend/.env.local`)
| Variable | Description | Example |
| :--- | :--- | :--- |
| `NEXT_PUBLIC_API_URL` | Root URL of the FastAPI backend | `http://localhost:8000` |

---

## API Endpoints

All endpoints are prefixed with `/api`.

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Health check — returns 503 if model is not loaded |
| `GET` | `/api/image-model/status` | Returns image model loading status and metadata |
| `POST` | `/api/image-predict` | Predict using V5-B production model (default) |
| `POST` | `/api/image-predict?model=v14` | Predict using V14 research ensemble via query parameter |
| `POST` | `/api/image-predict-v14` | Predict using V14 research ensemble via dedicated endpoint |

### Request Format
All prediction endpoints accept `multipart/form-data` with a single `file` field containing a JPEG or PNG breast ultrasound image.

### Response
Prediction responses include: predicted class, softmax probability distribution, confidence score, prediction status (`normal`, `review_required`, or `unsupported_or_review_required`), image quality assessment, and Grad-CAM heatmap data (base64-encoded).

---

## Model Evaluation

Evaluation and audit reports are stored under `backend/models/evaluation_reports/`. Reports contain:

- Per-class precision, recall, F1-score, and support
- Confusion matrices
- Model architecture and training configuration
- SHA256 file hashes for reproducibility
- Runtime validation and model loading tests
- Frontend and backend integration verification (where applicable)

### Production vs. Research Inference

| Property | V5-B (Production) | V14 (Research) |
| :--- | :--- | :--- |
| **Endpoint** | `/api/image-predict` | `/api/image-predict-v14` |
| **Architecture** | MobileNetV2 | MobileNetV2 + EfficientNetB0 Ensemble |
| **Status** | Active production default | Opt-in research only |
| **Promotion** | Currently deployed | Not automatically promoted |

V14 is accessible for research comparison but all default production inference uses V5-B. Model promotion requires explicit validation against held-out evaluation sets and manual deployment.

---

## Deployment

### Backend (Render, Railway, or VPS)
1. Set root directory to `backend/`.
2. **Build command:** `pip install -r requirements.txt`
3. **Start command:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
4. Set `BACKEND_CORS_ORIGINS` to your production frontend URL.
5. Ensure ≥ 1 GB RAM for TensorFlow initialization and model loading.
6. Health probe: `GET /api/health`

### Frontend (Vercel)
1. Import repository and set root directory to `frontend/`.
2. Add environment variable: `NEXT_PUBLIC_API_URL` = your deployed backend URL.
3. Deploy. Vercel handles build optimization automatically.

---

## Limitations

1. **Dataset Limitations:** The BUSI dataset contains 776 cleaned scans from approximately 600 patients. This is a relatively small dataset and may not capture the full diversity of breast ultrasound presentations across populations, equipment, and imaging protocols.
2. **Ultrasound Modality Limitations:** The system expects standard grayscale B-mode breast ultrasound scans. Color Doppler overlays, burned-in graphic annotations, measurement calipers, or non-ultrasound photographs will trigger modality safeguards and may be rejected.
3. **Model Generalization Limitations:** Models are trained on a single institutional dataset. Performance on ultrasound scans from different equipment, imaging protocols, or patient populations has not been independently validated.
4. **Prediction Uncertainty:** All predictions are probabilistic softmax estimates. Low-confidence predictions or close class margins are flagged but may still be incorrect. No prediction should be treated as a definitive clinical diagnosis.
5. **Grad-CAM Limitations:** Grad-CAM heatmaps display feature importance activation zones. They do not generate clinical segmentations, delineate pathological tumor boundaries, or provide pixel-level diagnostic masks.

---

## Research Status

| Property | Value |
| :--- | :--- |
| **Project Status** | Experimental Prototype |
| **Primary Model** | V5-B MobileNetV2 |
| **Research Model** | V14 MobileNetV2 + EfficientNetB0 Ensemble |
| **Input Modality** | Grayscale Breast Ultrasound |
| **Classes** | Benign / Malignant / Normal |
| **Explainability** | Grad-CAM |
| **Backend** | FastAPI + TensorFlow |
| **Frontend** | Next.js + React + TypeScript |

---

## License

Add the applicable software license for this repository.

The BUSI (Breast Ultrasound Images) dataset has its own licensing and usage requirements as specified by its authors (Al-Dhabyani et al., 2020). Users must respect the original dataset license terms.

---

## Disclaimer

This system is provided strictly for research, educational, and experimental decision-support purposes. It is not a clinically validated diagnostic tool, has not received regulatory clearance or certification, and must not be relied upon for medical diagnosis or treatment decisions. Always seek the advice of qualified healthcare professionals for clinical interpretation of medical imaging.
