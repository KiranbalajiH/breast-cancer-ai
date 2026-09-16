Searched for "RAP"
Viewed README.md:1-100
Viewed README.md:100-146

Here is the rephrased and updated content formatted as a **Repository Architecture & Platform (RAP)** specification document, fully aligned with your project's image-only ultrasound classification architecture:

---

# ⚡ Breast Cancer AI Prediction System (Experimental Prototype)

> [!WARNING]
> **IMPORTANT CLINICAL DISCLAIMER:** This project is intended for research, education, and experimental decision-support purposes only. It is **not** a clinically approved diagnostic system, is not medically certified, and must **not** be used as a substitute for professional medical advice, diagnosis, or treatment.

This platform provides a production-grade machine learning pipeline and an interactive visual workspace to predict breast cancer risk from breast ultrasound scans using deep computer vision models (V5-B MobileNetV2 Production Model & V14 Ensemble Research Model) alongside class-specific Grad-CAM visual heatmaps.

---

## 📌 Repository Architecture & Platform (RAP) Layout

```
breast-cancer-ai/
├── backend/                       # FastAPI Backend Microservice
│   ├── app/                       # Core service logic (lifespan, config, image pipeline, API endpoints)
│   ├── data/                      # Dataset manifest metadata & manifest definitions
│   ├── models/                    # Active production & candidate ML weight artifacts (.keras)
│   │   └── evaluation_reports/    # Independent validation & audit reports
│   └── archive/                   # Historical experiment iterations, candidate weights, & legacy reports
├── dataset/                       # BUSI (Breast Ultrasound Images) Dataset
│   └── BUSI/                      # Cleaned ultrasound scans (benign, malignant, normal)
├── frontend/                      # Next.js 14 Frontend Workspace (React, TypeScript, Tailwind CSS)
│   ├── app/                       # Application router pages (Ultrasound workspace, analytics dashboard)
│   ├── components/                # Reusable UI widgets & visual viewers (Grad-CAM viewer, dropzone)
│   ├── public/                    # Static assets & quick-load sample ultrasound scans
│   └── lib/                       # API fetching layer, contract definitions, & type interfaces
├── RMD/                           # R Markdown exploratory dataset analysis & statistical reports
└── README.md                      # Main project documentation & RAP specification
```

---

## Supported Inputs & Target Classes

### 1. Supported Input Modality
* **Image Mode:** Breast ultrasound scans (Grayscale scans only; supported formats: `.png`, `.jpeg`, `.jpg`).

### 2. Mapped Output Classes
* `benign`: Non-cancerous tumor or normal-appearing benign lesion.
* `malignant`: High-risk or cancerous lesion requiring clinical evaluation.
* `normal`: Healthy breast tissue scan with no focal lesion identified.

---

## Core System Capabilities

1. **Dual-Model Deep Learning Predictions:** Classifies tumor risk using our primary production MobileNetV2 classifier (**V5-B**) featuring aspect-ratio letterboxed preprocessing, with an opt-in **V14 Research Ensemble** (MobileNetV2 + EfficientNetB0).
2. **Probability Distribution Reporting:** Returns exact Softmax probability scores across all target classes (`benign`, `malignant`, `normal`) alongside the primary class confidence score.
3. **Uncertainty & Low-Margin Safeguard Layer:** Flags low-confidence predictions (maximum probability $< 0.50$) or close class competition (top-two class margin $< 0.15$), transitioning prediction status to `review_required` with an explicit clinician alert.
4. **Deterministic Image Quality Validation:** Scans incoming images for extreme blurriness (Laplacian variance evaluation), underexposure, overexposure, or low contrast, marking low-fidelity uploads as `"poor"` quality.
5. **Modality & Annotation Safeguards:** Blocks invalid uploads containing burned-in color calipers, Doppler color sweeps, non-ultrasound photographs, or abnormal aspect-ratio documents, flagging status as `"unsupported_or_review_required"`.
6. **Visual Explainability (Grad-CAM):** Generates high-resolution class-activation heatmaps from the deep convolutional backbone (`Conv_1` layer) to highlight spatial regions influencing the network's prediction.
7. **Quick-Load Clinician Samples:** Includes built-in sample scans (`Sample Benign Scan`, `Sample Malignant Scan`, `Sample Normal Scan`) for immediate end-to-end platform validation.

---

## Local Setup & Environment Execution

### Prerequisites
- **Python:** 3.9+
- **Node.js:** 18+

### 1. Backend Microservice (FastAPI & TensorFlow)
Navigate to the `backend/` directory:
```bash
# Create and activate Python virtual environment
python -m venv venv

# Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# Mac/Linux:
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

#### Launch Backend Server:
```bash
python -m uvicorn app.main:app --reload
```
* **API Root:** `http://localhost:8000`
* **Interactive OpenAPI Docs:** `http://localhost:8000/docs`

---

### 2. Frontend Workspace (Next.js & Tailwind CSS)
Navigate to the `frontend/` directory:
```bash
# Install Node dependencies
npm install
```

#### Launch Frontend Dev Server:
```bash
npm run dev
```
* **Visual Workspace UI:** `http://localhost:3000`

---

## Environment Variables Configuration

### Backend Environment File (`backend/.env`)
```env
BACKEND_CORS_ORIGINS="http://localhost:3000,https://breast-cancer-ai.vercel.app"
```

### Frontend Environment File (`frontend/.env.local`)
```env
NEXT_PUBLIC_API_URL="http://localhost:8000"
```

---

## Production Deployment Workflow

### 1. Backend Microservice Deployment (Render, Railway, or VPS)
1. **Root Directory:** Set root path to `backend/`.
2. **Build Command:** `pip install -r requirements.txt`
3. **Start Command:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
4. **Environment Variables:** Set `BACKEND_CORS_ORIGINS` to match your production frontend URL.
5. **Memory Allocation:** Ensure $\ge 1\text{GB}$ RAM to support TensorFlow initialization and model weight pre-loading.
6. **Health Monitoring Endpoints:** Configure health probes to target `/api/health` or `/api/image-model/status`.

### 2. Frontend Application Deployment (Vercel)
1. Import repository into Vercel.
2. Set **Root Directory** to `frontend/`.
3. Configure Environment Variables:
   - `NEXT_PUBLIC_API_URL`: `<your-production-backend-url>` (e.g. `https://breast-cancer-backend.onrender.com`)
4. Trigger **Deploy**. Vercel will build, optimize static bundles, and deploy the application.

---

## Model Limitations & Operational Boundaries

1. **Grayscale Ultrasound Requirement:** Quality heuristics expect standard B-mode ultrasound inputs. Color Doppler sweeps, burned-in graphic text, or high-saturation annotations will trigger `"unsupported_or_review_required"`.
2. **Attribution-Only Heatmaps:** Grad-CAM displays feature importance activation zones; it does **not** generate clinical segmentations or delineate exact pathological tumor boundaries.
