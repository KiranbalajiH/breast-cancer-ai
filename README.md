# Breast Cancer AI Prediction System (Experimental Prototype)

> [!WARNING]
> **IMPORTANT CLINICAL DISCLAIMER:** This project is intended for research, education, and experimental decision-support purposes only. It is **not** a clinically approved diagnostic system, is not medically certified, and should not be used as a substitute for professional medical advice, diagnosis, or treatment.

This application provides a machine learning pipeline and a modern visual workspace to predict breast cancer risk from clinical data and breast ultrasound scans.

---

## Repository Structure

```
breast-cancer-ai/
├── backend/                       # FastAPI Backend Application
│   ├── app/                       # Source files (lifespan, config, prediction api)
│   ├── data/                      # Manifest metadata (no raw images at runtime)
│   ├── models/                    # Stored ML weights (keras, joblib models)
│   │   └── evaluation_reports/    # Independent model validation reports
│   └── tests/                     # Automated unit and safety tests
├── frontend/                      # Next.js Frontend (React, Tailwind CSS)
│   ├── app/                       # Next.js Pages (prediction workspace, analytics)
│   ├── components/                # Reusable UI widgets
│   └── lib/                       # API fetching and type interfaces
├── RMD/                           # R Markdown dataset analysis reports
└── README.md                      # Main project documentation
```

---

## Supported Inputs & Outputs

### 1. Supported Input
* **Image Mode:** Breast ultrasound scans (grayscale scans only).
* **Tabular Mode:** Wisconsin Breast Cancer Diagnostic cell nuclei measurements (30 measurements).

### 2. Mapped Output Classes (Image Mode)
* `benign` (Normal-appearing or non-cancerous lesion)
* `malignant` (High-risk or cancerous lesion)
* `normal` (Healthy breast tissue scan)

---

## Core Features

1. **AI Predictions:** Classifies tumor risk using a MobileNetV2 deep learning classifier for images and an RBF Support Vector Machine for tabular inputs.
2. **Confidence Reporting:** Returns exact Softmax probability distributions for all target classes alongside the primary class confidence score.
3. **Uncertainty Alert Layer:** Identifies low-confidence predictions (max probability < 0.50) or close competition (top-two class margin < 0.15), mapping status to `review_required` with a clinician warning.
4. **Deterministic Image Quality Validation:** Analyzes uploads for extreme blurriness, darkness, brightness, or low contrast, alerting when inputs are of `"poor"` quality.
5. **Unsupported Image Safeguards:** Prevents processing of colored photographs, documents, or extreme aspect ratio screenshots, marking status as `"unsupported_or_review_required"`.
6. **Visual Explainability (Grad-CAM):** Generates heatmap overlays showing the image regions that most heavily influenced the MobileNetV2 prediction.

---

## Local Setup & Run Instructions

Ensure you have Python 3.9+ and Node.js 18+ installed on your system.

### 1. Backend Setup (FastAPI & TensorFlow)
Navigate to the `backend/` directory:
```bash
# Create and activate virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate.ps1
# Mac/Linux:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

#### Run Backend Server:
From the `backend/` folder:
```bash
uvicorn app.main:app --reload
```
The backend starts at `http://localhost:8000`. Swagger API documentation is available at `http://localhost:8000/docs`.

### 2. Frontend Setup (Next.js & Tailwind CSS)
Navigate to the `frontend/` directory:
```bash
# Install packages
npm install
```

#### Run Frontend Server:
From the `frontend/` folder:
```bash
npm run dev
```
The workspace UI opens at `http://localhost:3000`.

---

## Environment Variables

### Backend Configuration (`backend/.env`)
* `BACKEND_CORS_ORIGINS`: Comma-separated allowed frontend origins (e.g. `"http://localhost:3000,https://breast-cancer-ai.vercel.app"`).

### Frontend Configuration (`frontend/.env.local`)
* `NEXT_PUBLIC_API_URL`: Root URL of the running FastAPI server (e.g. `"http://localhost:8000"`).

---

## Production Deployment Workflow

### 1. Backend Deployment (e.g., Render, Railway, or VPS)
Connect your GitHub repository to your Python hosting platform of choice:
1. **Repository root path:** Select `backend` as the base directory (or root if using Docker/monorepo build configs).
2. **Build Command:** `pip install -r requirements.txt`
3. **Start Command:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
4. **Environment Variables:** Configure `BACKEND_CORS_ORIGINS` to point to your Vercel frontend URL.
5. **Memory Limit:** Ensure the hosting environment has $\ge$ 1GB of RAM to prevent out-of-memory issues during TensorFlow library initialization and model pre-loading.
6. **Health Probe Paths:** Use `/api/health` or `/api/model-status` for health checks.

### 2. Frontend Deployment (Vercel)
1. Go to Vercel and import your GitHub repository.
2. Select `frontend` as the **Root Directory**.
3. In **Environment Variables**, add:
   * **Key:** `NEXT_PUBLIC_API_URL`
   * **Value:** `<your-deployed-backend-url>` (e.g. `https://breast-cancer-backend.onrender.com`)
4. Click **Deploy**. Vercel will automatically compile, optimize, and serve the application statically.

---

## Model Limitations

1. **Grayscale Expectation:** Modality verification heuristics assume scans are grayscale. High-saturation color overlays, annotations, or Doppler color sweeps will trigger `"unsupported_or_review_required"` alerts.
2. **Attribution Only:** Grad-CAM displays feature contribution zones. It does **not** trace exact physical tumor borders or boundaries.
