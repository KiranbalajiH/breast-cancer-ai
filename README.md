# Breast Cancer Detection ML Application - Phase 1

> **DISCLAIMER:** This project is intended for educational and research purposes only. It is not a medical diagnostic system and should not be used as a substitute for professional medical advice, diagnosis, or treatment.

## Project Overview
This is Phase 1 of a Machine Learning application for predicting breast cancer malignancy. It uses the standard Breast Cancer Wisconsin Diagnostic dataset. The project contains a reproducible ML pipeline to evaluate multiple classification models, selecting the best one based on robust evaluation metrics (prioritizing recall for malignant cases). It also includes a FastAPI backend that serves the trained model to make predictions.

## Machine Learning Methodology
### Preprocessing
All data is processed using a Scikit-Learn `Pipeline` which includes a `StandardScaler`. This is crucial for distance-based models (KNN, SVM) and gradient-descent-based models (Neural Networks) to prevent data leakage during cross-validation and to normalize input features.

### Models Compared
1. Logistic Regression
2. Random Forest
3. K-Nearest Neighbors (KNN)
4. Support Vector Machine (RBF kernel)
5. Gaussian Naive Bayes
6. Neural Network (`MLPClassifier`)
7. LDA + Neural Network

### Evaluation and Selection Strategy
Models are evaluated using Stratified 5-Fold Cross-Validation. The selection strategy uses a custom weighted score prioritizing:
1. Malignant Recall (50% weight): To minimize false negatives.
2. F1 Score (30% weight): To balance precision and recall.
3. ROC-AUC (20% weight): For overall classifier performance.

### Final Model Performance
The best model selected by the pipeline was **SVM (RBF)**.
Key cross-validation metrics for this model:
- **Accuracy:** 0.9772
- **Precision:** 0.9810
- **Recall (Malignant):** 0.9576
- **Specificity:** 0.9888
- **F1 Score:** 0.9688
- **ROC-AUC:** 0.9945

## Backend Architecture
The backend is built with **FastAPI** following a modular structure:
- `app/api/`: API endpoint definitions (routing).
- `app/core/`: Application settings and config.
- `app/schemas/`: Pydantic models for request/response validation.
- `app/services/`: Business logic, specifically `model_service.py` to load the model into memory exactly once at startup.
- `models/`: Contains the joblib exported ML pipeline, metadata, and cross-validation comparisons.

## Setup Instructions

### 1. Create a Virtual Environment
```bash
python -m venv venv
# Windows:
.\venv\Scripts\Activate.ps1
# Mac/Linux:
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r backend/requirements.txt
```

### 3. Train the Model
This step will download the sklearn dataset, compare models, and save the best one.
```bash
python backend/training/train.py
```

### 4. Start the FastAPI Server
```bash
cd backend
uvicorn app.main:app --reload
```
The server will run at `http://localhost:8000`. Swagger documentation is available at `http://localhost:8000/docs`.

## API Endpoints

### 1. Health Check
`GET /api/health`
Returns the status of the server and whether the ML model is successfully loaded.

### 2. Prediction
`POST /api/predict`
Accepts a JSON payload of 30 cell nucleus features and returns the malignancy prediction.

**Example Request:**
```json
{
  "mean radius": 14.2,
  "mean texture": 19.2,
  "mean perimeter": 90.0,
  "mean area": 600.0,
  "mean smoothness": 0.1,
  "mean compactness": 0.1,
  "mean concavity": 0.05,
  "mean concave points": 0.05,
  "mean symmetry": 0.15,
  "mean fractal dimension": 0.06,
  "radius error": 0.3,
  "texture error": 1.0,
  "perimeter error": 2.5,
  "area error": 30.0,
  "smoothness error": 0.005,
  "compactness error": 0.01,
  "concavity error": 0.01,
  "concave points error": 0.01,
  "symmetry error": 0.01,
  "fractal dimension error": 0.003,
  "worst radius": 16.0,
  "worst texture": 25.0,
  "worst perimeter": 105.0,
  "worst area": 800.0,
  "worst smoothness": 0.13,
  "worst compactness": 0.2,
  "worst concavity": 0.2,
  "worst concave points": 0.1,
  "worst symmetry": 0.25,
  "worst fractal dimension": 0.08
}
```

## Troubleshooting
- **Model not loaded:** Ensure `backend/models/breast_cancer_model.joblib` exists. If not, run `train.py`.
- **Dependency Issues:** Make sure you are using Python 3.9+ and have activated the virtual environment before installing `requirements.txt`.

---

# Phase 2: Modern Frontend UI

A full-stack, responsive, and aesthetically pleasing modern frontend has been built using **Next.js**, **Tailwind CSS**, and **Recharts**.

## Features
- **Live Backend Communication:** The frontend communicates with the real FastAPI backend, fetching model status, generating real predictions, and rendering analytics based on the real trained model data.
- **Analytics Dashboard:** A comprehensive view of model performance metrics, interactive bar charts for model comparison, permutation feature importance graphs, and confusion matrix rendering.
- **AI Prediction Interface:** An intuitive and fully validated form to submit 30 diagnostic features, featuring animated AI processing steps and probability distributions.

## Starting the Frontend
1. Open a new terminal.
2. Navigate to the frontend directory: `cd frontend`
3. Start the Next.js development server:
```bash
npm run dev
```
4. Access the application at [http://localhost:3000](http://localhost:3000).

> **Note:** Ensure the FastAPI backend is running simultaneously on `http://localhost:8000`.
