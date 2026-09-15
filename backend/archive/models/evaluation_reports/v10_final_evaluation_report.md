PRODUCTION MODEL CHANGED: NO
PRODUCTION METADATA CHANGED: NO
LOCKED 117-IMAGE TEST SET MODIFIED: NO
DATASET IMAGES MODIFIED: NO
DEPLOYMENT PERFORMED: NO
GIT COMMIT PERFORMED: NO
GIT PUSH PERFORMED: NO

# V10 Final Evaluation & Malignant Sensitivity Optimization Report

**Date**: September 02, 2026  
**Project**: Breast Cancer AI  Ultrasound Image Classifier  
**Dataset**: `dataset/BUSI` (776 Clean Trainable Scans)  
**Production Model Status**: `backend/models/breast_image_classifier.keras` remains **unmodified** and **active** (V5-B).  
**Candidate Models Location**: `backend/models/candidates/`

---

## 1. Executive Summary & Production Status
- **Baseline Benchmark**: V5-B (`backend/models/breast_image_classifier.keras`)
  - Historical 117 Test: Accuracy 80.34%, Macro F1 79.75%, Malignant Recall 87.50% (28/32, 4 FN, 8 FP), Malignant Precision 77.78%, Malignant F1 82.35%.
  - Grouped Test: Accuracy 70.09%, Macro F1 69.46%, Malignant Recall 80.65% (25/31, 6 FN, 15 FP), Malignant Precision 62.50%, Malignant F1 70.42%.
- **V10 Objective**: Reduce malignant false negatives and beat V5-B on grouped malignant recall (>80.65%) while preserving precision and macro F1.

---

## 2. V10 Candidate Configurations
- **V10-A (Control Baseline)**: MobileNetV2 with aspect-ratio preserving letterbox preprocessing (Unweighted).
- **V10-B (Moderate Class Weighting)**: V5-B + Class weights `{0: 1.0, 1: 1.6, 2: 1.1}`.
- **V10-C (Focal Loss)**: V5-B + Sparse Categorical Focal Loss ($\gamma=2.0, \alpha=[1.0, 1.5, 1.1]$).
- **V10-D (Malignant-Focused Augmentation)**: V5-B + Controlled conservative malignant augmentation.
- **V10-E (Combined Strategy)**: Moderate Class Weighting (`{0: 1.0, 1: 1.6, 2: 1.1}`) + Malignant-Focused Augmentation.

---

## 3. Validation Performance & Threshold Tuning

| Candidate | Val Accuracy | Val Macro F1 | Val Malignant Recall | Val Malignant Precision | Val FN | Val FP | Selected Threshold |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **V10_MOBILENETV2_A** | 78.86% | 78.29% | 78.12% | 65.79% | 7 | 13 | 0.5 |
| **V10_MOBILENETV2_B** | 72.36% | 73.04% | 87.50% | 54.90% | 4 | 23 | 0.45 |
| **V10_MOBILENETV2_C** | 80.49% | 79.69% | 87.50% | 68.29% | 4 | 13 | 0.45 |
| **V10_MOBILENETV2_D** | 77.24% | 77.96% | 90.62% | 59.18% | 3 | 20 | 0.5 |
| **V10_MOBILENETV2_E** | 64.23% | 67.13% | 96.88% | 44.93% | 1 | 38 | 0.5 |

---

## 4. Frozen Test Set Results

### A. Historical Locked 117-Image Test Results
| Candidate | Accuracy | Macro F1 | Malignant Recall | Malignant Precision | Malignant F1 | Malignant FN | Malignant FP | Total Errors |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **V10_MOBILENETV2_A** | 73.50% | 73.01% | 81.25% | 72.22% | 76.47% | 6 | 10 | 31 |
| **V10_MOBILENETV2_B** | 78.63% | 77.90% | 87.50% | 77.78% | 82.35% | 4 | 8 | 25 |
| **V10_MOBILENETV2_C** | 73.50% | 73.28% | 84.38% | 65.85% | 73.97% | 5 | 14 | 31 |
| **V10_MOBILENETV2_D** | 76.92% | 76.57% | 93.75% | 69.77% | 80.00% | 2 | 13 | 27 |
| **V10_MOBILENETV2_E** | 66.67% | 67.45% | 100.00% | 55.17% | 71.11% | 0 | 26 | 39 |

### B. Grouped Generalization Test Results
| Candidate | Accuracy | Macro F1 | Malignant Recall | Malignant Precision | Malignant F1 | Malignant FN | Malignant FP | Total Errors |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **V10_MOBILENETV2_A** | 68.22% | 66.52% | 74.19% | 65.71% | 69.70% | 8 | 12 | 34 |
| **V10_MOBILENETV2_B** | 71.03% | 70.21% | 80.65% | 64.10% | 71.43% | 6 | 14 | 31 |
| **V10_MOBILENETV2_C** | 57.01% | 56.69% | 74.19% | 46.00% | 56.79% | 8 | 27 | 46 |
| **V10_MOBILENETV2_D** | 67.29% | 67.23% | 87.10% | 56.25% | 68.35% | 4 | 21 | 35 |
| **V10_MOBILENETV2_E** | 58.88% | 60.34% | 90.32% | 45.90% | 60.87% | 3 | 33 | 44 |

---

## 5. Historical Test Comparison Across Iterations

| Model Iteration | Accuracy | Macro F1 | Malignant Precision | Malignant Recall | Malignant F1 | Malignant FN | Malignant FP | Total Errors |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **V1 Baseline** | 84.62% | 84.21% | 84.62% | 68.75% | 75.86% | 10 | 4 | 18 |
| **V3 Mild Weights** | 76.92% | 74.50% | 66.67% | 87.50% | 75.68% | 4 | 14 | 27 |
| **V5-B (Active Prod)** | 80.34% | 79.75% | 77.78% | **87.50%** | **82.35%** | **4** | 8 | 23 |
| **V6-B** | 81.20% | 80.54% | 78.57% | **87.50%** | 82.76% | **4** | 7 | 22 |
| **V7-D (Dual-View)** | 84.62% | 83.17% | 80.77% | 65.62% | 72.41% | 11 | 5 | 18 |
| **V9-D (Dual-View Aux)**| 87.18% | 86.46% | 81.25% | 81.25% | 81.25% | 6 | 6 | 15 |
| **V10-A** | 73.50% | 73.01% | 72.22% | 81.25% | 76.47% | 6 | 10 | 31 |
| **V10-B** | 78.63% | 77.90% | 77.78% | 87.50% | 82.35% | 4 | 8 | 25 |
| **V10-C** | 73.50% | 73.28% | 65.85% | 84.38% | 73.97% | 5 | 14 | 31 |
| **V10-D** | 76.92% | 76.57% | 69.77% | 93.75% | 80.00% | 2 | 13 | 27 |
| **V10-E** | 66.67% | 67.45% | 55.17% | 100.00% | 71.11% | 0 | 26 | 39 |

---

## 6. Grouped Generalization Comparison Across Iterations

| Model Iteration | Accuracy | Macro F1 | Malignant Precision | Malignant Recall | Malignant F1 | Malignant FN | Malignant FP | Total Errors |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **V5-B (Active Prod)** | 70.09% | 69.46% | 62.50% | **80.65%** | 70.42% | **6** | 15 | 32 |
| **V6-B** | 71.96% | 71.10% | 64.10% | **80.65%** | 71.43% | **6** | 14 | 30 |
| **V7-D (Dual-View)** | 76.64% | 74.95% | 69.57% | 51.61% | 59.26% | 15 | 7 | 25 |
| **V9-D (Dual-View Aux)**| 77.57% | 77.84% | 63.64% | 67.74% | 65.62% | 10 | 12 | 24 |
| **V10-A** | 68.22% | 66.52% | 65.71% | 74.19% | 69.70% | 8 | 12 | 34 |
| **V10-B** | 71.03% | 70.21% | 64.10% | 80.65% | 71.43% | 6 | 14 | 31 |
| **V10-C** | 57.01% | 56.69% | 46.00% | 74.19% | 56.79% | 8 | 27 | 46 |
| **V10-D** | 67.29% | 67.23% | 56.25% | 87.10% | 68.35% | 4 | 21 | 35 |
| **V10-E** | 58.88% | 60.34% | 45.90% | 90.32% | 60.87% | 3 | 33 | 44 |

---

## 7. Model Ranking & Recommendation
- **Best Validation Candidate**: `V10_MOBILENETV2_C`
- **Best Historical Test Candidate**: `V10_MOBILENETV2_D`
- **Best Grouped Test Candidate**: `V10_MOBILENETV2_D`
- **Promotion Decision**: **DO NOT PROMOTE**. Recommend retaining **V5-B** as active production model.

---

PRODUCTION MODEL CHANGED: NO
PRODUCTION METADATA CHANGED: NO
LOCKED 117-IMAGE TEST SET MODIFIED: NO
DATASET IMAGES MODIFIED: NO
DEPLOYMENT PERFORMED: NO
GIT COMMIT PERFORMED: NO
GIT PUSH PERFORMED: NO
