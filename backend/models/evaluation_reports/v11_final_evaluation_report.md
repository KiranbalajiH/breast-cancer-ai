PRODUCTION MODEL CHANGED: NO
PRODUCTION METADATA CHANGED: NO
LOCKED 117-IMAGE TEST SET MODIFIED: NO
DATASET IMAGES MODIFIED: NO
DEPLOYMENT PERFORMED: NO
GIT COMMIT PERFORMED: NO
GIT PUSH PERFORMED: NO

# V11 Final Evaluation & Precision/Recall Optimization Report

**Date**: September 02, 2026  
**Project**: Breast Cancer AI — Ultrasound Image Classifier  
**Dataset**: `dataset/BUSI` (776 Clean Trainable Scans)  
**Production Model Status**: `backend/models/breast_image_classifier.keras` remains **unmodified** and **active** (V5-B).  
**Candidate Models Location**: `backend/models/candidates/`

---

## 1. Executive Summary & Production Status
- **Baseline Benchmarks**:
  - **V5-B (Production)**: Grouped Accuracy 70.09%, Macro F1 69.46%, Malignant Recall 80.65% (6 FN, 15 FP), Malignant Precision 62.50%, Malignant F1 70.42%.
  - **V10-D (Prior Best Sensitivity)**: Grouped Accuracy 67.29%, Macro F1 67.23%, Malignant Recall 87.10% (4 FN, 21 FP), Malignant Precision 56.25%, Malignant F1 68.35%.
- **V11 Objective**: Recover precision while retaining V10-D's malignant sensitivity gain ($\ge 85\%$ Grouped Malignant Recall, $\ge 60\%$ Precision, $\ge 70\%$ F1, $\text{FN} \le 5$, substantially reduced FP below 21).

---

## 2. V11 Candidate Configurations
- **V11-A (V10-D Control)**: Direct reproduction of V10-D (100% malignant sample replication, unweighted).
- **V11-B (Reduced Augmentation)**: 50% malignant sample replication (milder augmentation ratio).
- **V11-C (Augmentation + Mild Class Emphasis)**: 100% malignant sample replication + class weights `{0: 1.0, 1: 1.25, 2: 1.05}`.
- **V11-D (Augmentation + Regularization)**: 100% malignant sample replication + 30% benign replication + Dropout(0.30).
- **V11-E (Composite Strategy)**: 50% malignant sample replication + class weights `{0: 1.0, 1: 1.20, 2: 1.0}` + Dropout(0.25).

---

## 3. Validation Performance & Threshold Tuning

| Candidate | Val Accuracy | Val Macro F1 | Val Malignant Recall | Val Malignant Precision | Val FN | Val FP | Selected Threshold |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **V11_MOBILENETV2_A** | 77.24% | 77.96% | 90.62% | 59.18% | 3 | 20 | 0.5 |
| **V11_MOBILENETV2_B** | 71.54% | 73.14% | 93.75% | 52.63% | 2 | 27 | 0.4 |
| **V11_MOBILENETV2_C** | 71.54% | 73.43% | 90.62% | 51.79% | 3 | 27 | 0.5 |
| **V11_MOBILENETV2_D** | 75.61% | 76.58% | 90.62% | 56.86% | 3 | 22 | 0.5 |
| **V11_MOBILENETV2_E** | 69.92% | 71.03% | 90.62% | 50.88% | 3 | 28 | 0.5 |

---

## 4. Frozen Test Set Results

### A. Historical Locked 117-Image Test Results
| Candidate | Accuracy | Macro F1 | Malignant Recall | Malignant Precision | Malignant F1 | Malignant FN | Malignant FP | Total Errors |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **V11_MOBILENETV2_A** | 76.92% | 76.57% | 93.75% | 69.77% | 80.00% | 2 | 13 | 27 |
| **V11_MOBILENETV2_B** | 78.63% | 78.45% | 96.88% | 70.45% | 81.58% | 1 | 13 | 25 |
| **V11_MOBILENETV2_C** | 74.36% | 74.22% | 93.75% | 65.22% | 76.92% | 2 | 16 | 30 |
| **V11_MOBILENETV2_D** | 73.50% | 72.64% | 87.50% | 68.29% | 76.71% | 4 | 13 | 31 |
| **V11_MOBILENETV2_E** | 70.94% | 70.30% | 90.62% | 63.04% | 74.36% | 3 | 17 | 34 |

### B. Grouped Generalization Test Results
| Candidate | Accuracy | Macro F1 | Malignant Recall | Malignant Precision | Malignant F1 | Malignant FN | Malignant FP | Total Errors |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **V11_MOBILENETV2_A** | 67.29% | 67.23% | 87.10% | 56.25% | 68.35% | 4 | 21 | 35 |
| **V11_MOBILENETV2_B** | 69.16% | 70.12% | 90.32% | 57.14% | 70.00% | 3 | 21 | 33 |
| **V11_MOBILENETV2_C** | 62.62% | 63.14% | 83.87% | 52.00% | 64.20% | 5 | 24 | 40 |
| **V11_MOBILENETV2_D** | 65.42% | 64.79% | 80.65% | 55.56% | 65.79% | 6 | 20 | 37 |
| **V11_MOBILENETV2_E** | 64.49% | 64.52% | 80.65% | 51.02% | 62.50% | 6 | 24 | 38 |

---

## 5. Historical Test Comparison Across Iterations

| Model Iteration | Accuracy | Macro F1 | Malignant Precision | Malignant Recall | Malignant F1 | Malignant FN | Malignant FP | Total Errors |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **V1 Baseline** | 84.62% | 84.21% | 84.62% | 68.75% | 75.86% | 10 | 4 | 18 |
| **V3 Mild Weights** | 76.92% | 74.50% | 66.67% | 87.50% | 75.68% | 4 | 14 | 27 |
| **V5-B (Active Prod)** | 80.34% | 79.75% | 77.78% | 87.50% | **82.35%** | 4 | 8 | 23 |
| **V6-B** | 81.20% | 80.54% | 78.57% | 87.50% | 82.76% | 4 | 7 | 22 |
| **V9-D (Dual-View Aux)**| 87.18% | 86.46% | 81.25% | 81.25% | 81.25% | 6 | 6 | 15 |
| **V10-D** | 76.92% | 76.57% | 69.77% | **93.75%** | 80.00% | **2** | 13 | 27 |
| **V11-A** | 76.92% | 76.57% | 69.77% | 93.75% | 80.00% | 2 | 13 | 27 |
| **V11-B** | 78.63% | 78.45% | 70.45% | 96.88% | 81.58% | 1 | 13 | 25 |
| **V11-C** | 74.36% | 74.22% | 65.22% | 93.75% | 76.92% | 2 | 16 | 30 |
| **V11-D** | 73.50% | 72.64% | 68.29% | 87.50% | 76.71% | 4 | 13 | 31 |
| **V11-E** | 70.94% | 70.30% | 63.04% | 90.62% | 74.36% | 3 | 17 | 34 |

---

## 6. Grouped Generalization Comparison Across Iterations

| Model Iteration | Accuracy | Macro F1 | Malignant Precision | Malignant Recall | Malignant F1 | Malignant FN | Malignant FP | Total Errors |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **V5-B (Active Prod)** | 70.09% | 69.46% | 62.50% | 80.65% | 70.42% | 6 | 15 | 32 |
| **V6-B** | 71.96% | 71.10% | 64.10% | 80.65% | 71.43% | 6 | 14 | 30 |
| **V9-D (Dual-View Aux)**| 77.57% | 77.84% | 63.64% | 67.74% | 65.62% | 10 | 12 | 24 |
| **V10-D** | 67.29% | 67.23% | 56.25% | **87.10%** | 68.35% | **4** | 21 | 35 |
| **V11-A** | 67.29% | 67.23% | 56.25% | 87.10% | 68.35% | 4 | 21 | 35 |
| **V11-B** | 69.16% | 70.12% | 57.14% | 90.32% | 70.00% | 3 | 21 | 33 |
| **V11-C** | 62.62% | 63.14% | 52.00% | 83.87% | 64.20% | 5 | 24 | 40 |
| **V11-D** | 65.42% | 64.79% | 55.56% | 80.65% | 65.79% | 6 | 20 | 37 |
| **V11-E** | 64.49% | 64.52% | 51.02% | 80.65% | 62.50% | 6 | 24 | 38 |

---

## 7. Model Ranking & Recommendation
- **Best Validation Candidate**: `V11_MOBILENETV2_A`
- **Best Historical Test Candidate**: `V11_MOBILENETV2_B`
- **Best Grouped Test Candidate**: `V11_MOBILENETV2_B`
- **Promotion Decision**: **DO NOT PROMOTE**. Recommend retaining **V5-B** as active production model.

---

PRODUCTION MODEL CHANGED: NO
PRODUCTION METADATA CHANGED: NO
LOCKED 117-IMAGE TEST SET MODIFIED: NO
DATASET IMAGES MODIFIED: NO
DEPLOYMENT PERFORMED: NO
GIT COMMIT PERFORMED: NO
GIT PUSH PERFORMED: NO
