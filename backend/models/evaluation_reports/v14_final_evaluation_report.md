PRODUCTION MODEL CHANGED: NO
PRODUCTION METADATA CHANGED: NO
LOCKED 117-IMAGE TEST SET MODIFIED: NO
DATASET IMAGES MODIFIED: NO
DEPLOYMENT PERFORMED: NO
GIT COMMIT PERFORMED: NO
GIT PUSH PERFORMED: NO

# V14 Minimal Frozen Ensemble (V11-B + V13-B0) Final Report

**Date**: September 02, 2026  
**Project**: Breast Cancer AI — Ultrasound Image Classifier  
**Dataset**: `dataset/BUSI` (776 Clean Trainable Scans)  
**Production Model Status**: `backend/models/breast_image_classifier.keras` remains **unmodified** and **active** (V5-B).  
**Ensemble Components**: Frozen Candidate V11-B (`v11_mobilenetv2_b.keras`) + Frozen Candidate V13-B0 (`v13_efficientnet_b0.keras`)

---

## 1. Executive Summary & V14 Objective
The V14 experiment tested a simple frozen probability ensemble of **V11-B** (MobileNetV2, high sensitivity) and **V13-B0** (EfficientNetB0, high precision) to determine whether combining their complementary predictions improves the grouped malignant operating point.

### Selected Policy (Fitted on Grouped Validation Data ONLY):
- **Selected V11-B Weight ($w^*$)**: `0.50`
- **Selected V13-B0 Weight ($1 - w^*$)**: `0.50`
- **Selected Malignant Threshold ($t^*$)**: `0.58`

---

## 2. Grouped Generalization Test Comparison (Primary Benchmark)

| Model / Ensemble Iteration | Architecture / Strategy | Accuracy | Macro F1 | Malignant Recall | Malignant Precision | Malignant F1 | Malignant FN | Malignant FP | Total Errors |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **V5-B (Active Prod)** | MobileNetV2 (Control) | 70.09% | 69.46% | 80.65% | 62.50% | 70.42% | 6 | 15 | 32 |
| **V11-B** | MobileNetV2 (50% Aug) | 69.16% | 70.12% | **90.32%** | 57.14% | 70.00% | **3** | 21 | 33 |
| **V13-B0** | EfficientNetB0 | **80.37%** | 75.82% | 67.74% | **84.00%** | 75.00% | 10 | **4** | **21** |
| **V14 Reference (50/50, $t=0.50$)** | Frozen Ensemble (50/50) | 79.44% | **76.84%** | 80.65% | 73.53% | **76.92%** | 6 | 9 | 22 |
| **V14 Selected ($w=0.50, t=0.58$)**| **Frozen Ensemble (Selected)** | 80.37% | 79.12% | 83.87% | 74.29% | 78.79% | 5 | 9 | 21 |

---

## 3. Locked Historical 117-Image Test Comparison (Frozen Benchmark)

| Model / Ensemble Iteration | Architecture / Strategy | Accuracy | Macro F1 | Malignant Recall | Malignant Precision | Malignant F1 | Malignant FN | Malignant FP | Total Errors |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **V5-B (Active Prod)** | MobileNetV2 (Control) | 80.34% | 79.75% | 87.50% | 77.78% | 82.35% | 4 | 8 | 23 |
| **V11-B** | MobileNetV2 (50% Aug) | 78.63% | 78.45% | **96.88%** | 70.45% | 81.58% | **1** | 13 | 25 |
| **V13-B0** | EfficientNetB0 | 86.32% | 84.00% | 78.12% | **89.29%** | 83.33% | 7 | **3** | 16 |
| **V14 Reference (50/50, $t=0.50$)** | Frozen Ensemble (50/50) | 86.32% | 85.06% | 87.50% | 84.85% | 86.15% | 4 | 5 | 16 |
| **V14 Selected ($w=0.50, t=0.58$)**| **Frozen Ensemble (Selected)** | 86.32% | 85.44% | 84.38% | 87.10% | 85.71% | 5 | 4 | 16 |

---

## 4. Per-Class Performance Breakdown (V14 Selected Policy)

### Grouped Generalization Test:
- **Benign**: Precision = 88.46%, Recall = 77.97%, F1 = 82.88%
- **Malignant**: Precision = 74.29%, Recall = 83.87%, F1 = 78.79%
- **Normal**: Precision = 70.00%, Recall = 82.35%, F1 = 75.68%

### Locked Historical Test Set:
- **Benign**: Precision = 91.67%, Recall = 84.62%, F1 = 88.00%
- **Malignant**: Precision = 87.10%, Recall = 84.38%, F1 = 85.71%
- **Normal**: Precision = 73.08%, Recall = 95.00%, F1 = 82.61%

---

## 5. Model Recommendation & Verdict

- **Promotion Recommendation**: **DO NOT PROMOTE**. Recommend retaining **V5-B** as active production model.

---

PRODUCTION MODEL CHANGED: NO
PRODUCTION METADATA CHANGED: NO
LOCKED 117-IMAGE TEST SET MODIFIED: NO
DATASET IMAGES MODIFIED: NO
DEPLOYMENT PERFORMED: NO
GIT COMMIT PERFORMED: NO
GIT PUSH PERFORMED: NO
