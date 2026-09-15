PRODUCTION MODEL CHANGED: NO
PRODUCTION METADATA CHANGED: NO
LOCKED 117-IMAGE TEST SET MODIFIED: NO
DATASET IMAGES MODIFIED: NO
DEPLOYMENT PERFORMED: NO
GIT COMMIT PERFORMED: NO
GIT PUSH PERFORMED: NO

# V13 EfficientNet-B0 Backbone Generalization Report

**Date**: September 02, 2026  
**Project**: Breast Cancer AI — Ultrasound Image Classifier  
**Dataset**: `dataset/BUSI` (776 Clean Trainable Scans)  
**Production Model Status**: `backend/models/breast_image_classifier.keras` remains **unmodified** and **active** (V5-B).  
**Evaluated Backbone**: EfficientNetB0 (`backend/models/candidates/v13_efficientnet_b0.keras`)

---

## 1. Executive Summary & Production Status
- **V13 Hypothesis**: Test whether replacing MobileNetV2 (2.26M params) with a stronger pretrained image backbone (**EfficientNetB0**, 4.05M params with Squeeze-and-Excitation channel attention) can improve malignant generalization ($\ge 85\%$ Malignant Recall, $\ge 60\%$ Precision, $\ge 70\%$ Accuracy).

---

## 2. Grouped Generalization Test Comparison Across Iterations

| Model Iteration | Backbone Architecture | Accuracy | Macro F1 | Malignant Recall | Malignant Precision | Malignant F1 | Malignant FN | Malignant FP | Total Errors |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **V5-B (Active Prod)** | MobileNetV2 | 70.09% | 69.46% | 80.65% | 62.50% | 70.42% | 6 | 15 | 32 |
| **V11-B** | MobileNetV2 (50% Aug) | 69.16% | 70.12% | 90.32% | 57.14% | 70.00% | 3 | 21 | 33 |
| **V13-B0** | **EfficientNetB0** | 80.37% | 75.82% | 67.74% | 84.00% | 75.00% | 10 | 4 | 21 |

---

## 3. Historical Locked 117-Image Test Comparison

| Model Iteration | Backbone Architecture | Accuracy | Macro F1 | Malignant Recall | Malignant Precision | Malignant F1 | Malignant FN | Malignant FP | Total Errors |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **V5-B (Active Prod)** | MobileNetV2 | 80.34% | 79.75% | 87.50% | 77.78% | 82.35% | 4 | 8 | 23 |
| **V11-B** | MobileNetV2 (50% Aug) | 78.63% | 78.45% | 96.88% | 70.45% | 81.58% | 1 | 13 | 25 |
| **V13-B0** | **EfficientNetB0** | 86.32% | 84.00% | 78.12% | 89.29% | 83.33% | 7 | 3 | 16 |

---

## 4. Per-Class Performance Breakdown (V13-B0)

### Grouped Generalization Test:
- **Benign**: Precision = 85.48%, Recall = 89.83%, F1 = 87.60%
- **Malignant**: Precision = 84.00%, Recall = 67.74%, F1 = 75.00%
- **Normal**: Precision = 60.00%, Recall = 70.59%, F1 = 64.86%

### Locked Historical Test Set:
- **Benign**: Precision = 85.92%, Recall = 93.85%, F1 = 89.71%
- **Malignant**: Precision = 89.29%, Recall = 78.12%, F1 = 83.33%
- **Normal**: Precision = 83.33%, Recall = 75.00%, F1 = 78.95%

---

## 5. Model Recommendation & Promotion Decision

- **Selected Validation Threshold**: `0.58`
- **Promotion Recommendation**: **DO NOT PROMOTE**. Recommend retaining **V5-B** as active production model.

---

PRODUCTION MODEL CHANGED: NO
PRODUCTION METADATA CHANGED: NO
LOCKED 117-IMAGE TEST SET MODIFIED: NO
DATASET IMAGES MODIFIED: NO
DEPLOYMENT PERFORMED: NO
GIT COMMIT PERFORMED: NO
GIT PUSH PERFORMED: NO
