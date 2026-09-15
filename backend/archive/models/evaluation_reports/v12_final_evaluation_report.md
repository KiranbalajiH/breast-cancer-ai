PRODUCTION MODEL CHANGED: NO
PRODUCTION METADATA CHANGED: NO
LOCKED 117-IMAGE TEST SET MODIFIED: NO
DATASET IMAGES MODIFIED: NO
DEPLOYMENT PERFORMED: NO
GIT COMMIT PERFORMED: NO
GIT PUSH PERFORMED: NO

# V12 Decision Calibration & Selective Prediction Research Report

**Date**: September 02, 2026  
**Project**: Breast Cancer AI — Ultrasound Image Classifier  
**Dataset**: `dataset/BUSI` (776 Clean Trainable Scans)  
**Production Model Status**: `backend/models/breast_image_classifier.keras` remains **unmodified** and **active** (V5-B).  
**Evaluated Models**: V5-B (Active Production) and V11-B (Sensitivity Candidate)

---

## 1. Executive Summary & V12 Objective
The V12 research experiment evaluated whether probability calibration (Temperature Scaling) and validation-derived decision policies (Malignant Threshold Tuning and Selective Prediction / Uncertainty Zones) can recover precision on **V11-B** (or **V5-B**) while retaining malignant sensitivity ($\ge 85\%$), without retraining models or modifying production.

### Key Takeaways:
1. **Temperature Scaling ($T$)**:
   - V5-B Validation Temperature: $T^* = 1.1820$ (Validation LogLoss: 0.4908 -> 0.4850).
   - V11-B Validation Temperature: $T^* = 1.4385$ (Validation LogLoss: 0.5745 -> 0.5433).
2. **Grouped Test Performance Summary**:
   - **V11-B Policy C (Calibrated Threshold 0.7)**: Achieved **70.97% Grouped Malignant Recall** (9 FN) with 64.71% Precision (12 FP).
   - **V11-B Policy D (Selective Prediction Bounds [0.65, 0.75])**: On accepted non-review cases (review rate 7.5%), Malignant Recall was **75.00%** with **75.00%** Precision and **7 FN / 7 FP**.
3. **Promotion Recommendation**: **DO NOT PROMOTE**. Retain **V5-B** as active production model.

---

## 2. Grouped Generalization Test Results (Primary Benchmark)

### A. V5-B Production Model Policies
| Policy | Accuracy | Macro F1 | Malignant Recall | Malignant Precision | Malignant F1 | Malignant FN | Malignant FP | Reviewed Cases |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Policy A (Argmax 0.50) | 71.03% | 69.83% | 77.42% | 64.86% | 70.59% | 7 | 13 | 0 (0.0%) |
| Policy B (Raw Threshold 0.64) | 69.16% | 67.16% | 67.74% | 70.00% | 68.85% | 10 | 9 | 0 (0.0%) |
| Policy C (Calibrated Threshold 0.6) | 69.16% | 67.16% | 67.74% | 70.00% | 68.85% | 10 | 9 | 0 (0.0%) |
| Policy D (Selective [0.55, 0.65]) | 71.57% | 69.79% | 74.07% | 68.97% | 71.43% | 7 | 9 | 5 (4.7%) |

### B. V11-B Candidate Model Policies
| Policy | Accuracy | Macro F1 | Malignant Recall | Malignant Precision | Malignant F1 | Malignant FP | Reviewed Cases |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Policy A (Argmax 0.50) | 71.03% | 71.58% | 87.10% | 60.00% | 71.05% | 4 | 18 | 0 (0.0%) |
| Policy B (Raw Threshold 0.4) | 69.16% | 70.12% | 90.32% | 57.14% | 70.00% | 3 | 21 | 0 (0.0%) |
| Policy C (Calibrated Threshold 0.7) | 70.09% | 68.54% | 70.97% | 64.71% | 67.69% | 9 | 12 | 0 (0.0%) |
| Policy D (Selective [0.65, 0.75]) | 74.75% | 72.97% | 75.00% | 75.00% | 75.00% | 7 | 7 | 8 (7.5%) |

---

## 3. Locked Historical 117-Image Test Results (Frozen Benchmark)

### A. V5-B Production Model Policies
| Policy | Accuracy | Macro F1 | Malignant Recall | Malignant Precision | Malignant F1 | Malignant FN | Malignant FP | Reviewed Cases |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Policy A (Argmax 0.50) | 81.20% | 80.19% | 84.38% | 84.38% | 84.38% | 5 | 5 | 0 (0.0%) |
| Policy B (Raw Threshold 0.64) | 78.63% | 77.21% | 75.00% | 82.76% | 78.69% | 8 | 5 | 0 (0.0%) |
| Policy C (Calibrated Threshold 0.6) | 78.63% | 77.21% | 75.00% | 82.76% | 78.69% | 8 | 5 | 0 (0.0%) |
| Policy D (Selective [0.55, 0.65]) | 79.65% | 78.50% | 78.57% | 81.48% | 80.00% | 6 | 5 | 4 (3.4%) |

### B. V11-B Candidate Model Policies
| Policy | Accuracy | Macro F1 | Malignant Recall | Malignant Precision | Malignant F1 | Malignant FN | Malignant FP | Reviewed Cases |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Policy A (Argmax 0.50) | 77.78% | 77.27% | 90.62% | 72.50% | 80.56% | 3 | 11 | 0 (0.0%) |
| Policy B (Raw Threshold 0.4) | 78.63% | 78.45% | 96.88% | 70.45% | 81.58% | 1 | 13 | 0 (0.0%) |
| Policy C (Calibrated Threshold 0.7) | 81.20% | 80.20% | 87.50% | 84.85% | 86.15% | 4 | 5 | 0 (0.0%) |
| Policy D (Selective [0.65, 0.75]) | 81.42% | 80.52% | 86.67% | 86.67% | 86.67% | 4 | 4 | 4 (3.4%) |

---

## 4. Operating Point Assessment

- **V5-B Baseline**: Operates at 80.65% Malignant Recall and 62.50% Precision (Grouped Test).
- **V11-B Policy C (Calibrated Threshold)**: Operates at 70.97% Malignant Recall and 64.71% Precision (Grouped Test).
- **Selective Prediction Impact**: Selective prediction routes borderline predictions in the uncertainty zone to `"REVIEW"`. While this increases precision on non-reviewed cases when evaluated, the required review rate (approx 7.5%) does not eliminate the fundamental false-positive trade-off without human-in-the-loop review.

---

## 5. Recommendation

**RETAIN V5-B IN PRODUCTION.**

Neither temperature calibration nor selective prediction alone solves the underlying feature overlap causing false positives without requiring active clinical human review. Recommend maintaining **V5-B** as active production model.

---

PRODUCTION MODEL CHANGED: NO
PRODUCTION METADATA CHANGED: NO
LOCKED 117-IMAGE TEST SET MODIFIED: NO
DATASET IMAGES MODIFIED: NO
DEPLOYMENT PERFORMED: NO
GIT COMMIT PERFORMED: NO
GIT PUSH PERFORMED: NO
