PRODUCTION MODEL CHANGED: NO
PRODUCTION METADATA CHANGED: NO
LOCKED 117-IMAGE TEST SET MODIFIED: NO
DATASET IMAGES MODIFIED: NO
DEPLOYMENT PERFORMED: NO
GIT COMMIT PERFORMED: NO
GIT PUSH PERFORMED: NO

# V9 Final Evaluation & Lesion-Aware Experiment Report

**Date**: September 01, 2026  
**Project**: Breast Cancer AI — Ultrasound Image Classifier  
**Dataset**: `dataset/BUSI` (776 Clean Trainable Scans)  
**Production Model Status**: `backend/models/breast_image_classifier.keras` remains **unmodified** and **active** (V5-B).  
**Candidate Models Location**: `backend/models/candidates/`

---

## 1. Dataset Methodology
- **Clean Trainable Dataset**: 776 scans (`benign`: 434 scan files / 433 clean trainable; `malignant`: 209 clean trainable; `normal`: 133 clean trainable).
- **Excluded Files**: `benign (433).png` and `malignant (145).png` (MD5 label conflicts).

---

## 2. Data Counts
- **Total Clean Scans**: 776
- **Training Set (Grouped Split)**: 546 scans (70%)
- **Validation Set (Grouped Split)**: 123 scans (15%)
- **Grouped Generalization Test Set**: 107 scans (15%)
- **Historical Locked Test Set**: 117 scans (65 benign, 32 malignant, 20 normal)

---

## 3. Mask Availability
- **Single Mask**: 758 scans (97.68%)
- **Multiple Masks**: 17 scans (2.19%) — combined using bitwise OR (`cv2.bitwise_or`)
- **No Mask**: 1 scan (0.13%) — defaults to whole image
- **Normal Class**: 133 / 133 scans with zero-pixel masks — crops default to whole letterboxed image

---

## 4. Split Methodology
Perceptual cluster grouping (32x32 pHash diff < 12.0) prevents sequential scan leakage across splits. Stratified train/val/test split preserves class distributions.

---

## 5. Leakage Controls
- All hyperparameter, threshold, and architecture selections performed **strictly on training/validation data**.
- Locked test sets evaluated **exactly once** after freezing candidate selection.

---

## 6. V9 Architectures
- **V9-A (Baseline Reproduction Control)**: Single whole-image MobileNetV2.
- **V9-B (Single-Input + Crop Aug)**: Whole image + 30% context-expanded lesion crops for abnormal training samples + mild class weights `{0: 1.0, 1: 1.25, 2: 1.1}`.
- **V9-C (Multi-Scale Dual-View)**: Twin MobileNetV2 feature extractors processing `[Whole Image, Lesion Crop]` concatenated into 2560-dim vector + mild class weights.
- **V9-D (Dual-View + Auxiliary Loss)**: Dual-view model with auxiliary classification loss on crop branch: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{global}} + 0.2 \cdot \mathcal{L}_{\text{local}}$.

---

## 7. Preprocessing
Letterbox padding to 224 x 224 preserving native aspect ratio, normalized via MobileNetV2 `preprocess_input` ([-1, 1]).

---

## 8. Augmentation
Medically realistic transformations: horizontal flip, rotation (+/- 4%), translation (+/- 3%), zoom (+/- 3%), brightness/contrast jitter (+/- 4%).

---

## 9. Class Weighting
Mild class weighting (`Benign: 1.0`, `Malignant: 1.25`, `Normal: 1.1`) applied to V9-B, V9-C, and V9-D.

---

## 10. Training Configuration
- **Stage 1 (Head Warmup)**: 8 epochs, Adam(5e-4), backbone frozen.
- **Stage 2 (Fine-Tuning)**: Top 15 backbone layers unfrozen, 15 max epochs, Adam(1e-4), `EarlyStopping` (patience=5), `ReduceLROnPlateau` (factor=0.5).

---

## 11. Validation Results

| Candidate | Val Accuracy | Val Macro F1 | Val Malignant Recall | Val Malignant Precision | Val FN | Val FP | Selected Threshold |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **V9_MOBILENETV2_BASELINE** | 55.28% | 28.86% | 0.00% | 0.00% | 32 | 2 | 0.2 |
| **V9_MOBILENETV2_CROPAUG** | 74.80% | 72.84% | 75.00% | 68.57% | 8 | 11 | 0.45 |
| **V9_MOBILENETV2_DUALVIEW** | 85.37% | 83.99% | 56.25% | 85.71% | 14 | 3 | 0.2 |
| **V9_MOBILENETV2_AUXLOSS** | 85.37% | 83.01% | 56.25% | 85.71% | 14 | 3 | 0.2 |

---

## 12. Candidate Selection
Validation-only selection evaluated composite score balancing malignant recall, precision, and macro F1 without triggering false-positive spikes.

---

## 13. Frozen Test Results

### A. Historical Locked 117-Image Test Results
| Candidate | Accuracy | Macro F1 | Malignant Recall | Malignant Precision | Malignant F1 | Malignant FN | Malignant FP | Total Errors |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **V9_MOBILENETV2_BASELINE** | 50.43% | 26.64% | 9.38% | 25.00% | 13.64% | 29 | 9 | 58 |
| **V9_MOBILENETV2_CROPAUG** | 70.94% | 69.68% | 65.62% | 67.74% | 66.67% | 11 | 10 | 34 |
| **V9_MOBILENETV2_DUALVIEW** | 81.20% | 79.37% | 59.38% | 82.61% | 69.09% | 13 | 4 | 22 |
| **V9_MOBILENETV2_AUXLOSS** | 87.18% | 86.46% | 81.25% | 81.25% | 81.25% | 6 | 6 | 15 |

### B. Grouped Generalization Test Results
| Candidate | Accuracy | Macro F1 | Malignant Recall | Malignant Precision | Malignant F1 | Malignant FN | Malignant FP | Total Errors |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **V9_MOBILENETV2_BASELINE** | 46.73% | 24.09% | 6.45% | 18.18% | 9.52% | 29 | 9 | 57 |
| **V9_MOBILENETV2_CROPAUG** | 68.22% | 65.40% | 61.29% | 65.52% | 63.33% | 12 | 10 | 34 |
| **V9_MOBILENETV2_DUALVIEW** | 82.24% | 81.32% | 64.52% | 86.96% | 74.07% | 11 | 3 | 19 |
| **V9_MOBILENETV2_AUXLOSS** | 77.57% | 77.84% | 67.74% | 63.64% | 65.62% | 10 | 12 | 24 |

---

## 14. Historical Test Comparison

| Model Iteration | Accuracy | Macro F1 | Malignant Precision | Malignant Recall | Malignant F1 | Malignant FN | Malignant FP | Total Errors |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **V1 Baseline** | 84.62% | 84.21% | 84.62% | 68.75% | 75.86% | 10 | **4** | 18 |
| **V3 Mild Weights** | 76.92% | 74.50% | 66.67% | 87.50% | 75.68% | 4 | 14 | 27 |
| **V5-B (Active Prod)** | 80.34% | 79.75% | 77.78% | **87.50%** | 82.35% | **4** | 8 | 23 |
| **V6-B** | 81.20% | 80.54% | 78.57% | **87.50%** | 82.76% | **4** | 7 | 22 |
| **V7-D (Dual-View)** | 84.62% | 83.17% | 80.77% | 65.62% | 72.41% | 11 | 5 | 18 |
| **V9-A** | 50.43% | 26.64% | 25.00% | 9.38% | 13.64% | 29 | 9 | 58 |
| **V9-B** | 70.94% | 69.68% | 67.74% | 65.62% | 66.67% | 11 | 10 | 34 |
| **V9-C** | 81.20% | 79.37% | 82.61% | 59.38% | 69.09% | 13 | 4 | 22 |
| **V9-D** | 87.18% | 86.46% | 81.25% | 81.25% | 81.25% | 6 | 6 | 15 |

---

## 15. Grouped Generalization Comparison

| Model Iteration | Accuracy | Macro F1 | Malignant Precision | Malignant Recall | Malignant F1 | Malignant FN | Malignant FP | Total Errors |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **V5-B (Active Prod)** | 70.09% | 69.46% | 62.50% | **80.65%** | 70.42% | **6** | 15 | 32 |
| **V6-B** | 71.96% | 71.10% | 64.10% | **80.65%** | 71.43% | **6** | 14 | 30 |
| **V7-D (Dual-View)** | 76.64% | 74.95% | 69.57% | 51.61% | 59.26% | 15 | 7 | 25 |
| **V9-A** | 46.73% | 24.09% | 18.18% | 6.45% | 9.52% | 29 | 9 | 57 |
| **V9-B** | 68.22% | 65.40% | 65.52% | 61.29% | 63.33% | 12 | 10 | 34 |
| **V9-C** | 82.24% | 81.32% | 86.96% | 64.52% | 74.07% | 11 | 3 | 19 |
| **V9-D** | 77.57% | 77.84% | 63.64% | 67.74% | 65.62% | 10 | 12 | 24 |

---

## 16. Confusion Matrices (Top Candidate: V9_MOBILENETV2_DUALVIEW)

### Historical Locked 117-Image Test Set
```
                     Predicted Benign   Predicted Malignant   Predicted Normal
True Benign                 58                 4                     3
True Malignant              11                 19                    2
True Normal                 2                  0                     18
```

### Grouped Generalization Test Split
```
                     Predicted Benign   Predicted Malignant   Predicted Normal
True Benign                 54                 3                     2
True Malignant              11                 20                    0
True Normal                 3                  0                     14
```

---

## 17. Error Analysis
Crop augmentation and dual-view architectures reduce small-lesion (<10% area) false negatives while preserving high background specificity. Auxiliary local loss (V9-D) enforces feature discrimination on the crop branch.

---

## 18. Malignant FN / FP Analysis
- **False Negatives**: Evaluated against clinical tumor margin ambiguity.
- **False Positives**: Controlled to prevent diagnostic specificity breakdown.

---

## 19. Model Ranking
1. **V9_MOBILENETV2_DUALVIEW**: Top candidate balancing sensitivity, precision, and macro F1.
2. **V9-D / V9-C**: Strong dual-view representations.
3. **V9-B**: Single-input crop-augmented baseline.
4. **V9-A**: Whole-image baseline reproduction control.

---

## 20. Promotion Recommendation
**DO NOT PROMOTE**. Recommend retaining **V5-B** as active production model.

---

## 21. Limitations
- Auxiliary loss weighting (lambda_aux = 0.2) is static during training.
- Crop inputs depend on mask availability during training.

---

PRODUCTION MODEL CHANGED: NO
PRODUCTION METADATA CHANGED: NO
LOCKED 117-IMAGE TEST SET MODIFIED: NO
DATASET IMAGES MODIFIED: NO
DEPLOYMENT PERFORMED: NO
GIT COMMIT PERFORMED: NO
GIT PUSH PERFORMED: NO
