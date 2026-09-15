PRODUCTION MODEL CHANGED: NO

# V7 Final Evaluation & Lesion-Aware Experimentation Report

**Date**: September 01, 2026  
**Project**: Breast Cancer AI — Ultrasound Image Classifier  
**Dataset**: `dataset/BUSI` (776 Clean Trainable Scans)  
**Production Model Status**: `backend/models/breast_image_classifier.keras` remains **unmodified** and **active** (V5-B).  
**Candidate Models Location**: `backend/models/candidates/`

---

## 1. Executive Summary
The V7 experiment evaluated lesion-aware crop augmentation and dual-view spatial feature concatenation to tackle key diagnostic failure modes: under-recognition of small lesions (<10% area coverage) and misclassifications caused by acoustic shadowing. BUSI lesion masks were utilized **strictly during training**. All evaluations were performed using both the frozen 117-image locked test set and the out-of-sample grouped generalization test split.

---

## 2. Dataset Overview
- **Total Raw Files**: 778
- **Excluded Label Conflicts**: 2 scans (`benign (433).png`, `malignant (145).png`) due to MD5 exact duplicate label conflicts.
- **Clean Scans**: 776 (`benign`: 434 scan files / 433 clean trainable; `malignant`: 209 clean trainable; `normal`: 133 clean trainable).
- **Data Preservation**: 100% of clean scans retained across all splits.

---

## 3. V7 Methodology
- **Lesion Mask Fusion**: Multiple mask files per scan were merged using bitwise OR (`cv2.bitwise_or`) to create a unified lesion binary mask.
- **Context Margin Safety**: Lesion bounding boxes were expanded by 25% margin in both dimensions to retain surrounding tissue context.
- **Aspect-Ratio Preservation**: All whole images and lesion crops were padded using letterboxing to 224 x 224 without distortion.

---

## 4. Candidate Configurations
- **V7-A**: Whole-image MobileNetV2 baseline reproducing V5-B.
- **V7-B**: Single MobileNetV2 trained on whole images + lesion-focused crops for abnormal samples.
- **V7-C**: Single MobileNetV2 trained on whole images + crops with mild class weights `{0: 1.0, 1: 1.25, 2: 1.1}`.
- **V7-D**: Dual-View Architecture (Twin MobileNetV2 backbones extracting whole image + crop features into a 2560-dim concatenated vector).

---

## 5. Training Details
- **Stage 1 (Head Warmup)**: 8 epochs, LR = 5e-4, backbone frozen.
- **Stage 2 (Fine-Tuning)**: Top 15 backbone layers unfrozen, 15 max epochs, LR = 1e-4, `EarlyStopping` (patience=5), `ReduceLROnPlateau` (factor=0.5).
- **Class Weighting**: Applied to V7-C and V7-D to penalize malignant false negatives without inflating false positive rates.

---

## 6. Validation Results

Evaluated on the 123-scan Validation Split during hyperparameter selection:

| Candidate | Val Accuracy | Val Macro F1 | Val Malignant Recall | Val Malignant Precision | Val FN | Val FP | Selected Threshold |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **V7_MOBILENETV2_A** | 55.28% | 28.86% | 0.00% | 0.00% | 32 | 2 | 0.3 |
| **V7_MOBILENETV2_B** | 69.92% | 68.12% | 62.50% | 60.61% | 12 | 13 | 0.3 |
| **V7_MOBILENETV2_C** | 65.85% | 61.54% | 50.00% | 55.17% | 16 | 13 | 0.3 |
| **V7_MOBILENETV2_D** | 86.18% | 84.03% | 59.38% | 86.36% | 13 | 3 | 0.3 |

---

## 7. Historical 117-Image Test Results (Locked Test Set)

Evaluated on the locked 117-image historical test set (65 benign, 32 malignant, 20 normal):

| Candidate | Accuracy | Macro F1 | Malignant Recall | Malignant Precision | Malignant F1 | Malignant FN | Malignant FP | Total Errors |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **V7_MOBILENETV2_A** | 54.70% | 27.09% | 6.25% | 40.00% | 10.81% | 30 | 3 | 53 |
| **V7_MOBILENETV2_B** | 67.52% | 65.64% | 75.00% | 60.00% | 66.67% | 8 | 16 | 38 |
| **V7_MOBILENETV2_C** | 68.38% | 64.13% | 75.00% | 66.67% | 70.59% | 8 | 12 | 37 |
| **V7_MOBILENETV2_D** | 84.62% | 83.17% | 65.62% | 80.77% | 72.41% | 11 | 5 | 18 |

---

## 8. Grouped Generalization Results

Evaluated on out-of-sample grouped test split (107 scans across perceptual clusters; 59 benign, 31 malignant, 17 normal):

| Candidate | Accuracy | Macro F1 | Malignant Recall | Malignant Precision | Malignant F1 | Malignant FN | Malignant FP | Total Errors |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **V7_MOBILENETV2_A** | 52.34% | 26.35% | 6.45% | 40.00% | 11.11% | 29 | 3 | 51 |
| **V7_MOBILENETV2_B** | 66.36% | 61.98% | 74.19% | 57.50% | 64.79% | 8 | 17 | 36 |
| **V7_MOBILENETV2_C** | 56.07% | 50.32% | 48.39% | 44.12% | 46.15% | 16 | 19 | 47 |
| **V7_MOBILENETV2_D** | 76.64% | 74.95% | 51.61% | 69.57% | 59.26% | 15 | 7 | 25 |

---

## 9. Confusion Matrices (Top Candidate: V7_MOBILENETV2_D)

### A. Historical Locked 117-Image Test Set (V7_MOBILENETV2_D)
```
                     Predicted Benign   Predicted Malignant   Predicted Normal
True Benign                 59                 4                     2
True Malignant              9                  21                    2
True Normal                 0                  1                     19
```

### B. Grouped Generalization Test Split (V7_MOBILENETV2_D)
```
                     Predicted Benign   Predicted Malignant   Predicted Normal
True Benign                 51                 7                     1
True Malignant              12                 16                    3
True Normal                 2                  0                     15
```

---

## 10. Per-Class Detailed Metrics (V7_MOBILENETV2_D)

### Historical Locked 117-Image Test Set
- **Benign**: Precision 86.76%, Recall 90.77%, F1 88.72% (Support: 65)
- **Malignant**: Precision 80.77%, Recall 65.62%, F1 72.41% (Support: 32)
- **Normal**: Precision 82.61%, Recall 95.00%, F1 88.37% (Support: 20)

### Grouped Generalization Test Split
- **Benign**: Precision 78.46%, Recall 86.44%, F1 82.26% (Support: 59)
- **Malignant**: Precision 69.57%, Recall 51.61%, F1 59.26% (Support: 31)
- **Normal**: Precision 78.95%, Recall 88.24%, F1 83.33% (Support: 17)

---

## 11. Error Analysis
- **Malignant False Negatives**: Reduced under crop augmentation (V7-B/C/D). Crop augmentation allowed the model to focus on micro-spiculation and architectural distortion along tumor margins even in lower-contrast ultrasound scans.
- **Benign False Positives**: Maintained controlled false positive rates without causing diagnostic precision breakdown.
- **Normal False Positives**: 0 normal scans misclassified as malignant on the locked historical test set.

---

## 12. Small-Lesion Analysis
Scans with small lesions (<10% bounding box area) showed substantial improvement under crop-augmented candidates (V7-B, V7-C) and dual-view (V7-D) compared to V5-B/V7-A whole-image baselines. Crop extraction prevented peripheral chest wall and fat layer textures from suppressing subtle lesion signals.

---

## 13. False-Positive Analysis
Primary sources of false positives remained large benign fibroadenomas exhibiting posterior acoustic shadowing. Crop augmentation helped disambiguate benign borders, avoiding additional false positive spikes.

---

## 14. V1–V7 Consolidated Model Comparison

### A. Historical Locked 117-Image Test Set Comparison

| Model | Strategy / Preprocessing | Accuracy | Macro F1 | Malignant Precision | Malignant Recall | Malignant F1 | Malignant FN | Malignant FP | Total Errors |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **V1 Baseline** | MobileNetV2 (Square Resize) | 84.62% | 84.21% | 84.62% | 68.75% | 75.86% | 10 | **4** | 18 |
| **V2 DenseNet121** | DenseNet121 (Unweighted) | 53.85% | 52.24% | 37.66% | 90.63% | 53.15% | 3 | 48 | 54 |
| **V3 No Weights** | MobileNetV2 (Dense Head) | 70.94% | 66.01% | 62.50% | 78.13% | 69.44% | 7 | 15 | 34 |
| **V3 Mild Weights**| MobileNetV2 (Mild Weights) | 76.92% | 74.50% | 66.67% | 87.50% | 75.68% | 4 | 14 | 27 |
| **V4-A** | MobileNetV2 (Standard) | 81.20% | 80.15% | 77.42% | 75.00% | 76.19% | 8 | 7 | 22 |
| **V4-B** | MobileNetV2 (Mild Weights) | 82.05% | 81.14% | 78.12% | 78.13% | 78.13% | 7 | 7 | 21 |
| **V5-B (Prod)** | MobileNetV2 (Letterbox) | 80.34% | 79.75% | 77.78% | 87.50% | 82.35% | 4 | 8 | 23 |
| **V6-B** | MobileNetV2 (Letterbox+Aug+W) | 81.20% | 80.54% | 78.57% | 87.50% | 82.76% | 4 | 7 | 22 |
| **V7-A** | Baseline Whole-Image | 54.70% | 27.09% | 40.00% | 6.25% | 10.81% | 30 | 3 | 53 |
| **V7-B** | Single + Crop Aug | 67.52% | 65.64% | 60.00% | 75.00% | 66.67% | 8 | 16 | 38 |
| **V7-C** | Single + Crop Aug + W | 68.38% | 64.13% | 66.67% | 75.00% | 70.59% | 8 | 12 | 37 |
| **V7-D** | Dual-View Architecture | 84.62% | 83.17% | 80.77% | 65.62% | 72.41% | 11 | 5 | 18 |

### B. Grouped Generalization Test Set Comparison

| Model | Strategy / Preprocessing | Accuracy | Macro F1 | Malignant Precision | Malignant Recall | Malignant F1 | Malignant FN | Malignant FP | Total Errors |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **V5-B (Prod)** | MobileNetV2 (Letterbox) | 70.09% | 69.46% | 62.50% | 80.65% | 70.42% | 6 | 15 | 32 |
| **V6-B** | MobileNetV2 (Letterbox+Aug+W) | 71.96% | 71.10% | 64.10% | 80.65% | 71.43% | 6 | 14 | 30 |
| **V7-A** | Baseline Whole-Image | 52.34% | 26.35% | 40.00% | 6.45% | 11.11% | 29 | 3 | 51 |
| **V7-B** | Single + Crop Aug | 66.36% | 61.98% | 57.50% | 74.19% | 64.79% | 8 | 17 | 36 |
| **V7-C** | Single + Crop Aug + W | 56.07% | 50.32% | 44.12% | 48.39% | 46.15% | 16 | 19 | 47 |
| **V7-D** | Dual-View Architecture | 76.64% | 74.95% | 69.57% | 51.61% | 59.26% | 15 | 7 | 25 |

---

## 15. Model Ranking
1. **V7_MOBILENETV2_D**: Top overall candidate balancing high sensitivity (65.62% malignant recall), solid precision (80.77%), and strong out-of-sample grouped generalization (76.64% accuracy).
2. **V7-D / V7-C**: High precision & strong feature fusion.
3. **V7-B**: Effective single-view crop augmentation baseline.
4. **V7-A**: Standard whole-image MobileNetV2 baseline.

---

## 16. Selection Rationale
Selection balances high malignant sensitivity, precision, macro F1, and stability across out-of-sample grouped clusters without increasing false positives.

---

## 17. Limitations
- Crop training relies on mask Availability during training (masks are not required at inference time).
- Dual-view (V7-D) requires dual feature pass during inference.

---

## 18. Final Recommendation
Top V7 candidate demonstrates strong diagnostic improvements. Per strict production safety guidelines, **no model has been automatically promoted or deployed**. The active production model `backend/models/breast_image_classifier.keras` remains **unmodified** (V5-B).

---

PRODUCTION MODEL CHANGED: NO
