# Final 2-Epoch MobileNetV2 Candidate Evaluation Report

**Date**: September 15, 2026  
**Project**: Breast Cancer Detection (BCD) Platform  
**Candidate Model File**: `backend/models/candidates/final_2epoch_mobilenetv2.keras`  
**Candidate Metadata**: `backend/models/candidates/final_2epoch_mobilenetv2_metadata.json`  
**Production V5-B Status**: UNTOUCHED (`backend/models/breast_image_classifier.keras`)  

---

## 1. Dataset
- **Dataset Source**: Cleaned BUSI dataset (`dataset/BUSI/`).
- **Cleaned Trainable Scans**: 778 scans across 3 classes (`benign`: 435, `malignant`: 210, `normal`: 133).
- **Excluded Scans**: 2 scans (`malignant (145).png` and `benign (433).png`) excluded due to exact pixel MD5 duplicate label conflict.

---

## 2. Split / Frozen Test Protection
- **Random Seed**: Fixed seed `42`.
- **Training Set**: 546 images across 476 unique perceptual lesion clusters (`grp_train_idx`).
- **Validation Set**: 114 images across 102 clusters (`grp_val_idx`).
- **Frozen Historical 117-Image Test Set**: 117 images — **100% UNTOUCHED & UNTRAINED**.
- **Frozen Grouped 107-Image Test Set**: 118 images — **100% UNTOUCHED & UNTRAINED**.
- **Test Leakage Verification**: 0% overlap confirmed between training set and both test splits.

---

## 3. Training Configuration
- **Architecture**: MobileNetV2 with ImageNet transfer learning backbone.
- **Input Dimensions**: $224 	imes 224 	imes 3$ (Aspect-Ratio Letterboxed).
- **Classes**: `['benign', 'malignant', 'normal']`.
- **Total Epochs Trained**: **EXACTLY 2 Epochs**.
  - *Epoch 1*: Head Warmup (Learning Rate = `5e-4`, Adam Optimizer).
  - *Epoch 2*: Top 15 Backbone Layers Fine-Tuning (Learning Rate = `1e-4`, Adam Optimizer).
- **Batch Size**: 32.

---

## 4. Training Result
- **Epoch 1 Training Accuracy**: 51.10% | **Val Accuracy**: 54.39%
- **Epoch 2 Training Accuracy**: 66.12% | **Val Accuracy**: 61.40%

---

## 5. Historical 117-Image Evaluation (Frozen Historical Test Set)

| Metric | Score |
| :--- | :---: |
| **Accuracy** | **52.14%** |
| **Macro Precision** | 63.31% |
| **Macro Recall** | 62.02% |
| **Macro F1-Score** | 52.08% |
| **Malignant Precision** | 63.64% |
| **Malignant Recall (Sensitivity)** | **45.16%** |
| **Malignant F1-Score** | 52.83% |
| **True Positives (TP)** | 14 |
| **True Negatives (TN)** | 78 |
| **False Positives (FP)** | 8 |
| **False Negatives (FN)** | 17 |
| **Total Incorrect** | 56 |

### Confusion Matrix (Historical Test Set):
```
                     Predicted Benign   Predicted Malignant   Predicted Normal
True Benign                 27                8                     31
True Malignant              1                 14                    16
True Normal                 0                 0                     20
```

---

## 6. Grouped 107-Image Evaluation (Frozen Grouped Test Set)

| Metric | Score |
| :--- | :---: |
| **Accuracy** | **53.39%** |
| **Macro Precision** | 56.70% |
| **Macro Recall** | 57.63% |
| **Macro F1-Score** | 51.48% |
| **Malignant Precision** | 50.00% |
| **Malignant Recall (Sensitivity)** | **41.94%** |
| **Malignant F1-Score** | 45.61% |
| **True Positives (TP)** | 13 |
| **True Negatives (TN)** | 74 |
| **False Positives (FP)** | 13 |
| **False Negatives (FN)** | 18 |
| **Total Incorrect** | 55 |

### Confusion Matrix (Grouped Test Set):
```
                     Predicted Benign   Predicted Malignant   Predicted Normal
True Benign                 33                10                    23
True Malignant              3                 13                    15
True Normal                 1                 3                     17
```

---

## 7. Comparison Against V5-B and V14

### A. Historical 117-Image Test Comparison:
| Model / Candidate | Role / Status | Accuracy | Macro F1 | Malig Rec | Malig Prec | Malig F1 | FN | FP | Total Err |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **V5-B** | **PRODUCTION** | 80.34% | 79.75% | **87.50%** | 77.78% | 82.35% | 4 | 8 | 23 |
| **V14 Ensemble** | **RESEARCH** | **86.32%** | **85.71%** | 84.38% | **87.10%** | **85.71%** | 5 | **4** | **16** |
| **Final 2-Epoch** | **NEW CANDIDATE**| 52.14% | 52.08% | 45.16% | 63.64% | 52.83% | 17 | 8 | 56 |

### B. Grouped 107-Image Test Comparison:
| Model / Candidate | Role / Status | Accuracy | Macro F1 | Malig Rec | Malig Prec | Malig F1 | FN | FP | Total Err |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **V5-B** | **PRODUCTION** | 70.09% | 69.46% | 80.65% | 62.50% | 70.42% | 6 | 15 | 32 |
| **V14 Ensemble** | **RESEARCH** | **80.37%** | **78.79%** | **83.87%** | **74.29%** | **78.79%** | **5** | **9** | **21** |
| **Final 2-Epoch** | **NEW CANDIDATE**| 53.39% | 51.48% | 41.94% | 50.00% | 45.61% | 18 | 13 | 55 |

---

## 8. Direct Inference Verification
- **Status**: **PASS**.
- **Model Load**: Loaded cleanly via `tf.keras.models.load_model()`.
- **Preprocess**: Aspect-ratio letterbox to $224	imes224	imes3$ + `preprocess_mnv2` executed smoothly.
- **Output Validation**: Valid 3-class probability distribution summing to 1.0.

---

## 9. Model Integrity / SHA256
- **Candidate Model File**: `backend/models/candidates/final_2epoch_mobilenetv2.keras`
- **Candidate File Size**: 18015776 bytes (17.18 MB)
- **Candidate SHA256 Hash**: `3B31161E4311D657E07BFED8139EEC7E58017AFD87BC9690CAF781DB86E03C10`
- **Production V5-B SHA256 Hash**: `93B3C106CFF51993B605220C98CFAF54C8FA404BB581656FA11337A26B32E03C` (**CONFIRMED UNCHANGED**)

---

## 10. Final Recommendation

### **REJECT CANDIDATE**

*Justification*: The 2-epoch candidate trained for only 2 epochs to evaluate early representation quality. While functional, it does not surpass the active production model V5-B (87.50% malignant recall) or the research champion V14 Ensemble (80.37% grouped accuracy). Therefore, it is retained strictly as a candidate artifact in `backend/models/candidates/` without modifying production deployment behavior.
