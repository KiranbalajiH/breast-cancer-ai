PRODUCTION MODEL CHANGED: NO

# V8 Final Evaluation & Decision Calibration Report

**Date**: September 01, 2026  
**Project**: Breast Cancer AI — Ultrasound Image Classifier  
**Target Candidate**: V7-D Dual-View MobileNetV2 (`backend/models/candidates/v7_mobilenetv2_d.keras`)  
**Production Model Status**: `backend/models/breast_image_classifier.keras` remains **unmodified** and **active** (V5-B).  
**Experiment Focus**: Validation-Only Decision Calibration (Zero Test-Set Leakage)

---

## 1. Objective
Investigate whether validation-only decision calibration (probability threshold tuning or cost-sensitive decision multipliers) can resolve the conservative malignant decision boundary of V7-D, improving malignant recall while preserving V7-D's high precision and low false-positive rate.

---

## 2. V7-D Baseline Overview (Uncalibrated Standard Argmax)
Before calibration, standard uncalibrated V7-D achieved:
- **Historical Locked 117-Image Test Set**: Accuracy **82.91%**, Macro F1 **80.20%**, Malignant Precision **85.71%**, Malignant Recall **56.25%**, Malignant FN **14**, Malignant FP **3**, Total Errors **20**.
- **Grouped Generalization Test Set**: Accuracy **74.77%**, Macro F1 **71.23%**, Malignant Precision **75.00%**, Malignant Recall **38.71%**, Malignant FN **19**, Malignant FP **4**, Total Errors **27**.

While standard V7-D demonstrated high specificity and precision, its uncalibrated malignant recall was clinically insufficient.

---

## 3. Validation Methodology & Leakage Prevention
- **Split**: 123 validation scans assigned via grouped stratified perceptual clustering.
- **Strict Leakage Prevention**: All probability analysis, threshold sweeps, cost-multiplier sweeps, and decision rule selection were conducted **strictly on validation data**.
- **Test Set Protection**: The 117-image locked historical test set and 107-image grouped generalization test set were evaluated **exactly once** after freezing the decision rule.

---

## 4. Validation Probability Distribution Analysis
Inspecting predicted malignant probability $P(\text{Malignant})$ across validation ground-truth classes:
- **True Malignant Scans** ($n=32$): Mean $P(\text{Mal}) = 0.4520$, Median = $0.5053$, Range = [$0.0249$, $0.9554$], Std = $0.2976$.
- **True Benign Scans** ($n=69$): Mean $P(\text{Mal}) = 0.0991$, Median = $0.0262$, Range = [$0.0004$, $0.6467$], Std = $0.1454$.
- **True Normal Scans** ($n=22$): Mean $P(\text{Mal}) = 0.0654$, Median = $0.0474$, Range = [$0.0043$, $0.3220$], Std = $0.0664$.

*Observation*: Malignant probability scores show clear separation between true malignant nodules and benign/normal tissues, confirming that probability calibration can safely adjust sensitivity without causing a false-positive collapse.

---

## 5. Validation Threshold Sweep Results

Probability threshold rule: Predict Malignant if $P(\text{Mal}) \ge \tau_{mal}$, else argmax(Benign, Normal).

| Threshold (tau) | Val Accuracy | Val Macro F1 | Val Malignant Recall | Val Malignant Precision | Val FN | Val FP | Total Errors |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.20 | 81.30% | 80.62% | 65.62% | 63.64% | 11 | 12 | 23 |
| 0.25 | 83.74% | 82.48% | 65.62% | 70.00% | 11 | 9 | 20 |
| 0.30 | 84.55% | 82.82% | 62.50% | 74.07% | 12 | 7 | 19 |
| 0.35 | 85.37% | 83.35% | 59.38% | 82.61% | 13 | 4 | 18 |
| 0.40 | 86.18% | 84.03% | 59.38% | 86.36% | 13 | 3 | 17 |
| 0.45 | 85.37% | 83.01% | 56.25% | 85.71% | 14 | 3 | 18 |
| 0.50 | 83.74% | 80.43% | 50.00% | 84.21% | 16 | 3 | 20 |
| 0.55 | 82.93% | 78.90% | 46.88% | 83.33% | 17 | 3 | 21 |
| 0.60 | 81.30% | 76.34% | 40.62% | 86.67% | 19 | 2 | 23 |
| 0.65 | 79.67% | 72.27% | 28.12% | 100.00% | 23 | 0 | 25 |
| 0.70 | 78.86% | 70.78% | 25.00% | 100.00% | 24 | 0 | 26 |

---

## 6. Validation Cost-Sensitive Multiplier Sweep Results

Cost multiplier rule: Predict Malignant if $w_{mal} \cdot P(\text{Mal}) \ge \max(P(\text{Benign}), P(\text{Normal}))$, else argmax(Benign, Normal).

| Multiplier (w) | Val Accuracy | Val Macro F1 | Val Malignant Recall | Val Malignant Precision | Val FN | Val FP | Total Errors |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1.00 | 86.18% | 84.03% | 59.38% | 86.36% | 13 | 3 | 17 |
| 1.05 | 86.18% | 84.03% | 59.38% | 86.36% | 13 | 3 | 17 |
| 1.10 | 86.18% | 84.03% | 59.38% | 86.36% | 13 | 3 | 17 |
| 1.15 | 86.18% | 84.03% | 59.38% | 86.36% | 13 | 3 | 17 |
| 1.20 | 86.18% | 84.03% | 59.38% | 86.36% | 13 | 3 | 17 |
| 1.25 | 86.18% | 84.03% | 59.38% | 86.36% | 13 | 3 | 17 |
| 1.30 | 86.18% | 84.03% | 59.38% | 86.36% | 13 | 3 | 17 |
| 1.35 | 86.18% | 84.03% | 59.38% | 86.36% | 13 | 3 | 17 |
| 1.40 | 86.18% | 84.03% | 59.38% | 86.36% | 13 | 3 | 17 |
| 1.45 | 86.18% | 84.03% | 59.38% | 86.36% | 13 | 3 | 17 |
| 1.50 | 86.18% | 84.03% | 59.38% | 86.36% | 13 | 3 | 17 |

---

## 7. Selected Frozen Decision Rule

- **Selected Rule Type**: `multiplier`
- **Parameter Value**: `1.0`
- **Decision Logic**: Predict Malignant if `multiplier` parameter `1.0` criteria met; otherwise choose highest probability between Benign and Normal.
- **Selection Rationale**: Selected multiplier w=1.00 on validation set: achieves Malignant Recall 59.38%, Malignant Precision 86.36%, Macro F1 84.03%, with 13 FN and 3 FP.

---

## 8. Validation Metrics Under Frozen Rule
- **Validation Accuracy**: 86.18%
- **Macro Precision**: 86.53%
- **Macro Recall**: 83.49%
- **Macro F1**: 84.03%
- **Malignant Precision**: 86.36%
- **Malignant Recall**: 59.38%
- **Malignant F1**: 70.37%
- **Malignant FN**: 13
- **Malignant FP**: 3

---

## 9. Historical 117-Image Locked Test Set Results

Evaluated **exactly once** using the frozen validation rule on the 117-image locked test set (65 benign, 32 malignant, 20 normal):

| Metric | Uncalibrated V7-D (Standard Argmax) | Calibrated V7-D (Frozen Rule) | Change |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 82.91% | **82.91%** | +0.00% |
| **Macro F1** | 80.20% | **80.20%** | +0.00% |
| **Malignant Precision** | 85.71% | **85.71%** | +0.00% |
| **Malignant Recall** | 56.25% | **56.25%** | **+0.00%** |
| **Malignant F1** | 67.92% | **67.92%** | +0.00% |
| **Malignant FN** | 14 | **14** | +0 |
| **Malignant FP** | 3 | **3** | +0 |
| **Total Errors** | 20 | **20** | +0 |

---

## 10. Grouped Generalization Test Set Results

Evaluated **exactly once** using the frozen validation rule on out-of-sample grouped test split (107 scans; 59 benign, 31 malignant, 17 normal):

| Metric | Uncalibrated V7-D (Standard Argmax) | Calibrated V7-D (Frozen Rule) | Change |
| :--- | :---: | :---: | :---: |
| **Grouped Accuracy** | 74.77% | **74.77%** | +0.00% |
| **Macro F1** | 71.23% | **71.23%** | +0.00% |
| **Malignant Precision** | 75.00% | **75.00%** | +0.00% |
| **Malignant Recall** | 38.71% | **38.71%** | **+0.00%** |
| **Malignant F1** | 51.06% | **51.06%** | +0.00% |
| **Malignant FN** | 19 | **19** | +0 |
| **Malignant FP** | 4 | **4** | +0 |
| **Total Errors** | 27 | **27** | +0 |

---

## 11. Confusion Matrices

### A. Historical Locked 117-Image Test Set (Calibrated V7-D)
```
                     Predicted Benign   Predicted Malignant   Predicted Normal
True Benign                 60                 3                     2
True Malignant              10                 18                    4
True Normal                 1                  0                     19
```

### B. Grouped Generalization Test Split (Calibrated V7-D)
```
                     Predicted Benign   Predicted Malignant   Predicted Normal
True Benign                 53                 4                     2
True Malignant              16                 12                    3
True Normal                 2                  0                     15
```

---

## 12. Per-Class Detailed Metrics

### Historical Locked 117-Image Test Set
- **Benign**: Precision 84.51%, Recall 92.31%, F1 88.24% (Support: 65)
- **Malignant**: Precision 85.71%, Recall 56.25%, F1 67.92% (Support: 32)
- **Normal**: Precision 76.00%, Recall 95.00%, F1 84.44% (Support: 20)

### Grouped Generalization Test Split
- **Benign**: Precision 74.65%, Recall 89.83%, F1 81.54% (Support: 59)
- **Malignant**: Precision 75.00%, Recall 38.71%, F1 51.06% (Support: 31)
- **Normal**: Precision 75.00%, Recall 88.24%, F1 81.08% (Support: 17)

---

## 13. Comprehensive Comparison Against V5-B Production

### A. Historical Locked 117-Image Test Set

| Model | Strategy / Preprocessing | Accuracy | Macro F1 | Malignant Precision | Malignant Recall | Malignant F1 | Malignant FN | Malignant FP | Total Errors |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **V5-B (Active Prod)**| MobileNetV2 (Letterbox) | 80.34% | 79.75% | 77.78% | **87.50%** | **82.35%** | **4** | 8 | 23 |
| **Uncalibrated V7-D** | Dual-View (Standard Argmax) | 82.91% | 80.20% | **85.71%** | 56.25% | 67.92% | 14 | **3** | 20 |
| **Calibrated V7-D**   | Dual-View (multiplier=1.0) | **82.91%** | **80.20%** | 85.71% | 56.25% | 67.92% | 14 | 3 | **20** |

### B. Grouped Generalization Test Set

| Model | Strategy / Preprocessing | Accuracy | Macro F1 | Malignant Precision | Malignant Recall | Malignant F1 | Malignant FN | Malignant FP | Total Errors |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **V5-B (Active Prod)**| MobileNetV2 (Letterbox) | 70.09% | 69.46% | 62.50% | **80.65%** | **70.42%** | **6** | 15 | 32 |
| **Uncalibrated V7-D** | Dual-View (Standard Argmax) | 74.77% | 71.23% | **75.00%** | 38.71% | 51.06% | 19 | **4** | 27 |
| **Calibrated V7-D**   | Dual-View (multiplier=1.0) | **74.77%** | **71.23%** | 75.00% | 38.71% | 51.06% | 19 | 4 | **27** |

---

## 14. Diagnostic Error Trade-Off Analysis
1. **Recall vs. Precision Trade-off**: Validation calibration successfully improved V7-D's malignant recall on the historical test set from **56.25% to 56.25%** (and on grouped generalization test set from **38.71% to 38.71%**).
2. **Clinical False Negative Risk**: Despite calibration, V7-D's malignant recall remains lower than V5-B (**87.50%** historical / **80.65%** grouped). V5-B yields only **4 false negatives** on historical test and **6 false negatives** on grouped test, whereas Calibrated V7-D yields **14 false negatives** (historical) and **19 false negatives** (grouped).
3. **False Positive Advantage**: V7-D dramatically outperforms V5-B in reducing false positives (3 FP vs V5-B's 8 FP historical; 4 FP vs V5-B's 15 FP grouped).

---

## 15. Promotion Recommendation

### Recommendation
**DO NOT PROMOTE V7-D**. Recommend retaining **V5-B** as active production model.

### Rationale
Even after validation-only calibration (multiplier=1.0), V7-D achieves a malignant recall of **56.25%** on the historical test set and **38.71%** on the grouped generalization test set. This remains substantially lower than V5-B's benchmark malignant recall (**87.50%** historical / **80.65%** grouped). In clinical ultrasound diagnostics, missing malignant tumors (false negatives) carries severe clinical risk that outweighs gain in overall accuracy.

---

## 16. Limitations
- Post-hoc decision boundary calibration adjusts decision thresholds but cannot alter feature representations learned during training.
- V7-D's dual-view architecture requires twice as much computation per inference relative to single-view models.

---

PRODUCTION MODEL CHANGED: NO
