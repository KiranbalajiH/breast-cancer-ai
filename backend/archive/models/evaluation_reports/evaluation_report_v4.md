# V4 MODEL EVALUATION REPORT — BALANCED MALIGNANT SENSITIVITY

**Date**: 2026-08-27 20:05:07

**Evaluation Split**: Untouched 117-Image Test Set (65 Benign, 32 Malignant, 20 Normal)


---

## Executive Summary

The objective of the V4 experiment was to improve **malignant sensitivity (recall)** beyond V1 (68.75%) without causing an excessive increase in false positives or degrading overall classification quality (84.62% Accuracy, 84.21% Macro F1).


### Model Performance Overview Table

| Model | Accuracy | Macro F1 | Malignant Precision | Malignant Recall | Malignant F1 | Malignant FN | Malignant FP | Total Errors |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **V1 MobileNetV2 Baseline** | 84.62% | 84.21% | 84.62% | 68.75% | 75.86% | 10 | 4 | 18 |
| **V4-E: MobileNetV2 (Frozen Backbone / Mild Weighting 1.25x)** | 81.20% | 78.72% | 84.62% | 68.75% | 75.86% | 10 | 4 | 22 |
| **V4-A: MobileNetV2 (No Weights / Fine-Tuned)** | 75.21% | 72.93% | 71.43% | 78.12% | 74.63% | 7 | 10 | 29 |
| **V4-B: MobileNetV2 (Mild Weighting 1.25x / Fine-Tuned)** | 75.21% | 72.23% | 73.33% | 68.75% | 70.97% | 10 | 8 | 29 |
| **V4-C: MobileNetV2 (Balanced Loss Weighting / Fine-Tuned)** | 76.07% | 73.68% | 76.92% | 62.50% | 68.97% | 12 | 6 | 28 |
| **V3 MobileNetV2 No Weights** | 70.94% | 66.01% | 62.50% | 78.12% | 69.44% | 7 | 15 | 34 |
| **V3 MobileNetV2 Mild Weights** | 69.23% | 65.40% | 58.97% | 71.88% | 64.79% | 9 | 16 | 36 |
| **V4-D: EfficientNetB0 (Mild Weighting)** | 64.10% | 54.64% | 48.00% | 75.00% | 58.54% | 8 | 26 | 42 |
| **V2 DenseNet121** | 53.85% | 52.24% | 37.66% | 90.63% | 53.21% | 3 | 48 | 54 |

---

## Per-Class Breakdown & Confusion Matrices

### V1 MobileNetV2 Baseline
- **Overall Accuracy**: 84.62%
- **Macro Precision / Recall / F1**: 85.90% / 83.17% / 84.21%
- **Benign P / R / F1**: 83.10% / 90.77% / 86.76% (Support: 65)
- **Malignant P / R / F1**: 84.62% / 68.75% / 75.86% (Support: 32)
- **Normal P / R / F1**: 90.00% / 90.00% / 90.00% (Support: 20)
- **False Negatives (Malignant missed)**: 10
- **False Positives (Malignant over-predicted)**: 4
- **Total Incorrect Predictions**: 18

**Confusion Matrix (Row=True, Col=Pred)**:
```
               Pred Benign   Pred Malignant   Pred Normal
True Benign        59             4              2     
True Malignant     10             22             0     
True Normal         2             0             18     
```

### V4-E: MobileNetV2 (Frozen Backbone / Mild Weighting 1.25x)
- **Overall Accuracy**: 81.20%
- **Macro Precision / Recall / F1**: 80.44% / 77.66% / 78.72%
- **Benign P / R / F1**: 81.69% / 89.23% / 85.29% (Support: 65)
- **Malignant P / R / F1**: 84.62% / 68.75% / 75.86% (Support: 32)
- **Normal P / R / F1**: 75.00% / 75.00% / 75.00% (Support: 20)
- **False Negatives (Malignant missed)**: 10
- **False Positives (Malignant over-predicted)**: 4
- **Total Incorrect Predictions**: 22

**Confusion Matrix (Row=True, Col=Pred)**:
```
               Pred Benign   Pred Malignant   Pred Normal
True Benign        58             3              4     
True Malignant      9             22             1     
True Normal         4             1             15     
```

### V4-A: MobileNetV2 (No Weights / Fine-Tuned)
- **Overall Accuracy**: 75.21%
- **Macro Precision / Recall / F1**: 71.78% / 74.50% / 72.93%
- **Benign P / R / F1**: 83.05% / 75.38% / 79.03% (Support: 65)
- **Malignant P / R / F1**: 71.43% / 78.12% / 74.63% (Support: 32)
- **Normal P / R / F1**: 60.87% / 70.00% / 65.12% (Support: 20)
- **False Negatives (Malignant missed)**: 7
- **False Positives (Malignant over-predicted)**: 10
- **Total Incorrect Predictions**: 29

**Confusion Matrix (Row=True, Col=Pred)**:
```
               Pred Benign   Pred Malignant   Pred Normal
True Benign        49             8              8     
True Malignant      6             25             1     
True Normal         4             2             14     
```

### V4-B: MobileNetV2 (Mild Weighting 1.25x / Fine-Tuned)
- **Overall Accuracy**: 75.21%
- **Macro Precision / Recall / F1**: 71.82% / 72.92% / 72.23%
- **Benign P / R / F1**: 81.25% / 80.00% / 80.62% (Support: 65)
- **Malignant P / R / F1**: 73.33% / 68.75% / 70.97% (Support: 32)
- **Normal P / R / F1**: 60.87% / 70.00% / 65.12% (Support: 20)
- **False Negatives (Malignant missed)**: 10
- **False Positives (Malignant over-predicted)**: 8
- **Total Incorrect Predictions**: 29

**Confusion Matrix (Row=True, Col=Pred)**:
```
               Pred Benign   Pred Malignant   Pred Normal
True Benign        52             6              7     
True Malignant      8             22             2     
True Normal         4             2             14     
```

### V4-C: MobileNetV2 (Balanced Loss Weighting / Fine-Tuned)
- **Overall Accuracy**: 76.07%
- **Macro Precision / Recall / F1**: 73.39% / 75.83% / 73.68%
- **Benign P / R / F1**: 82.54% / 80.00% / 81.25% (Support: 65)
- **Malignant P / R / F1**: 76.92% / 62.50% / 68.97% (Support: 32)
- **Normal P / R / F1**: 60.71% / 85.00% / 70.83% (Support: 20)
- **False Negatives (Malignant missed)**: 12
- **False Positives (Malignant over-predicted)**: 6
- **Total Incorrect Predictions**: 28

**Confusion Matrix (Row=True, Col=Pred)**:
```
               Pred Benign   Pred Malignant   Pred Normal
True Benign        52             5              8     
True Malignant      9             20             3     
True Normal         2             1             17     
```

### V3 MobileNetV2 No Weights
- **Overall Accuracy**: 70.94%
- **Macro Precision / Recall / F1**: 67.18% / 66.17% / 66.01%
- **Benign P / R / F1**: 79.03% / 75.38% / 77.17% (Support: 65)
- **Malignant P / R / F1**: 62.50% / 78.12% / 69.44% (Support: 32)
- **Normal P / R / F1**: 60.00% / 45.00% / 51.43% (Support: 20)
- **False Negatives (Malignant missed)**: 7
- **False Positives (Malignant over-predicted)**: 15
- **Total Incorrect Predictions**: 34

**Confusion Matrix (Row=True, Col=Pred)**:
```
               Pred Benign   Pred Malignant   Pred Normal
True Benign        49             11             5     
True Malignant      6             25             1     
True Normal         7             4              9     
```

### V3 MobileNetV2 Mild Weights
- **Overall Accuracy**: 69.23%
- **Macro Precision / Recall / F1**: 65.00% / 66.39% / 65.40%
- **Benign P / R / F1**: 81.03% / 72.31% / 76.42% (Support: 65)
- **Malignant P / R / F1**: 58.97% / 71.88% / 64.79% (Support: 32)
- **Normal P / R / F1**: 55.00% / 55.00% / 55.00% (Support: 20)
- **False Negatives (Malignant missed)**: 9
- **False Positives (Malignant over-predicted)**: 16
- **Total Incorrect Predictions**: 36

**Confusion Matrix (Row=True, Col=Pred)**:
```
               Pred Benign   Pred Malignant   Pred Normal
True Benign        47             10             8     
True Malignant      8             23             1     
True Normal         3             6             11     
```

### V4-D: EfficientNetB0 (Mild Weighting)
- **Overall Accuracy**: 64.10%
- **Macro Precision / Recall / F1**: 63.91% / 55.77% / 54.64%
- **Benign P / R / F1**: 77.05% / 72.31% / 74.60% (Support: 65)
- **Malignant P / R / F1**: 48.00% / 75.00% / 58.54% (Support: 32)
- **Normal P / R / F1**: 66.67% / 20.00% / 30.77% (Support: 20)
- **False Negatives (Malignant missed)**: 8
- **False Positives (Malignant over-predicted)**: 26
- **Total Incorrect Predictions**: 42

**Confusion Matrix (Row=True, Col=Pred)**:
```
               Pred Benign   Pred Malignant   Pred Normal
True Benign        47             18             0     
True Malignant      6             24             2     
True Normal         8             8              4     
```

### V2 DenseNet121
- **Overall Accuracy**: 53.85%
- **Macro Precision / Recall / F1**: 65.89% / 55.72% / 52.24%
- **Benign P / R / F1**: 90.00% / 41.54% / 56.84% (Support: 65)
- **Malignant P / R / F1**: 37.66% / 90.63% / 53.21% (Support: 32)
- **Normal P / R / F1**: 70.00% / 35.00% / 46.67% (Support: 20)
- **False Negatives (Malignant missed)**: 3
- **False Positives (Malignant over-predicted)**: 48
- **Total Incorrect Predictions**: 54

**Confusion Matrix (Row=True, Col=Pred)**:
```
               Pred Benign   Pred Malignant   Pred Normal
True Benign        27             36             2     
True Malignant      2             29             1     
True Normal         1             12             7     
```


---

## Detailed Candidate Evaluation & Selection Rationale

### V1 MobileNetV2 Baseline
- **Status**: Current Production Baseline Model
- **Analysis**: Maintains the highest overall accuracy (84.62%), macro F1 (84.21%), and malignant precision (84.62%) with only 4 false positives. However, malignant recall is 68.75% (10 false negatives out of 32 malignant cases).

### V4-E: MobileNetV2 (Frozen Backbone / Mild Weighting 1.25x)
- **Status**: V4 Candidate
- **Recommendation**: **REJECTED / SUBOPTIMAL**. Does not present a sufficient net gain over V1 baseline (Accuracy: 81.2%, Malignant Recall: 68.8%, Malignant Precision: 84.6%).

### V4-A: MobileNetV2 (No Weights / Fine-Tuned)
- **Status**: V4 Candidate
- **Recommendation**: **REJECTED / SUBOPTIMAL**. Does not present a sufficient net gain over V1 baseline (Accuracy: 75.2%, Malignant Recall: 78.1%, Malignant Precision: 71.4%).

### V4-B: MobileNetV2 (Mild Weighting 1.25x / Fine-Tuned)
- **Status**: V4 Candidate
- **Recommendation**: **REJECTED / SUBOPTIMAL**. Does not present a sufficient net gain over V1 baseline (Accuracy: 75.2%, Malignant Recall: 68.8%, Malignant Precision: 73.3%).

### V4-C: MobileNetV2 (Balanced Loss Weighting / Fine-Tuned)
- **Status**: V4 Candidate
- **Recommendation**: **REJECTED / SUBOPTIMAL**. Does not present a sufficient net gain over V1 baseline (Accuracy: 76.1%, Malignant Recall: 62.5%, Malignant Precision: 76.9%).

### V3 MobileNetV2 No Weights
### V3 MobileNetV2 Mild Weights
### V4-D: EfficientNetB0 (Mild Weighting)
- **Status**: V4 Candidate
- **Recommendation**: **REJECTED / SUBOPTIMAL**. Does not present a sufficient net gain over V1 baseline (Accuracy: 64.1%, Malignant Recall: 75.0%, Malignant Precision: 48.0%).

### V2 DenseNet121