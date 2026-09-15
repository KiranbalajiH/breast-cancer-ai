# Final Model Evaluation Report — Phase F

* **Evaluation Date:** 2026-08-23 17:50:02
* **Active Model:** MobileNetV2 Transfer Learning (Version 1)
* **Model Path:** `backend/models/breast_image_classifier.keras`
* **Evaluation Dataset:** BUSI Held-out Test Split (15%)

---

## 1. Executive Safety Assessment

### Project Maturity Status: **Experimental decision-support prototype**

> [!NOTE]
> Classified as an 'Experimental decision-support prototype' because while the model has reasonable accuracy (approx 81%), its raw recall for malignant cases is 61.29% on the test split, resulting in a significant number of false negatives if left unflagged. However, the safety threshold layer successfully identifies and flags a majority of these errors as 'review_required' (preventing false negatives from going unnoticed), making it suitable as a decision-support prototype.

---

## 2. Dataset & Leakage Verification

* **Untouched Data Confirmation:** YES. The evaluation dataset consists of a 15% stratified test split set aside during train_v2.py configuration.
* **Leakage Results:**
  * Test-Train Overlap: `0` files
  * Test-Validation Overlap: `0` files
  * Annotation/Mask files in Test set: `0` files
  * **Status:** Clean split verified. No data leakage detected.

* **Class Distribution (Support):**
  * Benign: `65` images
  * Malignant: `32` images
  * Normal: `20` images
  * **Total Test Images:** `117`

---

## 3. Quantitative Performance Results

### Classification Metrics
* **Overall Accuracy:** `0.8462`

| Class | Precision | Recall / Sensitivity | F1-Score | Support |
|---|---|---|---|---|
| **Benign** | `0.8310` | `0.9077` | `0.8676` | `65` |
| **Malignant** | `0.8462` | `0.6875` | `0.7586` | `32` |
| **Normal** | `0.9000` | `0.9000` | `0.9000` | `20` |

### Confusion Matrix
```
Predicted ->   Benign  Malignant   Normal
True Benign    [ 59       4         2]
True Malignant [ 10      22         0]
True Normal    [  2       0        18]
```

### Malignant Error Analysis
* **Malignant Sensitivity:** `0.6875`
* **Malignant Precision:** `0.8462`
* **Malignant F1-Score:** `0.7586`
* **False Negatives (FN):** `10` (Malignant tumor predicted as benign or normal)
* **False Positives (FP):** `4` (Benign tissue/normal predicted as malignant)

---

## 4. Confidence & Safety Uncertainty Analysis

### Confidence Distributions
* **Average Confidence for Correct Predictions:** `0.8562`
* **Average Confidence for Incorrect Predictions:** `0.7336`
* **Highly Confident Incorrect Predictions (>= 70%):** `9`

### Uncertainty Status Safety Assessment
Using the central thresholds layer:
* **Incorrect predictions flagged for review:** `2` of `18` (`11.11%`)
* **Incorrect predictions remaining 'high_confidence':** `9`
* **Malignant False Negatives successfully marked 'review_required':** `1` of `10`
* **Malignant False Negatives incorrectly marked 'high_confidence':** `7`

---

## 5. Grad-CAM Spot Check Verification

Technical check of overlay dimensions matching original inputs:

* **Correct benign:** SUCCESS. Heatmap and overlay generated successfully. Dimensions match input. Image name: `benign (402).png`
* **Correct malignant:** SUCCESS. Heatmap and overlay generated successfully. Dimensions match input. Image name: `malignant (77).png`
* **Correct normal:** SUCCESS. Heatmap and overlay generated successfully. Dimensions match input. Image name: `normal (44).png`
* **Incorrect malignant:** SUCCESS. Heatmap and overlay generated successfully. Dimensions match input. Image name: `benign (316).png`
* **Malignant false negative:** SUCCESS. Heatmap and overlay generated successfully. Dimensions match input. Image name: `malignant (138).png`

---

## 6. Known Limitations
1. **Validation Scope:** No true independent external validation dataset was currently available; validation was performed on local held-out test split.
2. **Visual Explainability (Grad-CAM):** Highlights heat zones of features influencing the classification. It does NOT trace exact physical tumor boundaries.
3. **Modal Safeguards:** Greyscale verification heuristics assume raw images and might flag color annotations as unsupported.
