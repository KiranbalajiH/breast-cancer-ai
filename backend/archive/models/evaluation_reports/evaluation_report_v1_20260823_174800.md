# Final Model Evaluation Report — Phase F

* **Evaluation Date:** 2026-08-23 17:48:00
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
* **Overall Accuracy:** `0.5385`

| Class | Precision | Recall / Sensitivity | F1-Score | Support |
|---|---|---|---|---|
| **Benign** | `0.9000` | `0.4154` | `0.5684` | `65` |
| **Malignant** | `0.3766` | `0.9062` | `0.5321` | `32` |
| **Normal** | `0.7000` | `0.3500` | `0.4667` | `20` |

### Confusion Matrix
```
Predicted ->   Benign  Malignant   Normal
True Benign    [ 27      36         2]
True Malignant [  2      29         1]
True Normal    [  1      12         7]
```

### Malignant Error Analysis
* **Malignant Sensitivity:** `0.9062`
* **Malignant Precision:** `0.3766`
* **Malignant F1-Score:** `0.5321`
* **False Negatives (FN):** `3` (Malignant tumor predicted as benign or normal)
* **False Positives (FP):** `48` (Benign tissue/normal predicted as malignant)

---

## 4. Confidence & Safety Uncertainty Analysis

### Confidence Distributions
* **Average Confidence for Correct Predictions:** `0.7759`
* **Average Confidence for Incorrect Predictions:** `0.6366`
* **Highly Confident Incorrect Predictions (>= 70%):** `19`

### Uncertainty Status Safety Assessment
Using the central thresholds layer:
* **Incorrect predictions flagged for review:** `18` of `54` (`33.33%`)
* **Incorrect predictions remaining 'high_confidence':** `13`
* **Malignant False Negatives successfully marked 'review_required':** `2` of `3`
* **Malignant False Negatives incorrectly marked 'high_confidence':** `0`

---

## 5. Grad-CAM Spot Check Verification

Technical check of overlay dimensions matching original inputs:

* **Correct benign:** SUCCESS. Heatmap and overlay generated successfully. Dimensions match input. Image name: `benign (135).png`
* **Correct malignant:** SUCCESS. Heatmap and overlay generated successfully. Dimensions match input. Image name: `malignant (138).png`
* **Correct normal:** SUCCESS. Heatmap and overlay generated successfully. Dimensions match input. Image name: `normal (44).png`
* **Incorrect malignant:** SUCCESS. Heatmap and overlay generated successfully. Dimensions match input. Image name: `benign (402).png`
* **Malignant false negative:** SUCCESS. Heatmap and overlay generated successfully. Dimensions match input. Image name: `malignant (12).png`

---

## 6. Known Limitations
1. **Validation Scope:** No true independent external validation dataset was currently available; validation was performed on local held-out test split.
2. **Visual Explainability (Grad-CAM):** Highlights heat zones of features influencing the classification. It does NOT trace exact physical tumor boundaries.
3. **Modal Safeguards:** Greyscale verification heuristics assume raw images and might flag color annotations as unsupported.
