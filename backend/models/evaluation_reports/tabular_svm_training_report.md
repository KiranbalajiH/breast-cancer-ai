# Tabular SVM Model Training & Evaluation Report

**Training Date:** 2026-09-06 16:35:02  
**Dataset:** Wisconsin Breast Cancer Diagnostic Dataset (`data.csv`)  
**Model Architecture:** `StandardScaler` + `SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)` Pipeline  
**Model File:** `backend/models/breast_cancer_model.joblib`  
**Metadata File:** `backend/models/metadata.json`  

---

## A. Dataset Summary
* **Source:** `C:\Users\kiran\Downloads\data.csv`
* **Total Dataset Samples:** 569
* **Features Used:** 30 numerical breast cancer cell measurements
* **Target:** `diagnosis` (`B` -> 0 [Benign], `M` -> 1 [Malignant])
* **Class Counts:** Benign = 357 (62.74%), Malignant = 212 (37.26%)

---

## B. Preprocessing & Column Handling
* **Dropped Columns:** `id` (patient identifier) and `Unnamed: 32` (trailing empty CSV column).
* **Column Renaming:** Mapped dataset headers (`radius_mean`, `radius_se`, `radius_worst`, etc.) to standard backend feature names (`mean radius`, `radius error`, `worst radius`, etc.).
* **Scaling:** `StandardScaler` fitted **strictly on `X_train`** inside the `scikit-learn` Pipeline.

---

## C. Train / Test Split
* **Split Strategy:** Stratified 80/20 train/test split (`random_state=42`).
* **Training Set:** 455 samples (Benign: 285, Malignant: 170)
* **Test Set:** 114 samples (Benign: 72, Malignant: 42)

---

## D. Model Configuration
* **Pipeline Structure:**
  ```python
  Pipeline([
      ('scaler', StandardScaler()),
      ('svm', SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42))
  ])
  ```
* **Hyperparameters:** Default `C=1.0`, `gamma='scale'`, `kernel='rbf'`. No multi-architecture sweep or hyperparameter search was conducted.

---

## E. Evaluation Metrics (Held-Out Test Set: 114 Samples)

| Metric | Score |
| :--- | :--- |
| **Accuracy** | **97.37%** (0.9737) |
| **ROC-AUC** | **0.9947** |
| **Macro Precision** | 0.9800 |
| **Macro Recall** | 0.9643 |
| **Macro F1-Score** | 0.9713 |

---

## F. Confusion Matrix

```
               Predicted Benign (0)   Predicted Malignant (1)
Actual Benign      TN = 72                 FP =  0
Actual Malignant   FN =  3                 TP = 39
```

* **True Negatives (TN):** 72
* **False Positives (FP):** 0
* **False Negatives (FN):** 3
* **True Positives (TP):** 39

---

## G. Malignant-Class Performance (Class 1 / 'M')

* **Malignant Precision:** **100.00%** (1.0000)
* **Malignant Recall (Sensitivity):** **92.86%** (0.9286)
* **Malignant F1-Score:** **0.9630**

```text
Classification Report:
              precision    recall  f1-score   support

      Benign       0.96      1.00      0.98        72
   Malignant       1.00      0.93      0.96        42

    accuracy                           0.97       114
   macro avg       0.98      0.96      0.97       114
weighted avg       0.97      0.97      0.97       114

```

---

## H. Data Leakage Safeguards
1. `id` and `Unnamed: 32` were excluded prior to splitting or training.
2. `StandardScaler` was fitted strictly on `X_train` within the Pipeline; `X_test` remained completely unseen during scaler fitting.
3. Test metrics were computed exclusively on the held-out 114 test samples.

---

## I. Saved Model Information
* **Model Saved Path:** `backend/models/breast_cancer_model.joblib`
* **Previous Model SHA256:** `d7c77690b83e43f116eae3df33a387aee5ce866b7f6f44db3dbb1a42e2c41347`
* **New Model SHA256:** `c5cfade06da37519cfb30de3ba8dd914af7d0c580b392944079b3711fc056f16`

---

## J. Final Recommendation & Verdict

**TRAINING COMPLETE**
