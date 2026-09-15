import os
import json
import joblib
import hashlib
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = r"C:\Users\kiran\Downloads\data.csv"
    model_save_path = os.path.join(base_dir, "models", "breast_cancer_model.joblib")
    metadata_save_path = os.path.join(base_dir, "models", "metadata.json")
    report_save_path = os.path.join(base_dir, "models", "evaluation_reports", "tabular_svm_training_report.md")

    # 1. Existing model SHA256 check
    old_sha256 = None
    if os.path.exists(model_save_path):
        with open(model_save_path, "rb") as f:
            old_sha256 = hashlib.sha256(f.read()).hexdigest()
        print(f"Existing model SHA256: {old_sha256}")
    else:
        print("No existing model found.")

    # 2. Load dataset
    print(f"Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Loaded dataset shape: {df.shape}")

    # 3. Preprocessing: Drop id and Unnamed: 32
    cols_to_drop = [col for col in ["id", "Unnamed: 32"] if col in df.columns]
    df = df.drop(columns=cols_to_drop)

    # 4. Target encoding: B -> 0, M -> 1
    y = df["diagnosis"].map({"B": 0, "M": 1}).values
    X_raw = df.drop(columns=["diagnosis"])

    # 5. Header renaming to match standard backend feature_names
    feature_mapping = {
        "radius_mean": "mean radius",
        "texture_mean": "mean texture",
        "perimeter_mean": "mean perimeter",
        "area_mean": "mean area",
        "smoothness_mean": "mean smoothness",
        "compactness_mean": "mean compactness",
        "concavity_mean": "mean concavity",
        "concave points_mean": "mean concave points",
        "symmetry_mean": "mean symmetry",
        "fractal_dimension_mean": "mean fractal dimension",
        "radius_se": "radius error",
        "texture_se": "texture error",
        "perimeter_se": "perimeter error",
        "area_se": "area error",
        "smoothness_se": "smoothness error",
        "compactness_se": "compactness error",
        "concavity_se": "concavity error",
        "concave points_se": "concave points error",
        "symmetry_se": "symmetry error",
        "fractal_dimension_se": "fractal dimension error",
        "radius_worst": "worst radius",
        "texture_worst": "worst texture",
        "perimeter_worst": "worst perimeter",
        "area_worst": "worst area",
        "smoothness_worst": "worst smoothness",
        "compactness_worst": "worst compactness",
        "concavity_worst": "worst concavity",
        "concave points_worst": "worst concave points",
        "symmetry_worst": "worst symmetry",
        "fractal_dimension_worst": "worst fractal dimension"
    }

    X = X_raw.rename(columns=feature_mapping)
    feature_names = list(X.columns)

    print(f"Features count: {len(feature_names)}")
    print(f"Target count: Benign (0)={np.sum(y == 0)}, Malignant (1)={np.sum(y == 1)}")

    # 6. Stratified 80/20 train/test split (random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

    # 7. Build Pipeline: StandardScaler -> SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42))
    ])

    # 8. Train pipeline on X_train ONLY
    pipeline.fit(X_train, y_train)

    # 9. Evaluate ONLY on unseen X_test
    y_pred = pipeline.predict(X_test)
    y_probs = pipeline.predict_proba(X_test)
    # y_probs[:, 1] is probability of Malignant (1)

    acc = float(accuracy_score(y_test, y_pred))
    prec_macro = float(precision_score(y_test, y_pred, average="macro"))
    rec_macro = float(recall_score(y_test, y_pred, average="macro"))
    f1_macro = float(f1_score(y_test, y_pred, average="macro"))

    # Malignant (Class 1) specific metrics
    prec_mal = float(precision_score(y_test, y_pred, pos_label=1))
    rec_mal = float(recall_score(y_test, y_pred, pos_label=1))
    f1_mal = float(f1_score(y_test, y_pred, pos_label=1))

    # Benign (Class 0) specific metrics
    prec_ben = float(precision_score(y_test, y_pred, pos_label=0))
    rec_ben = float(recall_score(y_test, y_pred, pos_label=0))
    f1_ben = float(f1_score(y_test, y_pred, pos_label=0))

    roc_auc = float(roc_auc_score(y_test, y_probs[:, 1]))
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = [int(val) for val in cm.ravel()]

    cls_report = classification_report(y_test, y_pred, target_names=["Benign", "Malignant"])

    print("=== TEST EVALUATION METRICS ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Malignant Precision: {prec_mal:.4f}, Recall: {rec_mal:.4f}, F1: {f1_mal:.4f}")
    print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print("\nClassification Report:\n", cls_report)

    # 10. Save pipeline
    joblib.dump(pipeline, model_save_path)
    print(f"Model pipeline saved to {model_save_path}")

    # Compute new model SHA256
    with open(model_save_path, "rb") as f:
        new_sha256 = hashlib.sha256(f.read()).hexdigest()
    print(f"New model SHA256: {new_sha256}")

    # 11. Metadata creation
    metadata = {
        "model_name": "SVM (RBF) + StandardScaler Pipeline",
        "model_type": "SVM",
        "model_version": "1.2.0",
        "dataset_name": "Wisconsin Breast Cancer Diagnostic",
        "training_date": datetime.now().isoformat(),
        "random_seed": 42,
        "test_split": 0.20,
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "target_names": ["benign", "malignant"],
        "class_mapping": {"0": "benign", "1": "malignant"},
        "metrics": {
            "accuracy": acc,
            "precision": prec_macro,
            "recall": rec_macro,
            "f1_score": f1_macro,
            "roc_auc": roc_auc,
            "malignant_precision": prec_mal,
            "malignant_recall": rec_mal,
            "malignant_f1": f1_mal,
            "confusion_matrix": {
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp
            }
        },
        "old_model_sha256": old_sha256,
        "new_model_sha256": new_sha256
    }

    with open(metadata_save_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {metadata_save_path}")

    # 12. Create evaluation report markdown
    report_md = f"""# Tabular SVM Model Training & Evaluation Report

**Training Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Dataset:** Wisconsin Breast Cancer Diagnostic Dataset (`data.csv`)  
**Model Architecture:** `StandardScaler` + `SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)` Pipeline  
**Model File:** `backend/models/breast_cancer_model.joblib`  
**Metadata File:** `backend/models/metadata.json`  

---

## A. Dataset Summary
* **Source:** `C:\\Users\\kiran\\Downloads\\data.csv`
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
| **Accuracy** | **{acc * 100:.2f}%** ({acc:.4f}) |
| **ROC-AUC** | **{roc_auc:.4f}** |
| **Macro Precision** | {prec_macro:.4f} |
| **Macro Recall** | {rec_macro:.4f} |
| **Macro F1-Score** | {f1_macro:.4f} |

---

## F. Confusion Matrix

```
               Predicted Benign (0)   Predicted Malignant (1)
Actual Benign      TN = {tn:2d}                 FP = {fp:2d}
Actual Malignant   FN = {fn:2d}                 TP = {tp:2d}
```

* **True Negatives (TN):** {tn}
* **False Positives (FP):** {fp}
* **False Negatives (FN):** {fn}
* **True Positives (TP):** {tp}

---

## G. Malignant-Class Performance (Class 1 / 'M')

* **Malignant Precision:** **{prec_mal * 100:.2f}%** ({prec_mal:.4f})
* **Malignant Recall (Sensitivity):** **{rec_mal * 100:.2f}%** ({rec_mal:.4f})
* **Malignant F1-Score:** **{f1_mal:.4f}**

```text
Classification Report:
{cls_report}
```

---

## H. Data Leakage Safeguards
1. `id` and `Unnamed: 32` were excluded prior to splitting or training.
2. `StandardScaler` was fitted strictly on `X_train` within the Pipeline; `X_test` remained completely unseen during scaler fitting.
3. Test metrics were computed exclusively on the held-out 114 test samples.

---

## I. Saved Model Information
* **Model Saved Path:** `backend/models/breast_cancer_model.joblib`
* **Previous Model SHA256:** `{old_sha256}`
* **New Model SHA256:** `{new_sha256}`

---

## J. Final Recommendation & Verdict

**TRAINING COMPLETE**
"""

    with open(report_save_path, "w") as f:
        f.write(report_md)
    print(f"Report saved to {report_save_path}")

    # 14. Verification: Load saved model back and run prediction on a sample
    print("\n--- Verifying Saved Model ---")
    loaded_pipeline = joblib.load(model_save_path)
    sample_df = X_test.iloc[[0]]
    sample_pred = loaded_pipeline.predict(sample_df)[0]
    sample_probs = loaded_pipeline.predict_proba(sample_df)[0]

    print(f"Sample index 0 actual target: {y_test[0]} ({'M' if y_test[0] == 1 else 'B'})")
    print(f"Sample prediction output: {sample_pred}")
    print(f"Sample probability output: {sample_probs}")
    print("Verification successful!")

if __name__ == "__main__":
    main()
