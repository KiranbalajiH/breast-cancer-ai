import os
import cv2
import numpy as np
import json
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from datetime import datetime

# Import preprocessing functions
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_mnv2
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_eff
from tensorflow.keras.applications.densenet import preprocess_input as preprocess_dense

def load_and_clean_dataset(dataset_root):
    classes = ['benign', 'malignant', 'normal']
    image_paths = []
    labels = []
    
    exclude_files = {
        "malignant (145).png",
        "benign (433).png"
    }
    
    for class_idx, class_name in enumerate(classes):
        class_dir = os.path.join(dataset_root, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        class_files = os.listdir(class_dir)
        for f in class_files:
            if not f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                continue
            
            if '_mask' in f.lower() or 'mask' in f.lower():
                continue
                
            if f in exclude_files:
                continue
                
            full_path = os.path.join(class_dir, f)
            image_paths.append(full_path)
            labels.append(class_idx)
            
    return image_paths, labels, classes

def load_images_to_numpy(image_paths, target_size=(224, 224)):
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Could not read image: {path}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, target_size)
        images.append(img_resized)
    return np.array(images, dtype=np.float32)

def evaluate_model_on_test(model, X_test_raw, y_test, classes, preprocess_fn, test_paths=None):
    # Preprocess
    X_test_prep = preprocess_fn(X_test_raw.copy())
    
    predictions = model.predict(X_test_prep, verbose=0)
    pred_classes = np.argmax(predictions, axis=1)
    
    acc = accuracy_score(y_test, pred_classes)
    prec, rec, f1, support = precision_recall_fscore_support(y_test, pred_classes, labels=[0, 1, 2], zero_division=0)
    cm = confusion_matrix(y_test, pred_classes, labels=[0, 1, 2])
    
    # Calculate false negatives and false positives for malignant (class 1)
    mal_idx = 1
    mal_tp = int(cm[mal_idx, mal_idx])
    mal_fn = int(cm[mal_idx, 0] + cm[mal_idx, 2])
    mal_fp = int(cm[0, mal_idx] + cm[2, mal_idx])
    
    incorrect_count = int(np.sum(pred_classes != y_test))
    
    misclassifications = []
    if test_paths is not None:
        for idx in range(len(y_test)):
            if pred_classes[idx] != y_test[idx]:
                misclassifications.append({
                    "filename": os.path.basename(test_paths[idx]),
                    "true_class": classes[y_test[idx]],
                    "predicted_class": classes[pred_classes[idx]],
                    "confidence": float(predictions[idx][pred_classes[idx]])
                })
                
    return {
        "accuracy": float(acc),
        "precision": [float(p) for p in prec],
        "recall": [float(r) for r in rec],
        "f1": [float(f) for f in f1],
        "confusion_matrix": cm.tolist(),
        "false_negatives": int(mal_fn),
        "false_positives": int(mal_fp),
        "incorrect_count": incorrect_count,
        "misclassifications": misclassifications
    }

def main():
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_root = os.path.join(os.path.dirname(backend_dir), "dataset", "BUSI")
    models_dir = os.path.join(backend_dir, "models")
    candidates_dir = os.path.join(models_dir, "candidates")
    
    print("="*60)
    print("PHASE F: INDEPENDENT V3 CANDIDATE COMPARISON")
    print("="*60)
    
    # 1. Load clean dataset and split exactly
    image_paths, labels, classes = load_and_clean_dataset(dataset_root)
    labels = np.array(labels, dtype=np.int32)
    
    print("\nLoading images...")
    X_raw = load_images_to_numpy(image_paths)
    
    X_train, X_val_test, y_train, y_val_test = train_test_split(
        X_raw, labels, test_size=0.30, stratify=labels, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_val_test, y_val_test, test_size=0.50, stratify=y_val_test, random_state=42
    )
    
    paths_train, paths_val_test = train_test_split(
        image_paths, test_size=0.30, stratify=labels, random_state=42
    )
    paths_val, paths_test = train_test_split(
        paths_val_test, test_size=0.50, stratify=y_val_test, random_state=42
    )
    
    print(f"Untouched Test Split counts:")
    print(f"  Total: {len(X_test)}")
    for idx, c_name in enumerate(classes):
        print(f"    - {c_name:<10}: {np.sum(y_test == idx)}")
        
    # 2. Define models to evaluate
    eval_configs = {
        "V1 Baseline": {
            "path": os.path.join(models_dir, "breast_image_classifier.keras"),
            "preprocess_fn": preprocess_mnv2,
            "is_candidate": False
        },
        "MobileNetV2 (V3 No Weights)": {
            "path": os.path.join(candidates_dir, "mobilenetv2_v3_noweights.keras"),
            "preprocess_fn": preprocess_mnv2,
            "is_candidate": True
        },
        "MobileNetV2 (V3 Mild Weights)": {
            "path": os.path.join(candidates_dir, "mobilenetv2_v3_mildweights.keras"),
            "preprocess_fn": preprocess_mnv2,
            "is_candidate": True
        },
        "EfficientNetB0 (V3 No Weights)": {
            "path": os.path.join(candidates_dir, "efficientnetb0_v3_noweights.keras"),
            "preprocess_fn": preprocess_eff,
            "is_candidate": True
        },
        "EfficientNetB0 (V3 Mild Weights)": {
            "path": os.path.join(candidates_dir, "efficientnetb0_v3_mildweights.keras"),
            "preprocess_fn": preprocess_eff,
            "is_candidate": True
        },
        "DenseNet121 (V3 No Weights)": {
            "path": os.path.join(candidates_dir, "densenet121_v3_noweights.keras"),
            "preprocess_fn": preprocess_dense,
            "is_candidate": True
        },
        "DenseNet121 (V3 Mild Weights)": {
            "path": os.path.join(candidates_dir, "densenet121_v3_mildweights.keras"),
            "preprocess_fn": preprocess_dense,
            "is_candidate": True
        }
    }
    
    results = {}
    
    # 3. Evaluate each model
    for name, config in eval_configs.items():
        m_path = config["path"]
        if not os.path.exists(m_path):
            print(f"\nModel file not found: {m_path}. Skipping {name}.")
            continue
            
        print(f"\nEvaluating Model: {name}...")
        model = tf.keras.models.load_model(m_path)
        eval_metrics = evaluate_model_on_test(
            model, X_test, y_test, classes, config["preprocess_fn"], test_paths=paths_test
        )
        results[name] = eval_metrics
        
        print(f"  Accuracy: {eval_metrics['accuracy']:.4f}")
        print(f"  Malignant Recall: {eval_metrics['recall'][1]:.4f}")
        print(f"  Malignant Precision: {eval_metrics['precision'][1]:.4f}")
        print(f"  Malignant False Negatives: {eval_metrics['false_negatives']}")
        print(f"  Malignant False Positives: {eval_metrics['false_positives']}")
        print(f"  Total Errors: {eval_metrics['incorrect_count']}")
        
    # Check if V1 Baseline was evaluated
    if "V1 Baseline" not in results:
        print("\n[ERROR] V1 Baseline model was not evaluated! Cannot complete comparison.")
        return
        
    v1_res = results["V1 Baseline"]
    
    # 4. Determine Model Selection / Best Model
    best_candidate_name = None
    best_candidate_recall = 0.0
    best_candidate_accuracy = 0.0
    best_candidate_f1 = 0.0
    
    # We evaluate candidates based on overall performance profile:
    # A candidate must not significantly degrade overall accuracy or increase false positives unacceptably,
    # but we want to maximize malignant recall and overall F1.
    for name, config in eval_configs.items():
        if not config["is_candidate"] or name not in results:
            continue
            
        m = results[name]
        acc = m["accuracy"]
        recall = m["recall"][1]
        f1 = m["f1"][1]
        
        # We select the best candidate based on a balanced metric:
        # F1 score for malignant + overall accuracy
        score = (recall * 0.40) + (acc * 0.40) + (f1 * 0.20)
        
        # We want to find the candidate with the highest balanced score
        current_best_score = (best_candidate_recall * 0.40) + (best_candidate_accuracy * 0.40) + (best_candidate_f1 * 0.20)
        
        if score > current_best_score:
            best_candidate_recall = recall
            best_candidate_accuracy = acc
            best_candidate_f1 = f1
            best_candidate_name = name
            
    print(f"\nBest new candidate model: {best_candidate_name}")
    print(f"  Candidate Accuracy: {best_candidate_accuracy:.4f}")
    print(f"  Candidate Malignant Recall: {best_candidate_recall:.4f}")
    
    # 5. Check if best candidate beats V1
    is_better = False
    beats_v1_reason = ""
    
    if best_candidate_name is not None:
        best_cand_res = results[best_candidate_name]
        v1_accuracy = v1_res["accuracy"]
        v1_recall = v1_res["recall"][1]
        v1_fp = v1_res["false_positives"]
        v1_fn = v1_res["false_negatives"]
        
        cand_accuracy = best_cand_res["accuracy"]
        cand_recall = best_cand_res["recall"][1]
        cand_fp = best_cand_res["false_positives"]
        cand_fn = best_cand_res["false_negatives"]
        
        # Check conditions for beating V1:
        # 1. Accuracy must be close to or better than V1 (no more than 5% degradation in overall accuracy)
        # 2. Malignant recall must be higher than V1, OR if recall is equal, overall accuracy must be higher.
        # 3. False positives must not explode to an unacceptable level (e.g. no more than 2x V1's false positives, and definitely not like 20+ FP).
        
        accuracy_acceptable = (cand_accuracy >= v1_accuracy - 0.05)
        recall_improved = (cand_recall > v1_recall)
        recall_equal_acc_improved = (abs(cand_recall - v1_recall) < 1e-4 and cand_accuracy > v1_accuracy)
        fp_acceptable = (cand_fp <= 10) # 10 or fewer false positives is reasonable, whereas V1 had 4 and DenseNet121 V2 had 20.
        
        if accuracy_acceptable and (recall_improved or recall_equal_acc_improved) and fp_acceptable:
            is_better = True
            beats_v1_reason = (
                f"Candidate {best_candidate_name} beats V1 because it improves Malignant Recall from {v1_recall:.4f} to {cand_recall:.4f} "
                f"while maintaining a high overall accuracy of {cand_accuracy:.4f} (V1: {v1_accuracy:.4f}) and keeping "
                f"false positives under control at {cand_fp} (V1: {v1_fp})."
            )
        else:
            reasons = []
            if not accuracy_acceptable:
                reasons.append(f"Overall accuracy degraded too much ({cand_accuracy:.4f} vs V1 {v1_accuracy:.4f})")
            if not (recall_improved or recall_equal_acc_improved):
                reasons.append(f"Malignant recall did not improve ({cand_recall:.4f} vs V1 {v1_recall:.4f})")
            if not fp_acceptable:
                reasons.append(f"False positives exploded to an unacceptable level ({cand_fp} vs V1 {v1_fp})")
            beats_v1_reason = "Candidate does NOT beat V1: " + ", ".join(reasons)
            
    if is_better:
        print(f"\nPROMOTION VERDICT: YES! {beats_v1_reason}")
    else:
        print(f"\nPROMOTION VERDICT: NO. V1 remains the best model. Reason: {beats_v1_reason}")
        
    # 6. Save JSON Comparison Report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_json_path = os.path.join(candidates_dir, f"comparison_report_{timestamp}.json")
    report_md_path = os.path.join(candidates_dir, f"comparison_report_{timestamp}.md")
    
    # Save the latest comparison report in a static filename as well
    static_report_json = os.path.join(candidates_dir, "comparison_report.json")
    static_report_md = os.path.join(candidates_dir, "comparison_report.md")
    
    report_data = {
        "evaluation_date": datetime.now().isoformat(),
        "dataset_used": "BUSI",
        "total_test_images": len(X_test),
        "test_class_distribution": {
            classes[i]: int(np.sum(y_test == i)) for i in range(len(classes))
        },
        "best_candidate": best_candidate_name,
        "beats_v1": is_better,
        "beats_v1_reason": beats_v1_reason,
        "results": results
    }
    
    with open(report_json_path, 'w') as f:
        json.dump(report_data, f, indent=4)
    with open(static_report_json, 'w') as f:
        json.dump(report_data, f, indent=4)
    print(f"\nSaved comparison JSON report to: {report_json_path}")
    
    # 7. Generate formatted Markdown report
    md_content = f"""# V3 Candidate Model Comparison Report

* **Evaluation Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
* **Dataset Used:** BUSI Dataset (Held-out Test Split, random_state=42)
* **Total Test Images:** {len(X_test)}
* **Test Split Counts:**
  * Benign: {np.sum(y_test == 0)}
  * Malignant: {np.sum(y_test == 1)}
  * Normal: {np.sum(y_test == 2)}

---

## 1. Promotion Summary

* **Best Candidate Model:** `{best_candidate_name if best_candidate_name else "None"}`
* **Beats V1 Baseline:** **{"YES" if is_better else "NO"}**
* **Recommendation:** **{"PROMOTION ALLOWED" if is_better else "V1 remains the best model."}**

> [!NOTE]
> **Safety Justification / Reason:**
> {beats_v1_reason}

---

## 2. Quantitative Model Comparison

| Model | Accuracy | Malignant Recall (Sens) | Malignant Precision | Malignant F1 | False Negatives (FN) | False Positives (FP) | Total Errors |
|---|---|---|---|---|---|---|---|
"""
    for name, m in results.items():
        is_best_str = "**" if name == best_candidate_name else ""
        md_content += (
            f"| {is_best_str}{name}{is_best_str} | {m['accuracy']:.4f} | {m['recall'][1]:.4f} | {m['precision'][1]:.4f} | "
            f"{m['f1'][1]:.4f} | {m['false_negatives']} | {m['false_positives']} | {m['incorrect_count']} |\n"
        )
        
    md_content += """
---

## 3. Per-Class Detailed Metrics

"""
    for name, m in results.items():
        md_content += f"### {name}\n\n"
        md_content += "| Class | Precision | Recall | F1-Score | Support |\n|---|---|---|---|---|\n"
        for i, c_name in enumerate(classes):
            support_count = int(np.sum(y_test == i))
            md_content += f"| **{c_name.capitalize()}** | {m['precision'][i]:.4f} | {m['recall'][i]:.4f} | {m['f1'][i]:.4f} | {support_count} |\n"
        
        md_content += f"\n**Confusion Matrix:**\n```\n"
        md_content += f"Predicted ->   Benign  Malignant   Normal\n"
        md_content += f"True Benign    [{m['confusion_matrix'][0][0]:>3d}     {m['confusion_matrix'][0][1]:>3d}       {m['confusion_matrix'][0][2]:>3d}]\n"
        md_content += f"True Malignant [{m['confusion_matrix'][1][0]:>3d}     {m['confusion_matrix'][1][1]:>3d}       {m['confusion_matrix'][1][2]:>3d}]\n"
        md_content += f"True Normal    [{m['confusion_matrix'][2][0]:>3d}     {m['confusion_matrix'][2][1]:>3d}       {m['confusion_matrix'][2][2]:>3d}]\n"
        md_content += "```\n\n"
        md_content += "---\n\n"
        
    with open(report_md_path, 'w') as f:
        f.write(md_content)
    with open(static_report_md, 'w') as f:
        f.write(md_content)
        
    print(f"Saved formatted Markdown report to: {report_md_path}")
    print("\nEvaluation comparison completed successfully.")

if __name__ == "__main__":
    main()
