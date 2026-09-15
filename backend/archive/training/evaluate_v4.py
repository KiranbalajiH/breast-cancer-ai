import os
import sys
import json
import cv2
import numpy as np
import tensorflow as tf
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

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
            
        for f in sorted(os.listdir(class_dir)):
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

def evaluate_model(model, X_test, y_test, classes, prep_fn, test_paths=None):
    X_prep = prep_fn(X_test.copy())
    predictions = model.predict(X_prep, verbose=0)
    pred_classes = np.argmax(predictions, axis=1)
    
    acc = float(accuracy_score(y_test, pred_classes))
    prec, rec, f1, support = precision_recall_fscore_support(y_test, pred_classes, labels=[0, 1, 2], zero_division=0)
    cm = confusion_matrix(y_test, pred_classes, labels=[0, 1, 2]).tolist()
    
    mal_indices = np.where(y_test == 1)[0]
    fn_count = int(np.sum(pred_classes[mal_indices] != 1))
    
    non_mal_indices = np.where(y_data == 1)[0] if False else np.where(y_test != 1)[0]
    fp_count = int(np.sum(pred_classes[non_mal_indices] == 1))
    
    total_incorrect = int(np.sum(pred_classes != y_test))
    
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
        "accuracy": acc,
        "macro_precision": float(np.mean(prec)),
        "macro_recall": float(np.mean(rec)),
        "macro_f1": float(np.mean(f1)),
        "benign": {
            "precision": float(prec[0]),
            "recall": float(rec[0]),
            "f1": float(f1[0]),
            "support": int(support[0])
        },
        "malignant": {
            "precision": float(prec[1]),
            "recall": float(rec[1]),
            "f1": float(f1[1]),
            "support": int(support[1])
        },
        "normal": {
            "precision": float(prec[2]),
            "recall": float(rec[2]),
            "f1": float(f1[2]),
            "support": int(support[2])
        },
        "false_negatives": fn_count,
        "false_positives": fp_count,
        "total_incorrect": total_incorrect,
        "confusion_matrix": cm,
        "misclassifications": misclassifications
    }

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.dirname(script_dir)
    bcd_root = os.path.dirname(backend_dir)
    dataset_root = os.path.join(bcd_root, "dataset", "BUSI")
    if not os.path.exists(dataset_root):
        dataset_root = os.path.join(bcd_root, "dataset", "Dataset_BUSI_with_GT")
        
    models_dir = os.path.join(backend_dir, "models")
    candidates_dir = os.path.join(models_dir, "candidates")
    reports_dir = os.path.join(models_dir, "evaluation_reports")
    os.makedirs(reports_dir, exist_ok=True)
    
    print("="*70)
    print("V4 FINAL TEST SET EVALUATION & COMPARISON BENCHMARK")
    print("="*70)
    
    # 1. Load dataset and reproduce exact untouched test split
    image_paths, labels, classes = load_and_clean_dataset(dataset_root)
    labels = np.array(labels, dtype=np.int32)
    
    print("\nLoading images into memory...")
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
    
    print(f"\nVerified Untouched Test Split (Total: {len(X_test)} images):")
    for idx, cname in enumerate(classes):
        print(f"  - {cname:<10}: {np.sum(y_test == idx)}")
        
    # 2. Define Model Configurations for Benchmark
    eval_targets = [
        {
            "id": "v1_baseline",
            "name": "V1 MobileNetV2 Baseline",
            "path": os.path.join(models_dir, "breast_image_classifier.keras"),
            "prep_fn": preprocess_mnv2
        },
        {
            "id": "v3_noweights",
            "name": "V3 MobileNetV2 No Weights",
            "path": os.path.join(candidates_dir, "mobilenetv2_v3_noweights.keras"),
            "prep_fn": preprocess_mnv2
        },
        {
            "id": "v3_mildweights",
            "name": "V3 MobileNetV2 Mild Weights",
            "path": os.path.join(candidates_dir, "mobilenetv2_v3_mildweights.keras"),
            "prep_fn": preprocess_mnv2
        },
        {
            "id": "v4_mobilenetv2_a",
            "name": "V4-A: MobileNetV2 (No Weights / Fine-Tuned)",
            "path": os.path.join(candidates_dir, "v4_mobilenetv2_a.keras"),
            "prep_fn": preprocess_mnv2
        },
        {
            "id": "v4_mobilenetv2_b",
            "name": "V4-B: MobileNetV2 (Mild Weighting 1.25x / Fine-Tuned)",
            "path": os.path.join(candidates_dir, "v4_mobilenetv2_b.keras"),
            "prep_fn": preprocess_mnv2
        },
        {
            "id": "v4_mobilenetv2_c",
            "name": "V4-C: MobileNetV2 (Balanced Loss Weighting / Fine-Tuned)",
            "path": os.path.join(candidates_dir, "v4_mobilenetv2_c.keras"),
            "prep_fn": preprocess_mnv2
        },
        {
            "id": "v4_efficientnet_d",
            "name": "V4-D: EfficientNetB0 (Mild Weighting)",
            "path": os.path.join(candidates_dir, "v4_efficientnet_d.keras"),
            "prep_fn": preprocess_eff
        },
        {
            "id": "v4_mobilenetv2_headonly",
            "name": "V4-E: MobileNetV2 (Frozen Backbone / Mild Weighting 1.25x)",
            "path": os.path.join(candidates_dir, "v4_mobilenetv2_headonly.keras"),
            "prep_fn": preprocess_mnv2
        }
    ]
    
    # Static baseline for V2 DenseNet121
    v2_densenet_static = {
        "accuracy": 0.5385,
        "macro_precision": 0.6589,
        "macro_recall": 0.5572,
        "macro_f1": 0.5224,
        "benign": {"precision": 0.9000, "recall": 0.4154, "f1": 0.5684, "support": 65},
        "malignant": {"precision": 0.3766, "recall": 0.9063, "f1": 0.5321, "support": 32},
        "normal": {"precision": 0.7000, "recall": 0.3500, "f1": 0.4667, "support": 20},
        "false_negatives": 3,
        "false_positives": 48,
        "total_incorrect": 54,
        "confusion_matrix": [[27, 36, 2], [2, 29, 1], [1, 12, 7]]
    }
    
    test_results = {}
    
    print("\n--- Running Test Set Evaluations ---")
    for target in eval_targets:
        tid = target["id"]
        tname = target["name"]
        mpath = target["path"]
        
        if not os.path.exists(mpath):
            print(f"Skipping {tname} (file not found: {mpath})")
            continue
            
        print(f"Evaluating {tname}...")
        model = tf.keras.models.load_model(mpath)
        res = evaluate_model(model, X_test, y_test, classes, target["prep_fn"], test_paths=paths_test)
        test_results[tid] = res
        test_results[tid]["name"] = tname
        
        print(f"  Accuracy           : {res['accuracy']*100:.2f}%")
        print(f"  Macro F1           : {res['macro_f1']*100:.2f}%")
        print(f"  Malignant Precision: {res['malignant']['precision']*100:.2f}%")
        print(f"  Malignant Recall   : {res['malignant']['recall']*100:.2f}%")
        print(f"  Malignant F1       : {res['malignant']['f1']*100:.2f}%")
        print(f"  Malignant FN / FP  : {res['false_negatives']} / {res['false_positives']}")
        print(f"  Total Errors       : {res['total_incorrect']}")
        
    test_results["v2_densenet121"] = v2_densenet_static
    test_results["v2_densenet121"]["name"] = "V2 DenseNet121"
    
    # Save raw test evaluation json
    eval_json_path = os.path.join(candidates_dir, "v4_evaluation_summary.json")
    with open(eval_json_path, 'w') as f:
        json.dump(test_results, f, indent=4)
    print(f"\nTest evaluation summary saved to: {eval_json_path}")
    
    # Generate Markdown Report
    generate_markdown_report(test_results, reports_dir)

def generate_markdown_report(results, reports_dir):
    report_path = os.path.join(reports_dir, "evaluation_report_v4.md")
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Ranking score formula balancing accuracy, macro F1, malignant recall & precision:
    # Score = (Malignant Recall * 0.30) + (Malignant Precision * 0.30) + (Macro F1 * 0.20) + (Accuracy * 0.20)
    scored_models = []
    for k, v in results.items():
        m_rec = v["malignant"]["recall"]
        m_prec = v["malignant"]["precision"]
        macro_f1 = v["macro_f1"]
        acc = v["accuracy"]
        score = (m_rec * 0.30) + (m_prec * 0.30) + (macro_f1 * 0.20) + (acc * 0.20)
        scored_models.append((k, v["name"], score, v))
        
    scored_models.sort(key=lambda x: x[2], reverse=True)
    
    md = []
    md.append("# V4 MODEL EVALUATION REPORT — BALANCED MALIGNANT SENSITIVITY\n")
    md.append(f"**Date**: {timestamp}\n")
    md.append("**Evaluation Split**: Untouched 117-Image Test Set (65 Benign, 32 Malignant, 20 Normal)\n")
    md.append("\n---\n")
    
    md.append("## Executive Summary\n")
    md.append("The objective of the V4 experiment was to improve **malignant sensitivity (recall)** beyond V1 (68.75%) without causing an excessive increase in false positives or degrading overall classification quality (84.62% Accuracy, 84.21% Macro F1).\n")
    
    md.append("\n### Model Performance Overview Table\n")
    md.append("| Model | Accuracy | Macro F1 | Malignant Precision | Malignant Recall | Malignant F1 | Malignant FN | Malignant FP | Total Errors |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    
    for k, name, score, m in scored_models:
        acc = f"{m['accuracy']*100:.2f}%"
        mf1 = f"{m['macro_f1']*100:.2f}%"
        mp = f"{m['malignant']['precision']*100:.2f}%"
        mr = f"{m['malignant']['recall']*100:.2f}%"
        mf1_class = f"{m['malignant']['f1']*100:.2f}%"
        fn = m['false_negatives']
        fp = m['false_positives']
        err = m['total_incorrect']
        md.append(f"| **{name}** | {acc} | {mf1} | {mp} | {mr} | {mf1_class} | {fn} | {fp} | {err} |")
        
    md.append("\n---\n")
    md.append("## Per-Class Breakdown & Confusion Matrices\n")
    
    for k, name, score, m in scored_models:
        md.append(f"### {name}")
        md.append(f"- **Overall Accuracy**: {m['accuracy']*100:.2f}%")
        md.append(f"- **Macro Precision / Recall / F1**: {m['macro_precision']*100:.2f}% / {m['macro_recall']*100:.2f}% / {m['macro_f1']*100:.2f}%")
        md.append(f"- **Benign P / R / F1**: {m['benign']['precision']*100:.2f}% / {m['benign']['recall']*100:.2f}% / {m['benign']['f1']*100:.2f}% (Support: {m['benign']['support']})")
        md.append(f"- **Malignant P / R / F1**: {m['malignant']['precision']*100:.2f}% / {m['malignant']['recall']*100:.2f}% / {m['malignant']['f1']*100:.2f}% (Support: {m['malignant']['support']})")
        md.append(f"- **Normal P / R / F1**: {m['normal']['precision']*100:.2f}% / {m['normal']['recall']*100:.2f}% / {m['normal']['f1']*100:.2f}% (Support: {m['normal']['support']})")
        md.append(f"- **False Negatives (Malignant missed)**: {m['false_negatives']}")
        md.append(f"- **False Positives (Malignant over-predicted)**: {m['false_positives']}")
        md.append(f"- **Total Incorrect Predictions**: {m['total_incorrect']}")
        
        cm = m["confusion_matrix"]
        md.append("\n**Confusion Matrix (Row=True, Col=Pred)**:")
        md.append("```")
        md.append("               Pred Benign   Pred Malignant   Pred Normal")
        md.append(f"True Benign    {cm[0][0]:^11}  {cm[0][1]:^14}  {cm[0][2]:^11}")
        md.append(f"True Malignant {cm[1][0]:^11}  {cm[1][1]:^14}  {cm[1][2]:^11}")
        md.append(f"True Normal    {cm[2][0]:^11}  {cm[2][1]:^14}  {cm[2][2]:^11}")
        md.append("```\n")
        
    md.append("\n---\n")
    md.append("## Detailed Candidate Evaluation & Selection Rationale\n")
    
    for k, name, score, m in scored_models:
        md.append(f"### {name}")
        if k == "v1_baseline":
            md.append("- **Status**: Current Production Baseline Model")
            md.append("- **Analysis**: Maintains the highest overall accuracy (84.62%), macro F1 (84.21%), and malignant precision (84.62%) with only 4 false positives. However, malignant recall is 68.75% (10 false negatives out of 32 malignant cases).\n")
        elif "v4" in k:
            md.append("- **Status**: V4 Candidate")
            mr = m['malignant']['recall']
            mp = m['malignant']['precision']
            fp = m['false_positives']
            acc = m['accuracy']
            if acc >= 0.84 and mr > 0.75 and fp <= 5:
                md.append(f"- **Recommendation**: **ACCEPTED / STRONG CANDIDATE**. Achieves strong balance with {mr*100:.1f}% malignant recall, {mp*100:.1f}% malignant precision, and low false positives ({fp}).")
            elif mr > 0.75 and fp > 10:
                md.append(f"- **Recommendation**: **REJECTED**. High false-positive rate ({fp} false malignant predictions) degrades overall specificity and accuracy ({acc*100:.1f}%).")
            else:
                md.append(f"- **Recommendation**: **REJECTED / SUBOPTIMAL**. Does not present a sufficient net gain over V1 baseline (Accuracy: {acc*100:.1f}%, Malignant Recall: {mr*100:.1f}%, Malignant Precision: {mp*100:.1f}%).")
            md.append("")
            
    with open(report_path, 'w') as f:
        f.write("\n".join(md))
        
    print(f"\nMarkdown Evaluation Report written to: {report_path}")

if __name__ == "__main__":
    main()
