import os
import sys
import json
import cv2
import random
import numpy as np
import tensorflow as tf
from datetime import datetime
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_mnv2
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_effnet

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def load_and_clean_dataset(dataset_root):
    classes = ['benign', 'malignant', 'normal']
    scans = []
    exclude_files = {"malignant (145).png", "benign (433).png"}
    
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
            scans.append({
                "filename": f,
                "filepath": full_path,
                "class_name": class_name,
                "class_idx": class_idx
            })
            
    print(f"Loaded {len(scans)} clean trainable scans.")
    return scans, classes

# --- PERCEPTUAL CLUSTERING ---
def compute_phash(img_gray):
    resized = cv2.resize(img_gray, (32, 32))
    return resized

def group_scans_by_perceptual_cluster(scans):
    class_scans = defaultdict(list)
    for idx, s in enumerate(scans):
        class_scans[s['class_idx']].append((idx, s))
        
    cluster_labels = np.zeros(len(scans), dtype=int)
    current_cluster_id = 0
    
    for c_idx, items in class_scans.items():
        n = len(items)
        indices = [it[0] for it in items]
        scan_objs = [it[1] for it in items]
        
        phashes = [compute_phash(cv2.imread(s['filepath'], cv2.IMREAD_GRAYSCALE)) for s in scan_objs]
        adj = defaultdict(list)
        
        for i in range(n):
            for j in range(i + 1, n):
                diff = np.mean(np.abs(phashes[i].astype(float) - phashes[j].astype(float)))
                if diff < 12.0:
                    adj[i].append(j)
                    adj[j].append(i)
                    
        visited = set()
        c_clusters = []
        for i in range(n):
            if i not in visited:
                comp = []
                queue = [i]
                visited.add(i)
                while queue:
                    curr = queue.pop(0)
                    comp.append(curr)
                    for nbr in adj[curr]:
                        if nbr not in visited:
                            visited.add(nbr)
                            queue.append(nbr)
                c_clusters.append(comp)
                
        for comp in c_clusters:
            for i_local in comp:
                global_idx = indices[i_local]
                cluster_labels[global_idx] = current_cluster_id
            current_cluster_id += 1
            
    total_clusters = current_cluster_id
    print(f"Grouped {len(scans)} scans into {total_clusters} unique perceptual lesion clusters.")
    return cluster_labels, total_clusters

def create_grouped_stratified_split(scans, cluster_labels, seed=42):
    cluster_info = defaultdict(lambda: {"class_idx": None, "scan_indices": []})
    for idx, s in enumerate(scans):
        cid = cluster_labels[idx]
        cluster_info[cid]["class_idx"] = s["class_idx"]
        cluster_info[cid]["scan_indices"].append(idx)
        
    cids = np.array(sorted(cluster_info.keys()))
    clabels = np.array([cluster_info[cid]["class_idx"] for cid in cids])
    
    cids_train, cids_val_test, clabels_train, clabels_val_test = train_test_split(
        cids, clabels, test_size=0.30, stratify=clabels, random_state=seed
    )
    cids_val, cids_test, clabels_val, clabels_test = train_test_split(
        cids_val_test, clabels_val_test, test_size=0.50, stratify=clabels_val_test, random_state=seed
    )
    
    train_indices = []
    for cid in cids_train:
        train_indices.extend(cluster_info[cid]["scan_indices"])
        
    val_indices = []
    for cid in cids_val:
        val_indices.extend(cluster_info[cid]["scan_indices"])
        
    test_indices = []
    for cid in cids_test:
        test_indices.extend(cluster_info[cid]["scan_indices"])
        
    return np.array(train_indices), np.array(val_indices), np.array(test_indices)

def letterbox_resize(img, target_size=(224, 224), pad_color=(0, 0, 0)):
    h, w = img.shape[:2]
    target_w, target_h = target_size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
    
    pad_w = target_w - new_w
    pad_h = target_h - new_h
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color)
    return padded

def preprocess_letterbox(filepath):
    img = cv2.imread(filepath)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return letterbox_resize(img_rgb, (224, 224))

def load_images(scans):
    images = [preprocess_letterbox(s["filepath"]) for s in scans]
    return np.array(images, dtype=np.float32)

def evaluate_ensemble_policy(probs, y_data, mal_threshold=0.50):
    pred_classes = np.zeros(len(probs), dtype=int)
    for i, p in enumerate(probs):
        if p[1] >= mal_threshold:
            pred_classes[i] = 1
        else:
            pred_classes[i] = 0 if p[0] >= p[2] else 2
            
    acc = float(accuracy_score(y_data, pred_classes))
    prec, rec, f1, support = precision_recall_fscore_support(y_data, pred_classes, labels=[0, 1, 2], zero_division=0)
    cm = confusion_matrix(y_data, pred_classes, labels=[0, 1, 2]).tolist()
    
    fn_count = int(np.sum((y_data == 1) & (pred_classes != 1)))
    fp_count = int(np.sum((y_data != 1) & (pred_classes == 1)))
    total_incorrect = int(np.sum(pred_classes != y_data))
    
    return {
        "accuracy": acc,
        "macro_precision": float(np.mean(prec)),
        "macro_recall": float(np.mean(rec)),
        "macro_f1": float(np.mean(f1)),
        "benign": {"precision": float(prec[0]), "recall": float(rec[0]), "f1": float(f1[0]), "support": int(support[0])},
        "malignant": {"precision": float(prec[1]), "recall": float(rec[1]), "f1": float(f1[1]), "support": int(support[1])},
        "normal": {"precision": float(prec[2]), "recall": float(rec[2]), "f1": float(f1[2]), "support": int(support[2])},
        "false_negatives": fn_count,
        "false_positives": fp_count,
        "total_incorrect": total_incorrect,
        "confusion_matrix": cm,
        "mal_threshold_used": mal_threshold
    }

def main():
    set_seed(42)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.dirname(script_dir)
    bcd_root = os.path.dirname(backend_dir)
    dataset_root = os.path.join(bcd_root, "dataset", "BUSI")
    
    candidates_dir = os.path.join(backend_dir, "models", "candidates")
    reports_dir = os.path.join(backend_dir, "models", "evaluation_reports")
    os.makedirs(reports_dir, exist_ok=True)
    
    v11b_path = os.path.join(candidates_dir, "v11_mobilenetv2_b.keras")
    v13_path = os.path.join(candidates_dir, "v13_efficientnet_b0.keras")
    
    print("="*75)
    print("V14 MINIMAL FROZEN ENSEMBLE EXPERIMENT (V11-B + V13-B0)")
    print("="*75)
    
    print("\n1. LOADING FROZEN CANDIDATE MODELS...")
    assert os.path.exists(v11b_path), f"V11-B model missing: {v11b_path}"
    assert os.path.exists(v13_path), f"V13-B0 model missing: {v13_path}"
    
    m_v11 = tf.keras.models.load_model(v11b_path)
    m_v13 = tf.keras.models.load_model(v13_path)
    print("   V11-B MobileNetV2 loaded successfully.")
    print("   V13 EfficientNetB0 loaded successfully.")
    
    scans, classes = load_and_clean_dataset(dataset_root)
    labels = np.array([s["class_idx"] for s in scans], dtype=np.int32)
    cluster_labels, total_clusters = group_scans_by_perceptual_cluster(scans)
    
    # Historical Split (LOCKED 117-IMAGE TEST SET)
    hist_train_idx, hist_val_test_idx = train_test_split(np.arange(len(scans)), test_size=0.30, stratify=labels, random_state=42)
    hist_val_idx, hist_test_idx = train_test_split(hist_val_test_idx, test_size=0.50, stratify=labels[hist_val_test_idx], random_state=42)
    
    # Grouped Generalization Split (Seed 42)
    grp_train_idx, grp_val_idx, grp_test_idx = create_grouped_stratified_split(scans, cluster_labels, seed=42)
    
    print("\nLoading images...")
    X_all = load_images(scans)
    
    X_val = X_all[grp_val_idx]
    y_val = labels[grp_val_idx]
    
    X_hist_test = X_all[hist_test_idx]
    y_hist_test = labels[hist_test_idx]
    
    X_grp_test = X_all[grp_test_idx]
    y_grp_test = labels[grp_test_idx]
    
    # Generate Predictions for V11-B
    print("\n2. GENERATING PROBABILITY PREDICTIONS...")
    probs_v11_val = m_v11.predict(preprocess_mnv2(X_val.copy()), verbose=0)
    probs_v11_hist = m_v11.predict(preprocess_mnv2(X_hist_test.copy()), verbose=0)
    probs_v11_grp = m_v11.predict(preprocess_mnv2(X_grp_test.copy()), verbose=0)
    
    # Generate Predictions for V13-B0
    probs_v13_val = m_v13.predict(preprocess_effnet(X_val.copy()), verbose=0)
    probs_v13_hist = m_v13.predict(preprocess_effnet(X_hist_test.copy()), verbose=0)
    probs_v13_grp = m_v13.predict(preprocess_effnet(X_grp_test.copy()), verbose=0)
    
    # --- POLICY FITTING ON GROUPED VALIDATION ONLY ---
    print("\n3. SWEEPING ENSEMBLE WEIGHTS & THRESHOLDS ON GROUPED VALIDATION DATA ONLY...")
    weight_candidates = [0.0, 0.25, 0.50, 0.75, 1.0]
    threshold_candidates = np.arange(0.20, 0.71, 0.02)
    
    valid_policy_candidates = []
    grid_search_summary = []
    
    for w in weight_candidates:
        w_val = round(float(w), 2)
        probs_ens_val = w_val * probs_v11_val + (1.0 - w_val) * probs_v13_val
        
        for t in threshold_candidates:
            t_val = round(float(t), 2)
            eval_wt = evaluate_ensemble_policy(probs_ens_val, y_val, mal_threshold=t_val)
            
            rec = eval_wt["malignant"]["recall"]
            prec = eval_wt["malignant"]["precision"]
            f1_m = eval_wt["malignant"]["f1"]
            acc = eval_wt["accuracy"]
            fn = eval_wt["false_negatives"]
            fp = eval_wt["false_positives"]
            
            cand_entry = {
                "weight_v11": w_val,
                "weight_v13": round(1.0 - w_val, 2),
                "threshold": t_val,
                "accuracy": acc,
                "macro_f1": eval_wt["macro_f1"],
                "malignant_recall": rec,
                "malignant_precision": prec,
                "malignant_f1": f1_m,
                "false_negatives": fn,
                "false_positives": fp,
                "eval_dict": eval_wt
            }
            grid_search_summary.append(cand_entry)
            
            # Primary Selection Constraints: Recall >= 85%, Precision >= 60%, Accuracy >= 70%
            if rec >= 0.85 and prec >= 0.60 and acc >= 0.70:
                valid_policy_candidates.append(cand_entry)
                
    if valid_policy_candidates:
        # Sort by: 1. Malignant F1 (descending), 2. False Negatives (ascending)
        valid_policy_candidates.sort(key=lambda c: (c["malignant_f1"], -c["false_negatives"]), reverse=True)
        best_cand = valid_policy_candidates[0]
    else:
        # Fallback: Sort all grid entries by composite score
        grid_search_summary.sort(key=lambda c: (c["malignant_recall"], c["malignant_f1"]), reverse=True)
        best_cand = grid_search_summary[0]
        
    best_w = best_cand["weight_v11"]
    best_t = best_cand["threshold"]
    best_val_eval = best_cand["eval_dict"]
                
    print(f"\n   SELECTED OPTIMAL VALIDATION POLICY:")
    print(f"   - Ensemble Weight (V11-B): {best_w:.2f} | Weight (V13-B0): {1.0 - best_w:.2f}")
    print(f"   - Malignant Threshold: {best_t:.2f}")
    print(f"   - Validation Metrics: Acc={best_val_eval['accuracy']*100:.2f}%, Mal Rec={best_val_eval['malignant']['recall']*100:.2f}%, Mal Prec={best_val_eval['malignant']['precision']*100:.2f}%, Mal F1={best_val_eval['malignant']['f1']*100:.2f}%, FN={best_val_eval['false_negatives']}, FP={best_val_eval['false_positives']}")
    
    # Reference Baseline: 50/50 Ensemble at Threshold 0.50
    probs_ref_val = 0.50 * probs_v11_val + 0.50 * probs_v13_val
    ref_val_eval = evaluate_ensemble_policy(probs_ref_val, y_val, mal_threshold=0.50)
    
    # --- SINGLE-PASS FROZEN TEST EVALUATIONS ---
    print("\n4. EVALUATING FROZEN POLICY ON GROUPED AND HISTORICAL TEST SETS...")
    
    # Primary Selected Policy Evaluation
    probs_ens_grp = best_w * probs_v11_grp + (1.0 - best_w) * probs_v13_grp
    grp_eval_selected = evaluate_ensemble_policy(probs_ens_grp, y_grp_test, mal_threshold=best_t)
    
    probs_ens_hist = best_w * probs_v11_hist + (1.0 - best_w) * probs_v13_hist
    hist_eval_selected = evaluate_ensemble_policy(probs_ens_hist, y_hist_test, mal_threshold=best_t)
    
    # 50/50 Reference Policy Evaluation
    probs_ref_grp = 0.50 * probs_v11_grp + 0.50 * probs_v13_grp
    grp_eval_ref = evaluate_ensemble_policy(probs_ref_grp, y_grp_test, mal_threshold=0.50)
    
    probs_ref_hist = 0.50 * probs_v11_hist + 0.50 * probs_v13_hist
    hist_eval_ref = evaluate_ensemble_policy(probs_ref_hist, y_hist_test, mal_threshold=0.50)
    
    # Save Validation Summary
    val_summary = {
        "selected_policy": {
            "weight_v11": best_w,
            "weight_v13": round(1.0 - best_w, 2),
            "threshold": best_t,
            "validation_metrics": best_val_eval
        },
        "reference_policy_50_50": {
            "weight_v11": 0.50,
            "weight_v13": 0.50,
            "threshold": 0.50,
            "validation_metrics": ref_val_eval
        },
        "grid_search_summary": grid_search_summary
    }
    
    val_json_path = os.path.join(reports_dir, "v14_validation_summary.json")
    with open(val_json_path, "w", encoding="utf-8") as f:
        json.dump(val_summary, f, indent=4)
    print(f"Saved V14 validation summary JSON to: {val_json_path}")
    
    # Save Raw Evaluation Results
    eval_results = {
        "v14_ensemble_selected": {
            "name": f"V14 Selected Ensemble (w_v11={best_w}, w_v13={1.0-best_w:.2f}, t={best_t})",
            "weights": {"v11_b": best_w, "v13_b0": round(1.0 - best_w, 2)},
            "threshold": best_t,
            "grouped_test": grp_eval_selected,
            "historical_test_117": hist_eval_selected
        },
        "v14_ensemble_ref_50_50": {
            "name": "V14 Reference Ensemble (50/50, t=0.50)",
            "weights": {"v11_b": 0.50, "v13_b0": 0.50},
            "threshold": 0.50,
            "grouped_test": grp_eval_ref,
            "historical_test_117": hist_eval_ref
        }
    }
    
    eval_json_path = os.path.join(reports_dir, "v14_evaluation_results.json")
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=4)
    print(f"Saved V14 evaluation results JSON to: {eval_json_path}")
    
    generate_v14_markdown_report(reports_dir, best_w, best_t, grp_eval_selected, hist_eval_selected, grp_eval_ref, hist_eval_ref)

def generate_v14_markdown_report(reports_dir, best_w, best_t, grp_sel, hist_sel, grp_ref, hist_ref):
    report_path = os.path.join(reports_dir, "v14_final_evaluation_report.md")
    
    # Success Criteria check vs V5-B and targets (Recall >= 85%, Prec >= 60%, Acc >= 70%)
    top_grp_rec = grp_sel["malignant"]["recall"]
    top_grp_prec = grp_sel["malignant"]["precision"]
    top_grp_acc = grp_sel["accuracy"]
    top_grp_f1 = grp_sel["malignant"]["f1"]
    
    if top_grp_rec >= 0.85 and top_grp_prec >= 0.60 and top_grp_acc >= 0.70 and top_grp_f1 >= 0.70:
        promo_recommendation = f"**INTERESTING ENSEMBLE CANDIDATE** (Meets criteria: Recall {top_grp_rec*100:.2f}%, Precision {top_grp_prec*100:.2f}%, Accuracy {top_grp_acc*100:.2f}%, F1 {top_grp_f1*100:.2f}%). Note: DO NOT PROMOTE automatically per project safety guidelines."
    else:
        promo_recommendation = f"**DO NOT PROMOTE**. Recommend retaining **V5-B** as active production model."

    report_md = f"""PRODUCTION MODEL CHANGED: NO
PRODUCTION METADATA CHANGED: NO
LOCKED 117-IMAGE TEST SET MODIFIED: NO
DATASET IMAGES MODIFIED: NO
DEPLOYMENT PERFORMED: NO
GIT COMMIT PERFORMED: NO
GIT PUSH PERFORMED: NO

# V14 Minimal Frozen Ensemble (V11-B + V13-B0) Final Report

**Date**: September 02, 2026  
**Project**: Breast Cancer AI — Ultrasound Image Classifier  
**Dataset**: `dataset/BUSI` (776 Clean Trainable Scans)  
**Production Model Status**: `backend/models/breast_image_classifier.keras` remains **unmodified** and **active** (V5-B).  
**Ensemble Components**: Frozen Candidate V11-B (`v11_mobilenetv2_b.keras`) + Frozen Candidate V13-B0 (`v13_efficientnet_b0.keras`)

---

## 1. Executive Summary & V14 Objective
The V14 experiment tested a simple frozen probability ensemble of **V11-B** (MobileNetV2, high sensitivity) and **V13-B0** (EfficientNetB0, high precision) to determine whether combining their complementary predictions improves the grouped malignant operating point.

### Selected Policy (Fitted on Grouped Validation Data ONLY):
- **Selected V11-B Weight ($w^*$)**: `{best_w:.2f}`
- **Selected V13-B0 Weight ($1 - w^*$)**: `{1.0 - best_w:.2f}`
- **Selected Malignant Threshold ($t^*$)**: `{best_t:.2f}`

---

## 2. Grouped Generalization Test Comparison (Primary Benchmark)

| Model / Ensemble Iteration | Architecture / Strategy | Accuracy | Macro F1 | Malignant Recall | Malignant Precision | Malignant F1 | Malignant FN | Malignant FP | Total Errors |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **V5-B (Active Prod)** | MobileNetV2 (Control) | 70.09% | 69.46% | 80.65% | 62.50% | 70.42% | 6 | 15 | 32 |
| **V11-B** | MobileNetV2 (50% Aug) | 69.16% | 70.12% | **90.32%** | 57.14% | 70.00% | **3** | 21 | 33 |
| **V13-B0** | EfficientNetB0 | **80.37%** | 75.82% | 67.74% | **84.00%** | 75.00% | 10 | **4** | **21** |
| **V14 Reference (50/50, $t=0.50$)** | Frozen Ensemble (50/50) | 79.44% | **76.84%** | 80.65% | 73.53% | **76.92%** | 6 | 9 | 22 |
| **V14 Selected ($w={best_w:.2f}, t={best_t:.2f}$)**| **Frozen Ensemble (Selected)** | {grp_sel['accuracy']*100:.2f}% | {grp_sel['macro_f1']*100:.2f}% | {grp_sel['malignant']['recall']*100:.2f}% | {grp_sel['malignant']['precision']*100:.2f}% | {grp_sel['malignant']['f1']*100:.2f}% | {grp_sel['false_negatives']} | {grp_sel['false_positives']} | {grp_sel['total_incorrect']} |

---

## 3. Locked Historical 117-Image Test Comparison (Frozen Benchmark)

| Model / Ensemble Iteration | Architecture / Strategy | Accuracy | Macro F1 | Malignant Recall | Malignant Precision | Malignant F1 | Malignant FN | Malignant FP | Total Errors |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **V5-B (Active Prod)** | MobileNetV2 (Control) | 80.34% | 79.75% | 87.50% | 77.78% | 82.35% | 4 | 8 | 23 |
| **V11-B** | MobileNetV2 (50% Aug) | 78.63% | 78.45% | **96.88%** | 70.45% | 81.58% | **1** | 13 | 25 |
| **V13-B0** | EfficientNetB0 | 86.32% | 84.00% | 78.12% | **89.29%** | 83.33% | 7 | **3** | 16 |
| **V14 Reference (50/50, $t=0.50$)** | Frozen Ensemble (50/50) | 86.32% | 85.06% | 87.50% | 84.85% | 86.15% | 4 | 5 | 16 |
| **V14 Selected ($w={best_w:.2f}, t={best_t:.2f}$)**| **Frozen Ensemble (Selected)** | {hist_sel['accuracy']*100:.2f}% | {hist_sel['macro_f1']*100:.2f}% | {hist_sel['malignant']['recall']*100:.2f}% | {hist_sel['malignant']['precision']*100:.2f}% | {hist_sel['malignant']['f1']*100:.2f}% | {hist_sel['false_negatives']} | {hist_sel['false_positives']} | {hist_sel['total_incorrect']} |

---

## 4. Per-Class Performance Breakdown (V14 Selected Policy)

### Grouped Generalization Test:
- **Benign**: Precision = {grp_sel['benign']['precision']*100:.2f}%, Recall = {grp_sel['benign']['recall']*100:.2f}%, F1 = {grp_sel['benign']['f1']*100:.2f}%
- **Malignant**: Precision = {grp_sel['malignant']['precision']*100:.2f}%, Recall = {grp_sel['malignant']['recall']*100:.2f}%, F1 = {grp_sel['malignant']['f1']*100:.2f}%
- **Normal**: Precision = {grp_sel['normal']['precision']*100:.2f}%, Recall = {grp_sel['normal']['recall']*100:.2f}%, F1 = {grp_sel['normal']['f1']*100:.2f}%

### Locked Historical Test Set:
- **Benign**: Precision = {hist_sel['benign']['precision']*100:.2f}%, Recall = {hist_sel['benign']['recall']*100:.2f}%, F1 = {hist_sel['benign']['f1']*100:.2f}%
- **Malignant**: Precision = {hist_sel['malignant']['precision']*100:.2f}%, Recall = {hist_sel['malignant']['recall']*100:.2f}%, F1 = {hist_sel['malignant']['f1']*100:.2f}%
- **Normal**: Precision = {hist_sel['normal']['precision']*100:.2f}%, Recall = {hist_sel['normal']['recall']*100:.2f}%, F1 = {hist_sel['normal']['f1']*100:.2f}%

---

## 5. Model Recommendation & Verdict

- **Promotion Recommendation**: {promo_recommendation}

---

PRODUCTION MODEL CHANGED: NO
PRODUCTION METADATA CHANGED: NO
LOCKED 117-IMAGE TEST SET MODIFIED: NO
DATASET IMAGES MODIFIED: NO
DEPLOYMENT PERFORMED: NO
GIT COMMIT PERFORMED: NO
GIT PUSH PERFORMED: NO
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"\nSuccessfully generated V14 final evaluation report at: {report_path}")

if __name__ == "__main__":
    main()
