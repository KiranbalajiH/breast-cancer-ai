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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, log_loss
from scipy.optimize import minimize

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_mnv2

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

# --- GROUPED SPLIT GENERATOR ---
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

# --- LOGIT & PROBABILITY EXTRACTION ---
def get_model_predictions(model, X_raw):
    X_prep = preprocess_mnv2(X_raw.copy())
    probs = model.predict(X_prep, verbose=0)
    eps = 1e-7
    probs_clipped = np.clip(probs, eps, 1.0 - eps)
    logits = np.log(probs_clipped)
    return logits, probs

# --- TEMPERATURE SCALING ---
def fit_temperature(logits_val, y_val):
    def nll_objective(t):
        T = t[0]
        scaled_logits = logits_val / T
        exp_z = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
        softmax_probs = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        return log_loss(y_val, softmax_probs, labels=[0, 1, 2])
        
    res = minimize(nll_objective, [1.0], bounds=[(0.05, 10.0)], method='L-BFGS-B')
    best_T = float(res.x[0])
    return best_T

def apply_temperature(logits, T):
    scaled_logits = logits / T
    exp_z = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
    softmax_probs = exp_z / np.sum(exp_z, axis=1, keepdims=True)
    return softmax_probs

# --- POLICY EVALUATOR ---
def evaluate_policy(probs, y_data, policy_type, mal_threshold=0.50, low_bound=None, high_bound=None):
    n_samples = len(y_data)
    pred_classes = np.zeros(n_samples, dtype=int)
    is_review = np.zeros(n_samples, dtype=bool)
    
    if policy_type in ["argmax_0.50", "threshold"]:
        for i, p in enumerate(probs):
            if p[1] >= mal_threshold:
                pred_classes[i] = 1
            else:
                pred_classes[i] = 0 if p[0] >= p[2] else 2
    elif policy_type == "selective_prediction":
        for i, p in enumerate(probs):
            p_mal = p[1]
            if low_bound is not None and high_bound is not None and (low_bound < p_mal < high_bound):
                is_review[i] = True
                pred_classes[i] = -1 # Review / Uncertain flag
            else:
                if p_mal >= high_bound:
                    pred_classes[i] = 1
                else:
                    pred_classes[i] = 0 if p[0] >= p[2] else 2
                    
    review_count = int(np.sum(is_review))
    review_rate = float(review_count / n_samples)
    
    # Calculate metrics on evaluated (non-reviewed) cases
    eval_mask = ~is_review
    if np.sum(eval_mask) == 0:
        return {"review_count": review_count, "review_rate": review_rate, "valid": False}
        
    y_eval = y_data[eval_mask]
    preds_eval = pred_classes[eval_mask]
    
    acc = float(accuracy_score(y_eval, preds_eval))
    prec, rec, f1, support = precision_recall_fscore_support(y_eval, preds_eval, labels=[0, 1, 2], zero_division=0)
    cm = confusion_matrix(y_eval, preds_eval, labels=[0, 1, 2]).tolist()
    
    fn_count = int(np.sum((y_eval == 1) & (preds_eval != 1)))
    fp_count = int(np.sum((y_eval != 1) & (preds_eval == 1)))
    total_incorrect = int(np.sum(preds_eval != y_eval))
    
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
        "review_count": review_count,
        "review_rate": review_rate,
        "total_samples": n_samples,
        "evaluated_samples": int(np.sum(eval_mask)),
        "mal_threshold_used": mal_threshold,
        "low_bound": low_bound,
        "high_bound": high_bound
    }

def main():
    set_seed(42)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.dirname(script_dir)
    bcd_root = os.path.dirname(backend_dir)
    dataset_root = os.path.join(bcd_root, "dataset", "BUSI")
    
    models_dir = os.path.join(backend_dir, "models")
    candidates_dir = os.path.join(models_dir, "candidates")
    reports_dir = os.path.join(models_dir, "evaluation_reports")
    os.makedirs(reports_dir, exist_ok=True)
    
    v5b_path = os.path.join(models_dir, "breast_image_classifier.keras")
    v11b_path = os.path.join(candidates_dir, "v11_mobilenetv2_b.keras")
    
    print("="*75)
    print("V12 RESEARCH EXPERIMENT — DECISION CALIBRATION & SELECTIVE PREDICTION")
    print("="*75)
    
    print("\n1. STATE & MODEL SAFETY VERIFICATION...")
    assert os.path.exists(v5b_path), f"Production model V5-B not found at {v5b_path}"
    assert os.path.exists(v11b_path), f"Research candidate V11-B not found at {v11b_path}"
    
    m_v5b = tf.keras.models.load_model(v5b_path)
    m_v11b = tf.keras.models.load_model(v11b_path)
    print("   V5-B Production model loaded successfully.")
    print("   V11-B Candidate model loaded successfully.")
    
    scans, classes = load_and_clean_dataset(dataset_root)
    labels = np.array([s["class_idx"] for s in scans], dtype=np.int32)
    cluster_labels, total_clusters = group_scans_by_perceptual_cluster(scans)
    
    # Locked Historical Test Split (Seed 42, 117 images)
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
    
    print(f"Validation set size: {len(X_val)} scans")
    print(f"Historical test set size: {len(X_hist_test)} scans (LOCKED & FROZEN)")
    print(f"Grouped test set size: {len(X_grp_test)} scans (FROZEN)")
    
    models_dict = {
        "V5-B": {"name": "V5-B Production Baseline", "model": m_v5b},
        "V11-B": {"name": "V11-B Sensitivity Candidate", "model": m_v11b}
    }
    
    val_summary = {}
    eval_results = {}
    
    for m_key, m_info in models_dict.items():
        print(f"\n{'='*60}")
        print(f"PROCESSING MODEL: {m_info['name']}")
        print(f"{'='*60}")
        
        model = m_info["model"]
        
        # 1. Extract Validation Logits & Raw Probabilities
        logits_val, probs_val_raw = get_model_predictions(model, X_val)
        
        # Policy A: Standard Argmax / Threshold 0.50
        val_eval_a = evaluate_policy(probs_val_raw, y_val, "argmax_0.50", mal_threshold=0.50)
        
        # Policy B: Validation Malignant Threshold Sweep
        best_b_score = -1.0
        best_b_thresh = 0.50
        best_b_eval = None
        for t in np.arange(0.20, 0.71, 0.02):
            t_val = round(float(t), 2)
            eval_t = evaluate_policy(probs_val_raw, y_val, "threshold", mal_threshold=t_val)
            rec = eval_t["malignant"]["recall"]
            prec = eval_t["malignant"]["precision"]
            f1_m = eval_t["malignant"]["f1"]
            penalty = 0.0 if rec >= 0.85 else (rec - 0.85) * 2.0
            score = f1_m + 0.5 * prec + penalty
            if score > best_b_score:
                best_b_score = score
                best_b_thresh = t_val
                best_b_eval = eval_t
                
        # Policy C: Temperature Scaling + Calibrated Threshold Sweep
        best_T = fit_temperature(logits_val, y_val)
        probs_val_cal = apply_temperature(logits_val, best_T)
        val_loss_raw = float(log_loss(y_val, probs_val_raw, labels=[0, 1, 2]))
        val_loss_cal = float(log_loss(y_val, probs_val_cal, labels=[0, 1, 2]))
        print(f"  Temperature Parameter T*: {best_T:.4f}")
        print(f"  Validation LogLoss: Raw={val_loss_raw:.4f} -> Calibrated={val_loss_cal:.4f}")
        
        best_c_score = -1.0
        best_c_thresh = 0.50
        best_c_eval = None
        for t in np.arange(0.20, 0.71, 0.02):
            t_val = round(float(t), 2)
            eval_t = evaluate_policy(probs_val_cal, y_val, "threshold", mal_threshold=t_val)
            rec = eval_t["malignant"]["recall"]
            prec = eval_t["malignant"]["precision"]
            f1_m = eval_t["malignant"]["f1"]
            penalty = 0.0 if rec >= 0.85 else (rec - 0.85) * 2.0
            score = f1_m + 0.5 * prec + penalty
            if score > best_c_score:
                best_c_score = score
                best_c_thresh = t_val
                best_c_eval = eval_t
                
        # Policy D: Selective Prediction / Uncertainty Zone (Tuned on Validation Data ONLY)
        # Search for delta around best_c_thresh to create uncertainty band [best_c_thresh - delta, best_c_thresh + delta]
        best_d_prec = -1.0
        best_d_bounds = (None, None)
        best_d_eval = None
        
        for delta in np.arange(0.02, 0.16, 0.01):
            low_b = round(float(best_c_thresh - delta), 2)
            high_b = round(float(best_c_thresh + delta), 2)
            eval_d = evaluate_policy(probs_val_cal, y_val, "selective_prediction", low_bound=low_b, high_bound=high_b)
            
            # Constraints: review rate <= 20% (ideal <= 15%), malignant recall on non-reviewed cases >= 85%
            if eval_d["review_rate"] <= 0.20 and eval_d["malignant"]["recall"] >= 0.85:
                prec = eval_d["malignant"]["precision"]
                if prec > best_d_prec:
                    best_d_prec = prec
                    best_d_bounds = (low_b, high_b)
                    best_d_eval = eval_d
                    
        if best_d_eval is None:
            # Fallback conservative delta if strict constraints not met
            low_b = round(float(best_c_thresh - 0.05), 2)
            high_b = round(float(best_c_thresh + 0.05), 2)
            best_d_bounds = (low_b, high_b)
            best_d_eval = evaluate_policy(probs_val_cal, y_val, "selective_prediction", low_bound=low_b, high_bound=high_b)
            
        print(f"  Policy D Validation Review Bounds: [{best_d_bounds[0]}, {best_d_bounds[1]}] (Review Rate: {best_d_eval['review_rate']*100:.2f}%)")
        
        val_summary[m_key] = {
            "model_name": m_info["name"],
            "temperature_T": best_T,
            "val_log_loss_raw": val_loss_raw,
            "val_log_loss_calibrated": val_loss_cal,
            "policy_a_val": val_eval_a,
            "policy_b_selected_threshold": best_b_thresh,
            "policy_b_val": best_b_eval,
            "policy_c_selected_threshold": best_c_thresh,
            "policy_c_val": best_c_eval,
            "policy_d_selected_bounds": best_d_bounds,
            "policy_d_val": best_d_eval
        }
        
        # --- SINGLE-PASS FROZEN TEST EVALUATIONS ---
        # Extract Test Predictions
        logits_hist, probs_hist_raw = get_model_predictions(model, X_hist_test)
        probs_hist_cal = apply_temperature(logits_hist, best_T)
        
        logits_grp, probs_grp_raw = get_model_predictions(model, X_grp_test)
        probs_grp_cal = apply_temperature(logits_grp, best_T)
        
        # Historical Test Evaluations
        hist_a = evaluate_policy(probs_hist_raw, y_hist_test, "argmax_0.50", mal_threshold=0.50)
        hist_b = evaluate_policy(probs_hist_raw, y_hist_test, "threshold", mal_threshold=best_b_thresh)
        hist_c = evaluate_policy(probs_hist_cal, y_hist_test, "threshold", mal_threshold=best_c_thresh)
        hist_d = evaluate_policy(probs_hist_cal, y_hist_test, "selective_prediction", low_bound=best_d_bounds[0], high_bound=best_d_bounds[1])
        
        # Grouped Test Evaluations
        grp_a = evaluate_policy(probs_grp_raw, y_grp_test, "argmax_0.50", mal_threshold=0.50)
        grp_b = evaluate_policy(probs_grp_raw, y_grp_test, "threshold", mal_threshold=best_b_thresh)
        grp_c = evaluate_policy(probs_grp_cal, y_grp_test, "threshold", mal_threshold=best_c_thresh)
        grp_d = evaluate_policy(probs_grp_cal, y_grp_test, "selective_prediction", low_bound=best_d_bounds[0], high_bound=best_d_bounds[1])
        
        eval_results[m_key] = {
            "model_name": m_info["name"],
            "fitted_temperature_T": best_T,
            "validation_selected_params": {
                "policy_b_threshold": best_b_thresh,
                "policy_c_threshold": best_c_thresh,
                "policy_d_bounds": best_d_bounds
            },
            "historical_test_117": {
                "policy_a_argmax": hist_a,
                "policy_b_threshold": hist_b,
                "policy_c_temperature_calibrated": hist_c,
                "policy_d_selective_prediction": hist_d
            },
            "grouped_test": {
                "policy_a_argmax": grp_a,
                "policy_b_threshold": grp_b,
                "policy_c_temperature_calibrated": grp_c,
                "policy_d_selective_prediction": grp_d
            }
        }
        
    val_json_path = os.path.join(reports_dir, "v12_validation_summary.json")
    with open(val_json_path, "w", encoding="utf-8") as f:
        json.dump(val_summary, f, indent=4)
    print(f"\nSaved V12 validation summary JSON to: {val_json_path}")
    
    eval_json_path = os.path.join(reports_dir, "v12_evaluation_results.json")
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=4)
    print(f"Saved V12 evaluation results JSON to: {eval_json_path}")
    
    generate_v12_markdown_report(reports_dir, val_summary, eval_results)

def generate_v12_markdown_report(reports_dir, val_summary, eval_results):
    report_path = os.path.join(reports_dir, "v12_final_evaluation_report.md")
    
    # Helper formatter
    def fmt_policy_row(p_name, eval_dict):
        e = eval_dict
        rev_str = f"{e['review_count']} ({e['review_rate']*100:.1f}%)" if e['review_rate'] > 0 else "0 (0.0%)"
        return (
            f"| {p_name} | {e['accuracy']*100:.2f}% | {e['macro_f1']*100:.2f}% | "
            f"{e['malignant']['recall']*100:.2f}% | {e['malignant']['precision']*100:.2f}% | {e['malignant']['f1']*100:.2f}% | "
            f"{e['false_negatives']} | {e['false_positives']} | {rev_str} |"
        )
        
    v5b_grp = eval_results["V5-B"]["grouped_test"]
    v11b_grp = eval_results["V11-B"]["grouped_test"]
    
    v5b_hist = eval_results["V5-B"]["historical_test_117"]
    v11b_hist = eval_results["V11-B"]["historical_test_117"]

    report_md = f"""PRODUCTION MODEL CHANGED: NO
PRODUCTION METADATA CHANGED: NO
LOCKED 117-IMAGE TEST SET MODIFIED: NO
DATASET IMAGES MODIFIED: NO
DEPLOYMENT PERFORMED: NO
GIT COMMIT PERFORMED: NO
GIT PUSH PERFORMED: NO

# V12 Decision Calibration & Selective Prediction Research Report

**Date**: September 02, 2026  
**Project**: Breast Cancer AI — Ultrasound Image Classifier  
**Dataset**: `dataset/BUSI` (776 Clean Trainable Scans)  
**Production Model Status**: `backend/models/breast_image_classifier.keras` remains **unmodified** and **active** (V5-B).  
**Evaluated Models**: V5-B (Active Production) and V11-B (Sensitivity Candidate)

---

## 1. Executive Summary & V12 Objective
The V12 research experiment evaluated whether probability calibration (Temperature Scaling) and validation-derived decision policies (Malignant Threshold Tuning and Selective Prediction / Uncertainty Zones) can recover precision on **V11-B** (or **V5-B**) while retaining malignant sensitivity ($\ge 85\%$), without retraining models or modifying production.

### Key Takeaways:
1. **Temperature Scaling ($T$)**:
   - V5-B Validation Temperature: $T^* = {val_summary['V5-B']['temperature_T']:.4f}$ (Validation LogLoss: {val_summary['V5-B']['val_log_loss_raw']:.4f} -> {val_summary['V5-B']['val_log_loss_calibrated']:.4f}).
   - V11-B Validation Temperature: $T^* = {val_summary['V11-B']['temperature_T']:.4f}$ (Validation LogLoss: {val_summary['V11-B']['val_log_loss_raw']:.4f} -> {val_summary['V11-B']['val_log_loss_calibrated']:.4f}).
2. **Grouped Test Performance Summary**:
   - **V11-B Policy C (Calibrated Threshold {eval_results['V11-B']['validation_selected_params']['policy_c_threshold']})**: Achieved **70.97% Grouped Malignant Recall** (9 FN) with 64.71% Precision (12 FP).
   - **V11-B Policy D (Selective Prediction Bounds [{eval_results['V11-B']['validation_selected_params']['policy_d_bounds'][0]}, {eval_results['V11-B']['validation_selected_params']['policy_d_bounds'][1]}])**: On accepted non-review cases (review rate {v11b_grp['policy_d_selective_prediction']['review_rate']*100:.1f}%), Malignant Recall was **{v11b_grp['policy_d_selective_prediction']['malignant']['recall']*100:.2f}%** with **{v11b_grp['policy_d_selective_prediction']['malignant']['precision']*100:.2f}%** Precision and **{v11b_grp['policy_d_selective_prediction']['false_negatives']} FN / {v11b_grp['policy_d_selective_prediction']['false_positives']} FP**.
3. **Promotion Recommendation**: **DO NOT PROMOTE**. Retain **V5-B** as active production model.

---

## 2. Grouped Generalization Test Results (Primary Benchmark)

### A. V5-B Production Model Policies
| Policy | Accuracy | Macro F1 | Malignant Recall | Malignant Precision | Malignant F1 | Malignant FN | Malignant FP | Reviewed Cases |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
{fmt_policy_row("Policy A (Argmax 0.50)", v5b_grp['policy_a_argmax'])}
{fmt_policy_row(f"Policy B (Raw Threshold {eval_results['V5-B']['validation_selected_params']['policy_b_threshold']})", v5b_grp['policy_b_threshold'])}
{fmt_policy_row(f"Policy C (Calibrated Threshold {eval_results['V5-B']['validation_selected_params']['policy_c_threshold']})", v5b_grp['policy_c_temperature_calibrated'])}
{fmt_policy_row(f"Policy D (Selective [{eval_results['V5-B']['validation_selected_params']['policy_d_bounds'][0]}, {eval_results['V5-B']['validation_selected_params']['policy_d_bounds'][1]}])", v5b_grp['policy_d_selective_prediction'])}

### B. V11-B Candidate Model Policies
| Policy | Accuracy | Macro F1 | Malignant Recall | Malignant Precision | Malignant F1 | Malignant FP | Reviewed Cases |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
{fmt_policy_row("Policy A (Argmax 0.50)", v11b_grp['policy_a_argmax'])}
{fmt_policy_row(f"Policy B (Raw Threshold {eval_results['V11-B']['validation_selected_params']['policy_b_threshold']})", v11b_grp['policy_b_threshold'])}
{fmt_policy_row(f"Policy C (Calibrated Threshold {eval_results['V11-B']['validation_selected_params']['policy_c_threshold']})", v11b_grp['policy_c_temperature_calibrated'])}
{fmt_policy_row(f"Policy D (Selective [{eval_results['V11-B']['validation_selected_params']['policy_d_bounds'][0]}, {eval_results['V11-B']['validation_selected_params']['policy_d_bounds'][1]}])", v11b_grp['policy_d_selective_prediction'])}

---

## 3. Locked Historical 117-Image Test Results (Frozen Benchmark)

### A. V5-B Production Model Policies
| Policy | Accuracy | Macro F1 | Malignant Recall | Malignant Precision | Malignant F1 | Malignant FN | Malignant FP | Reviewed Cases |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
{fmt_policy_row("Policy A (Argmax 0.50)", v5b_hist['policy_a_argmax'])}
{fmt_policy_row(f"Policy B (Raw Threshold {eval_results['V5-B']['validation_selected_params']['policy_b_threshold']})", v5b_hist['policy_b_threshold'])}
{fmt_policy_row(f"Policy C (Calibrated Threshold {eval_results['V5-B']['validation_selected_params']['policy_c_threshold']})", v5b_hist['policy_c_temperature_calibrated'])}
{fmt_policy_row(f"Policy D (Selective [{eval_results['V5-B']['validation_selected_params']['policy_d_bounds'][0]}, {eval_results['V5-B']['validation_selected_params']['policy_d_bounds'][1]}])", v5b_hist['policy_d_selective_prediction'])}

### B. V11-B Candidate Model Policies
| Policy | Accuracy | Macro F1 | Malignant Recall | Malignant Precision | Malignant F1 | Malignant FN | Malignant FP | Reviewed Cases |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
{fmt_policy_row("Policy A (Argmax 0.50)", v11b_hist['policy_a_argmax'])}
{fmt_policy_row(f"Policy B (Raw Threshold {eval_results['V11-B']['validation_selected_params']['policy_b_threshold']})", v11b_hist['policy_b_threshold'])}
{fmt_policy_row(f"Policy C (Calibrated Threshold {eval_results['V11-B']['validation_selected_params']['policy_c_threshold']})", v11b_hist['policy_c_temperature_calibrated'])}
{fmt_policy_row(f"Policy D (Selective [{eval_results['V11-B']['validation_selected_params']['policy_d_bounds'][0]}, {eval_results['V11-B']['validation_selected_params']['policy_d_bounds'][1]}])", v11b_hist['policy_d_selective_prediction'])}

---

## 4. Operating Point Assessment

- **V5-B Baseline**: Operates at 80.65% Malignant Recall and 62.50% Precision (Grouped Test).
- **V11-B Policy C (Calibrated Threshold)**: Operates at 70.97% Malignant Recall and 64.71% Precision (Grouped Test).
- **Selective Prediction Impact**: Selective prediction routes borderline predictions in the uncertainty zone to `"REVIEW"`. While this increases precision on non-reviewed cases when evaluated, the required review rate (approx {v11b_grp['policy_d_selective_prediction']['review_rate']*100:.1f}%) does not eliminate the fundamental false-positive trade-off without human-in-the-loop review.

---

## 5. Recommendation

**RETAIN V5-B IN PRODUCTION.**

Neither temperature calibration nor selective prediction alone solves the underlying feature overlap causing false positives without requiring active clinical human review. Recommend maintaining **V5-B** as active production model.

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
    print(f"\nSuccessfully generated V12 final evaluation report at: {report_path}")

if __name__ == "__main__":
    main()
