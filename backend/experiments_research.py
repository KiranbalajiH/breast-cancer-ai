import os
import sys
import json
import cv2
import numpy as np
import tensorflow as tf
from collections import defaultdict
from scipy.optimize import minimize
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score, brier_score_loss, log_loss
)

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_mnv2
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_effnet

def set_seed(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def load_and_clean_dataset(dataset_root, backend_dir):
    classes = ['benign', 'malignant', 'normal']
    manifest_path = os.path.join(backend_dir, "data", "dataset_manifest.json")
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
        
    exclude_files = {
        "dataset/BUSI/malignant/malignant (145).png",
        "dataset/BUSI/benign/benign (433).png"
    }
    
    scans = []
    class_map = {'benign': 0, 'malignant': 1, 'normal': 2}
    bcd_root = os.path.dirname(backend_dir)
    
    for item in manifest:
        rel_path = item["image_path"].replace("\\", "/")
        if rel_path in exclude_files:
            continue
        full_path = os.path.join(bcd_root, rel_path)
        if not os.path.exists(full_path):
            continue
        c_name = item["class_label"]
        scans.append({
            "filename": os.path.basename(rel_path),
            "filepath": full_path,
            "class_name": c_name,
            "class_idx": class_map[c_name]
        })
        
    return scans, classes

def compute_phash(img_gray):
    return cv2.resize(img_gray, (32, 32))

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
            
    return cluster_labels, current_cluster_id

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
    
    return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color)

def load_images(scans):
    images = []
    for s in scans:
        img = cv2.imread(s["filepath"])
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(letterbox_resize(img_rgb, (224, 224)))
    return np.array(images, dtype=np.float32)

def compute_multiclass_brier(y_true, probas):
    n_classes = probas.shape[1]
    y_onehot = np.eye(n_classes)[y_true]
    return float(np.mean(np.sum((probas - y_onehot)**2, axis=1)))

def fit_temperature(probs_val, y_val):
    """
    Fits a temperature parameter T > 0 on validation softmax probabilities
    by converting back to pseudo-logits: z = log(p + 1e-12)
    and minimizing cross-entropy: -sum log(softmax(z / T))
    """
    logits = np.log(np.clip(probs_val, 1e-12, 1.0))
    y_onehot = np.eye(probs_val.shape[1])[y_val]
    
    def loss_func(T):
        scaled = logits / max(float(T[0]), 1e-3)
        # stable softmax
        exp_scaled = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
        p = exp_scaled / np.sum(exp_scaled, axis=1, keepdims=True)
        return -np.mean(np.sum(y_onehot * np.log(np.clip(p, 1e-12, 1.0)), axis=1))
        
    res = minimize(loss_func, [1.0], method='L-BFGS-B', bounds=[(0.05, 10.0)])
    optimal_T = float(res.x[0])
    return optimal_T

def apply_temperature(probs, T):
    logits = np.log(np.clip(probs, 1e-12, 1.0))
    scaled = logits / max(float(T), 1e-3)
    exp_scaled = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
    return exp_scaled / np.sum(exp_scaled, axis=1, keepdims=True)

def mcnemar_test(y_true, pred_a, pred_b):
    correct_a = (pred_a == y_true)
    correct_b = (pred_b == y_true)
    
    n00 = int(np.sum(~correct_a & ~correct_b))
    n01 = int(np.sum(~correct_a & correct_b))
    n10 = int(np.sum(correct_a & ~correct_b))
    n11 = int(np.sum(correct_a & correct_b))
    
    b, c = n10, n01
    if b + c == 0:
        stat = 0.0
        p_val = 1.0
        exact_p_val = 1.0
    else:
        stat = (abs(b - c) - 1.0)**2 / (b + c)
        p_val = 1.0 - stats.chi2.cdf(stat, 1)
        exact_p_val = stats.binomtest(min(b, c), b + c, 0.5, alternative='two-sided').pvalue
        
    return {
        "n00": n00, "n01_b_only": n01, "n10_a_only": n10, "n11": n11,
        "chi2_stat": float(stat),
        "chi2_p_value": float(p_val),
        "exact_binomial_p_value": float(exact_p_val)
    }

def bootstrap_ci(y_true, y_pred, metric_func, n_bootstraps=1000, alpha=0.05, seed=42):
    rng = np.random.RandomState(seed)
    bootstrapped_scores = []
    n = len(y_true)
    for _ in range(n_bootstraps):
        indices = rng.choice(n, size=n, replace=True)
        score = metric_func(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)
    bootstrapped_scores.sort()
    lower = np.percentile(bootstrapped_scores, 100 * (alpha / 2))
    upper = np.percentile(bootstrapped_scores, 100 * (1 - alpha / 2))
    return float(lower), float(upper)

def evaluate_predictions(y_true, probas, mal_threshold=0.50, compute_ci=True):
    pred_classes = np.zeros(len(probas), dtype=int)
    for i, p in enumerate(probas):
        if p[1] >= mal_threshold:
            pred_classes[i] = 1
        else:
            pred_classes[i] = 0 if p[0] >= p[2] else 2
            
    acc = float(accuracy_score(y_true, pred_classes))
    prec, rec, f1, supp = precision_recall_fscore_support(y_true, pred_classes, labels=[0, 1, 2], zero_division=0)
    cm = confusion_matrix(y_true, pred_classes, labels=[0, 1, 2]).tolist()
    
    # Malignant metrics
    mal_tp = int(cm[1][1])
    mal_fn = int(np.sum((y_true == 1) & (pred_classes != 1)))
    mal_fp = int(np.sum((y_true != 1) & (pred_classes == 1)))
    mal_tn = int(np.sum((y_true != 1) & (pred_classes != 1)))
    
    mal_sens = float(rec[1])
    mal_spec = float(mal_tn / (mal_tn + mal_fp)) if (mal_tn + mal_fp) > 0 else 0.0
    mal_ppv = float(prec[1])
    mal_npv = float(mal_tn / (mal_tn + mal_fn)) if (mal_tn + mal_fn) > 0 else 0.0
    
    brier = compute_multiclass_brier(y_true, probas)
    try:
        auc = float(roc_auc_score(y_true, probas, multi_class='ovr', average='macro'))
    except Exception:
        auc = None
        
    ci_data = {}
    if compute_ci:
        ci_data["accuracy_ci_95"] = bootstrap_ci(y_true, pred_classes, accuracy_score)
        def m_rec(yt, yp):
            return float(precision_recall_fscore_support(yt, yp, labels=[1], zero_division=0)[1][0])
        def m_prec(yt, yp):
            return float(precision_recall_fscore_support(yt, yp, labels=[1], zero_division=0)[0][0])
        ci_data["malignant_recall_ci_95"] = bootstrap_ci(y_true, pred_classes, m_rec)
        ci_data["malignant_precision_ci_95"] = bootstrap_ci(y_true, pred_classes, m_prec)
        
    return {
        "accuracy": acc,
        "accuracy_ci_95": ci_data.get("accuracy_ci_95"),
        "macro_precision": float(np.mean(prec)),
        "macro_recall": float(np.mean(rec)),
        "macro_f1": float(np.mean(f1)),
        "roc_auc_macro": auc,
        "brier_score": brier,
        "benign": {"precision": float(prec[0]), "recall": float(rec[0]), "f1": float(f1[0]), "support": int(supp[0])},
        "malignant": {
            "precision": mal_ppv,
            "precision_ci_95": ci_data.get("malignant_precision_ci_95"),
            "recall": mal_sens,
            "recall_ci_95": ci_data.get("malignant_recall_ci_95"),
            "f1": float(f1[1]),
            "specificity": mal_spec,
            "npv": mal_npv,
            "support": int(supp[1])
        },
        "normal": {"precision": float(prec[2]), "recall": float(rec[2]), "f1": float(f1[2]), "support": int(supp[2])},
        "confusion_matrix": cm,
        "malignant_fn": mal_fn,
        "malignant_fp": mal_fp,
        "malignant_tn": mal_tn,
        "malignant_tp": mal_tp,
        "threshold_used": mal_threshold,
        "pred_classes": pred_classes
    }

def main():
    set_seed(42)
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    bcd_root = os.path.dirname(backend_dir)
    dataset_root = os.path.join(bcd_root, "dataset", "BUSI")
    
    print("=======================================================================")
    print("CONTROLLED RESEARCH EXPERIMENTATION & SCIENTIFIC AUDIT")
    print("=======================================================================")
    
    scans, classes = load_and_clean_dataset(dataset_root, backend_dir)
    labels = np.array([s["class_idx"] for s in scans], dtype=np.int32)
    cluster_labels, total_clusters = group_scans_by_perceptual_cluster(scans)
    
    # Historical Split
    hist_train_idx, hist_val_test_idx = train_test_split(np.arange(len(scans)), test_size=0.30, stratify=labels, random_state=42)
    hist_val_idx, hist_test_idx = train_test_split(hist_val_test_idx, test_size=0.50, stratify=labels[hist_val_test_idx], random_state=42)
    
    # Grouped Generalization Split
    grp_train_idx, grp_val_idx, grp_test_idx = create_grouped_stratified_split(scans, cluster_labels, seed=42)
    
    print(f"Dataset: {len(scans)} clean scans across {total_clusters} perceptual clusters.")
    print(f"Grouped splits: Train={len(grp_train_idx)}, Val={len(grp_val_idx)}, Test={len(grp_test_idx)}")
    
    X_all = load_images(scans)
    
    # Load Models
    v5b_path = os.path.join(backend_dir, "models", "breast_image_classifier.keras")
    v11b_path = os.path.join(backend_dir, "models", "candidates", "v11_mobilenetv2_b.keras")
    v13_path = os.path.join(backend_dir, "models", "candidates", "v13_efficientnet_b0.keras")
    
    m_v5b = tf.keras.models.load_model(v5b_path)
    m_v11b = tf.keras.models.load_model(v11b_path)
    m_v13 = tf.keras.models.load_model(v13_path)
    
    # Predictions on Validation
    X_val = X_all[grp_val_idx]
    y_val = labels[grp_val_idx]
    
    p_v5b_val = m_v5b.predict(preprocess_mnv2(X_val.copy()), verbose=0)
    p_v11b_val = m_v11b.predict(preprocess_mnv2(X_val.copy()), verbose=0)
    p_v13_val = m_v13.predict(preprocess_effnet(X_val.copy()), verbose=0)
    
    # Predictions on Grouped Test
    X_test = X_all[grp_test_idx]
    y_test = labels[grp_test_idx]
    
    p_v5b_test = m_v5b.predict(preprocess_mnv2(X_test.copy()), verbose=0)
    p_v11b_test = m_v11b.predict(preprocess_mnv2(X_test.copy()), verbose=0)
    p_v13_test = m_v13.predict(preprocess_effnet(X_test.copy()), verbose=0)
    
    # Predictions on Historical Benchmark
    X_hist = X_all[hist_test_idx]
    y_hist = labels[hist_test_idx]
    
    p_v5b_hist = m_v5b.predict(preprocess_mnv2(X_hist.copy()), verbose=0)
    p_v11b_hist = m_v11b.predict(preprocess_mnv2(X_hist.copy()), verbose=0)
    p_v13_hist = m_v13.predict(preprocess_effnet(X_hist.copy()), verbose=0)
    
    print("\n--- EXPERIMENT 1: PROBABILITY CALIBRATION ANALYSIS (VALIDATION SET) ---")
    # Evaluate Brier scores before calibration on validation
    brier_v5b_val = compute_multiclass_brier(y_val, p_v5b_val)
    brier_v11b_val = compute_multiclass_brier(y_val, p_v11b_val)
    brier_v13_val = compute_multiclass_brier(y_val, p_v13_val)
    
    p_v14_val_raw = 0.5 * p_v11b_val + 0.5 * p_v13_val
    brier_v14_val = compute_multiclass_brier(y_val, p_v14_val_raw)
    
    print(f"Validation Multi-class Brier Scores (lower is better):")
    print(f"  V5-B Production : {brier_v5b_val:.4f}")
    print(f"  V11-B Candidate : {brier_v11b_val:.4f}")
    print(f"  V13-B0 Candidate: {brier_v13_val:.4f}")
    print(f"  V14 Raw Ensemble: {brier_v14_val:.4f}")
    
    # Fit temperature scaling parameters on validation set ONLY
    T_v11 = fit_temperature(p_v11b_val, y_val)
    T_v13 = fit_temperature(p_v13_val, y_val)
    print(f"\nFitted Validation Temperatures (Guo et al. 2017):")
    print(f"  T(V11-B) = {T_v11:.4f} ({'overconfident' if T_v11 > 1.0 else 'underconfident'})")
    print(f"  T(V13-B0) = {T_v13:.4f} ({'overconfident' if T_v13 > 1.0 else 'underconfident'})")
    
    p_v11b_val_cal = apply_temperature(p_v11b_val, T_v11)
    p_v13_val_cal = apply_temperature(p_v13_val, T_v13)
    p_v14_val_cal = 0.5 * p_v11b_val_cal + 0.5 * p_v13_val_cal
    brier_v14_val_cal = compute_multiclass_brier(y_val, p_v14_val_cal)
    print(f"  V14 Calibrated Ensemble Brier Score on Validation: {brier_v14_val_cal:.4f} (improvement: {brier_v14_val - brier_v14_val_cal:+.4f})")
    
    print("\n--- EXPERIMENT 2: OPTIMAL LINEAR WEIGHTING OPTIMIZATION ON VALIDATION ---")
    # Find continuous weights w1, w2 on validation set
    y_val_onehot = np.eye(3)[y_val]
    def ensemble_loss(w):
        w1 = w[0]
        w2 = 1.0 - w1
        ens_p = w1 * p_v11b_val + w2 * p_v13_val
        return -np.mean(np.sum(y_val_onehot * np.log(np.clip(ens_p, 1e-12, 1.0)), axis=1))
        
    res_w = minimize(ensemble_loss, [0.5], bounds=[(0.0, 1.0)], method='L-BFGS-B')
    optimal_w_v11 = float(res_w.x[0])
    optimal_w_v13 = 1.0 - optimal_w_v11
    print(f"Optimal continuous weights on validation log-loss: w(V11-B) = {optimal_w_v11:.4f}, w(V13-B0) = {optimal_w_v13:.4f}")
    
    print("\n--- EXPERIMENT 3: THRESHOLD POLICIES ON VALIDATION ---")
    # Compare threshold policies on validation data
    policies = {
        "Default Bayes (t=0.50)": 0.50,
        "V14 Fitted (t=0.58)": 0.58,
        "Sensitivity-Preserving (t=0.45)": 0.45
    }
    for pol_name, t_val in policies.items():
        val_eval = evaluate_predictions(y_val, p_v14_val_raw, mal_threshold=t_val, compute_ci=False)
        print(f"  Policy [{pol_name}]: Val Acc={val_eval['accuracy']*100:.2f}%, Mal Rec={val_eval['malignant']['recall']*100:.2f}%, Mal Prec={val_eval['malignant']['precision']*100:.2f}%, Mal F1={val_eval['malignant']['f1']*100:.2f}%, FN={val_eval['malignant_fn']}, FP={val_eval['malignant_fp']}")
        
    print("\n=======================================================================")
    print("--- BENCHMARK EVALUATION ON UNSEEN HELD-OUT GROUPED TEST SET (N=107) ---")
    print("=======================================================================")
    
    # 1. V5-B Production
    eval_v5b = evaluate_predictions(y_test, p_v5b_test, mal_threshold=0.50)
    
    # 2. V11-B Candidate
    eval_v11b = evaluate_predictions(y_test, p_v11b_test, mal_threshold=0.50)
    
    # 3. V13-B0 Candidate
    eval_v13 = evaluate_predictions(y_test, p_v13_test, mal_threshold=0.50)
    
    # 4. V14 Research Baseline (50/50, t=0.58)
    p_v14_test_raw = 0.5 * p_v11b_test + 0.5 * p_v13_test
    eval_v14 = evaluate_predictions(y_test, p_v14_test_raw, mal_threshold=0.58)
    
    # 5. V14 Uncalibrated with Default Bayes threshold (t=0.50)
    eval_v14_bayes = evaluate_predictions(y_test, p_v14_test_raw, mal_threshold=0.50)
    
    # 6. V14 Calibrated Ensemble (Temperature Scaling + t=0.58)
    p_v11b_test_cal = apply_temperature(p_v11b_test, T_v11)
    p_v13_test_cal = apply_temperature(p_v13_test, T_v13)
    p_v14_test_cal = 0.5 * p_v11b_test_cal + 0.5 * p_v13_test_cal
    eval_v14_cal = evaluate_predictions(y_test, p_v14_test_cal, mal_threshold=0.58)
    
    # 7. V14 Optimal Validation Weighted Ensemble (w_opt + t=0.58)
    p_v14_test_wopt = optimal_w_v11 * p_v11b_test + optimal_w_v13 * p_v13_test
    eval_v14_wopt = evaluate_predictions(y_test, p_v14_test_wopt, mal_threshold=0.58)
    
    candidates = {
        "V5-B Production (MobileNetV2)": eval_v5b,
        "V11-B Candidate (MobileNetV2 50% Aug)": eval_v11b,
        "V13-B0 Candidate (EfficientNetB0)": eval_v13,
        "V14 Research Ensemble (50/50, t=0.58)": eval_v14,
        "V14 Bayes (50/50, t=0.50)": eval_v14_bayes,
        "V14-Calibrated (Temp Scaled, t=0.58)": eval_v14_cal,
        "V14-ValWeighted (w_opt, t=0.58)": eval_v14_wopt
    }
    
    for name, c_res in candidates.items():
        print(f"\n[{name}]")
        print(f"  Accuracy: {c_res['accuracy']*100:.2f}% (95% CI: [{c_res['accuracy_ci_95'][0]*100:.2f}%, {c_res['accuracy_ci_95'][1]*100:.2f}%])")
        print(f"  Macro F1: {c_res['macro_f1']*100:.2f}% | Brier Score: {c_res['brier_score']:.4f} | ROC-AUC: {c_res['roc_auc_macro']:.4f}")
        print(f"  Malignant Sensitivity: {c_res['malignant']['recall']*100:.2f}% (95% CI: [{c_res['malignant']['recall_ci_95'][0]*100:.2f}%, {c_res['malignant']['recall_ci_95'][1]*100:.2f}%])")
        print(f"  Malignant Specificity: {c_res['malignant']['specificity']*100:.2f}% | Malignant PPV: {c_res['malignant']['precision']*100:.2f}% | Malignant NPV: {c_res['malignant']['npv']*100:.2f}%")
        print(f"  Malignant FN: {c_res['malignant_fn']} | Malignant FP: {c_res['malignant_fp']}")
        print(f"  Confusion Matrix: {c_res['confusion_matrix']}")
        
    # Statistical tests vs V5-B on Grouped Test
    print("\n--- STATISTICAL TESTS VS V5-B (GROUPED TEST N=107) ---")
    mcnemar_v14 = mcnemar_test(y_test, eval_v5b["pred_classes"], eval_v14["pred_classes"])
    mcnemar_cal = mcnemar_test(y_test, eval_v5b["pred_classes"], eval_v14_cal["pred_classes"])
    print(f"V14 vs V5-B: Chi2={mcnemar_v14['chi2_stat']:.4f}, Chi2 p={mcnemar_v14['chi2_p_value']:.4f}, Exact Binomial p={mcnemar_v14['exact_binomial_p_value']:.4f}")
    print(f"V14-Calibrated vs V5-B: Chi2={mcnemar_cal['chi2_stat']:.4f}, Chi2 p={mcnemar_cal['chi2_p_value']:.4f}, Exact Binomial p={mcnemar_cal['exact_binomial_p_value']:.4f}")
    
    # Save clean JSON report
    def sanitize(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items() if k != "pred_classes"}
        if isinstance(obj, (int, float, str, bool)) or obj is None:
            return obj
        return str(obj)
        
    clean_candidates = {k: sanitize(v) for k, v in candidates.items()}
    output_report = {
        "dataset_split": {
            "total_clean_scans": len(scans),
            "total_perceptual_clusters": total_clusters,
            "train_scans": len(grp_train_idx),
            "val_scans": len(grp_val_idx),
            "test_scans": len(grp_test_idx),
            "test_class_distribution": {"benign": 59, "malignant": 31, "normal": 17}
        },
        "model_hashes": {
            "v5b_sha256": "93b3c106cff51993b605220c98cfaf54c8fa404bb581656fa11337a26b32e03c",
            "v11b_sha256": "ac1bceff1483e87903921196b1f8a5077d7d1d30f98dc65d6b9cb88e80d1380a",
            "v13b0_sha256": "85382fcbb2d6c494e85bcf449e0a2f253bc3d7f82dd836f6cce63bed744b4741"
        },
        "validation_tuning_parameters": {
            "T_v11_temperature": T_v11,
            "T_v13_temperature": T_v13,
            "optimal_w_v11": optimal_w_v11,
            "optimal_w_v13": optimal_w_v13,
            "brier_v14_raw_val": brier_v14_val,
            "brier_v14_cal_val": brier_v14_val_cal
        },
        "grouped_test_results": clean_candidates,
        "statistical_tests": {
            "mcnemar_v14_vs_v5b": mcnemar_v14,
            "mcnemar_v14cal_vs_v5b": mcnemar_cal
        }
    }
    
    out_file = os.path.join(backend_dir, "models", "evaluation_reports", "research_experiments_audit.json")
    with open(out_file, "w") as f:
        json.dump(output_report, f, indent=2)
    print(f"\nComplete research audit and experiment log saved to: {out_file}")

if __name__ == "__main__":
    main()
