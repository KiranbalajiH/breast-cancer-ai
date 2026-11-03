import os
import sys
import json
import cv2
import numpy as np
import tensorflow as tf
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score, brier_score_loss
)
from scipy import stats

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

def bootstrap_ci(y_true, y_pred, metric_func, n_bootstraps=1000, alpha=0.05, seed=42):
    rng = np.random.RandomState(seed)
    bootstrapped_scores = []
    n = len(y_true)
    for _ in range(n_bootstraps):
        indices = rng.choice(n, size=n, replace=True)
        # ensure all classes present in bootstrap sample if metric requires it
        score = metric_func(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)
    bootstrapped_scores.sort()
    lower = np.percentile(bootstrapped_scores, 100 * (alpha / 2))
    upper = np.percentile(bootstrapped_scores, 100 * (1 - alpha / 2))
    return float(lower), float(upper)

def mcnemar_test(y_true, pred_a, pred_b):
    # Table of concordant/discordant predictions
    correct_a = (pred_a == y_true)
    correct_b = (pred_b == y_true)
    
    n00 = int(np.sum(~correct_a & ~correct_b)) # both wrong
    n01 = int(np.sum(~correct_a & correct_b))  # A wrong, B right
    n10 = int(np.sum(correct_a & ~correct_b))  # A right, B wrong
    n11 = int(np.sum(correct_a & correct_b))   # both right
    
    # McNemar with continuity correction
    b = n10
    c = n01
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

def main():
    set_seed(42)
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    bcd_root = os.path.dirname(backend_dir)
    dataset_root = os.path.join(bcd_root, "dataset", "BUSI")
    
    print("Loading BUSI dataset...")
    scans, classes = load_and_clean_dataset(dataset_root, backend_dir)
    labels = np.array([s["class_idx"] for s in scans], dtype=np.int32)
    cluster_labels, total_clusters = group_scans_by_perceptual_cluster(scans)
    
    # Historical Split (LOCKED 117-IMAGE TEST SET)
    hist_train_idx, hist_val_test_idx = train_test_split(np.arange(len(scans)), test_size=0.30, stratify=labels, random_state=42)
    hist_val_idx, hist_test_idx = train_test_split(hist_val_test_idx, test_size=0.50, stratify=labels[hist_val_test_idx], random_state=42)
    
    # Grouped Generalization Split (Seed 42)
    grp_train_idx, grp_val_idx, grp_test_idx = create_grouped_stratified_split(scans, cluster_labels, seed=42)
    
    print("Pre-loading images...")
    X_all = load_images(scans)
    
    # Load Models
    v5b_path = os.path.join(backend_dir, "models", "breast_image_classifier.keras")
    v11b_path = os.path.join(backend_dir, "models", "candidates", "v11_mobilenetv2_b.keras")
    v13_path = os.path.join(backend_dir, "models", "candidates", "v13_efficientnet_b0.keras")
    
    print("Loading model weights...")
    m_v5b = tf.keras.models.load_model(v5b_path)
    m_v11b = tf.keras.models.load_model(v11b_path)
    m_v13 = tf.keras.models.load_model(v13_path)
    
    splits = {
        "Grouped Test Set (Unseen Clusters, N=107)": {
            "indices": grp_test_idx,
            "y": labels[grp_test_idx],
            "X": X_all[grp_test_idx]
        },
        "Historical Benchmark (Locked Random, N=117)": {
            "indices": hist_test_idx,
            "y": labels[hist_test_idx],
            "X": X_all[hist_test_idx]
        }
    }
    
    results = {}
    
    for split_name, s_data in splits.items():
        print(f"\n=======================================================")
        print(f"BENCHMARKING ON: {split_name}")
        print(f"=======================================================")
        y_true = s_data["y"]
        X_raw = s_data["X"]
        
        # Inferences
        p_v5b = m_v5b.predict(preprocess_mnv2(X_raw.copy()), verbose=0)
        p_v11b = m_v11b.predict(preprocess_mnv2(X_raw.copy()), verbose=0)
        p_v13 = m_v13.predict(preprocess_effnet(X_raw.copy()), verbose=0)
        
        # V14 ensemble (50/50, t=0.58)
        p_v14 = 0.5 * p_v11b + 0.5 * p_v13
        
        # Predictions
        pred_v5b = np.argmax(p_v5b, axis=1)
        pred_v11b = np.argmax(p_v11b, axis=1)
        pred_v13 = np.argmax(p_v13, axis=1)
        
        pred_v14 = np.zeros(len(p_v14), dtype=int)
        for i, p in enumerate(p_v14):
            if p[1] >= 0.58:
                pred_v14[i] = 1
            else:
                pred_v14[i] = 0 if p[0] >= p[2] else 2
                
        # Compute Metrics for V5-B and V14
        models_eval = {
            "V5-B Production (MobileNetV2)": (pred_v5b, p_v5b),
            "V11-B Candidate (MobileNetV2 50% Aug)": (pred_v11b, p_v11b),
            "V13-B0 Candidate (EfficientNetB0)": (pred_v13, p_v13),
            "V14 Ensemble (50/50, t=0.58)": (pred_v14, p_v14)
        }
        
        split_results = {}
        for m_name, (pred, proba) in models_eval.items():
            acc = accuracy_score(y_true, pred)
            acc_ci = bootstrap_ci(y_true, pred, accuracy_score)
            
            prec, rec, f1, supp = precision_recall_fscore_support(y_true, pred, labels=[0, 1, 2], zero_division=0)
            
            # Malignant metrics
            def mal_rec_func(yt, yp):
                return float(precision_recall_fscore_support(yt, yp, labels=[1], zero_division=0)[1][0])
            def mal_prec_func(yt, yp):
                return float(precision_recall_fscore_support(yt, yp, labels=[1], zero_division=0)[0][0])
                
            mal_rec_ci = bootstrap_ci(y_true, pred, mal_rec_func)
            mal_prec_ci = bootstrap_ci(y_true, pred, mal_prec_func)
            
            cm = confusion_matrix(y_true, pred, labels=[0, 1, 2]).tolist()
            
            # ROC-AUC (one-vs-rest macro)
            try:
                auc_score = float(roc_auc_score(y_true, proba, multi_class='ovr', average='macro'))
            except Exception:
                auc_score = None
                
            # Malignant Specificity and NPV (one-vs-rest)
            mal_tp = int(cm[1][1])
            mal_fn = int(np.sum((y_true == 1) & (pred != 1)))
            mal_fp = int(np.sum((y_true != 1) & (pred == 1)))
            mal_tn = int(np.sum((y_true != 1) & (pred != 1)))
            mal_spec = float(mal_tn / (mal_tn + mal_fp)) if (mal_tn + mal_fp) > 0 else 0.0
            mal_npv = float(mal_tn / (mal_tn + mal_fn)) if (mal_tn + mal_fn) > 0 else 0.0

            split_results[m_name] = {
                "accuracy": float(acc),
                "accuracy_ci_95": acc_ci,
                "macro_precision": float(np.mean(prec)),
                "macro_recall": float(np.mean(rec)),
                "macro_f1": float(np.mean(f1)),
                "roc_auc_macro": auc_score,
                "benign": {"precision": float(prec[0]), "recall": float(rec[0]), "f1": float(f1[0]), "support": int(supp[0])},
                "malignant": {
                    "precision": float(prec[1]), "precision_ci_95": mal_prec_ci,
                    "recall": float(rec[1]), "recall_ci_95": mal_rec_ci,
                    "f1": float(f1[1]), "support": int(supp[1]),
                    "specificity": mal_spec,
                    "npv": mal_npv
                },
                "normal": {"precision": float(prec[2]), "recall": float(rec[2]), "f1": float(f1[2]), "support": int(supp[2])},
                "confusion_matrix": cm,
                "malignant_fn": mal_fn,
                "malignant_fp": mal_fp,
                "malignant_tn": mal_tn,
                "malignant_tp": mal_tp
            }
            
            print(f"[{m_name}]")
            print(f"  Accuracy: {acc*100:.2f}% (95% CI: [{acc_ci[0]*100:.2f}%, {acc_ci[1]*100:.2f}%])")
            print(f"  Macro F1: {np.mean(f1)*100:.2f}% | Malignant Recall: {rec[1]*100:.2f}% (95% CI: [{mal_rec_ci[0]*100:.2f}%, {mal_rec_ci[1]*100:.2f}%]) | Malignant Prec: {prec[1]*100:.2f}%")
            print(f"  Malignant Specificity: {mal_spec*100:.2f}% | Malignant NPV: {mal_npv*100:.2f}%")
            print(f"  Malignant FN: {mal_fn} | Malignant FP: {mal_fp}")
            print(f"  Confusion Matrix: {cm}")
            
        # Statistical Test between V5-B and V14
        mcnemar = mcnemar_test(y_true, pred_v5b, pred_v14)
        print(f"\nMcNemar Test (V5-B vs V14 Ensemble) on {split_name}:")
        print(f"  Discordant: V5-B only right = {mcnemar['n10_a_only']}, V14 only right = {mcnemar['n01_b_only']}")
        print(f"  Chi2 stat = {mcnemar['chi2_stat']:.4f}, Chi2 p-value = {mcnemar['chi2_p_value']:.4f}")
        print(f"  Exact Binomial p-value = {mcnemar['exact_binomial_p_value']:.4f}")
        split_results["mcnemar_v5b_vs_v14"] = mcnemar
        
        results[split_name] = split_results
        
    out_path = os.path.join(backend_dir, "models", "evaluation_reports", "rigorous_benchmark_audit_v2.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nRigorous evaluation saved to: {out_path}")

if __name__ == "__main__":
    main()
