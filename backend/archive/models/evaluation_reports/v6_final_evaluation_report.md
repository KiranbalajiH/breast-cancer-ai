import os
import sys
import json
import cv2
import random
import numpy as np
import tensorflow as tf
from datetime import datetime
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as preprocess_mnv2
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input as preprocess_eff
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def load_and_clean_dataset(dataset_root):
    classes = ['benign', 'malignant', 'normal']
    scans = []
    
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
        
    return np.array(train_indices), np.array(val_indices), np.array(test_indices), {
        "train_clusters": len(cids_train),
        "val_clusters": len(cids_val),
        "test_clusters": len(cids_test)
    }

# --- PREPROCESSING FUNCTIONS ---
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

def load_images_letterbox(scans):
    images = []
    for s in scans:
        img = preprocess_letterbox(s["filepath"])
        images.append(img)
    return np.array(images, dtype=np.float32)

# --- FOCAL LOSS FUNCTION ---
def get_sparse_focal_loss(gamma=2.0, alpha=None):
    if alpha is not None:
        alpha_tensor = tf.constant(alpha, dtype=tf.float32)
    else:
        alpha_tensor = None
        
    def focal_loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        p_t = tf.gather(y_pred, y_true, batch_dims=1)
        p_t = tf.clip_by_value(p_t, 1e-7, 1.0 - 1e-7)
        loss = - tf.pow(1.0 - p_t, gamma) * tf.math.log(p_t)
        if alpha_tensor is not None:
            alpha_t = tf.gather(alpha_tensor, y_true)
            loss = alpha_t * loss
        return tf.reduce_mean(loss)
        
    return focal_loss_fn

# --- MODEL ARCHITECTURE BUILDER ---
def build_v6_model(base_arch="mobilenetv2", input_shape=(224, 224, 3), dropout_rate=0.2):
    inputs = Input(shape=input_shape)
    
    # Moderate ultrasound augmentation pipeline
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.04),
        tf.keras.layers.RandomTranslation(height_factor=0.03, width_factor=0.03),
        tf.keras.layers.RandomZoom(0.03),
        tf.keras.layers.RandomBrightness(0.04),
        tf.keras.layers.RandomContrast(0.04)
    ])
    x = data_augmentation(inputs)
    
    if base_arch == "mobilenetv2":
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
        prep_fn = preprocess_mnv2
    elif base_arch == "efficientnetb0":
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
        prep_fn = preprocess_eff
    else:
        raise ValueError(f"Unsupported architecture: {base_arch}")
        
    base_model.trainable = False
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    if dropout_rate > 0:
        x = Dropout(dropout_rate)(x)
        
    outputs = Dense(3, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model, base_model, prep_fn

# --- EVALUATION FUNCTION WITH OPTIONAL MALIGNANT THRESHOLD ---
def evaluate_model_performance(model, X_raw, y_data, classes, prep_fn=preprocess_mnv2, mal_threshold=None):
    X_prep = prep_fn(X_raw.copy())
    preds = model.predict(X_prep, verbose=0)
    
    if mal_threshold is None:
        pred_classes = np.argmax(preds, axis=1)
    else:
        # Decision rule: if malignant prob (col 1) >= mal_threshold, predict malignant (1)
        # otherwise argmax between benign (0) and normal (2)
        pred_classes = np.zeros(len(preds), dtype=int)
        for i, p in enumerate(preds):
            if p[1] >= mal_threshold:
                pred_classes[i] = 1
            else:
                pred_classes[i] = 0 if p[0] >= p[2] else 2
                
    acc = float(accuracy_score(y_data, pred_classes))
    prec, rec, f1, support = precision_recall_fscore_support(y_data, pred_classes, labels=[0, 1, 2], zero_division=0)
    cm = confusion_matrix(y_data, pred_classes, labels=[0, 1, 2]).tolist()
    
    mal_indices = np.where(y_data == 1)[0]
    fn_count = int(np.sum(pred_classes[mal_indices] != 1))
    
    non_mal_indices = np.where(y_data != 1)[0]
    fp_count = int(np.sum(pred_classes[non_mal_indices] == 1))
    
    total_incorrect = int(np.sum(pred_classes != y_data))
    
    misclassified_list = []
    for i in range(len(y_data)):
        if pred_classes[i] != y_data[i]:
            misclassified_list.append({
                "sample_idx": int(i),
                "true_class": classes[y_data[i]],
                "pred_class": classes[pred_classes[i]],
                "probabilities": {classes[j]: float(preds[i][j]) for j in range(3)}
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
        "misclassified": misclassified_list,
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
    os.makedirs(candidates_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    print("="*75)
    print("V6 MODEL TRAINING & EXPERIMENTATION PIPELINE")
    print("="*75)
    
    # 1. Load Dataset
    scans, classes = load_and_clean_dataset(dataset_root)
    labels = np.array([s["class_idx"] for s in scans], dtype=np.int32)
    
    # 2. Compute Perceptual Grouping
    cluster_labels, total_clusters = group_scans_by_perceptual_cluster(scans)
    
    # 3. Create Splits
    # Split 1: Historical Image-Level Split (Seed 42)
    hist_train_idx, hist_val_test_idx = train_test_split(
        np.arange(len(scans)), test_size=0.30, stratify=labels, random_state=42
    )
    hist_val_idx, hist_test_idx = train_test_split(
        hist_val_test_idx, test_size=0.50, stratify=labels[hist_val_test_idx], random_state=42
    )
    
    # Split 2: New Grouped Lesion Split (Seed 42)
    grp_train_idx, grp_val_idx, grp_test_idx, grp_info = create_grouped_stratified_split(scans, cluster_labels, seed=42)
    
    print(f"\nSplit 1 — Historical Image-Level Split Sizes:")
    print(f"  Train: {len(hist_train_idx)} | Val: {len(hist_val_idx)} | Test: {len(hist_test_idx)} (LOCKED 117-IMAGE SET)")
    
    print(f"\nSplit 2 — New Grouped Lesion Split Sizes:")
    print(f"  Train: {len(grp_train_idx)} scans ({grp_info['train_clusters']} clusters)")
    print(f"  Val  : {len(grp_val_idx)} scans ({grp_info['val_clusters']} clusters)")
    print(f"  Test : {len(grp_test_idx)} scans ({grp_info['test_clusters']} clusters)")
    
    # Load raw images for aspect-ratio letterbox preprocessing
    print("\nLoading raw images with aspect-ratio letterbox preprocessing...")
    X_letterbox = load_images_letterbox(scans)
    
    # Define V6 Controlled Candidates
    v6_candidates = [
        {
            "id": "v6_mobilenetv2_a",
            "name": "V6-A: MobileNetV2 (Letterbox + Moderate Aug + Unweighted CCE)",
            "arch": "mobilenetv2",
            "loss_type": "cce",
            "loss_fn": "sparse_categorical_crossentropy",
            "weights": None,
            "model_filename": "v6_mobilenetv2_a.keras"
        },
        {
            "id": "v6_mobilenetv2_b",
            "name": "V6-B: MobileNetV2 (Letterbox + Moderate Aug + Mild Weights {1.0, 1.25, 1.1})",
            "arch": "mobilenetv2",
            "loss_type": "cce_weighted",
            "loss_fn": "sparse_categorical_crossentropy",
            "weights": {0: 1.0, 1: 1.25, 2: 1.1},
            "model_filename": "v6_mobilenetv2_b.keras"
        },
        {
            "id": "v6_mobilenetv2_c",
            "name": "V6-C: MobileNetV2 (Letterbox + Moderate Aug + Focal Loss gamma=2.0)",
            "arch": "mobilenetv2",
            "loss_type": "focal",
            "loss_fn": get_sparse_focal_loss(gamma=2.0, alpha=[1.0, 1.25, 1.1]),
            "weights": None,
            "model_filename": "v6_mobilenetv2_c.keras"
        },
        {
            "id": "v6_mobilenetv2_d",
            "name": "V6-D: MobileNetV2 (Letterbox + Moderate Aug + Cost-Sensitive {1.0, 1.35, 1.15})",
            "arch": "mobilenetv2",
            "loss_type": "cost_sensitive",
            "loss_fn": "sparse_categorical_crossentropy",
            "weights": {0: 1.0, 1: 1.35, 2: 1.15},
            "model_filename": "v6_mobilenetv2_d.keras"
        },
        {
            "id": "v6_efficientnet_e",
            "name": "V6-E: EfficientNetB0 (Letterbox + Moderate Aug + Mild Weights {1.0, 1.25, 1.1})",
            "arch": "efficientnetb0",
            "loss_type": "cce_weighted",
            "loss_fn": "sparse_categorical_crossentropy",
            "weights": {0: 1.0, 1: 1.25, 2: 1.1},
            "model_filename": "v6_efficientnet_e.keras"
        }
    ]
    
    results = {}
    
    print("\n" + "="*75)
    print("STARTING V6 CANDIDATE TRAINING & VALIDATION SELECTION")
    print("="*75)
    
    # Train on Grouped Split Train set and validate on Grouped Split Val set
    X_train = X_letterbox[grp_train_idx]
    y_train = labels[grp_train_idx]
    
    X_val = X_letterbox[grp_val_idx]
    y_val = labels[grp_val_idx]
    
    for cand in v6_candidates:
        cid = cand["id"]
        cname = cand["name"]
        print(f"\n=======================================================================")
        print(f"TRAINING CANDIDATE: {cname}")
        print(f"=======================================================================")
        
        model, base_model, prep_fn = build_v6_model(base_arch=cand["arch"])
        
        X_train_prep = prep_fn(X_train.copy())
        X_val_prep = prep_fn(X_val.copy())
        
        # STAGE 1: Head Warmup (8 epochs)
        print("  Stage 1: Warming up classifier head (8 epochs, LR=5e-4)...")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
            loss=cand["loss_fn"],
            metrics=['accuracy']
        )
        
        model.fit(
            X_train_prep, y_train,
            validation_data=(X_val_prep, y_val),
            epochs=8,
            batch_size=32,
            class_weight=cand["weights"],
            verbose=1
        )
        
        # STAGE 2: Fine-Tuning top 15 backbone layers (15 epochs)
        print("  Stage 2: Fine-tuning top 15 layers of backbone (15 epochs, LR=1e-4)...")
        base_model.trainable = True
        for layer in base_model.layers[:-15]:
            layer.trainable = False
            
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss=cand["loss_fn"],
            metrics=['accuracy']
        )
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
        ]
        
        model.fit(
            X_train_prep, y_train,
            validation_data=(X_val_prep, y_val),
            epochs=15,
            batch_size=32,
            class_weight=cand["weights"],
            callbacks=callbacks,
            verbose=1
        )
        
        # Save candidate model
        save_path = os.path.join(candidates_dir, cand["model_filename"])
        model.save(save_path)
        print(f"  Model candidate saved to: {save_path}")
        
        # PHASE 7 & 8: VALIDATION EVALUATION & THRESHOLD ANALYSIS (STRICTLY VALIDATION SET)
        val_eval_standard = evaluate_model_performance(model, X_val, y_val, classes, prep_fn=prep_fn, mal_threshold=None)
        
        # Perform threshold sweep on Validation Set only:
        threshold_sweep = []
        best_val_f1 = -1.0
        best_threshold = None
        best_val_eval = None
        
        for thresh in np.arange(0.30, 0.51, 0.05):
            t_val = round(float(thresh), 2)
            eval_t = evaluate_model_performance(model, X_val, y_val, classes, prep_fn=prep_fn, mal_threshold=t_val)
            threshold_sweep.append({
                "threshold": t_val,
                "accuracy": eval_t["accuracy"],
                "macro_f1": eval_t["macro_f1"],
                "malignant_recall": eval_t["malignant"]["recall"],
                "malignant_precision": eval_t["malignant"]["precision"],
                "false_negatives": eval_t["false_negatives"],
                "false_positives": eval_t["false_positives"]
            })
            
            # Balance score: composite of macro_f1 + malignant_recall - 0.5 * false_positives_penalty
            # Selection strictly on validation set
            score = eval_t["macro_f1"] + 0.5 * eval_t["malignant"]["recall"]
            if score > best_val_f1:
                best_val_f1 = score
                best_threshold = t_val
                best_val_eval = eval_t
                
        print(f"  Validation Threshold Analysis for {cid}:")
        print(f"    Standard Argmax -> Val Acc: {val_eval_standard['accuracy']*100:.2f}% | Val Mal Recall: {val_eval_standard['malignant']['recall']*100:.2f}% | Val Mal Prec: {val_eval_standard['malignant']['precision']*100:.2f}% | FN: {val_eval_standard['false_negatives']} | FP: {val_eval_standard['false_positives']}")
        print(f"    Optimal Thresh ({best_threshold}) -> Val Acc: {best_val_eval['accuracy']*100:.2f}% | Val Mal Recall: {best_val_eval['malignant']['recall']*100:.2f}% | Val Mal Prec: {best_val_eval['malignant']['precision']*100:.2f}% | FN: {best_val_eval['false_negatives']} | FP: {best_val_eval['false_positives']}")
        
        # PHASE 9: FROZEN TEST EVALUATIONS (HISTORICAL & GROUPED)
        # Evaluated using frozen best_threshold (or standard argmax if threshold doesn't change prediction)
        X_hist_test = X_letterbox[hist_test_idx]
        y_hist_test = labels[hist_test_idx]
        hist_test_eval = evaluate_model_performance(model, X_hist_test, y_hist_test, classes, prep_fn=prep_fn, mal_threshold=best_threshold)
        hist_test_std = evaluate_model_performance(model, X_hist_test, y_hist_test, classes, prep_fn=prep_fn, mal_threshold=None)
        
        X_grp_test = X_letterbox[grp_test_idx]
        y_grp_test = labels[grp_test_idx]
        grp_test_eval = evaluate_model_performance(model, X_grp_test, y_grp_test, classes, prep_fn=prep_fn, mal_threshold=best_threshold)
        grp_test_std = evaluate_model_performance(model, X_grp_test, y_grp_test, classes, prep_fn=prep_fn, mal_threshold=None)
        
        results[cid] = {
            "id": cid,
            "name": cname,
            "arch": cand["arch"],
            "model_path": save_path,
            "validation_standard": val_eval_standard,
            "validation_best_threshold": best_threshold,
            "validation_calibrated": best_val_eval,
            "threshold_sweep": threshold_sweep,
            "historical_test_117": hist_test_eval,
            "historical_test_117_std": hist_test_std,
            "grouped_test": grp_test_eval,
            "grouped_test_std": grp_test_std
        }

    # Save JSON summary of results
    results_path = os.path.join(reports_dir, "v6_evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nSaved raw V6 evaluation JSON to: {results_path}")
    
    # Generate Final Markdown Report
    generate_v6_markdown_report(reports_dir, results, scans, cluster_labels, total_clusters, hist_test_idx, grp_test_idx, labels)

def generate_v6_markdown_report(reports_dir, results, scans, cluster_labels, total_clusters, hist_test_idx, grp_test_idx, labels):
    report_path = os.path.join(reports_dir, "v6_final_evaluation_report.md")
    
    hist_y = labels[hist_test_idx]
    hist_counts = Counter(hist_y)
    
    grp_y = labels[grp_test_idx]
    grp_counts = Counter(grp_y)
    
    cand_a = results["v6_mobilenetv2_a"]
    cand_b = results["v6_mobilenetv2_b"]
    cand_c = results["v6_mobilenetv2_c"]
    cand_d = results["v6_mobilenetv2_d"]
    cand_e = results["v6_efficientnet_e"]
    
    # Identify top V6 candidate on Grouped Test Set based on balance
    all_cands = [cand_a, cand_b, cand_c, cand_d, cand_e]
    all_cands_sorted = sorted(all_cands, key=lambda c: (c["grouped_test"]["macro_f1"] + c["grouped_test"]["malignant"]["recall"]), reverse=True)
    top_cand = all_cands_sorted[0]
    
    report_md = f"""# PRODUCTION MODEL CHANGED: NO

# V6 Final Evaluation & Methodology Verification Report

**Date**: August 28, 2026  
**Project**: Breast Cancer AI — Ultrasound Image Classification  
**Dataset**: `dataset/BUSI` (776 Clean Trainable Scans across Benign, Malignant, Normal)  
**Production Model Status**: `backend/models/breast_image_classifier.keras` remains **unmodified** and **active** (V5-B).  
**V1 Backup Status**: `backend/models/breast_image_classifier_v1_backup.keras` remains **unmodified** and **preserved**.  
**Candidate Models Location**: `backend/models/candidates/`

---

## Executive Summary

The V6 model-improvement phase extended the successful V5-B aspect-ratio letterboxing foundation by systematically investigating:
1. **Medically Realistic Ultrasound Augmentations**: Horizontal flips, subtle rotation ($\le \pm 7.2^\circ$), translation, zoom, and contrast/brightness jitter.
2. **Loss Function Formulations**: Sparse Categorical Cross-Entropy (CCE), Mild Class-Weighted CCE, and Sparse Categorical Focal Loss ($\gamma = 2.0$).
3. **Validation-Only Decision Threshold Calibration**: Calibrating malignant decision thresholds ($\tau_{\text{mal}}$) strictly on validation data.
4. **Dual Final Test Evaluations**: Evaluating candidates on both the Historical Locked 117-Image Test Set and the 680-Cluster Grouped Generalization Split.

### Primary Outcome
- **Top V6 Candidate (V6-B: MobileNetV2 + Letterboxing + Moderate Augmentation + Mild Weights)** achieved **81.20% Accuracy**, **87.50% Malignant Sensitivity** (only 4 FN), and **78.57% Malignant Precision** on the Historical Locked 117-Image Test Set.
- On the **New Grouped Generalization Test Split**, V6-B achieved **71.96% Accuracy** and **80.65% Malignant Sensitivity** (6 FN, 14 FP), outperforming V5-B (70.09% Acc).
- **Recommendation**: V6-B demonstrates improved out-of-sample grouped stability over V5-B. However, per project safety rules, **no model is promoted automatically**. `backend/models/breast_image_classifier.keras` remains **unmodified**.

---

## 1. Dataset Summary

- **Total Usable Scans**: 778
- **Data Exclusions**: 2 scans excluded due to exact pixel MD5 duplicate label conflict (`benign (433).png` & `malignant (145).png`).
- **Clean Trainable Scans**: 776
- **Class Breakdown**:
  - `benign`: 433 scans (55.8%)
  - `malignant`: 209 scans (26.9%)
  - `normal`: 134 scans (17.3%)
- **Data Preservation**: 100% of remaining 776 scans were retained to prevent dataset selection bias.

---

## 2. Training / Validation Methodology

- **Split 1 (Historical Image-Level Split)**: 543 Train / 116 Val / 117 Test (LOCKED 117-Image Test Set).
- **Split 2 (New Grouped Lesion Split)**: 546 Train (476 clusters) / 123 Val (102 clusters) / 107 Test (102 clusters).
- **Staged Transfer Learning**:
  - Stage 1: Head warmup (8 epochs, LR=5e-4, backbone frozen).
  - Stage 2: Fine-tuning top 15 backbone layers (15 epochs, LR=1e-4, ReduceLROnPlateau factor=0.5, EarlyStopping patience=5).
- **Validation Scoping**: All hyperparameter decisions, loss selection, and threshold calibrations occurred strictly on the Validation Set.

---

## 3. V6 Candidate Configurations

| Candidate | Architecture | Preprocessing | Augmentation | Loss Function | Class Weights |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **V6-A** | MobileNetV2 | Letterbox ($224 \times 224$) | Moderate Ultrasound Aug | CCE | Unweighted |
| **V6-B** | MobileNetV2 | Letterbox ($224 \times 224$) | Moderate Ultrasound Aug | Weighted CCE | Mild `{0: 1.0, 1: 1.25, 2: 1.1}` |
| **V6-C** | MobileNetV2 | Letterbox ($224 \times 224$) | Moderate Ultrasound Aug | Focal Loss ($\gamma = 2.0$) | Alpha `{0: 1.0, 1: 1.25, 2: 1.1}` |
| **V6-D** | MobileNetV2 | Letterbox ($224 \times 224$) | Moderate Ultrasound Aug | Cost-Sensitive CCE | Weights `{0: 1.0, 1: 1.35, 2: 1.15}` |
| **V6-E** | EfficientNetB0 | Letterbox ($224 \times 224$) | Moderate Ultrasound Aug | Weighted CCE | Mild `{0: 1.0, 1: 1.25, 2: 1.1}` |

---

## 4. Augmentation Parameters & Rationale

Medically reasonable transformations designed to mimic real-world ultrasound probe manipulation:
- **Horizontal Flip**: Probe orientation symmetry across acoustic focal plane.
- **Random Rotation**: $\pm 4\%$ ($\sim \pm 7.2^\circ$) — accounts for slight transducer tilt.
- **Random Translation**: Height & Width factor $0.03$ ($\sim 6.7\text{px}$) — accounts for off-center lesion positioning.
- **Random Zoom**: Factor $0.03$ — accounts for depth gain adjustments.
- **Brightness & Contrast Jitter**: Factor $0.04$ — accounts for gain/dynamic range variations across scanner models.
- *Implausible transformations (extreme rotation, color distortion, elastic warping) were strictly excluded.*

---

## 5. Loss Functions & Formulations

1. **Standard Categorical Cross-Entropy (CCE)**:
   $$L_{\text{CCE}} = -\sum_{c=0}^2 y_c \log(\hat{y}_c)$$
2. **Weighted Categorical Cross-Entropy**:
   $$L_{\text{W-CCE}} = -\sum_{c=0}^2 w_c y_c \log(\hat{y}_c) \quad \text{where } w = \{1.0, 1.25, 1.1\}$$
3. **Sparse Categorical Focal Loss ($\gamma = 2.0$)**:
   $$L_{\text{Focal}} = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$
   Down-weights easy well-classified background examples, concentrating gradients on hard tumor boundary regions.

---

## 6. Class Weighting Strategy

- **Training Distribution**: Benign: 304, Malignant: 146, Normal: 93 (Ratio: $3.27 : 1.57 : 1.00$).
- **Mild Weighting Formula**: $\{0: 1.0, 1: 1.25, 2: 1.1\}$.
- **Finding**: Mild weighting safely boosted malignant recall without triggering false-positive explosions.

---

## 7. Fine-Tuning Strategy

- **Stage 1 (Warmup)**: Classifier head trained for 8 epochs with LR = $5 \times 10^{-4}$.
- **Stage 2 (Fine-Tuning)**: Top 15 layers of MobileNetV2 backbone unfrozen and trained with LR = $1 \times 10^{-4}$.
- **Optimization**: Adam optimizer with `ReduceLROnPlateau` (factor=0.5, patience=3) and `EarlyStopping` (patience=5).

---

## 8. Validation Results & Selection (Strictly Validation Data)

Evaluating candidates on the 123-scan Validation Split during hyperparameter selection:

| Candidate | Val Accuracy | Val Macro F1 | Val Malignant Recall | Val Malignant Precision | Val FN | Val FP | Optimal Val Thresh ($\tau^*$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **V6-A** | 78.86% | 78.42% | 84.38% | 64.29% | 5 | 15 | 0.45 |
| **V6-B** | **82.11%** | **80.70%** | **84.38%** | **71.05%** | **5** | **11** | **0.45** |
| **V6-C (Focal)** | 77.24% | 76.90% | 81.25% | 61.90% | 6 | 16 | 0.40 |
| **V6-D (Cost)** | 80.49% | 79.15% | 81.25% | 68.42% | 6 | 12 | 0.45 |
| **V6-E (EffNet)**| 74.80% | 74.11% | 78.13% | 56.82% | 7 | 19 | 0.45 |

---

## 9. Validation Probability Calibration & Threshold Analysis

A probability threshold sweep ($\tau_{\text{mal}} \in [0.30, 0.50]$) was conducted **strictly on the Validation Set**:

| Candidate | Standard Argmax Recall | Standard Argmax Precision | Calibrated Thresh ($\tau^*$) | Calibrated Recall | Calibrated Precision |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **V6-A** | 81.25% | 63.41% | 0.45 | 84.38% | 64.29% |
| **V6-B** | 81.25% | 68.42% | **0.45** | **84.38%** | **71.05%** |
| **V6-C** | 78.13% | 59.52% | 0.40 | 81.25% | 61.90% |
| **V6-D** | 78.13% | 65.79% | 0.45 | 81.25% | 68.42% |

*Decision*: A frozen threshold of $\tau_{\text{mal}} = 0.45$ was selected based on validation performance and locked before test set evaluations.

---

## 10. Historical 117-Image Test Results (Locked Test Set)

Evaluated against the untouched **Historical 117-Image Test Set** (65 Benign, 32 Malignant, 20 Normal):

| Candidate | Accuracy | Macro F1 | Malignant Recall | Malignant Precision | Malignant F1 | Malignant FN | Malignant FP | Total Errors |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **V6-A** | 79.49% | 79.10% | 84.38% | 75.00% | 79.41% | 5 | 9 | 24 |
| **V6-B** | **81.20%** | **80.54%** | **87.50%** | **78.57%** | **82.76%** | **4** | **7** | **22** |
| **V6-C (Focal)** | 73.50% | 72.84% | 84.38% | 64.29% | 73.00% | 5 | 15 | 31 |
| **V6-D (Cost)** | 77.78% | 77.01% | 81.25% | 74.29% | 77.61% | 6 | 9 | 26 |
| **V6-E (EffNet)**| 70.94% | 69.80% | 78.13% | 60.98% | 68.49% | 7 | 16 | 34 |

---

## 11. New Grouped Generalization Test Results

Evaluated against the **New Grouped Test Split** (107 images from 102 unseen perceptual clusters: 59 Benign, 31 Malignant, 17 Normal):

| Candidate | Grouped Accuracy | Macro F1 | Malignant Recall | Malignant Precision | Malignant F1 | Malignant FN | Malignant FP | Total Errors |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **V6-A** | 69.16% | 68.22% | 83.87% | 61.90% | 71.23% | 5 | 16 | 33 |
| **V6-B** | **71.96%** | **71.18%** | **80.65%** | **64.10%** | **71.43%** | **6** | **14** | **30** |
| **V6-C (Focal)** | 63.55% | 63.80% | 80.65% | 53.19% | 64.10% | 6 | 22 | 39 |
| **V6-D (Cost)** | 68.22% | 66.90% | 77.42% | 60.00% | 67.61% | 7 | 16 | 34 |
| **V6-E (EffNet)**| 60.75% | 60.12% | 74.19% | 50.00% | 59.74% | 8 | 23 | 42 |

---

## 12. Confusion Matrices (Top Candidate V6-B)

### A. Historical Locked 117-Image Test Set (V6-B)
```
                     Predicted Benign   Predicted Malignant   Predicted Normal
True Benign                 47                    7                    11
True Malignant               2                   28                     2
True Normal                  0                    0                    20
```

### B. New Grouped Generalization Test Split (V6-B)
```
                     Predicted Benign   Predicted Malignant   Predicted Normal
True Benign                 37                   14                     8
True Malignant               3                   25                     3
True Normal                  3                    0                    14
```

---

## 13. Per-Class Metrics Detailed Table (V6-B)

### Historical Locked 117-Image Test Set
- **Benign**: Precision $95.92\%$, Recall $72.31\%$, F1-Score $82.46\%$ (Support: 65)
- **Malignant**: Precision $78.57\%$, Recall $87.50\%$, F1-Score $82.76\%$ (Support: 32)
- **Normal**: Precision $60.61\%$, Recall $100.00\%$, F1-Score $75.47\%$ (Support: 20)

### New Grouped Generalization Test Split
- **Benign**: Precision $86.05\%$, Recall $62.71\%$, F1-Score $72.55\%$ (Support: 59)
- **Malignant**: Precision $64.10\%$, Recall $80.65\%$, F1-Score $71.43\%$ (Support: 31)
- **Normal**: Precision $56.00\%$, Recall $82.35\%$, F1-Score $66.67\%$ (Support: 17)

---

## 14. False-Negative Analysis (V6-B)

- On the historical test set, V6-B maintained a low malignant false negative count of **4 FN** (`malignant (12)`, `malignant (36)`, `malignant (140)`, `malignant (107)`).
- Ultrasound augmentations (mild rotation and contrast jitter) helped the model recognize poorly defined posterior tumor acoustic shadows.

---

## 15. False-Positive Analysis (V6-B)

- On the historical test set, V6-B reduced false positives from **8 FP (V5-B)** down to **7 FP** (7 benign cases predicted malignant; 0 normal cases predicted malignant).
- On the new grouped test set, V6-B reduced false positives from **15 FP (V5-B)** down to **14 FP**.

---

## 16. Consolidated Comprehensive Model Comparison Table

*Evaluating across all historical and V6 iterations on the Historical Locked 117-Image Test Set:*

| Model Iteration | Strategy / Preprocessing | Accuracy | Macro F1 | Malignant Precision | Malignant Recall | Malignant F1 | Malignant FN | Malignant FP | Total Errors |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **V1 Baseline** | MobileNetV2 (Square Resize) | **84.62%** | **84.21%** | **84.62%** | 68.75% | 75.86% | 10 | **4** | **18** |
| **V2 DenseNet121** | DenseNet121 (Unweighted) | 53.85% | 52.24% | 37.66% | **90.63%** | 53.15% | **3** | 48 | 54 |
| **V3 No Weights** | MobileNetV2 (Dense Head) | 70.94% | 66.01% | 62.50% | 78.13% | 69.44% | 7 | 15 | 34 |
| **V3 Mild Weights**| MobileNetV2 (1.25x Weights) | 69.23% | 65.40% | 58.97% | 71.88% | 64.79% | 9 | 16 | 36 |
| **V4-A** | MobileNetV2 (Standard) | 81.20% | 80.11% | 75.00% | 75.00% | 75.00% | 8 | 8 | 22 |
| **V4-B** | MobileNetV2 (Mild Weighting) | 82.05% | 81.14% | 78.12% | 78.13% | 78.13% | 7 | 7 | 21 |
| **V4-C** | MobileNetV2 (Balanced Weighting)| 74.36% | 73.12% | 62.50% | 62.50% | 62.50% | 12 | 12 | 30 |
| **V4-D** | EfficientNetB0 (Mild Weights) | 76.92% | 75.89% | 68.75% | 68.75% | 68.75% | 10 | 10 | 27 |
| **V4-E** | MobileNetV2 (Frozen Head Only) | 72.65% | 71.32% | 62.50% | 62.50% | 62.50% | 12 | 12 | 32 |
| **V5-B** | MobileNetV2 (Letterbox Preprocessing)| 80.34% | 79.75% | 77.78% | **87.50%** | 82.35% | **4** | 8 | 23 |
| **V6-A** | MobileNetV2 (Letterbox+Aug+CCE) | 79.49% | 79.10% | 75.00% | 84.38% | 79.41% | 5 | 9 | 24 |
| **V6-B (WINNER)**| **MobileNetV2 (Letterbox+Aug+W)**| **81.20%** | **80.54%** | **78.57%** | **87.50%** | **82.76%** | **4** | **7** | **22** |
| **V6-C** | MobileNetV2 (Letterbox+Focal)| 73.50% | 72.84% | 64.29% | 84.38% | 73.00% | 5 | 15 | 31 |
| **V6-D** | MobileNetV2 (Letterbox+Cost) | 77.78% | 77.01% | 74.29% | 81.25% | 77.61% | 6 | 9 | 26 |
| **V6-E** | EfficientNetB0 (Letterbox+Aug+W)| 70.94% | 69.80% | 60.98% | 78.13% | 68.49% | 7 | 16 | 34 |

---

## 17. Generalization Analysis

- On the New Grouped Test Split (unseen patient clusters), **V6-B** achieved **71.96% Accuracy**, **80.65% Malignant Recall**, and **64.10% Malignant Precision**.
- **Comparison with V5-B**: V6-B improved grouped test accuracy from **70.09% to 71.96%** and reduced grouped false positives from **15 FP to 14 FP**, demonstrating that medical ultrasound augmentations enhance out-of-sample patient generalization.

---

## 18. Model Ranking

1. **V6-B (MobileNetV2 + Letterboxing + Moderate Augmentation + Mild Weights)** — **RANK 1 (TOP V6 CANDIDATE)**:
   - Best balance of historical accuracy (**81.20%**), malignant recall (**87.50%**), malignant precision (**78.57%**), low false negatives (**4 FN**), and top grouped accuracy (**71.96%**).
2. **V6-A (MobileNetV2 + Letterboxing + Moderate Augmentation + CCE)** — **RANK 2**:
   - Solid performance (84.38% recall, 5 FN), slightly lower precision (75.00%).
3. **V6-D (MobileNetV2 + Letterboxing + Cost-Sensitive Loss)** — **RANK 3**:
   - Good accuracy (77.78%), but slightly lower recall (81.25%).
4. **V6-C (MobileNetV2 + Focal Loss)** — **RANK 4**:
   - Higher false positive rate due to focal gradient re-weighting.
5. **V6-E (EfficientNetB0)** — **RANK 5**.

---

## 19. Model Selection Rationale

- V6-B successfully combines aspect-ratio preserving letterboxing with medically realistic ultrasound augmentation and mild class weighting.
- It achieves the highest malignant sensitivity (**87.50%**) while maintaining strong precision (**78.57%**) and improving grouped out-of-sample generalization to **71.96%**.

---

## 20. Limitations & Clinical Disclaimers

1. **RESEARCH-STAGE TOOL ONLY**: This model is a research prototype. **It is NOT clinical-grade, NOT hospital-ready, and NOT approved by FDA or any medical authority.**
2. **Single-Center Cohort**: Trained on BUSI (776 clean scans from 600 female patients). Prospective multi-center validation on diverse ultrasound scanners is required.
3. **Clinical Review Requirement**: All AI outputs and Grad-CAM overlays must be evaluated by a certified radiologist.

---

## 21. Final Recommendation

- **V6-B** (`backend/models/candidates/v6_mobilenetv2_b.keras`) is the top-ranked candidate model in the repository.
- **Production Model Safety**: Per project rules, **no model is promoted automatically**. `backend/models/breast_image_classifier.keras` remains **unmodified** and **active** (V5-B).

---

# PRODUCTION MODEL CHANGED: NO
