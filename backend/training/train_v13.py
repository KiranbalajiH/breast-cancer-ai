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

from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input as preprocess_effnet
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

# --- MODEL ARCHITECTURE BUILDER ---
def build_v13_model(input_shape=(224, 224, 3), dropout_rate=0.20):
    inputs = Input(shape=input_shape)
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomTranslation(height_factor=0.03, width_factor=0.03),
        tf.keras.layers.RandomZoom(0.03)
    ])
    x = data_augmentation(inputs)
    
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False
    
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    if dropout_rate > 0:
        x = Dropout(dropout_rate)(x)
        
    outputs = Dense(3, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model, base_model

# --- EVALUATION FUNCTION ---
def evaluate_model(model, X_raw, y_data, classes, mal_threshold=None):
    X_prep = preprocess_effnet(X_raw.copy())
    preds = model.predict(X_prep, verbose=0)
    
    if mal_threshold is None:
        pred_classes = np.argmax(preds, axis=1)
    else:
        pred_classes = np.zeros(len(preds), dtype=int)
        for i, p in enumerate(preds):
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
    os.makedirs(candidates_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    print("="*75)
    print("V13 EFFICIENTNET-B0 BACKBONE GENERALIZATION EXPERIMENT")
    print("="*75)
    
    scans, classes = load_and_clean_dataset(dataset_root)
    labels = np.array([s["class_idx"] for s in scans], dtype=np.int32)
    cluster_labels, total_clusters = group_scans_by_perceptual_cluster(scans)
    
    # Historical Split (LOCKED 117-IMAGE TEST SET)
    hist_train_idx, hist_val_test_idx = train_test_split(np.arange(len(scans)), test_size=0.30, stratify=labels, random_state=42)
    hist_val_idx, hist_test_idx = train_test_split(hist_val_test_idx, test_size=0.50, stratify=labels[hist_val_test_idx], random_state=42)
    
    # Grouped Generalization Split (Seed 42)
    grp_train_idx, grp_val_idx, grp_test_idx = create_grouped_stratified_split(scans, cluster_labels, seed=42)
    
    print("\nLoading images with aspect-ratio preserving letterbox preprocessing...")
    X_all = load_images(scans)
    
    X_train = X_all[grp_train_idx]
    y_train = labels[grp_train_idx]
    
    X_val = X_all[grp_val_idx]
    y_val = labels[grp_val_idx]
    
    X_train_prep = preprocess_effnet(X_train.copy())
    X_val_prep = preprocess_effnet(X_val.copy())
    
    save_path = os.path.join(candidates_dir, "v13_efficientnet_b0.keras")
    
    if os.path.exists(save_path):
        print(f"\n=======================================================================")
        print(f"FOUND EXISTING CANDIDATE MODEL: Loading {save_path}")
        print(f"=======================================================================")
        model = tf.keras.models.load_model(save_path)
    else:
        print(f"\n=======================================================================")
        print(f"TRAINING V13 CANDIDATE: EfficientNetB0 Transfer Learning")
        print(f"=======================================================================")
        
        model, base_model = build_v13_model(dropout_rate=0.20)
        
        print("  Stage 1: Warming up classifier head (8 epochs, LR=5e-4)...")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
            loss="sparse_categorical_crossentropy",
            metrics=['accuracy']
        )
        model.fit(
            X_train_prep, y_train,
            validation_data=(X_val_prep, y_val),
            epochs=8,
            batch_size=32,
            verbose=1
        )
        
        print("  Stage 2: Fine-tuning top 20 backbone layers (15 epochs max, LR=1e-4)...")
        base_model.trainable = True
        for layer in base_model.layers[:-20]:
            layer.trainable = False
            
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss="sparse_categorical_crossentropy",
            metrics=['accuracy']
        )
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, mode='min'),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, mode='min')
        ]
        model.fit(
            X_train_prep, y_train,
            validation_data=(X_val_prep, y_val),
            epochs=15,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        model.save(save_path)
        print(f"  Model saved to: {save_path}")
        
    # --- VALIDATION EVALUATION & THRESHOLD SWEEP ---
    val_eval_std = evaluate_model(model, X_val, y_val, classes, mal_threshold=None)
    
    best_val_score = -1.0
    best_thresh = None
    best_val_eval = None
    threshold_sweep = []
    
    for t in np.arange(0.20, 0.71, 0.02):
        t_val = round(float(t), 2)
        eval_t = evaluate_model(model, X_val, y_val, classes, mal_threshold=t_val)
        threshold_sweep.append({
            "threshold": t_val,
            "accuracy": eval_t["accuracy"],
            "macro_f1": eval_t["macro_f1"],
            "malignant_recall": eval_t["malignant"]["recall"],
            "malignant_precision": eval_t["malignant"]["precision"],
            "false_negatives": eval_t["false_negatives"],
            "false_positives": eval_t["false_positives"]
        })
        rec = eval_t["malignant"]["recall"]
        prec = eval_t["malignant"]["precision"]
        f1_m = eval_t["malignant"]["f1"]
        
        recall_penalty = 0.0 if rec >= 0.85 else (rec - 0.85) * 2.0
        score = f1_m + 0.5 * prec + recall_penalty
        
        if score > best_val_score:
            best_val_score = score
            best_thresh = t_val
            best_val_eval = eval_t
            
    # --- FROZEN TEST SET EVALUATIONS ---
    X_hist_test = X_all[hist_test_idx]
    y_hist_test = labels[hist_test_idx]
    
    X_grp_test = X_all[grp_test_idx]
    y_grp_test = labels[grp_test_idx]
    
    hist_test_eval = evaluate_model(model, X_hist_test, y_hist_test, classes, mal_threshold=best_thresh)
    hist_test_std = evaluate_model(model, X_hist_test, y_hist_test, classes, mal_threshold=None)
    
    grp_test_eval = evaluate_model(model, X_grp_test, y_grp_test, classes, mal_threshold=best_thresh)
    grp_test_std = evaluate_model(model, X_grp_test, y_grp_test, classes, mal_threshold=None)
    
    results = {
        "v13_efficientnet_b0": {
            "id": "v13_efficientnet_b0",
            "name": "V13: EfficientNet-B0 Pretrained Backbone",
            "model_path": save_path,
            "validation_standard": val_eval_std,
            "best_val_threshold": best_thresh,
            "validation_calibrated": best_val_eval,
            "threshold_sweep": threshold_sweep,
            "historical_test_117": hist_test_eval,
            "historical_test_117_std": hist_test_std,
            "grouped_test": grp_test_eval,
            "grouped_test_std": grp_test_std
        }
    }
    
    val_summary = {
        "v13_efficientnet_b0": {
            "id": "v13_efficientnet_b0",
            "name": "V13: EfficientNet-B0 Pretrained Backbone",
            "validation_standard": val_eval_std,
            "best_val_threshold": best_thresh,
            "validation_calibrated": best_val_eval
        }
    }
    
    val_json_out_path = os.path.join(reports_dir, "v13_validation_summary.json")
    with open(val_json_out_path, "w", encoding="utf-8") as f:
        json.dump(val_summary, f, indent=4)
    print(f"\nSaved V13 validation summary JSON to: {val_json_out_path}")
    
    json_out_path = os.path.join(reports_dir, "v13_evaluation_results.json")
    with open(json_out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"Saved V13 evaluation raw JSON to: {json_out_path}")
    
    generate_v13_markdown_report(reports_dir, results)

def generate_v13_markdown_report(reports_dir, results):
    report_path = os.path.join(reports_dir, "v13_final_evaluation_report.md")
    res_b0 = results["v13_efficientnet_b0"]
    
    v13_grp = res_b0["grouped_test"]
    v13_hist = res_b0["historical_test_117"]
    
    # Check promotion condition vs V5-B baseline (Grouped Malignant Recall >= 85% AND Malignant Precision >= 60% AND Grouped Acc >= 70%)
    top_grp_rec = v13_grp["malignant"]["recall"]
    top_grp_prec = v13_grp["malignant"]["precision"]
    top_grp_acc = v13_grp["accuracy"]
    
    if top_grp_rec >= 0.85 and top_grp_prec >= 0.60 and top_grp_acc >= 0.70:
        promo_recommendation = f"**PROMOTABLE FOR REVIEW** (Meets criteria: Recall {top_grp_rec*100:.2f}%, Precision {top_grp_prec*100:.2f}%, Accuracy {top_grp_acc*100:.2f}%). Note: NO automatic promotion will occur per project safety guidelines."
    else:
        promo_recommendation = f"**DO NOT PROMOTE**. Recommend retaining **V5-B** as active production model."

    report_md = f"""PRODUCTION MODEL CHANGED: NO
PRODUCTION METADATA CHANGED: NO
LOCKED 117-IMAGE TEST SET MODIFIED: NO
DATASET IMAGES MODIFIED: NO
DEPLOYMENT PERFORMED: NO
GIT COMMIT PERFORMED: NO
GIT PUSH PERFORMED: NO

# V13 EfficientNet-B0 Backbone Generalization Report

**Date**: September 02, 2026  
**Project**: Breast Cancer AI — Ultrasound Image Classifier  
**Dataset**: `dataset/BUSI` (776 Clean Trainable Scans)  
**Production Model Status**: `backend/models/breast_image_classifier.keras` remains **unmodified** and **active** (V5-B).  
**Evaluated Backbone**: EfficientNetB0 (`backend/models/candidates/v13_efficientnet_b0.keras`)

---

## 1. Executive Summary & Production Status
- **V13 Hypothesis**: Test whether replacing MobileNetV2 (2.26M params) with a stronger pretrained image backbone (**EfficientNetB0**, 4.05M params with Squeeze-and-Excitation channel attention) can improve malignant generalization ($\ge 85\%$ Malignant Recall, $\ge 60\%$ Precision, $\ge 70\%$ Accuracy).

---

## 2. Grouped Generalization Test Comparison Across Iterations

| Model Iteration | Backbone Architecture | Accuracy | Macro F1 | Malignant Recall | Malignant Precision | Malignant F1 | Malignant FN | Malignant FP | Total Errors |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **V5-B (Active Prod)** | MobileNetV2 | 70.09% | 69.46% | 80.65% | 62.50% | 70.42% | 6 | 15 | 32 |
| **V11-B** | MobileNetV2 (50% Aug) | 69.16% | 70.12% | 90.32% | 57.14% | 70.00% | 3 | 21 | 33 |
| **V13-B0** | **EfficientNetB0** | {v13_grp['accuracy']*100:.2f}% | {v13_grp['macro_f1']*100:.2f}% | {v13_grp['malignant']['recall']*100:.2f}% | {v13_grp['malignant']['precision']*100:.2f}% | {v13_grp['malignant']['f1']*100:.2f}% | {v13_grp['false_negatives']} | {v13_grp['false_positives']} | {v13_grp['total_incorrect']} |

---

## 3. Historical Locked 117-Image Test Comparison

| Model Iteration | Backbone Architecture | Accuracy | Macro F1 | Malignant Recall | Malignant Precision | Malignant F1 | Malignant FN | Malignant FP | Total Errors |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **V5-B (Active Prod)** | MobileNetV2 | 80.34% | 79.75% | 87.50% | 77.78% | 82.35% | 4 | 8 | 23 |
| **V11-B** | MobileNetV2 (50% Aug) | 78.63% | 78.45% | 96.88% | 70.45% | 81.58% | 1 | 13 | 25 |
| **V13-B0** | **EfficientNetB0** | {v13_hist['accuracy']*100:.2f}% | {v13_hist['macro_f1']*100:.2f}% | {v13_hist['malignant']['recall']*100:.2f}% | {v13_hist['malignant']['precision']*100:.2f}% | {v13_hist['malignant']['f1']*100:.2f}% | {v13_hist['false_negatives']} | {v13_hist['false_positives']} | {v13_hist['total_incorrect']} |

---

## 4. Per-Class Performance Breakdown (V13-B0)

### Grouped Generalization Test:
- **Benign**: Precision = {v13_grp['benign']['precision']*100:.2f}%, Recall = {v13_grp['benign']['recall']*100:.2f}%, F1 = {v13_grp['benign']['f1']*100:.2f}%
- **Malignant**: Precision = {v13_grp['malignant']['precision']*100:.2f}%, Recall = {v13_grp['malignant']['recall']*100:.2f}%, F1 = {v13_grp['malignant']['f1']*100:.2f}%
- **Normal**: Precision = {v13_grp['normal']['precision']*100:.2f}%, Recall = {v13_grp['normal']['recall']*100:.2f}%, F1 = {v13_grp['normal']['f1']*100:.2f}%

### Locked Historical Test Set:
- **Benign**: Precision = {v13_hist['benign']['precision']*100:.2f}%, Recall = {v13_hist['benign']['recall']*100:.2f}%, F1 = {v13_hist['benign']['f1']*100:.2f}%
- **Malignant**: Precision = {v13_hist['malignant']['precision']*100:.2f}%, Recall = {v13_hist['malignant']['recall']*100:.2f}%, F1 = {v13_hist['malignant']['f1']*100:.2f}%
- **Normal**: Precision = {v13_hist['normal']['precision']*100:.2f}%, Recall = {v13_hist['normal']['recall']*100:.2f}%, F1 = {v13_hist['normal']['f1']*100:.2f}%

---

## 5. Model Recommendation & Promotion Decision

- **Selected Validation Threshold**: `{res_b0['best_val_threshold']}`
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
    print(f"\nSuccessfully generated V13 final evaluation report at: {report_path}")

if __name__ == "__main__":
    main()
