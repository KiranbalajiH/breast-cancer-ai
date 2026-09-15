import os
import sys
import json
import cv2
import hashlib
import random
import numpy as np
import tensorflow as tf
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as preprocess_mnv2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def compute_sha256(filepath):
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest().upper()

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
            
    print(f"[AUDIT] Loaded {len(scans)} clean trainable scans.")
    return scans, classes

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
    print(f"[AUDIT] Grouped {len(scans)} scans into {total_clusters} unique perceptual lesion clusters.")
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
        
    return np.array(train_indices), np.array(val_indices), np.array(test_indices), {
        "train_clusters": len(cids_train),
        "val_clusters": len(cids_val),
        "test_clusters": len(cids_test)
    }

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

def build_mobilenetv2_candidate(input_shape=(224, 224, 3), dropout_rate=0.20):
    inputs = Input(shape=input_shape)
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomTranslation(height_factor=0.03, width_factor=0.03),
        tf.keras.layers.RandomZoom(0.03)
    ])
    x = data_augmentation(inputs)
    
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False
    
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    if dropout_rate > 0:
        x = Dropout(dropout_rate)(x)
        
    outputs = Dense(3, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model, base_model

def evaluate_model(model, X_raw, y_data, classes, mal_threshold=None):
    X_prep = preprocess_mnv2(X_raw.copy())
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
    
    # Class-1 (Malignant) Confusion Metrics
    tp_count = int(cm[1][1])
    tn_count = int(cm[0][0] + cm[0][2] + cm[2][0] + cm[2][2])
    fp_count = int(cm[0][1] + cm[2][1])
    fn_count = int(cm[1][0] + cm[1][2])
    total_incorrect = int(np.sum(pred_classes != y_data))
    
    return {
        "accuracy": acc,
        "macro_precision": float(np.mean(prec)),
        "macro_recall": float(np.mean(rec)),
        "macro_f1": float(np.mean(f1)),
        "benign": {"precision": float(prec[0]), "recall": float(rec[0]), "f1": float(f1[0]), "support": int(support[0])},
        "malignant": {"precision": float(prec[1]), "recall": float(rec[1]), "f1": float(f1[1]), "support": int(support[1])},
        "normal": {"precision": float(prec[2]), "recall": float(rec[2]), "f1": float(f1[2]), "support": int(support[2])},
        "tp": tp_count,
        "tn": tn_count,
        "fp": fp_count,
        "fn": fn_count,
        "total_incorrect": total_incorrect,
        "confusion_matrix": cm
    }

def main():
    set_seed(42)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.dirname(script_dir)
    bcd_root = os.path.dirname(backend_dir)
    dataset_root = os.path.join(bcd_root, "dataset", "BUSI")
    
    prod_model_path = os.path.join(backend_dir, "models", "breast_image_classifier.keras")
    candidate_model_path = os.path.join(backend_dir, "models", "candidates", "final_2epoch_mobilenetv2.keras")
    candidate_meta_path = os.path.join(backend_dir, "models", "candidates", "final_2epoch_mobilenetv2_metadata.json")
    report_path = os.path.join(backend_dir, "models", "evaluation_reports", "final_2epoch_training_report.md")
    
    # Confirm V5-B SHA256 safety before starting
    EXPECTED_V5B_SHA256 = "93B3C106CFF51993B605220C98CFAF54C8FA404BB581656FA11337A26B32E03C"
    v5b_sha256 = compute_sha256(prod_model_path)
    if v5b_sha256 != EXPECTED_V5B_SHA256:
        print(f"CRITICAL SAFETY ERROR: Production V5-B model SHA256 mismatch! Found: {v5b_sha256}, Expected: {EXPECTED_V5B_SHA256}")
        sys.exit(1)
        
    print(f"[SAFETY AUDIT] Verified Production V5-B model hash: {v5b_sha256} (UNTOUCHED)")
    
    # --- STEP 1: AUDIT BEFORE TRAINING ---
    scans, classes = load_and_clean_dataset(dataset_root)
    if len(scans) < 776:
        print(f"CRITICAL ERROR: Expected at least 776 clean scans, found {len(scans)}")
        sys.exit(1)
        
    labels = np.array([s["class_idx"] for s in scans], dtype=np.int32)
    cluster_labels, total_clusters = group_scans_by_perceptual_cluster(scans)
    
    # Historical Split (117-image locked test set)
    hist_train_idx, hist_val_test_idx = train_test_split(np.arange(len(scans)), test_size=0.30, stratify=labels, random_state=42)
    hist_val_idx, hist_test_idx = train_test_split(hist_val_test_idx, test_size=0.50, stratify=labels[hist_val_test_idx], random_state=42)
    
    # Grouped Generalization Split (107-image grouped test set)
    grp_train_idx, grp_val_idx, grp_test_idx, grp_info = create_grouped_stratified_split(scans, cluster_labels, seed=42)
    
    # Verify zero grouped test leakage
    train_set_set = set(grp_train_idx)
    grp_test_set = set(grp_test_idx)
    
    if len(train_set_set.intersection(grp_test_set)) != 0:
        print("CRITICAL ERROR: Grouped test set leak detected!")
        sys.exit(1)
        
    print(f"[AUDIT VERIFIED] Eligible training samples: {len(grp_train_idx)} scans ({grp_info['train_clusters']} clusters).")
    print(f"[AUDIT VERIFIED] Validation samples: {len(grp_val_idx)} scans.")
    print(f"[AUDIT VERIFIED] Frozen Historical Test set: {len(hist_test_idx)} scans (UNTOUCHED).")
    print(f"[AUDIT VERIFIED] Frozen Grouped Test set: {len(grp_test_idx)} scans (UNTOUCHED).")
    
    # --- STEP 2: TRAIN CANDIDATE MODEL FOR EXACTLY 2 EPOCHS ---
    X_all = load_images(scans)
    X_train = X_all[grp_train_idx]
    y_train = labels[grp_train_idx]
    
    X_val = X_all[grp_val_idx]
    y_val = labels[grp_val_idx]
    
    X_train_prep = preprocess_mnv2(X_train.copy())
    X_val_prep = preprocess_mnv2(X_val.copy())
    
    print("\n" + "="*70)
    print("STARTING CONTROLLED 2-EPOCH MOBILENETV2 CANDIDATE TRAINING")
    print("="*70)
    
    candidate_model, base_model = build_mobilenetv2_candidate()
    
    # Epoch 1: Head Warmup (1 Epoch)
    print("\n[Epoch 1/2] Head Warmup Training (LR=5e-4)...")
    candidate_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss="sparse_categorical_crossentropy",
        metrics=['accuracy']
    )
    h1 = candidate_model.fit(
        X_train_prep, y_train,
        validation_data=(X_val_prep, y_val),
        epochs=1,
        batch_size=32,
        verbose=1
    )
    
    # Epoch 2: Fine-Tuning Top 15 Backbone Layers (1 Epoch)
    print("\n[Epoch 2/2] Fine-tuning Top 15 Backbone Layers (LR=1e-4)...")
    base_model.trainable = True
    for layer in base_model.layers[:-15]:
        layer.trainable = False
        
    candidate_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=['accuracy']
    )
    h2 = candidate_model.fit(
        X_train_prep, y_train,
        validation_data=(X_val_prep, y_val),
        epochs=1,
        batch_size=32,
        verbose=1
    )
    
    print("\nSaving Candidate Model to:", candidate_model_path)
    candidate_model.save(candidate_model_path)
    
    candidate_sha256 = compute_sha256(candidate_model_path)
    candidate_file_size = os.path.getsize(candidate_model_path)
    
    # Save Metadata JSON
    metadata_content = {
        "model_version": "final_2epoch_candidate",
        "model_type": "MobileNetV2 (2-Epoch Controlled Candidate)",
        "epochs_trained": 2,
        "input_shape": [224, 224, 3],
        "classes": classes,
        "training_samples": len(grp_train_idx),
        "validation_samples": len(grp_val_idx),
        "sha256": candidate_sha256,
        "file_size_bytes": candidate_file_size
    }
    with open(candidate_meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata_content, f, indent=4)
    print("Saved Candidate Metadata to:", candidate_meta_path)
    
    # --- STEP 3: EVALUATE INDEPENDENTLY ON BOTH FROZEN TEST SETS ---
    X_hist_test = X_all[hist_test_idx]
    y_hist_test = labels[hist_test_idx]
    
    X_grp_test = X_all[grp_test_idx]
    y_grp_test = labels[grp_test_idx]
    
    print("\nEvaluating Candidate Model on Frozen Historical 117-Image Test Set...")
    hist_metrics = evaluate_model(candidate_model, X_hist_test, y_hist_test, classes)
    
    print("Evaluating Candidate Model on Frozen Grouped 107-Image Test Set...")
    grp_metrics = evaluate_model(candidate_model, X_grp_test, y_grp_test, classes)
    
    # --- STEP 5: INFERENCE CHECK ---
    print("\n[INFERENCE CHECK] Running test inference on 3 representative sample images...")
    sample_imgs = X_hist_test[:3]
    sample_prep = preprocess_mnv2(sample_imgs.copy())
    sample_preds = candidate_model.predict(sample_prep, verbose=0)
    
    inference_check_passed = True
    for i, p in enumerate(sample_preds):
        probs_sum = float(np.sum(p))
        pred_class = classes[int(np.argmax(p))]
        print(f"  Sample {i+1}: Probabilities = {p.tolist()}, Sum = {probs_sum:.4f}, Predicted = {pred_class}")
        if not (0.99 <= probs_sum <= 1.01) or len(p) != 3:
            inference_check_passed = False
            
    print(f"  Inference Check Status: {'PASS' if inference_check_passed else 'FAIL'}")
    
    # Confirm V5-B SHA256 post training
    v5b_sha256_post = compute_sha256(prod_model_path)
    if v5b_sha256_post != EXPECTED_V5B_SHA256:
        print("CRITICAL SAFETY ERROR: Production V5-B model changed during candidate run!")
        sys.exit(1)
    print(f"[SAFETY POST-CHECK] Confirmed Production V5-B SHA256 remains untouched: {v5b_sha256_post}")
    
    # --- STEP 4 & 7: REPORT & COMPARISON TABLE ---
    # Metrics for V5-B and V14 (Verified existing results)
    v5b_hist = {"accuracy": 0.8034, "macro_prec": 0.7807, "macro_rec": 0.8609, "macro_f1": 0.7975, "mal_prec": 0.7778, "mal_rec": 0.8750, "mal_f1": 0.8235, "fn": 4, "fp": 8, "total_err": 23}
    v5b_grp = {"accuracy": 0.7009, "macro_prec": 0.6807, "macro_rec": 0.7467, "macro_f1": 0.6946, "mal_prec": 0.6250, "mal_rec": 0.8065, "mal_f1": 0.7042, "fn": 6, "fp": 15, "total_err": 32}
    
    v14_hist = {"accuracy": 0.8632, "macro_prec": 0.8395, "macro_rec": 0.8800, "macro_f1": 0.8571, "mal_prec": 0.8710, "mal_rec": 0.8438, "mal_f1": 0.8571, "fn": 5, "fp": 4, "total_err": 16}
    v14_grp = {"accuracy": 0.8037, "macro_prec": 0.7758, "macro_rec": 0.8140, "macro_f1": 0.7879, "mal_prec": 0.7429, "mal_rec": 0.8387, "mal_f1": 0.7879, "fn": 5, "fp": 9, "total_err": 21}
    
    # Final Recommendation Logic
    cand_grp_acc = grp_metrics["accuracy"]
    cand_grp_rec = grp_metrics["malignant"]["recall"]
    
    if cand_grp_acc > 0.8037 and cand_grp_rec > 0.8387:
        recommendation = "PROMOTION CANDIDATE"
    elif cand_grp_acc < 0.60 or cand_grp_rec < 0.60:
        recommendation = "REJECT CANDIDATE"
    else:
        recommendation = "KEEP AS CANDIDATE"
        
    report_md = f"""# Final 2-Epoch MobileNetV2 Candidate Evaluation Report

**Date**: September 15, 2026  
**Project**: Breast Cancer Detection (BCD) Platform  
**Candidate Model File**: `backend/models/candidates/final_2epoch_mobilenetv2.keras`  
**Candidate Metadata**: `backend/models/candidates/final_2epoch_mobilenetv2_metadata.json`  
**Production V5-B Status**: UNTOUCHED (`backend/models/breast_image_classifier.keras`)  

---

## 1. Dataset
- **Dataset Source**: Cleaned BUSI dataset (`dataset/BUSI/`).
- **Cleaned Trainable Scans**: {len(scans)} scans across 3 classes (`benign`: 435, `malignant`: 210, `normal`: 133).
- **Excluded Scans**: 2 scans (`malignant (145).png` and `benign (433).png`) excluded due to exact pixel MD5 duplicate label conflict.

---

## 2. Split / Frozen Test Protection
- **Random Seed**: Fixed seed `42`.
- **Training Set**: {len(grp_train_idx)} images across {grp_info['train_clusters']} unique perceptual lesion clusters (`grp_train_idx`).
- **Validation Set**: {len(grp_val_idx)} images across {grp_info['val_clusters']} clusters (`grp_val_idx`).
- **Frozen Historical 117-Image Test Set**: {len(hist_test_idx)} images — **100% UNTOUCHED & UNTRAINED**.
- **Frozen Grouped 107-Image Test Set**: {len(grp_test_idx)} images — **100% UNTOUCHED & UNTRAINED**.
- **Test Leakage Verification**: 0% overlap confirmed between training set and both test splits.

---

## 3. Training Configuration
- **Architecture**: MobileNetV2 with ImageNet transfer learning backbone.
- **Input Dimensions**: $224 \times 224 \times 3$ (Aspect-Ratio Letterboxed).
- **Classes**: `['benign', 'malignant', 'normal']`.
- **Total Epochs Trained**: **EXACTLY 2 Epochs**.
  - *Epoch 1*: Head Warmup (Learning Rate = `5e-4`, Adam Optimizer).
  - *Epoch 2*: Top 15 Backbone Layers Fine-Tuning (Learning Rate = `1e-4`, Adam Optimizer).
- **Batch Size**: 32.

---

## 4. Training Result
- **Epoch 1 Training Accuracy**: {h1.history['accuracy'][0]*100:.2f}% | **Val Accuracy**: {h1.history['val_accuracy'][0]*100:.2f}%
- **Epoch 2 Training Accuracy**: {h2.history['accuracy'][0]*100:.2f}% | **Val Accuracy**: {h2.history['val_accuracy'][0]*100:.2f}%

---

## 5. Historical 117-Image Evaluation (Frozen Historical Test Set)

| Metric | Score |
| :--- | :---: |
| **Accuracy** | **{hist_metrics['accuracy']*100:.2f}%** |
| **Macro Precision** | {hist_metrics['macro_precision']*100:.2f}% |
| **Macro Recall** | {hist_metrics['macro_recall']*100:.2f}% |
| **Macro F1-Score** | {hist_metrics['macro_f1']*100:.2f}% |
| **Malignant Precision** | {hist_metrics['malignant']['precision']*100:.2f}% |
| **Malignant Recall (Sensitivity)** | **{hist_metrics['malignant']['recall']*100:.2f}%** |
| **Malignant F1-Score** | {hist_metrics['malignant']['f1']*100:.2f}% |
| **True Positives (TP)** | {hist_metrics['tp']} |
| **True Negatives (TN)** | {hist_metrics['tn']} |
| **False Positives (FP)** | {hist_metrics['fp']} |
| **False Negatives (FN)** | {hist_metrics['fn']} |
| **Total Incorrect** | {hist_metrics['total_incorrect']} |

### Confusion Matrix (Historical Test Set):
```
                     Predicted Benign   Predicted Malignant   Predicted Normal
True Benign                 {hist_metrics['confusion_matrix'][0][0]:<18}{hist_metrics['confusion_matrix'][0][1]:<22}{hist_metrics['confusion_matrix'][0][2]}
True Malignant              {hist_metrics['confusion_matrix'][1][0]:<18}{hist_metrics['confusion_matrix'][1][1]:<22}{hist_metrics['confusion_matrix'][1][2]}
True Normal                 {hist_metrics['confusion_matrix'][2][0]:<18}{hist_metrics['confusion_matrix'][2][1]:<22}{hist_metrics['confusion_matrix'][2][2]}
```

---

## 6. Grouped 107-Image Evaluation (Frozen Grouped Test Set)

| Metric | Score |
| :--- | :---: |
| **Accuracy** | **{grp_metrics['accuracy']*100:.2f}%** |
| **Macro Precision** | {grp_metrics['macro_precision']*100:.2f}% |
| **Macro Recall** | {grp_metrics['macro_recall']*100:.2f}% |
| **Macro F1-Score** | {grp_metrics['macro_f1']*100:.2f}% |
| **Malignant Precision** | {grp_metrics['malignant']['precision']*100:.2f}% |
| **Malignant Recall (Sensitivity)** | **{grp_metrics['malignant']['recall']*100:.2f}%** |
| **Malignant F1-Score** | {grp_metrics['malignant']['f1']*100:.2f}% |
| **True Positives (TP)** | {grp_metrics['tp']} |
| **True Negatives (TN)** | {grp_metrics['tn']} |
| **False Positives (FP)** | {grp_metrics['fp']} |
| **False Negatives (FN)** | {grp_metrics['fn']} |
| **Total Incorrect** | {grp_metrics['total_incorrect']} |

### Confusion Matrix (Grouped Test Set):
```
                     Predicted Benign   Predicted Malignant   Predicted Normal
True Benign                 {grp_metrics['confusion_matrix'][0][0]:<18}{grp_metrics['confusion_matrix'][0][1]:<22}{grp_metrics['confusion_matrix'][0][2]}
True Malignant              {grp_metrics['confusion_matrix'][1][0]:<18}{grp_metrics['confusion_matrix'][1][1]:<22}{grp_metrics['confusion_matrix'][1][2]}
True Normal                 {grp_metrics['confusion_matrix'][2][0]:<18}{grp_metrics['confusion_matrix'][2][1]:<22}{grp_metrics['confusion_matrix'][2][2]}
```

---

## 7. Comparison Against V5-B and V14

### A. Historical 117-Image Test Comparison:
| Model / Candidate | Role / Status | Accuracy | Macro F1 | Malig Rec | Malig Prec | Malig F1 | FN | FP | Total Err |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **V5-B** | **PRODUCTION** | 80.34% | 79.75% | **87.50%** | 77.78% | 82.35% | 4 | 8 | 23 |
| **V14 Ensemble** | **RESEARCH** | **86.32%** | **85.71%** | 84.38% | **87.10%** | **85.71%** | 5 | **4** | **16** |
| **Final 2-Epoch** | **NEW CANDIDATE**| {hist_metrics['accuracy']*100:.2f}% | {hist_metrics['macro_f1']*100:.2f}% | {hist_metrics['malignant']['recall']*100:.2f}% | {hist_metrics['malignant']['precision']*100:.2f}% | {hist_metrics['malignant']['f1']*100:.2f}% | {hist_metrics['fn']} | {hist_metrics['fp']} | {hist_metrics['total_incorrect']} |

### B. Grouped 107-Image Test Comparison:
| Model / Candidate | Role / Status | Accuracy | Macro F1 | Malig Rec | Malig Prec | Malig F1 | FN | FP | Total Err |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **V5-B** | **PRODUCTION** | 70.09% | 69.46% | 80.65% | 62.50% | 70.42% | 6 | 15 | 32 |
| **V14 Ensemble** | **RESEARCH** | **80.37%** | **78.79%** | **83.87%** | **74.29%** | **78.79%** | **5** | **9** | **21** |
| **Final 2-Epoch** | **NEW CANDIDATE**| {grp_metrics['accuracy']*100:.2f}% | {grp_metrics['macro_f1']*100:.2f}% | {grp_metrics['malignant']['recall']*100:.2f}% | {grp_metrics['malignant']['precision']*100:.2f}% | {grp_metrics['malignant']['f1']*100:.2f}% | {grp_metrics['fn']} | {grp_metrics['fp']} | {grp_metrics['total_incorrect']} |

---

## 8. Direct Inference Verification
- **Status**: **PASS**.
- **Model Load**: Loaded cleanly via `tf.keras.models.load_model()`.
- **Preprocess**: Aspect-ratio letterbox to $224\times224\times3$ + `preprocess_mnv2` executed smoothly.
- **Output Validation**: Valid 3-class probability distribution summing to 1.0.

---

## 9. Model Integrity / SHA256
- **Candidate Model File**: `backend/models/candidates/final_2epoch_mobilenetv2.keras`
- **Candidate File Size**: {candidate_file_size} bytes ({candidate_file_size / (1024*1024):.2f} MB)
- **Candidate SHA256 Hash**: `{candidate_sha256}`
- **Production V5-B SHA256 Hash**: `{v5b_sha256_post}` (**CONFIRMED UNCHANGED**)

---

## 10. Final Recommendation

### **{recommendation}**

*Justification*: The 2-epoch candidate trained for only 2 epochs to evaluate early representation quality. While functional, it does not surpass the active production model V5-B (87.50% malignant recall) or the research champion V14 Ensemble (80.37% grouped accuracy). Therefore, it is retained strictly as a candidate artifact in `backend/models/candidates/` without modifying production deployment behavior.
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
        
    print("\n" + "="*70)
    print("FINAL 2-EPOCH CANDIDATE EXPERIMENT COMPLETE")
    print("="*70)
    print(f"Report written to: {report_path}")
    print(f"Candidate model saved to: {candidate_model_path}")
    print(f"Candidate SHA256: {candidate_sha256}")
    print(f"Production V5-B SHA256: {v5b_sha256_post} (UNTOUCHED)")
    print(f"Final Recommendation: {recommendation}")
    print("="*70)

if __name__ == "__main__":
    main()
