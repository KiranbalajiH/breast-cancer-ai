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

def load_images(scans):
    images = [preprocess_letterbox(s["filepath"]) for s in scans]
    return np.array(images, dtype=np.float32)

# --- KERAS 3 SERIALIZABLE FOCAL LOSS ---
@tf.keras.utils.register_keras_serializable(package="Custom", name="SparseCategoricalFocalLoss")
class SparseCategoricalFocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=None, name="sparse_categorical_focal_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.gamma = float(gamma)
        self.alpha = alpha if alpha is not None else [1.0, 1.5, 1.1]
        
    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_true_one_hot = tf.one_hot(y_true, depth=3)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        alpha_tensor = tf.constant(self.alpha, dtype=tf.float32)
        alpha_factor = tf.reduce_sum(alpha_tensor * y_true_one_hot, axis=-1)
        
        p_t = tf.reduce_sum(y_pred * y_true_one_hot, axis=-1)
        modulating_factor = tf.pow(1.0 - p_t, self.gamma)
        
        focal_loss = - alpha_factor * modulating_factor * tf.math.log(p_t)
        return tf.reduce_mean(focal_loss)
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "gamma": self.gamma,
            "alpha": self.alpha
        })
        return config

# --- MODEL ARCHITECTURE BUILDER ---
def build_v10_model(input_shape=(224, 224, 3), dropout_rate=0.2):
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

# --- EVALUATION FUNCTION ---
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
    print("V10 MALIGNANT SENSITIVITY OPTIMIZATION EXPERIMENT")
    print("="*75)
    
    scans, classes = load_and_clean_dataset(dataset_root)
    labels = np.array([s["class_idx"] for s in scans], dtype=np.int32)
    cluster_labels, total_clusters = group_scans_by_perceptual_cluster(scans)
    
    # Historical Split (LOCKED 117-IMAGE TEST SET)
    hist_train_idx, hist_val_test_idx = train_test_split(np.arange(len(scans)), test_size=0.30, stratify=labels, random_state=42)
    hist_val_idx, hist_test_idx = train_test_split(hist_val_test_idx, test_size=0.50, stratify=labels[hist_val_test_idx], random_state=42)
    
    # Grouped Generalization Split (Seed 42)
    grp_train_idx, grp_val_idx, grp_test_idx, grp_info = create_grouped_stratified_split(scans, cluster_labels, seed=42)
    
    print("\nLoading images with aspect-ratio preserving letterbox preprocessing...")
    X_all = load_images(scans)
    
    X_train = X_all[grp_train_idx]
    y_train = labels[grp_train_idx]
    
    X_val = X_all[grp_val_idx]
    y_val = labels[grp_val_idx]
    
    X_train_prep = preprocess_mnv2(X_train.copy())
    X_val_prep = preprocess_mnv2(X_val.copy())
    
    # Prepare malignant-focused augmented dataset for V10-D and V10-E
    mal_train_mask = (y_train == 1)
    X_train_mal = X_train[mal_train_mask]
    y_train_mal = y_train[mal_train_mask]
    
    # Double malignant training samples (conservative replication with augmentation in pipeline)
    X_train_aug_mal = np.concatenate([X_train, X_train_mal], axis=0)
    y_train_aug_mal = np.concatenate([y_train, y_train_mal], axis=0)
    X_train_prep_aug_mal = preprocess_mnv2(X_train_aug_mal.copy())
    
    v10_candidates_def = [
        {
            "id": "v10_mobilenetv2_a",
            "name": "V10-A: V5-B Control Baseline (Unweighted)",
            "weights": None,
            "loss": "sparse_categorical_crossentropy",
            "use_mal_aug": False
        },
        {
            "id": "v10_mobilenetv2_b",
            "name": "V10-B: Moderate Malignant Weighting {1.0, 1.6, 1.1}",
            "weights": {0: 1.0, 1: 1.6, 2: 1.1},
            "loss": "sparse_categorical_crossentropy",
            "use_mal_aug": False
        },
        {
            "id": "v10_mobilenetv2_c",
            "name": "V10-C: Sparse Categorical Focal Loss (gamma=2.0, alpha=[1.0, 1.5, 1.1])",
            "weights": None,
            "loss": SparseCategoricalFocalLoss(gamma=2.0, alpha=[1.0, 1.5, 1.1]),
            "use_mal_aug": False
        },
        {
            "id": "v10_mobilenetv2_d",
            "name": "V10-D: Controlled Malignant-Focused Augmentation",
            "weights": None,
            "loss": "sparse_categorical_crossentropy",
            "use_mal_aug": True
        },
        {
            "id": "v10_mobilenetv2_e",
            "name": "V10-E: Combined Moderate Weighting + Malignant-Focused Augmentation",
            "weights": {0: 1.0, 1: 1.6, 2: 1.1},
            "loss": "sparse_categorical_crossentropy",
            "use_mal_aug": True
        }
    ]
    
    results = {}
    
    for cand in v10_candidates_def:
        cid = cand["id"]
        cname = cand["name"]
        save_path = os.path.join(candidates_dir, f"{cid}.keras")
        
        if cand["use_mal_aug"]:
            X_tr_prep = X_train_prep_aug_mal
            y_tr = y_train_aug_mal
        else:
            X_tr_prep = X_train_prep
            y_tr = y_train
            
        if os.path.exists(save_path):
            print(f"\n=======================================================================")
            print(f"FOUND EXISTING CANDIDATE MODEL ({cid}): Loading {save_path}")
            print(f"=======================================================================")
            model = tf.keras.models.load_model(save_path, custom_objects={"SparseCategoricalFocalLoss": SparseCategoricalFocalLoss})
        else:
            print(f"\n=======================================================================")
            print(f"TRAINING V10 CANDIDATE: {cname}")
            print(f"=======================================================================")
            
            model, base_model = build_v10_model()
            
            print("  Stage 1: Warming up classifier head (8 epochs, LR=5e-4)...")
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
                loss=cand["loss"],
                metrics=['accuracy']
            )
            model.fit(
                X_tr_prep, y_tr,
                validation_data=(X_val_prep, y_val),
                epochs=8,
                batch_size=32,
                class_weight=cand["weights"],
                verbose=1
            )
            
            print("  Stage 2: Fine-tuning top 15 backbone layers (15 epochs max, LR=1e-4)...")
            base_model.trainable = True
            for layer in base_model.layers[:-15]:
                layer.trainable = False
                
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                loss=cand["loss"],
                metrics=['accuracy']
            )
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, mode='min'),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, mode='min')
            ]
            model.fit(
                X_tr_prep, y_tr,
                validation_data=(X_val_prep, y_val),
                epochs=15,
                batch_size=32,
                class_weight=cand["weights"],
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
        
        for t in np.arange(0.20, 0.51, 0.05):
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
            # Composite score: Macro F1 + 0.5 * Malignant Recall (validation only)
            score = eval_t["macro_f1"] + 0.5 * eval_t["malignant"]["recall"]
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
        
        results[cid] = {
            "id": cid,
            "name": cname,
            "model_path": save_path,
            "weights": str(cand["weights"]),
            "validation_standard": val_eval_std,
            "best_val_threshold": best_thresh,
            "validation_calibrated": best_val_eval,
            "threshold_sweep": threshold_sweep,
            "historical_test_117": hist_test_eval,
            "historical_test_117_std": hist_test_std,
            "grouped_test": grp_test_eval,
            "grouped_test_std": grp_test_std
        }
        
    val_summary = {cid: {
        "id": res["id"],
        "name": res["name"],
        "weights": res["weights"],
        "validation_standard": res["validation_standard"],
        "best_val_threshold": res["best_val_threshold"],
        "validation_calibrated": res["validation_calibrated"],
        "threshold_sweep": res["threshold_sweep"]
    } for cid, res in results.items()}
    val_json_out_path = os.path.join(reports_dir, "v10_validation_summary.json")
    with open(val_json_out_path, "w") as f:
        json.dump(val_summary, f, indent=4)
    print(f"\nSaved V10 validation summary JSON to: {val_json_out_path}")
    
    json_out_path = os.path.join(reports_dir, "v10_evaluation_results.json")
    with open(json_out_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved V10 evaluation raw JSON to: {json_out_path}")
    
    generate_v10_markdown_report(reports_dir, results, scans, labels, hist_test_idx, grp_test_idx)

def generate_v10_markdown_report(reports_dir, results, scans, labels, hist_test_idx, grp_test_idx):
    report_path = os.path.join(reports_dir, "v10_final_evaluation_report.md")
    
    cand_ids = ["v10_mobilenetv2_a", "v10_mobilenetv2_b", "v10_mobilenetv2_c", "v10_mobilenetv2_d", "v10_mobilenetv2_e"]
    cands_present = [results[cid] for cid in cand_ids if cid in results]
    
    # Sort candidates strictly by validation calibrated score
    cands_val_sorted = sorted(cands_present, key=lambda c: (c["validation_calibrated"]["macro_f1"] + 0.5 * c["validation_calibrated"]["malignant"]["recall"]), reverse=True)
    best_val_cand = cands_val_sorted[0]
    
    cands_hist_sorted = sorted(cands_present, key=lambda c: (c["historical_test_117"]["macro_f1"] + 0.5 * c["historical_test_117"]["malignant"]["recall"]), reverse=True)
    best_hist_cand = cands_hist_sorted[0]
    
    cands_grp_sorted = sorted(cands_present, key=lambda c: (c["grouped_test"]["macro_f1"] + 0.5 * c["grouped_test"]["malignant"]["recall"]), reverse=True)
    best_grp_cand = cands_grp_sorted[0]
    
    val_table_rows = []
    for c in cands_present:
        v_std = c["validation_calibrated"]
        val_table_rows.append(
            f"| **{c['id'].upper()}** | {v_std['accuracy']*100:.2f}% | {v_std['macro_f1']*100:.2f}% | "
            f"{v_std['malignant']['recall']*100:.2f}% | {v_std['malignant']['precision']*100:.2f}% | "
            f"{v_std['false_negatives']} | {v_std['false_positives']} | {c['best_val_threshold']} |"
        )
    val_table_str = "\n".join(val_table_rows)

    hist_table_rows = []
    for c in cands_present:
        h = c["historical_test_117"]
        hist_table_rows.append(
            f"| **{c['id'].upper()}** | {h['accuracy']*100:.2f}% | {h['macro_f1']*100:.2f}% | "
            f"{h['malignant']['recall']*100:.2f}% | {h['malignant']['precision']*100:.2f}% | "
            f"{h['malignant']['f1']*100:.2f}% | {h['false_negatives']} | {h['false_positives']} | {h['total_incorrect']} |"
        )
    hist_table_str = "\n".join(hist_table_rows)

    grp_table_rows = []
    for c in cands_present:
        g = c["grouped_test"]
        grp_table_rows.append(
            f"| **{c['id'].upper()}** | {g['accuracy']*100:.2f}% | {g['macro_f1']*100:.2f}% | "
            f"{g['malignant']['recall']*100:.2f}% | {g['malignant']['precision']*100:.2f}% | "
            f"{g['malignant']['f1']*100:.2f}% | {g['false_negatives']} | {g['false_positives']} | {g['total_incorrect']} |"
        )
    grp_table_str = "\n".join(grp_table_rows)

    # Check promotion condition: Grouped malignant recall > 80.65% (V5-B baseline) AND Macro F1 >= 70%
    v5b_grp_mal_rec = 0.8065
    top_grp_rec = best_val_cand["grouped_test"]["malignant"]["recall"]
    
    if top_grp_rec > v5b_grp_mal_rec and best_val_cand["grouped_test"]["macro_f1"] >= 0.70:
        promo_recommendation = f"**PROMOTABLE FOR PRODUCTION REVIEW**. {best_val_cand['id'].upper()} achieves higher malignant sensitivity ({top_grp_rec*100:.2f}% vs V5-B 80.65%) while maintaining strong macro F1."
    else:
        promo_recommendation = f"**DO NOT PROMOTE**. Recommend retaining **V5-B** as active production model."

    report_md = fr"""PRODUCTION MODEL CHANGED: NO
PRODUCTION METADATA CHANGED: NO
LOCKED 117-IMAGE TEST SET MODIFIED: NO
DATASET IMAGES MODIFIED: NO
DEPLOYMENT PERFORMED: NO
GIT COMMIT PERFORMED: NO
GIT PUSH PERFORMED: NO

# V10 Final Evaluation & Malignant Sensitivity Optimization Report

**Date**: September 02, 2026  
**Project**: Breast Cancer AI — Ultrasound Image Classifier  
**Dataset**: `dataset/BUSI` (776 Clean Trainable Scans)  
**Production Model Status**: `backend/models/breast_image_classifier.keras` remains **unmodified** and **active** (V5-B).  
**Candidate Models Location**: `backend/models/candidates/`

---

## 1. Executive Summary & Production Status
- **Baseline Benchmark**: V5-B (`backend/models/breast_image_classifier.keras`)
  - Historical 117 Test: Accuracy 80.34%, Macro F1 79.75%, Malignant Recall 87.50% (28/32, 4 FN, 8 FP), Malignant Precision 77.78%, Malignant F1 82.35%.
  - Grouped Test: Accuracy 70.09%, Macro F1 69.46%, Malignant Recall 80.65% (25/31, 6 FN, 15 FP), Malignant Precision 62.50%, Malignant F1 70.42%.
- **V10 Objective**: Reduce malignant false negatives and beat V5-B on grouped malignant recall (>80.65%) while preserving precision and macro F1.

---

## 2. V10 Candidate Configurations
- **V10-A (Control Baseline)**: MobileNetV2 with aspect-ratio preserving letterbox preprocessing (Unweighted).
- **V10-B (Moderate Class Weighting)**: V5-B + Class weights `{{0: 1.0, 1: 1.6, 2: 1.1}}`.
- **V10-C (Focal Loss)**: V5-B + Sparse Categorical Focal Loss ($\gamma=2.0, \alpha=[1.0, 1.5, 1.1]$).
- **V10-D (Malignant-Focused Augmentation)**: V5-B + Controlled conservative malignant augmentation.
- **V10-E (Combined Strategy)**: Moderate Class Weighting (`{{0: 1.0, 1: 1.6, 2: 1.1}}`) + Malignant-Focused Augmentation.

---

## 3. Validation Performance & Threshold Tuning

| Candidate | Val Accuracy | Val Macro F1 | Val Malignant Recall | Val Malignant Precision | Val FN | Val FP | Selected Threshold |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
{val_table_str}

---

## 4. Frozen Test Set Results

### A. Historical Locked 117-Image Test Results
| Candidate | Accuracy | Macro F1 | Malignant Recall | Malignant Precision | Malignant F1 | Malignant FN | Malignant FP | Total Errors |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
{hist_table_str}

### B. Grouped Generalization Test Results
| Candidate | Accuracy | Macro F1 | Malignant Recall | Malignant Precision | Malignant F1 | Malignant FN | Malignant FP | Total Errors |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
{grp_table_str}

---

## 5. Historical Test Comparison Across Iterations

| Model Iteration | Accuracy | Macro F1 | Malignant Precision | Malignant Recall | Malignant F1 | Malignant FN | Malignant FP | Total Errors |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **V1 Baseline** | 84.62% | 84.21% | 84.62% | 68.75% | 75.86% | 10 | 4 | 18 |
| **V3 Mild Weights** | 76.92% | 74.50% | 66.67% | 87.50% | 75.68% | 4 | 14 | 27 |
| **V5-B (Active Prod)** | 80.34% | 79.75% | 77.78% | **87.50%** | **82.35%** | **4** | 8 | 23 |
| **V6-B** | 81.20% | 80.54% | 78.57% | **87.50%** | 82.76% | **4** | 7 | 22 |
| **V7-D (Dual-View)** | 84.62% | 83.17% | 80.77% | 65.62% | 72.41% | 11 | 5 | 18 |
| **V9-D (Dual-View Aux)**| 87.18% | 86.46% | 81.25% | 81.25% | 81.25% | 6 | 6 | 15 |
| **V10-A** | {results.get('v10_mobilenetv2_a', {}).get('historical_test_117', {}).get('accuracy', 0)*100:.2f}% | {results.get('v10_mobilenetv2_a', {}).get('historical_test_117', {}).get('macro_f1', 0)*100:.2f}% | {results.get('v10_mobilenetv2_a', {}).get('historical_test_117', {}).get('malignant', {}).get('precision', 0)*100:.2f}% | {results.get('v10_mobilenetv2_a', {}).get('historical_test_117', {}).get('malignant', {}).get('recall', 0)*100:.2f}% | {results.get('v10_mobilenetv2_a', {}).get('historical_test_117', {}).get('malignant', {}).get('f1', 0)*100:.2f}% | {results.get('v10_mobilenetv2_a', {}).get('historical_test_117', {}).get('false_negatives', '-')} | {results.get('v10_mobilenetv2_a', {}).get('historical_test_117', {}).get('false_positives', '-')} | {results.get('v10_mobilenetv2_a', {}).get('historical_test_117', {}).get('total_incorrect', '-')} |
| **V10-B** | {results.get('v10_mobilenetv2_b', {}).get('historical_test_117', {}).get('accuracy', 0)*100:.2f}% | {results.get('v10_mobilenetv2_b', {}).get('historical_test_117', {}).get('macro_f1', 0)*100:.2f}% | {results.get('v10_mobilenetv2_b', {}).get('historical_test_117', {}).get('malignant', {}).get('precision', 0)*100:.2f}% | {results.get('v10_mobilenetv2_b', {}).get('historical_test_117', {}).get('malignant', {}).get('recall', 0)*100:.2f}% | {results.get('v10_mobilenetv2_b', {}).get('historical_test_117', {}).get('malignant', {}).get('f1', 0)*100:.2f}% | {results.get('v10_mobilenetv2_b', {}).get('historical_test_117', {}).get('false_negatives', '-')} | {results.get('v10_mobilenetv2_b', {}).get('historical_test_117', {}).get('false_positives', '-')} | {results.get('v10_mobilenetv2_b', {}).get('historical_test_117', {}).get('total_incorrect', '-')} |
| **V10-C** | {results.get('v10_mobilenetv2_c', {}).get('historical_test_117', {}).get('accuracy', 0)*100:.2f}% | {results.get('v10_mobilenetv2_c', {}).get('historical_test_117', {}).get('macro_f1', 0)*100:.2f}% | {results.get('v10_mobilenetv2_c', {}).get('historical_test_117', {}).get('malignant', {}).get('precision', 0)*100:.2f}% | {results.get('v10_mobilenetv2_c', {}).get('historical_test_117', {}).get('malignant', {}).get('recall', 0)*100:.2f}% | {results.get('v10_mobilenetv2_c', {}).get('historical_test_117', {}).get('malignant', {}).get('f1', 0)*100:.2f}% | {results.get('v10_mobilenetv2_c', {}).get('historical_test_117', {}).get('false_negatives', '-')} | {results.get('v10_mobilenetv2_c', {}).get('historical_test_117', {}).get('false_positives', '-')} | {results.get('v10_mobilenetv2_c', {}).get('historical_test_117', {}).get('total_incorrect', '-')} |
| **V10-D** | {results.get('v10_mobilenetv2_d', {}).get('historical_test_117', {}).get('accuracy', 0)*100:.2f}% | {results.get('v10_mobilenetv2_d', {}).get('historical_test_117', {}).get('macro_f1', 0)*100:.2f}% | {results.get('v10_mobilenetv2_d', {}).get('historical_test_117', {}).get('malignant', {}).get('precision', 0)*100:.2f}% | {results.get('v10_mobilenetv2_d', {}).get('historical_test_117', {}).get('malignant', {}).get('recall', 0)*100:.2f}% | {results.get('v10_mobilenetv2_d', {}).get('historical_test_117', {}).get('malignant', {}).get('f1', 0)*100:.2f}% | {results.get('v10_mobilenetv2_d', {}).get('historical_test_117', {}).get('false_negatives', '-')} | {results.get('v10_mobilenetv2_d', {}).get('historical_test_117', {}).get('false_positives', '-')} | {results.get('v10_mobilenetv2_d', {}).get('historical_test_117', {}).get('total_incorrect', '-')} |
| **V10-E** | {results.get('v10_mobilenetv2_e', {}).get('historical_test_117', {}).get('accuracy', 0)*100:.2f}% | {results.get('v10_mobilenetv2_e', {}).get('historical_test_117', {}).get('macro_f1', 0)*100:.2f}% | {results.get('v10_mobilenetv2_e', {}).get('historical_test_117', {}).get('malignant', {}).get('precision', 0)*100:.2f}% | {results.get('v10_mobilenetv2_e', {}).get('historical_test_117', {}).get('malignant', {}).get('recall', 0)*100:.2f}% | {results.get('v10_mobilenetv2_e', {}).get('historical_test_117', {}).get('malignant', {}).get('f1', 0)*100:.2f}% | {results.get('v10_mobilenetv2_e', {}).get('historical_test_117', {}).get('false_negatives', '-')} | {results.get('v10_mobilenetv2_e', {}).get('historical_test_117', {}).get('false_positives', '-')} | {results.get('v10_mobilenetv2_e', {}).get('historical_test_117', {}).get('total_incorrect', '-')} |

---

## 6. Grouped Generalization Comparison Across Iterations

| Model Iteration | Accuracy | Macro F1 | Malignant Precision | Malignant Recall | Malignant F1 | Malignant FN | Malignant FP | Total Errors |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **V5-B (Active Prod)** | 70.09% | 69.46% | 62.50% | **80.65%** | 70.42% | **6** | 15 | 32 |
| **V6-B** | 71.96% | 71.10% | 64.10% | **80.65%** | 71.43% | **6** | 14 | 30 |
| **V7-D (Dual-View)** | 76.64% | 74.95% | 69.57% | 51.61% | 59.26% | 15 | 7 | 25 |
| **V9-D (Dual-View Aux)**| 77.57% | 77.84% | 63.64% | 67.74% | 65.62% | 10 | 12 | 24 |
| **V10-A** | {results.get('v10_mobilenetv2_a', {}).get('grouped_test', {}).get('accuracy', 0)*100:.2f}% | {results.get('v10_mobilenetv2_a', {}).get('grouped_test', {}).get('macro_f1', 0)*100:.2f}% | {results.get('v10_mobilenetv2_a', {}).get('grouped_test', {}).get('malignant', {}).get('precision', 0)*100:.2f}% | {results.get('v10_mobilenetv2_a', {}).get('grouped_test', {}).get('malignant', {}).get('recall', 0)*100:.2f}% | {results.get('v10_mobilenetv2_a', {}).get('grouped_test', {}).get('malignant', {}).get('f1', 0)*100:.2f}% | {results.get('v10_mobilenetv2_a', {}).get('grouped_test', {}).get('false_negatives', '-')} | {results.get('v10_mobilenetv2_a', {}).get('grouped_test', {}).get('false_positives', '-')} | {results.get('v10_mobilenetv2_a', {}).get('grouped_test', {}).get('total_incorrect', '-')} |
| **V10-B** | {results.get('v10_mobilenetv2_b', {}).get('grouped_test', {}).get('accuracy', 0)*100:.2f}% | {results.get('v10_mobilenetv2_b', {}).get('grouped_test', {}).get('macro_f1', 0)*100:.2f}% | {results.get('v10_mobilenetv2_b', {}).get('grouped_test', {}).get('malignant', {}).get('precision', 0)*100:.2f}% | {results.get('v10_mobilenetv2_b', {}).get('grouped_test', {}).get('malignant', {}).get('recall', 0)*100:.2f}% | {results.get('v10_mobilenetv2_b', {}).get('grouped_test', {}).get('malignant', {}).get('f1', 0)*100:.2f}% | {results.get('v10_mobilenetv2_b', {}).get('grouped_test', {}).get('false_negatives', '-')} | {results.get('v10_mobilenetv2_b', {}).get('grouped_test', {}).get('false_positives', '-')} | {results.get('v10_mobilenetv2_b', {}).get('grouped_test', {}).get('total_incorrect', '-')} |
| **V10-C** | {results.get('v10_mobilenetv2_c', {}).get('grouped_test', {}).get('accuracy', 0)*100:.2f}% | {results.get('v10_mobilenetv2_c', {}).get('grouped_test', {}).get('macro_f1', 0)*100:.2f}% | {results.get('v10_mobilenetv2_c', {}).get('grouped_test', {}).get('malignant', {}).get('precision', 0)*100:.2f}% | {results.get('v10_mobilenetv2_c', {}).get('grouped_test', {}).get('malignant', {}).get('recall', 0)*100:.2f}% | {results.get('v10_mobilenetv2_c', {}).get('grouped_test', {}).get('malignant', {}).get('f1', 0)*100:.2f}% | {results.get('v10_mobilenetv2_c', {}).get('grouped_test', {}).get('false_negatives', '-')} | {results.get('v10_mobilenetv2_c', {}).get('grouped_test', {}).get('false_positives', '-')} | {results.get('v10_mobilenetv2_c', {}).get('grouped_test', {}).get('total_incorrect', '-')} |
| **V10-D** | {results.get('v10_mobilenetv2_d', {}).get('grouped_test', {}).get('accuracy', 0)*100:.2f}% | {results.get('v10_mobilenetv2_d', {}).get('grouped_test', {}).get('macro_f1', 0)*100:.2f}% | {results.get('v10_mobilenetv2_d', {}).get('grouped_test', {}).get('malignant', {}).get('precision', 0)*100:.2f}% | {results.get('v10_mobilenetv2_d', {}).get('grouped_test', {}).get('malignant', {}).get('recall', 0)*100:.2f}% | {results.get('v10_mobilenetv2_d', {}).get('grouped_test', {}).get('malignant', {}).get('f1', 0)*100:.2f}% | {results.get('v10_mobilenetv2_d', {}).get('grouped_test', {}).get('false_negatives', '-')} | {results.get('v10_mobilenetv2_d', {}).get('grouped_test', {}).get('false_positives', '-')} | {results.get('v10_mobilenetv2_d', {}).get('grouped_test', {}).get('total_incorrect', '-')} |
| **V10-E** | {results.get('v10_mobilenetv2_e', {}).get('grouped_test', {}).get('accuracy', 0)*100:.2f}% | {results.get('v10_mobilenetv2_e', {}).get('grouped_test', {}).get('macro_f1', 0)*100:.2f}% | {results.get('v10_mobilenetv2_e', {}).get('grouped_test', {}).get('malignant', {}).get('precision', 0)*100:.2f}% | {results.get('v10_mobilenetv2_e', {}).get('grouped_test', {}).get('malignant', {}).get('recall', 0)*100:.2f}% | {results.get('v10_mobilenetv2_e', {}).get('grouped_test', {}).get('malignant', {}).get('f1', 0)*100:.2f}% | {results.get('v10_mobilenetv2_e', {}).get('grouped_test', {}).get('false_negatives', '-')} | {results.get('v10_mobilenetv2_e', {}).get('grouped_test', {}).get('false_positives', '-')} | {results.get('v10_mobilenetv2_e', {}).get('grouped_test', {}).get('total_incorrect', '-')} |

---

## 7. Model Ranking & Recommendation
- **Best Validation Candidate**: `{best_val_cand['id'].upper()}`
- **Best Historical Test Candidate**: `{best_hist_cand['id'].upper()}`
- **Best Grouped Test Candidate**: `{best_grp_cand['id'].upper()}`
- **Promotion Decision**: {promo_recommendation}

---

PRODUCTION MODEL CHANGED: NO
PRODUCTION METADATA CHANGED: NO
LOCKED 117-IMAGE TEST SET MODIFIED: NO
DATASET IMAGES MODIFIED: NO
DEPLOYMENT PERFORMED: NO
GIT COMMIT PERFORMED: NO
GIT PUSH PERFORMED: NO
"""

    with open(report_path, "w") as f:
        f.write(report_md)
    print(f"\nSuccessfully generated V10 final evaluation report at: {report_path}")

if __name__ == "__main__":
    main()
