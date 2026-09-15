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
        
        save_path = os.path.join(candidates_dir, cand["model_filename"])
        model.save(save_path)
        print(f"  Model candidate saved to: {save_path}")
        
        # PHASE 7 & 8: VALIDATION EVALUATION & THRESHOLD ANALYSIS (STRICTLY VALIDATION SET)
        val_eval_standard = evaluate_model_performance(model, X_val, y_val, classes, prep_fn=prep_fn, mal_threshold=None)
        
        threshold_sweep = []
        best_val_score = -1.0
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
            
            score = eval_t["macro_f1"] + 0.5 * eval_t["malignant"]["recall"]
            if score > best_val_score:
                best_val_score = score
                best_threshold = t_val
                best_val_eval = eval_t
                
        print(f"  Validation Threshold Analysis for {cid}:")
        print(f"    Standard Argmax -> Val Acc: {val_eval_standard['accuracy']*100:.2f}% | Val Mal Recall: {val_eval_standard['malignant']['recall']*100:.2f}% | Val Mal Prec: {val_eval_standard['malignant']['precision']*100:.2f}% | FN: {val_eval_standard['false_negatives']} | FP: {val_eval_standard['false_positives']}")
        print(f"    Optimal Thresh ({best_threshold}) -> Val Acc: {best_val_eval['accuracy']*100:.2f}% | Val Mal Recall: {best_val_eval['malignant']['recall']*100:.2f}% | Val Mal Prec: {best_val_eval['malignant']['precision']*100:.2f}% | FN: {best_val_eval['false_negatives']} | FP: {best_val_eval['false_positives']}")
        
        # PHASE 9: FROZEN TEST EVALUATIONS
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

if __name__ == "__main__":
    main()
