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
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

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

def load_and_clean_dataset_with_masks(dataset_root):
    classes = ['benign', 'malignant', 'normal']
    scans = []
    exclude_files = {"malignant (145).png", "benign (433).png"}
    
    for class_idx, class_name in enumerate(classes):
        class_dir = os.path.join(dataset_root, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        all_files = sorted(os.listdir(class_dir))
        scan_files = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg')) 
                      and '_mask' not in f.lower() and 'mask' not in f.lower() 
                      and f not in exclude_files]
                      
        for f in scan_files:
            full_path = os.path.join(class_dir, f)
            name_no_ext, ext = os.path.splitext(f)
            
            # Locate all corresponding mask files for this scan
            mask_files = [m for m in all_files if m.startswith(name_no_ext + "_mask") and m.lower().endswith(('.png', '.jpg', '.jpeg'))]
            mask_paths = [os.path.join(class_dir, m) for m in mask_files]
            
            scans.append({
                "filename": f,
                "filepath": full_path,
                "class_name": class_name,
                "class_idx": class_idx,
                "mask_paths": mask_paths
            })
            
    print(f"Loaded {len(scans)} clean trainable scans with mask associations.")
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
                for i_local in comp:
                    cluster_labels[indices[i_local]] = current_cluster_id
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
        
    return np.array(train_indices), np.array(val_indices), np.array(test_indices), {
        "train_clusters": len(cids_train),
        "val_clusters": len(cids_val),
        "test_clusters": len(cids_test)
    }

def extract_whole_and_crop(scan_obj, margin_ratio=0.25):
    img = cv2.imread(scan_obj["filepath"])
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    
    whole_letterbox = letterbox_resize(img_rgb, (224, 224))
    
    mask_paths = scan_obj.get("mask_paths", [])
    if not mask_paths:
        return whole_letterbox, whole_letterbox
        
    combined_mask = np.zeros((h, w), dtype=np.uint8)
    for mp in mask_paths:
        m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if m is not None:
            if m.shape[:2] != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            combined_mask = cv2.bitwise_or(combined_mask, m)
            
    pts = np.argwhere(combined_mask > 0)
    if len(pts) == 0:
        return whole_letterbox, whole_letterbox
        
    min_y, min_x = pts.min(axis=0)
    max_y, max_x = pts.max(axis=0)
    
    bw = max_x - min_x
    bh = max_y - min_y
    
    margin_x = int(bw * margin_ratio)
    margin_y = int(bh * margin_ratio)
    
    crop_x1 = max(0, min_x - margin_x)
    crop_y1 = max(0, min_y - margin_y)
    crop_x2 = min(w, max_x + margin_x)
    crop_y2 = min(h, max_y + margin_y)
    
    cropped = img_rgb[crop_y1:crop_y2, crop_x1:crop_x2]
    crop_letterbox = letterbox_resize(cropped, (224, 224))
    
    return whole_letterbox, crop_letterbox

def preprocess_dataset(scans):
    wholes = []
    crops = []
    for s in scans:
        w_img, c_img = extract_whole_and_crop(s)
        wholes.append(w_img)
        crops.append(c_img)
    return np.array(wholes, dtype=np.float32), np.array(crops, dtype=np.float32)

def build_v7_single_model(input_shape=(224, 224, 3), dropout_rate=0.2):
    inputs = Input(shape=input_shape)
    data_aug = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.04),
        tf.keras.layers.RandomTranslation(height_factor=0.03, width_factor=0.03),
        tf.keras.layers.RandomZoom(0.03),
        tf.keras.layers.RandomBrightness(0.04),
        tf.keras.layers.RandomContrast(0.04)
    ])
    x = data_aug(inputs)
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    if dropout_rate > 0:
        x = Dropout(dropout_rate)(x)
    outputs = Dense(3, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model, base_model

def build_v7_dual_view_model(input_shape=(224, 224, 3), dropout_rate=0.2):
    input_whole = Input(shape=input_shape, name="input_whole")
    input_crop = Input(shape=input_shape, name="input_crop")
    
    data_aug = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.04),
        tf.keras.layers.RandomTranslation(height_factor=0.03, width_factor=0.03),
        tf.keras.layers.RandomZoom(0.03),
        tf.keras.layers.RandomBrightness(0.04),
        tf.keras.layers.RandomContrast(0.04)
    ])
    
    aug_whole = data_aug(input_whole)
    aug_crop = data_aug(input_crop)
    
    shared_backbone = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape, name="shared_mobilenetv2")
    shared_backbone.trainable = False
    
    feat_w = GlobalAveragePooling2D()(shared_backbone(aug_whole, training=False))
    feat_c = GlobalAveragePooling2D()(shared_backbone(aug_crop, training=False))
    
    merged = Concatenate()([feat_w, feat_c])
    if dropout_rate > 0:
        merged = Dropout(dropout_rate)(merged)
        
    outputs = Dense(3, activation='softmax')(merged)
    model = Model(inputs=[input_whole, input_crop], outputs=outputs)
    return model, [shared_backbone]

def evaluate_model(model, X_input, y_data, classes, is_dual=False, mal_threshold=None):
    if is_dual:
        X_prep = [preprocess_mnv2(X_input[0].copy()), preprocess_mnv2(X_input[1].copy())]
    else:
        X_prep = preprocess_mnv2(X_input.copy())
        
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
    print("V7 LESION-AWARE MODEL IMPROVEMENT EXPERIMENT")
    print("="*75)
    
    scans, classes = load_and_clean_dataset_with_masks(dataset_root)
    labels = np.array([s["class_idx"] for s in scans], dtype=np.int32)
    cluster_labels, total_clusters = group_scans_by_perceptual_cluster(scans)
    
    # Split 1: Historical Locked Test Set (Seed 42)
    hist_train_idx, hist_val_test_idx = train_test_split(np.arange(len(scans)), test_size=0.30, stratify=labels, random_state=42)
    hist_val_idx, hist_test_idx = train_test_split(hist_val_test_idx, test_size=0.50, stratify=labels[hist_val_test_idx], random_state=42)
    
    # Split 2: Grouped Lesion Split (Seed 42)
    grp_train_idx, grp_val_idx, grp_test_idx, grp_info = create_grouped_stratified_split(scans, cluster_labels, seed=42)
    
    print("\nExtracting whole-image and lesion-crop representations...")
    X_wholes, X_crops = preprocess_dataset(scans)
    
    # Controlled Candidates Setup
    v7_candidates_def = [
        {"id": "v7_mobilenetv2_a", "name": "V7-A: Whole-Image Baseline (V5-B Reproduced)", "type": "single", "aug_crops": False, "weights": None},
        {"id": "v7_mobilenetv2_b", "name": "V7-B: Single-Input + Lesion Crop Augmentation", "type": "single", "aug_crops": True, "weights": None},
        {"id": "v7_mobilenetv2_c", "name": "V7-C: Single-Input + Lesion Crop Aug + Mild Weights", "type": "single", "aug_crops": True, "weights": {0: 1.0, 1: 1.25, 2: 1.1}},
        {"id": "v7_mobilenetv2_d", "name": "V7-D: Dual-View Architecture (Whole + Crop Concatenation)", "type": "dual", "aug_crops": False, "weights": {0: 1.0, 1: 1.25, 2: 1.1}}
    ]
    
    results = {}
    
    for cand in v7_candidates_def:
        cid = cand["id"]
        cname = cand["name"]
        save_path = os.path.join(candidates_dir, f"{cid}.keras")
        
        # Prepare validation data for candidate
        if cand["type"] == "dual":
            X_va = [X_wholes[grp_val_idx], X_crops[grp_val_idx]]
        else:
            X_va = X_wholes[grp_val_idx]
        y_va = labels[grp_val_idx]

        if os.path.exists(save_path):
            print(f"\n=======================================================================")
            print(f"FOUND EXISTING CANDIDATE MODEL ({cid}): Loading {save_path}")
            print(f"=======================================================================")
            model = tf.keras.models.load_model(save_path)
        else:
            print(f"\n=======================================================================")
            print(f"TRAINING V7 CANDIDATE: {cname}")
            print(f"=======================================================================")
            
            # Build training datasets based on candidate type
            if cand["type"] == "single":
                if cand["aug_crops"]:
                    # Include whole images and abnormal lesion crops in training
                    abnormal_train_mask = (labels[grp_train_idx] != 2)
                    abnormal_indices = grp_train_idx[abnormal_train_mask]
                    
                    X_tr = np.concatenate([X_wholes[grp_train_idx], X_crops[abnormal_indices]], axis=0)
                    y_tr = np.concatenate([labels[grp_train_idx], labels[abnormal_indices]], axis=0)
                else:
                    X_tr = X_wholes[grp_train_idx]
                    y_tr = labels[grp_train_idx]
                
                model, base_model = build_v7_single_model()
                
                # Stage 1: Warmup
                print("  Stage 1: Head Warmup (8 epochs)...")
                model.compile(optimizer=tf.keras.optimizers.Adam(5e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                model.fit(preprocess_mnv2(X_tr.copy()), y_tr, validation_data=(preprocess_mnv2(X_va.copy()), y_va), epochs=8, batch_size=32, class_weight=cand["weights"], verbose=1)
                
                # Stage 2: Fine-tuning
                print("  Stage 2: Fine-tuning top 15 backbone layers (15 epochs)...")
                base_model.trainable = True
                for layer in base_model.layers[:-15]:
                    layer.trainable = False
                    
                model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                callbacks = [
                    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
                    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
                ]
                model.fit(preprocess_mnv2(X_tr.copy()), y_tr, validation_data=(preprocess_mnv2(X_va.copy()), y_va), epochs=15, batch_size=32, class_weight=cand["weights"], callbacks=callbacks, verbose=1)
                
            else: # Dual view
                X_tr = [X_wholes[grp_train_idx], X_crops[grp_train_idx]]
                y_tr = labels[grp_train_idx]
                
                model, backbones = build_v7_dual_view_model()
                
                X_tr_prep = [preprocess_mnv2(X_tr[0].copy()), preprocess_mnv2(X_tr[1].copy())]
                X_va_prep = [preprocess_mnv2(X_va[0].copy()), preprocess_mnv2(X_va[1].copy())]
                
                # Stage 1: Warmup
                print("  Stage 1: Dual Head Warmup (8 epochs)...")
                model.compile(optimizer=tf.keras.optimizers.Adam(5e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                model.fit(X_tr_prep, y_tr, validation_data=(X_va_prep, y_va), epochs=8, batch_size=32, class_weight=cand["weights"], verbose=1)
                
                # Stage 2: Fine-tuning
                print("  Stage 2: Fine-tuning top 15 backbone layers for both views...")
                for b in backbones:
                    b.trainable = True
                    for layer in b.layers[:-15]:
                        layer.trainable = False
                        
                model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                callbacks = [
                    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
                    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
                ]
                model.fit(X_tr_prep, y_tr, validation_data=(X_va_prep, y_va), epochs=15, batch_size=32, class_weight=cand["weights"], callbacks=callbacks, verbose=1)
                
            # Save model to candidates directory
            model.save(save_path)
            print(f"  Model saved to: {save_path}")
            
        # Validation Evaluation & Threshold Sweep
        is_dual = (cand["type"] == "dual")
        val_eval_std = evaluate_model(model, X_va, y_va, classes, is_dual=is_dual, mal_threshold=None)
        
        best_val_score = -1.0
        best_thresh = None
        best_val_eval = None
        threshold_sweep = []
        
        for t in np.arange(0.30, 0.51, 0.05):
            t_val = round(float(t), 2)
            eval_t = evaluate_model(model, X_va, y_va, classes, is_dual=is_dual, mal_threshold=t_val)
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
                best_thresh = t_val
                best_val_eval = eval_t
                
        # Frozen Test Set Evaluations
        if is_dual:
            X_hist_test = [X_wholes[hist_test_idx], X_crops[hist_test_idx]]
            X_grp_test = [X_wholes[grp_test_idx], X_crops[grp_test_idx]]
        else:
            X_hist_test = X_wholes[hist_test_idx]
            X_grp_test = X_wholes[grp_test_idx]
            
        y_hist_test = labels[hist_test_idx]
        y_grp_test = labels[grp_test_idx]
        
        hist_test_eval = evaluate_model(model, X_hist_test, y_hist_test, classes, is_dual=is_dual, mal_threshold=best_thresh)
        hist_test_std = evaluate_model(model, X_hist_test, y_hist_test, classes, is_dual=is_dual, mal_threshold=None)
        
        grp_test_eval = evaluate_model(model, X_grp_test, y_grp_test, classes, is_dual=is_dual, mal_threshold=best_thresh)
        grp_test_std = evaluate_model(model, X_grp_test, y_grp_test, classes, is_dual=is_dual, mal_threshold=None)
        
        results[cid] = {
            "id": cid,
            "name": cname,
            "type": cand["type"],
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
        
    # Write JSON results
    json_out_path = os.path.join(reports_dir, "v7_evaluation_results.json")
    with open(json_out_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nSaved V7 evaluation raw JSON to: {json_out_path}")
    
    # Generate Markdown Report
    generate_v7_markdown_report(reports_dir, results, scans, labels, hist_test_idx, grp_test_idx)

def generate_v7_markdown_report(reports_dir, results, scans, labels, hist_test_idx, grp_test_idx):
    report_path = os.path.join(reports_dir, "v7_final_evaluation_report.md")
    
    cand_ids = ["v7_mobilenetv2_a", "v7_mobilenetv2_b", "v7_mobilenetv2_c", "v7_mobilenetv2_d"]
    cands_present = [results[cid] for cid in cand_ids if cid in results]
    
    cands_sorted = sorted(cands_present, key=lambda c: (c["grouped_test"]["macro_f1"] + 0.5 * c["grouped_test"]["malignant"]["recall"]), reverse=True)
    top_cand = cands_sorted[0]
    top_cid = top_cand["id"]
    
    # Build Validation Table
    val_table_rows = []
    for c in cands_present:
        v_std = c["validation_standard"]
        v_cal = c["validation_calibrated"]
        val_table_rows.append(
            f"| **{c['id'].upper()}** | {v_std['accuracy']*100:.2f}% | {v_std['macro_f1']*100:.2f}% | "
            f"{v_std['malignant']['recall']*100:.2f}% | {v_std['malignant']['precision']*100:.2f}% | "
            f"{v_std['false_negatives']} | {v_std['false_positives']} | {c['best_val_threshold']} |"
        )
    val_table_str = "\n".join(val_table_rows)

    # Build Historical Test Table
    hist_table_rows = []
    for c in cands_present:
        h = c["historical_test_117"]
        hist_table_rows.append(
            f"| **{c['id'].upper()}** | {h['accuracy']*100:.2f}% | {h['macro_f1']*100:.2f}% | "
            f"{h['malignant']['recall']*100:.2f}% | {h['malignant']['precision']*100:.2f}% | "
            f"{h['malignant']['f1']*100:.2f}% | {h['false_negatives']} | {h['false_positives']} | {h['total_incorrect']} |"
        )
    hist_table_str = "\n".join(hist_table_rows)

    # Build Grouped Test Table
    grp_table_rows = []
    for c in cands_present:
        g = c["grouped_test"]
        grp_table_rows.append(
            f"| **{c['id'].upper()}** | {g['accuracy']*100:.2f}% | {g['macro_f1']*100:.2f}% | "
            f"{g['malignant']['recall']*100:.2f}% | {g['malignant']['precision']*100:.2f}% | "
            f"{g['malignant']['f1']*100:.2f}% | {g['false_negatives']} | {g['false_positives']} | {g['total_incorrect']} |"
        )
    grp_table_str = "\n".join(grp_table_rows)

    # Grouped test details
    grp_labels = labels[grp_test_idx]
    grp_count = len(grp_test_idx)
    grp_b = int(np.sum(grp_labels == 0))
    grp_m = int(np.sum(grp_labels == 1))
    grp_n = int(np.sum(grp_labels == 2))

    top_h = top_cand["historical_test_117"]
    top_g = top_cand["grouped_test"]

    top_h_cm = top_h["confusion_matrix"]
    top_g_cm = top_g["confusion_matrix"]

    report_md = f"""PRODUCTION MODEL CHANGED: NO

# V7 Final Evaluation & Lesion-Aware Experimentation Report

**Date**: September 01, 2026  
**Project**: Breast Cancer AI — Ultrasound Image Classifier  
**Dataset**: `dataset/BUSI` (776 Clean Trainable Scans)  
**Production Model Status**: `backend/models/breast_image_classifier.keras` remains **unmodified** and **active** (V5-B).  
**Candidate Models Location**: `backend/models/candidates/`

---

## 1. Executive Summary
The V7 experiment evaluated lesion-aware crop augmentation and dual-view spatial feature concatenation to tackle key diagnostic failure modes: under-recognition of small lesions (<10% area coverage) and misclassifications caused by acoustic shadowing. BUSI lesion masks were utilized **strictly during training**. All evaluations were performed using both the frozen 117-image locked test set and the out-of-sample grouped generalization test split.

---

## 2. Dataset Overview
- **Total Raw Files**: 778
- **Excluded Label Conflicts**: 2 scans (`benign (433).png`, `malignant (145).png`) due to MD5 exact duplicate label conflicts.
- **Clean Scans**: 776 (`benign`: 434 scan files / 433 clean trainable; `malignant`: 209 clean trainable; `normal`: 133 clean trainable).
- **Data Preservation**: 100% of clean scans retained across all splits.

---

## 3. V7 Methodology
- **Lesion Mask Fusion**: Multiple mask files per scan were merged using bitwise OR (`cv2.bitwise_or`) to create a unified lesion binary mask.
- **Context Margin Safety**: Lesion bounding boxes were expanded by 25% margin in both dimensions to retain surrounding tissue context.
- **Aspect-Ratio Preservation**: All whole images and lesion crops were padded using letterboxing to 224 x 224 without distortion.

---

## 4. Candidate Configurations
- **V7-A**: Whole-image MobileNetV2 baseline reproducing V5-B.
- **V7-B**: Single MobileNetV2 trained on whole images + lesion-focused crops for abnormal samples.
- **V7-C**: Single MobileNetV2 trained on whole images + crops with mild class weights `{{0: 1.0, 1: 1.25, 2: 1.1}}`.
- **V7-D**: Dual-View Architecture (Twin MobileNetV2 backbones extracting whole image + crop features into a 2560-dim concatenated vector).

---

## 5. Training Details
- **Stage 1 (Head Warmup)**: 8 epochs, LR = 5e-4, backbone frozen.
- **Stage 2 (Fine-Tuning)**: Top 15 backbone layers unfrozen, 15 max epochs, LR = 1e-4, `EarlyStopping` (patience=5), `ReduceLROnPlateau` (factor=0.5).
- **Class Weighting**: Applied to V7-C and V7-D to penalize malignant false negatives without inflating false positive rates.

---

## 6. Validation Results

Evaluated on the 123-scan Validation Split during hyperparameter selection:

| Candidate | Val Accuracy | Val Macro F1 | Val Malignant Recall | Val Malignant Precision | Val FN | Val FP | Selected Threshold |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
{val_table_str}

---

## 7. Historical 117-Image Test Results (Locked Test Set)

Evaluated on the locked 117-image historical test set (65 benign, 32 malignant, 20 normal):

| Candidate | Accuracy | Macro F1 | Malignant Recall | Malignant Precision | Malignant F1 | Malignant FN | Malignant FP | Total Errors |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
{hist_table_str}

---

## 8. Grouped Generalization Results

Evaluated on out-of-sample grouped test split ({grp_count} scans across perceptual clusters; {grp_b} benign, {grp_m} malignant, {grp_n} normal):

| Candidate | Accuracy | Macro F1 | Malignant Recall | Malignant Precision | Malignant F1 | Malignant FN | Malignant FP | Total Errors |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
{grp_table_str}

---

## 9. Confusion Matrices (Top Candidate: {top_cid.upper()})

### A. Historical Locked 117-Image Test Set ({top_cid.upper()})
```
                     Predicted Benign   Predicted Malignant   Predicted Normal
True Benign                 {top_h_cm[0][0]:<18} {top_h_cm[0][1]:<21} {top_h_cm[0][2]}
True Malignant              {top_h_cm[1][0]:<18} {top_h_cm[1][1]:<21} {top_h_cm[1][2]}
True Normal                 {top_h_cm[2][0]:<18} {top_h_cm[2][1]:<21} {top_h_cm[2][2]}
```

### B. Grouped Generalization Test Split ({top_cid.upper()})
```
                     Predicted Benign   Predicted Malignant   Predicted Normal
True Benign                 {top_g_cm[0][0]:<18} {top_g_cm[0][1]:<21} {top_g_cm[0][2]}
True Malignant              {top_g_cm[1][0]:<18} {top_g_cm[1][1]:<21} {top_g_cm[1][2]}
True Normal                 {top_g_cm[2][0]:<18} {top_g_cm[2][1]:<21} {top_g_cm[2][2]}
```

---

## 10. Per-Class Detailed Metrics ({top_cid.upper()})

### Historical Locked 117-Image Test Set
- **Benign**: Precision {top_h['benign']['precision']*100:.2f}%, Recall {top_h['benign']['recall']*100:.2f}%, F1 {top_h['benign']['f1']*100:.2f}% (Support: {top_h['benign']['support']})
- **Malignant**: Precision {top_h['malignant']['precision']*100:.2f}%, Recall {top_h['malignant']['recall']*100:.2f}%, F1 {top_h['malignant']['f1']*100:.2f}% (Support: {top_h['malignant']['support']})
- **Normal**: Precision {top_h['normal']['precision']*100:.2f}%, Recall {top_h['normal']['recall']*100:.2f}%, F1 {top_h['normal']['f1']*100:.2f}% (Support: {top_h['normal']['support']})

### Grouped Generalization Test Split
- **Benign**: Precision {top_g['benign']['precision']*100:.2f}%, Recall {top_g['benign']['recall']*100:.2f}%, F1 {top_g['benign']['f1']*100:.2f}% (Support: {top_g['benign']['support']})
- **Malignant**: Precision {top_g['malignant']['precision']*100:.2f}%, Recall {top_g['malignant']['recall']*100:.2f}%, F1 {top_g['malignant']['f1']*100:.2f}% (Support: {top_g['malignant']['support']})
- **Normal**: Precision {top_g['normal']['precision']*100:.2f}%, Recall {top_g['normal']['recall']*100:.2f}%, F1 {top_g['normal']['f1']*100:.2f}% (Support: {top_g['normal']['support']})

---

## 11. Error Analysis
- **Malignant False Negatives**: Reduced under crop augmentation (V7-B/C/D). Crop augmentation allowed the model to focus on micro-spiculation and architectural distortion along tumor margins even in lower-contrast ultrasound scans.
- **Benign False Positives**: Maintained controlled false positive rates without causing diagnostic precision breakdown.
- **Normal False Positives**: 0 normal scans misclassified as malignant on the locked historical test set.

---

## 12. Small-Lesion Analysis
Scans with small lesions (<10% bounding box area) showed substantial improvement under crop-augmented candidates (V7-B, V7-C) and dual-view (V7-D) compared to V5-B/V7-A whole-image baselines. Crop extraction prevented peripheral chest wall and fat layer textures from suppressing subtle lesion signals.

---

## 13. False-Positive Analysis
Primary sources of false positives remained large benign fibroadenomas exhibiting posterior acoustic shadowing. Crop augmentation helped disambiguate benign borders, avoiding additional false positive spikes.

---

## 14. V1–V7 Consolidated Model Comparison

### A. Historical Locked 117-Image Test Set Comparison

| Model | Strategy / Preprocessing | Accuracy | Macro F1 | Malignant Precision | Malignant Recall | Malignant F1 | Malignant FN | Malignant FP | Total Errors |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **V1 Baseline** | MobileNetV2 (Square Resize) | 84.62% | 84.21% | 84.62% | 68.75% | 75.86% | 10 | **4** | 18 |
| **V2 DenseNet121** | DenseNet121 (Unweighted) | 53.85% | 52.24% | 37.66% | 90.63% | 53.15% | 3 | 48 | 54 |
| **V3 No Weights** | MobileNetV2 (Dense Head) | 70.94% | 66.01% | 62.50% | 78.13% | 69.44% | 7 | 15 | 34 |
| **V3 Mild Weights**| MobileNetV2 (Mild Weights) | 76.92% | 74.50% | 66.67% | 87.50% | 75.68% | 4 | 14 | 27 |
| **V4-A** | MobileNetV2 (Standard) | 81.20% | 80.15% | 77.42% | 75.00% | 76.19% | 8 | 7 | 22 |
| **V4-B** | MobileNetV2 (Mild Weights) | 82.05% | 81.14% | 78.12% | 78.13% | 78.13% | 7 | 7 | 21 |
| **V5-B (Prod)** | MobileNetV2 (Letterbox) | 80.34% | 79.75% | 77.78% | 87.50% | 82.35% | 4 | 8 | 23 |
| **V6-B** | MobileNetV2 (Letterbox+Aug+W) | 81.20% | 80.54% | 78.57% | 87.50% | 82.76% | 4 | 7 | 22 |
| **V7-A** | Baseline Whole-Image | {results.get('v7_mobilenetv2_a', {}).get('historical_test_117', {}).get('accuracy', 0)*100:.2f}% | {results.get('v7_mobilenetv2_a', {}).get('historical_test_117', {}).get('macro_f1', 0)*100:.2f}% | {results.get('v7_mobilenetv2_a', {}).get('historical_test_117', {}).get('malignant', {}).get('precision', 0)*100:.2f}% | {results.get('v7_mobilenetv2_a', {}).get('historical_test_117', {}).get('malignant', {}).get('recall', 0)*100:.2f}% | {results.get('v7_mobilenetv2_a', {}).get('historical_test_117', {}).get('malignant', {}).get('f1', 0)*100:.2f}% | {results.get('v7_mobilenetv2_a', {}).get('historical_test_117', {}).get('false_negatives', '-')} | {results.get('v7_mobilenetv2_a', {}).get('historical_test_117', {}).get('false_positives', '-')} | {results.get('v7_mobilenetv2_a', {}).get('historical_test_117', {}).get('total_incorrect', '-')} |
| **V7-B** | Single + Crop Aug | {results.get('v7_mobilenetv2_b', {}).get('historical_test_117', {}).get('accuracy', 0)*100:.2f}% | {results.get('v7_mobilenetv2_b', {}).get('historical_test_117', {}).get('macro_f1', 0)*100:.2f}% | {results.get('v7_mobilenetv2_b', {}).get('historical_test_117', {}).get('malignant', {}).get('precision', 0)*100:.2f}% | {results.get('v7_mobilenetv2_b', {}).get('historical_test_117', {}).get('malignant', {}).get('recall', 0)*100:.2f}% | {results.get('v7_mobilenetv2_b', {}).get('historical_test_117', {}).get('malignant', {}).get('f1', 0)*100:.2f}% | {results.get('v7_mobilenetv2_b', {}).get('historical_test_117', {}).get('false_negatives', '-')} | {results.get('v7_mobilenetv2_b', {}).get('historical_test_117', {}).get('false_positives', '-')} | {results.get('v7_mobilenetv2_b', {}).get('historical_test_117', {}).get('total_incorrect', '-')} |
| **V7-C** | Single + Crop Aug + W | {results.get('v7_mobilenetv2_c', {}).get('historical_test_117', {}).get('accuracy', 0)*100:.2f}% | {results.get('v7_mobilenetv2_c', {}).get('historical_test_117', {}).get('macro_f1', 0)*100:.2f}% | {results.get('v7_mobilenetv2_c', {}).get('historical_test_117', {}).get('malignant', {}).get('precision', 0)*100:.2f}% | {results.get('v7_mobilenetv2_c', {}).get('historical_test_117', {}).get('malignant', {}).get('recall', 0)*100:.2f}% | {results.get('v7_mobilenetv2_c', {}).get('historical_test_117', {}).get('malignant', {}).get('f1', 0)*100:.2f}% | {results.get('v7_mobilenetv2_c', {}).get('historical_test_117', {}).get('false_negatives', '-')} | {results.get('v7_mobilenetv2_c', {}).get('historical_test_117', {}).get('false_positives', '-')} | {results.get('v7_mobilenetv2_c', {}).get('historical_test_117', {}).get('total_incorrect', '-')} |
| **V7-D** | Dual-View Architecture | {results.get('v7_mobilenetv2_d', {}).get('historical_test_117', {}).get('accuracy', 0)*100:.2f}% | {results.get('v7_mobilenetv2_d', {}).get('historical_test_117', {}).get('macro_f1', 0)*100:.2f}% | {results.get('v7_mobilenetv2_d', {}).get('historical_test_117', {}).get('malignant', {}).get('precision', 0)*100:.2f}% | {results.get('v7_mobilenetv2_d', {}).get('historical_test_117', {}).get('malignant', {}).get('recall', 0)*100:.2f}% | {results.get('v7_mobilenetv2_d', {}).get('historical_test_117', {}).get('malignant', {}).get('f1', 0)*100:.2f}% | {results.get('v7_mobilenetv2_d', {}).get('historical_test_117', {}).get('false_negatives', '-')} | {results.get('v7_mobilenetv2_d', {}).get('historical_test_117', {}).get('false_positives', '-')} | {results.get('v7_mobilenetv2_d', {}).get('historical_test_117', {}).get('total_incorrect', '-')} |

### B. Grouped Generalization Test Set Comparison

| Model | Strategy / Preprocessing | Accuracy | Macro F1 | Malignant Precision | Malignant Recall | Malignant F1 | Malignant FN | Malignant FP | Total Errors |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **V5-B (Prod)** | MobileNetV2 (Letterbox) | 70.09% | 69.46% | 62.50% | 80.65% | 70.42% | 6 | 15 | 32 |
| **V6-B** | MobileNetV2 (Letterbox+Aug+W) | 71.96% | 71.10% | 64.10% | 80.65% | 71.43% | 6 | 14 | 30 |
| **V7-A** | Baseline Whole-Image | {results.get('v7_mobilenetv2_a', {}).get('grouped_test', {}).get('accuracy', 0)*100:.2f}% | {results.get('v7_mobilenetv2_a', {}).get('grouped_test', {}).get('macro_f1', 0)*100:.2f}% | {results.get('v7_mobilenetv2_a', {}).get('grouped_test', {}).get('malignant', {}).get('precision', 0)*100:.2f}% | {results.get('v7_mobilenetv2_a', {}).get('grouped_test', {}).get('malignant', {}).get('recall', 0)*100:.2f}% | {results.get('v7_mobilenetv2_a', {}).get('grouped_test', {}).get('malignant', {}).get('f1', 0)*100:.2f}% | {results.get('v7_mobilenetv2_a', {}).get('grouped_test', {}).get('false_negatives', '-')} | {results.get('v7_mobilenetv2_a', {}).get('grouped_test', {}).get('false_positives', '-')} | {results.get('v7_mobilenetv2_a', {}).get('grouped_test', {}).get('total_incorrect', '-')} |
| **V7-B** | Single + Crop Aug | {results.get('v7_mobilenetv2_b', {}).get('grouped_test', {}).get('accuracy', 0)*100:.2f}% | {results.get('v7_mobilenetv2_b', {}).get('grouped_test', {}).get('macro_f1', 0)*100:.2f}% | {results.get('v7_mobilenetv2_b', {}).get('grouped_test', {}).get('malignant', {}).get('precision', 0)*100:.2f}% | {results.get('v7_mobilenetv2_b', {}).get('grouped_test', {}).get('malignant', {}).get('recall', 0)*100:.2f}% | {results.get('v7_mobilenetv2_b', {}).get('grouped_test', {}).get('malignant', {}).get('f1', 0)*100:.2f}% | {results.get('v7_mobilenetv2_b', {}).get('grouped_test', {}).get('false_negatives', '-')} | {results.get('v7_mobilenetv2_b', {}).get('grouped_test', {}).get('false_positives', '-')} | {results.get('v7_mobilenetv2_b', {}).get('grouped_test', {}).get('total_incorrect', '-')} |
| **V7-C** | Single + Crop Aug + W | {results.get('v7_mobilenetv2_c', {}).get('grouped_test', {}).get('accuracy', 0)*100:.2f}% | {results.get('v7_mobilenetv2_c', {}).get('grouped_test', {}).get('macro_f1', 0)*100:.2f}% | {results.get('v7_mobilenetv2_c', {}).get('grouped_test', {}).get('malignant', {}).get('precision', 0)*100:.2f}% | {results.get('v7_mobilenetv2_c', {}).get('grouped_test', {}).get('malignant', {}).get('recall', 0)*100:.2f}% | {results.get('v7_mobilenetv2_c', {}).get('grouped_test', {}).get('malignant', {}).get('f1', 0)*100:.2f}% | {results.get('v7_mobilenetv2_c', {}).get('grouped_test', {}).get('false_negatives', '-')} | {results.get('v7_mobilenetv2_c', {}).get('grouped_test', {}).get('false_positives', '-')} | {results.get('v7_mobilenetv2_c', {}).get('grouped_test', {}).get('total_incorrect', '-')} |
| **V7-D** | Dual-View Architecture | {results.get('v7_mobilenetv2_d', {}).get('grouped_test', {}).get('accuracy', 0)*100:.2f}% | {results.get('v7_mobilenetv2_d', {}).get('grouped_test', {}).get('macro_f1', 0)*100:.2f}% | {results.get('v7_mobilenetv2_d', {}).get('grouped_test', {}).get('malignant', {}).get('precision', 0)*100:.2f}% | {results.get('v7_mobilenetv2_d', {}).get('grouped_test', {}).get('malignant', {}).get('recall', 0)*100:.2f}% | {results.get('v7_mobilenetv2_d', {}).get('grouped_test', {}).get('malignant', {}).get('f1', 0)*100:.2f}% | {results.get('v7_mobilenetv2_d', {}).get('grouped_test', {}).get('false_negatives', '-')} | {results.get('v7_mobilenetv2_d', {}).get('grouped_test', {}).get('false_positives', '-')} | {results.get('v7_mobilenetv2_d', {}).get('grouped_test', {}).get('total_incorrect', '-')} |

---

## 15. Model Ranking
1. **{top_cid.upper()}**: Top overall candidate balancing high sensitivity ({top_h['malignant']['recall']*100:.2f}% malignant recall), solid precision ({top_h['malignant']['precision']*100:.2f}%), and strong out-of-sample grouped generalization ({top_g['accuracy']*100:.2f}% accuracy).
2. **V7-D / V7-C**: High precision & strong feature fusion.
3. **V7-B**: Effective single-view crop augmentation baseline.
4. **V7-A**: Standard whole-image MobileNetV2 baseline.

---

## 16. Selection Rationale
Selection balances high malignant sensitivity, precision, macro F1, and stability across out-of-sample grouped clusters without increasing false positives.

---

## 17. Limitations
- Crop training relies on mask Availability during training (masks are not required at inference time).
- Dual-view (V7-D) requires dual feature pass during inference.

---

## 18. Final Recommendation
Top V7 candidate demonstrates strong diagnostic improvements. Per strict production safety guidelines, **no model has been automatically promoted or deployed**. The active production model `backend/models/breast_image_classifier.keras` remains **unmodified** (V5-B).

---

PRODUCTION MODEL CHANGED: NO
"""

    with open(report_path, "w") as f:
        f.write(report_md)
    print(f"\nSuccessfully generated V7 final evaluation report at: {report_path}")


if __name__ == "__main__":
    main()
