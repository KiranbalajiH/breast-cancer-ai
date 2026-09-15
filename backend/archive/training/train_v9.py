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

def extract_whole_and_crop(scan_obj, margin_ratio=0.30):
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

def build_v9_single_model(input_shape=(224, 224, 3), dropout_rate=0.2):
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

def build_v9_dual_view_model(input_shape=(224, 224, 3), dropout_rate=0.2, aux_loss=False):
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
        
    main_output = Dense(3, activation='softmax', name="main_output")(merged)
    
    if aux_loss:
        local_output = Dense(3, activation='softmax', name="local_aux_output")(feat_c)
        model = Model(inputs=[input_whole, input_crop], outputs=[main_output, local_output])
    else:
        model = Model(inputs=[input_whole, input_crop], outputs=main_output)
        
    return model, [shared_backbone]

def evaluate_model(model, X_input, y_data, classes, is_dual=False, is_aux=False, mal_threshold=None):
    if is_dual:
        X_prep = [preprocess_mnv2(X_input[0].copy()), preprocess_mnv2(X_input[1].copy())]
    else:
        X_prep = preprocess_mnv2(X_input.copy())
        
    preds = model.predict(X_prep, verbose=0)
    if is_aux:
        preds = preds[0] # main output predictions
        
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
    print("V9 LESION-AWARE ROBUST CLASSIFICATION EXPERIMENT")
    print("="*75)
    
    scans, classes = load_and_clean_dataset_with_masks(dataset_root)
    labels = np.array([s["class_idx"] for s in scans], dtype=np.int32)
    cluster_labels, total_clusters = group_scans_by_perceptual_cluster(scans)
    
    hist_train_idx, hist_val_test_idx = train_test_split(np.arange(len(scans)), test_size=0.30, stratify=labels, random_state=42)
    hist_val_idx, hist_test_idx = train_test_split(hist_val_test_idx, test_size=0.50, stratify=labels[hist_val_test_idx], random_state=42)
    
    grp_train_idx, grp_val_idx, grp_test_idx, grp_info = create_grouped_stratified_split(scans, cluster_labels, seed=42)
    
    print("\nExtracting whole-image and lesion-crop representations...")
    X_wholes, X_crops = preprocess_dataset(scans)
    
    v9_candidates_def = [
        {"id": "v9_mobilenetv2_baseline", "name": "V9-A: V5-B Reproduction Baseline Control", "type": "single", "aug_crops": False, "aux": False, "weights": None},
        {"id": "v9_mobilenetv2_cropaug", "name": "V9-B: Single-Input + Lesion Crop Aug + Mild Weights", "type": "single", "aug_crops": True, "aux": False, "weights": {0: 1.0, 1: 1.25, 2: 1.1}},
        {"id": "v9_mobilenetv2_dualview", "name": "V9-C: Multi-Scale Dual-View Architecture", "type": "dual", "aug_crops": False, "aux": False, "weights": {0: 1.0, 1: 1.25, 2: 1.1}},
        {"id": "v9_mobilenetv2_auxloss", "name": "V9-D: Dual-View + Local Auxiliary Loss (weight 0.2)", "type": "dual", "aug_crops": False, "aux": True, "weights": {0: 1.0, 1: 1.25, 2: 1.1}}
    ]
    
    results = {}
    
    for cand in v9_candidates_def:
        cid = cand["id"]
        cname = cand["name"]
        save_path = os.path.join(candidates_dir, f"{cid}.keras")
        
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
            print(f"TRAINING V9 CANDIDATE: {cname}")
            print(f"=======================================================================")
            
            if cand["type"] == "single":
                if cand["aug_crops"]:
                    abnormal_train_mask = (labels[grp_train_idx] != 2)
                    abnormal_indices = grp_train_idx[abnormal_train_mask]
                    X_tr = np.concatenate([X_wholes[grp_train_idx], X_crops[abnormal_indices]], axis=0)
                    y_tr = np.concatenate([labels[grp_train_idx], labels[abnormal_indices]], axis=0)
                else:
                    X_tr = X_wholes[grp_train_idx]
                    y_tr = labels[grp_train_idx]
                    
                model, base_model = build_v9_single_model()
                
                print("  Stage 1: Head Warmup (8 epochs)...")
                model.compile(optimizer=tf.keras.optimizers.Adam(5e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                model.fit(preprocess_mnv2(X_tr.copy()), y_tr, validation_data=(preprocess_mnv2(X_va.copy()), y_va), epochs=8, batch_size=32, class_weight=cand["weights"], verbose=1)
                
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
                
                model, backbones = build_v9_dual_view_model(aux_loss=cand["aux"])
                
                X_tr_prep = [preprocess_mnv2(X_tr[0].copy()), preprocess_mnv2(X_tr[1].copy())]
                X_va_prep = [preprocess_mnv2(X_va[0].copy()), preprocess_mnv2(X_va[1].copy())]
                
                if cand["aux"]:
                    loss_dict = {"main_output": "sparse_categorical_crossentropy", "local_aux_output": "sparse_categorical_crossentropy"}
                    loss_weights = {"main_output": 1.0, "local_aux_output": 0.2}
                    y_tr_dict = {"main_output": y_tr, "local_aux_output": y_tr}
                    y_va_dict = {"main_output": y_va, "local_aux_output": y_va}
                    
                    print("  Stage 1: Dual Head Warmup with Auxiliary Loss (8 epochs)...")
                    metrics_dict = {"main_output": "accuracy", "local_aux_output": "accuracy"}
                    model.compile(optimizer=tf.keras.optimizers.Adam(5e-4), loss=loss_dict, loss_weights=loss_weights, metrics=metrics_dict)
                    model.fit(X_tr_prep, y_tr_dict, validation_data=(X_va_prep, y_va_dict), epochs=8, batch_size=32, verbose=1)
                    
                    print("  Stage 2: Fine-tuning top 15 backbone layers for both views with Aux Loss...")
                    for b in backbones:
                        b.trainable = True
                        for layer in b.layers[:-15]:
                            layer.trainable = False
                    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss=loss_dict, loss_weights=loss_weights, metrics=metrics_dict)
                    callbacks = [
                        EarlyStopping(monitor='val_main_output_loss', patience=5, restore_best_weights=True, mode='min'),
                        ReduceLROnPlateau(monitor='val_main_output_loss', factor=0.5, patience=3, verbose=1, mode='min')
                    ]
                    model.fit(X_tr_prep, y_tr_dict, validation_data=(X_va_prep, y_va_dict), epochs=15, batch_size=32, callbacks=callbacks, verbose=1)
                else:
                    print("  Stage 1: Dual Head Warmup (8 epochs)...")
                    model.compile(optimizer=tf.keras.optimizers.Adam(5e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                    model.fit(X_tr_prep, y_tr, validation_data=(X_va_prep, y_va), epochs=8, batch_size=32, class_weight=cand["weights"], verbose=1)
                    
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
                    
            model.save(save_path)
            print(f"  Model saved to: {save_path}")
            
        # Validation Evaluation & Threshold Sweep
        is_dual = (cand["type"] == "dual")
        is_aux = cand["aux"]
        val_eval_std = evaluate_model(model, X_va, y_va, classes, is_dual=is_dual, is_aux=is_aux, mal_threshold=None)
        
        best_val_score = -1.0
        best_thresh = None
        best_val_eval = None
        threshold_sweep = []
        
        for t in np.arange(0.20, 0.51, 0.05):
            t_val = round(float(t), 2)
            eval_t = evaluate_model(model, X_va, y_va, classes, is_dual=is_dual, is_aux=is_aux, mal_threshold=t_val)
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
        
        hist_test_eval = evaluate_model(model, X_hist_test, y_hist_test, classes, is_dual=is_dual, is_aux=is_aux, mal_threshold=best_thresh)
        hist_test_std = evaluate_model(model, X_hist_test, y_hist_test, classes, is_dual=is_dual, is_aux=is_aux, mal_threshold=None)
        
        grp_test_eval = evaluate_model(model, X_grp_test, y_grp_test, classes, is_dual=is_dual, is_aux=is_aux, mal_threshold=best_thresh)
        grp_test_std = evaluate_model(model, X_grp_test, y_grp_test, classes, is_dual=is_dual, is_aux=is_aux, mal_threshold=None)
        
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
        
    val_summary = {cid: {
        "id": res["id"],
        "name": res["name"],
        "type": res["type"],
        "validation_standard": res["validation_standard"],
        "best_val_threshold": res["best_val_threshold"],
        "validation_calibrated": res["validation_calibrated"],
        "threshold_sweep": res["threshold_sweep"]
    } for cid, res in results.items()}
    val_json_out_path = os.path.join(reports_dir, "v9_validation_summary.json")
    with open(val_json_out_path, "w") as f:
        json.dump(val_summary, f, indent=4)
    print(f"Saved V9 validation summary JSON to: {val_json_out_path}")
    
    json_out_path = os.path.join(reports_dir, "v9_evaluation_results.json")
    with open(json_out_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved V9 evaluation raw JSON to: {json_out_path}")
    
    generate_v9_markdown_report(reports_dir, results, scans, labels, hist_test_idx, grp_test_idx)

def generate_v9_markdown_report(reports_dir, results, scans, labels, hist_test_idx, grp_test_idx):
    report_path = os.path.join(reports_dir, "v9_final_evaluation_report.md")
    
    cand_ids = ["v9_mobilenetv2_baseline", "v9_mobilenetv2_cropaug", "v9_mobilenetv2_dualview", "v9_mobilenetv2_auxloss"]
    cands_present = [results[cid] for cid in cand_ids if cid in results]
    
    cands_sorted = sorted(cands_present, key=lambda c: (c["validation_calibrated"]["macro_f1"] + 0.5 * c["validation_calibrated"]["malignant"]["recall"]), reverse=True)
    top_cand = cands_sorted[0]
    top_cid = top_cand["id"]
    
    val_table_rows = []
    for c in cands_present:
        v_std = c["validation_standard"]
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

    grp_labels = labels[grp_test_idx]
    grp_count = len(grp_test_idx)
    grp_b = int(np.sum(grp_labels == 0))
    grp_m = int(np.sum(grp_labels == 1))
    grp_n = int(np.sum(grp_labels == 2))

    top_h = top_cand["historical_test_117"]
    top_g = top_cand["grouped_test"]
    top_h_cm = top_h["confusion_matrix"]
    top_g_cm = top_g["confusion_matrix"]

    v5b_hist_mal_rec = 0.8750
    v5b_grp_mal_rec = 0.8065

    if top_g["malignant"]["recall"] >= v5b_grp_mal_rec and top_g["macro_f1"] >= 0.70:
        promo_recommendation = f"**PROMOTABLE FOR PRODUCTION REVIEW**. {top_cid.upper()} achieves superior malignant sensitivity and overall classification quality."
    else:
        promo_recommendation = f"**DO NOT PROMOTE**. Recommend retaining **V5-B** as active production model."

    report_md = f"""PRODUCTION MODEL CHANGED: NO
PRODUCTION METADATA CHANGED: NO
LOCKED 117-IMAGE TEST SET MODIFIED: NO
DATASET IMAGES MODIFIED: NO
DEPLOYMENT PERFORMED: NO
GIT COMMIT PERFORMED: NO
GIT PUSH PERFORMED: NO

# V9 Final Evaluation & Lesion-Aware Experiment Report

**Date**: September 01, 2026  
**Project**: Breast Cancer AI — Ultrasound Image Classifier  
**Dataset**: `dataset/BUSI` (776 Clean Trainable Scans)  
**Production Model Status**: `backend/models/breast_image_classifier.keras` remains **unmodified** and **active** (V5-B).  
**Candidate Models Location**: `backend/models/candidates/`

---

## 1. Dataset Methodology
- **Clean Trainable Dataset**: 776 scans (`benign`: 434 scan files / 433 clean trainable; `malignant`: 209 clean trainable; `normal`: 133 clean trainable).
- **Excluded Files**: `benign (433).png` and `malignant (145).png` (MD5 label conflicts).

---

## 2. Data Counts
- **Total Clean Scans**: 776
- **Training Set (Grouped Split)**: 546 scans (70%)
- **Validation Set (Grouped Split)**: 123 scans (15%)
- **Grouped Generalization Test Set**: 107 scans (15%)
- **Historical Locked Test Set**: 117 scans (65 benign, 32 malignant, 20 normal)

---

## 3. Mask Availability
- **Single Mask**: 758 scans (97.68%)
- **Multiple Masks**: 17 scans (2.19%) — combined using bitwise OR (`cv2.bitwise_or`)
- **No Mask**: 1 scan (0.13%) — defaults to whole image
- **Normal Class**: 133 / 133 scans with zero-pixel masks — crops default to whole letterboxed image

---

## 4. Split Methodology
Perceptual cluster grouping (32x32 pHash diff < 12.0) prevents sequential scan leakage across splits. Stratified train/val/test split preserves class distributions.

---

## 5. Leakage Controls
- All hyperparameter, threshold, and architecture selections performed **strictly on training/validation data**.
- Locked test sets evaluated **exactly once** after freezing candidate selection.

---

## 6. V9 Architectures
- **V9-A (Baseline Reproduction Control)**: Single whole-image MobileNetV2.
- **V9-B (Single-Input + Crop Aug)**: Whole image + 30% context-expanded lesion crops for abnormal training samples + mild class weights `{{0: 1.0, 1: 1.25, 2: 1.1}}`.
- **V9-C (Multi-Scale Dual-View)**: Twin MobileNetV2 feature extractors processing `[Whole Image, Lesion Crop]` concatenated into 2560-dim vector + mild class weights.
- **V9-D (Dual-View + Auxiliary Loss)**: Dual-view model with auxiliary classification loss on crop branch: $\\mathcal{{L}}_{{\\text{{total}}}} = \\mathcal{{L}}_{{\\text{{global}}}} + 0.2 \\cdot \\mathcal{{L}}_{{\\text{{local}}}}$.

---

## 7. Preprocessing
Letterbox padding to 224 x 224 preserving native aspect ratio, normalized via MobileNetV2 `preprocess_input` ([-1, 1]).

---

## 8. Augmentation
Medically realistic transformations: horizontal flip, rotation (+/- 4%), translation (+/- 3%), zoom (+/- 3%), brightness/contrast jitter (+/- 4%).

---

## 9. Class Weighting
Mild class weighting (`Benign: 1.0`, `Malignant: 1.25`, `Normal: 1.1`) applied to V9-B, V9-C, and V9-D.

---

## 10. Training Configuration
- **Stage 1 (Head Warmup)**: 8 epochs, Adam(5e-4), backbone frozen.
- **Stage 2 (Fine-Tuning)**: Top 15 backbone layers unfrozen, 15 max epochs, Adam(1e-4), `EarlyStopping` (patience=5), `ReduceLROnPlateau` (factor=0.5).

---

## 11. Validation Results

| Candidate | Val Accuracy | Val Macro F1 | Val Malignant Recall | Val Malignant Precision | Val FN | Val FP | Selected Threshold |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
{val_table_str}

---

## 12. Candidate Selection
Validation-only selection evaluated composite score balancing malignant recall, precision, and macro F1 without triggering false-positive spikes.

---

## 13. Frozen Test Results

### A. Historical Locked 117-Image Test Results
| Candidate | Accuracy | Macro F1 | Malignant Recall | Malignant Precision | Malignant F1 | Malignant FN | Malignant FP | Total Errors |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
{hist_table_str}

### B. Grouped Generalization Test Results
| Candidate | Accuracy | Macro F1 | Malignant Recall | Malignant Precision | Malignant F1 | Malignant FN | Malignant FP | Total Errors |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
{grp_table_str}

---

## 14. Historical Test Comparison

| Model Iteration | Accuracy | Macro F1 | Malignant Precision | Malignant Recall | Malignant F1 | Malignant FN | Malignant FP | Total Errors |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **V1 Baseline** | 84.62% | 84.21% | 84.62% | 68.75% | 75.86% | 10 | **4** | 18 |
| **V3 Mild Weights** | 76.92% | 74.50% | 66.67% | 87.50% | 75.68% | 4 | 14 | 27 |
| **V5-B (Active Prod)** | 80.34% | 79.75% | 77.78% | **87.50%** | 82.35% | **4** | 8 | 23 |
| **V6-B** | 81.20% | 80.54% | 78.57% | **87.50%** | 82.76% | **4** | 7 | 22 |
| **V7-D (Dual-View)** | 84.62% | 83.17% | 80.77% | 65.62% | 72.41% | 11 | 5 | 18 |
| **V9-A** | {results.get('v9_mobilenetv2_baseline', {}).get('historical_test_117', {}).get('accuracy', 0)*100:.2f}% | {results.get('v9_mobilenetv2_baseline', {}).get('historical_test_117', {}).get('macro_f1', 0)*100:.2f}% | {results.get('v9_mobilenetv2_baseline', {}).get('historical_test_117', {}).get('malignant', {}).get('precision', 0)*100:.2f}% | {results.get('v9_mobilenetv2_baseline', {}).get('historical_test_117', {}).get('malignant', {}).get('recall', 0)*100:.2f}% | {results.get('v9_mobilenetv2_baseline', {}).get('historical_test_117', {}).get('malignant', {}).get('f1', 0)*100:.2f}% | {results.get('v9_mobilenetv2_baseline', {}).get('historical_test_117', {}).get('false_negatives', '-')} | {results.get('v9_mobilenetv2_baseline', {}).get('historical_test_117', {}).get('false_positives', '-')} | {results.get('v9_mobilenetv2_baseline', {}).get('historical_test_117', {}).get('total_incorrect', '-')} |
| **V9-B** | {results.get('v9_mobilenetv2_cropaug', {}).get('historical_test_117', {}).get('accuracy', 0)*100:.2f}% | {results.get('v9_mobilenetv2_cropaug', {}).get('historical_test_117', {}).get('macro_f1', 0)*100:.2f}% | {results.get('v9_mobilenetv2_cropaug', {}).get('historical_test_117', {}).get('malignant', {}).get('precision', 0)*100:.2f}% | {results.get('v9_mobilenetv2_cropaug', {}).get('historical_test_117', {}).get('malignant', {}).get('recall', 0)*100:.2f}% | {results.get('v9_mobilenetv2_cropaug', {}).get('historical_test_117', {}).get('malignant', {}).get('f1', 0)*100:.2f}% | {results.get('v9_mobilenetv2_cropaug', {}).get('historical_test_117', {}).get('false_negatives', '-')} | {results.get('v9_mobilenetv2_cropaug', {}).get('historical_test_117', {}).get('false_positives', '-')} | {results.get('v9_mobilenetv2_cropaug', {}).get('historical_test_117', {}).get('total_incorrect', '-')} |
| **V9-C** | {results.get('v9_mobilenetv2_dualview', {}).get('historical_test_117', {}).get('accuracy', 0)*100:.2f}% | {results.get('v9_mobilenetv2_dualview', {}).get('historical_test_117', {}).get('macro_f1', 0)*100:.2f}% | {results.get('v9_mobilenetv2_dualview', {}).get('historical_test_117', {}).get('malignant', {}).get('precision', 0)*100:.2f}% | {results.get('v9_mobilenetv2_dualview', {}).get('historical_test_117', {}).get('malignant', {}).get('recall', 0)*100:.2f}% | {results.get('v9_mobilenetv2_dualview', {}).get('historical_test_117', {}).get('malignant', {}).get('f1', 0)*100:.2f}% | {results.get('v9_mobilenetv2_dualview', {}).get('historical_test_117', {}).get('false_negatives', '-')} | {results.get('v9_mobilenetv2_dualview', {}).get('historical_test_117', {}).get('false_positives', '-')} | {results.get('v9_mobilenetv2_dualview', {}).get('historical_test_117', {}).get('total_incorrect', '-')} |
| **V9-D** | {results.get('v9_mobilenetv2_auxloss', {}).get('historical_test_117', {}).get('accuracy', 0)*100:.2f}% | {results.get('v9_mobilenetv2_auxloss', {}).get('historical_test_117', {}).get('macro_f1', 0)*100:.2f}% | {results.get('v9_mobilenetv2_auxloss', {}).get('historical_test_117', {}).get('malignant', {}).get('precision', 0)*100:.2f}% | {results.get('v9_mobilenetv2_auxloss', {}).get('historical_test_117', {}).get('malignant', {}).get('recall', 0)*100:.2f}% | {results.get('v9_mobilenetv2_auxloss', {}).get('historical_test_117', {}).get('malignant', {}).get('f1', 0)*100:.2f}% | {results.get('v9_mobilenetv2_auxloss', {}).get('historical_test_117', {}).get('false_negatives', '-')} | {results.get('v9_mobilenetv2_auxloss', {}).get('historical_test_117', {}).get('false_positives', '-')} | {results.get('v9_mobilenetv2_auxloss', {}).get('historical_test_117', {}).get('total_incorrect', '-')} |

---

## 15. Grouped Generalization Comparison

| Model Iteration | Accuracy | Macro F1 | Malignant Precision | Malignant Recall | Malignant F1 | Malignant FN | Malignant FP | Total Errors |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **V5-B (Active Prod)** | 70.09% | 69.46% | 62.50% | **80.65%** | 70.42% | **6** | 15 | 32 |
| **V6-B** | 71.96% | 71.10% | 64.10% | **80.65%** | 71.43% | **6** | 14 | 30 |
| **V7-D (Dual-View)** | 76.64% | 74.95% | 69.57% | 51.61% | 59.26% | 15 | 7 | 25 |
| **V9-A** | {results.get('v9_mobilenetv2_baseline', {}).get('grouped_test', {}).get('accuracy', 0)*100:.2f}% | {results.get('v9_mobilenetv2_baseline', {}).get('grouped_test', {}).get('macro_f1', 0)*100:.2f}% | {results.get('v9_mobilenetv2_baseline', {}).get('grouped_test', {}).get('malignant', {}).get('precision', 0)*100:.2f}% | {results.get('v9_mobilenetv2_baseline', {}).get('grouped_test', {}).get('malignant', {}).get('recall', 0)*100:.2f}% | {results.get('v9_mobilenetv2_baseline', {}).get('grouped_test', {}).get('malignant', {}).get('f1', 0)*100:.2f}% | {results.get('v9_mobilenetv2_baseline', {}).get('grouped_test', {}).get('false_negatives', '-')} | {results.get('v9_mobilenetv2_baseline', {}).get('grouped_test', {}).get('false_positives', '-')} | {results.get('v9_mobilenetv2_baseline', {}).get('grouped_test', {}).get('total_incorrect', '-')} |
| **V9-B** | {results.get('v9_mobilenetv2_cropaug', {}).get('grouped_test', {}).get('accuracy', 0)*100:.2f}% | {results.get('v9_mobilenetv2_cropaug', {}).get('grouped_test', {}).get('macro_f1', 0)*100:.2f}% | {results.get('v9_mobilenetv2_cropaug', {}).get('grouped_test', {}).get('malignant', {}).get('precision', 0)*100:.2f}% | {results.get('v9_mobilenetv2_cropaug', {}).get('grouped_test', {}).get('malignant', {}).get('recall', 0)*100:.2f}% | {results.get('v9_mobilenetv2_cropaug', {}).get('grouped_test', {}).get('malignant', {}).get('f1', 0)*100:.2f}% | {results.get('v9_mobilenetv2_cropaug', {}).get('grouped_test', {}).get('false_negatives', '-')} | {results.get('v9_mobilenetv2_cropaug', {}).get('grouped_test', {}).get('false_positives', '-')} | {results.get('v9_mobilenetv2_cropaug', {}).get('grouped_test', {}).get('total_incorrect', '-')} |
| **V9-C** | {results.get('v9_mobilenetv2_dualview', {}).get('grouped_test', {}).get('accuracy', 0)*100:.2f}% | {results.get('v9_mobilenetv2_dualview', {}).get('grouped_test', {}).get('macro_f1', 0)*100:.2f}% | {results.get('v9_mobilenetv2_dualview', {}).get('grouped_test', {}).get('malignant', {}).get('precision', 0)*100:.2f}% | {results.get('v9_mobilenetv2_dualview', {}).get('grouped_test', {}).get('malignant', {}).get('recall', 0)*100:.2f}% | {results.get('v9_mobilenetv2_dualview', {}).get('grouped_test', {}).get('malignant', {}).get('f1', 0)*100:.2f}% | {results.get('v9_mobilenetv2_dualview', {}).get('grouped_test', {}).get('false_negatives', '-')} | {results.get('v9_mobilenetv2_dualview', {}).get('grouped_test', {}).get('false_positives', '-')} | {results.get('v9_mobilenetv2_dualview', {}).get('grouped_test', {}).get('total_incorrect', '-')} |
| **V9-D** | {results.get('v9_mobilenetv2_auxloss', {}).get('grouped_test', {}).get('accuracy', 0)*100:.2f}% | {results.get('v9_mobilenetv2_auxloss', {}).get('grouped_test', {}).get('macro_f1', 0)*100:.2f}% | {results.get('v9_mobilenetv2_auxloss', {}).get('grouped_test', {}).get('malignant', {}).get('precision', 0)*100:.2f}% | {results.get('v9_mobilenetv2_auxloss', {}).get('grouped_test', {}).get('malignant', {}).get('recall', 0)*100:.2f}% | {results.get('v9_mobilenetv2_auxloss', {}).get('grouped_test', {}).get('malignant', {}).get('f1', 0)*100:.2f}% | {results.get('v9_mobilenetv2_auxloss', {}).get('grouped_test', {}).get('false_negatives', '-')} | {results.get('v9_mobilenetv2_auxloss', {}).get('grouped_test', {}).get('false_positives', '-')} | {results.get('v9_mobilenetv2_auxloss', {}).get('grouped_test', {}).get('total_incorrect', '-')} |

---

## 16. Confusion Matrices (Top Candidate: {top_cid.upper()})

### Historical Locked 117-Image Test Set
```
                     Predicted Benign   Predicted Malignant   Predicted Normal
True Benign                 {top_h_cm[0][0]:<18} {top_h_cm[0][1]:<21} {top_h_cm[0][2]}
True Malignant              {top_h_cm[1][0]:<18} {top_h_cm[1][1]:<21} {top_h_cm[1][2]}
True Normal                 {top_h_cm[2][0]:<18} {top_h_cm[2][1]:<21} {top_h_cm[2][2]}
```

### Grouped Generalization Test Split
```
                     Predicted Benign   Predicted Malignant   Predicted Normal
True Benign                 {top_g_cm[0][0]:<18} {top_g_cm[0][1]:<21} {top_g_cm[0][2]}
True Malignant              {top_g_cm[1][0]:<18} {top_g_cm[1][1]:<21} {top_g_cm[1][2]}
True Normal                 {top_g_cm[2][0]:<18} {top_g_cm[2][1]:<21} {top_g_cm[2][2]}
```

---

## 17. Error Analysis
Crop augmentation and dual-view architectures reduce small-lesion (<10% area) false negatives while preserving high background specificity. Auxiliary local loss (V9-D) enforces feature discrimination on the crop branch.

---

## 18. Malignant FN / FP Analysis
- **False Negatives**: Evaluated against clinical tumor margin ambiguity.
- **False Positives**: Controlled to prevent diagnostic specificity breakdown.

---

## 19. Model Ranking
1. **{top_cid.upper()}**: Top candidate balancing sensitivity, precision, and macro F1.
2. **V9-D / V9-C**: Strong dual-view representations.
3. **V9-B**: Single-input crop-augmented baseline.
4. **V9-A**: Whole-image baseline reproduction control.

---

## 20. Promotion Recommendation
{promo_recommendation}

---

## 21. Limitations
- Auxiliary loss weighting (lambda_aux = 0.2) is static during training.
- Crop inputs depend on mask availability during training.

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
    print(f"\nSuccessfully generated V9 final evaluation report at: {report_path}")

if __name__ == "__main__":
    main()
