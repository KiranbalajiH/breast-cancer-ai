import os
import sys
import json
import cv2
import random
import shutil
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

# --- MODEL ARCHITECTURE BUILDER ---
def build_v11_model(input_shape=(224, 224, 3), dropout_rate=0.20, aug_rotation=0.05, aug_zoom=0.03):
    inputs = Input(shape=input_shape)
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(aug_rotation),
        tf.keras.layers.RandomTranslation(height_factor=0.03, width_factor=0.03),
        tf.keras.layers.RandomZoom(aug_zoom)
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
    print("V11 MALIGNANT PRECISION / RECALL OPTIMIZATION EXPERIMENT")
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
    
    # Prepare malignant sample subsets for controlled augmentation levels
    mal_train_mask = (y_train == 1)
    X_train_mal = X_train[mal_train_mask]
    y_train_mal = y_train[mal_train_mask]
    
    # Full 100% Malignant Replication (Used in V10-D / V11-A)
    X_tr_aug_100 = np.concatenate([X_train, X_train_mal], axis=0)
    y_tr_aug_100 = np.concatenate([y_train, y_train_mal], axis=0)
    X_tr_prep_aug_100 = preprocess_mnv2(X_tr_aug_100.copy())
    
    # Reduced 50% Malignant Replication (Used in V11-B & V11-E)
    np.random.seed(42)
    half_mal_indices = np.random.choice(len(X_train_mal), size=len(X_train_mal)//2, replace=False)
    X_train_mal_50 = X_train_mal[half_mal_indices]
    y_train_mal_50 = y_train_mal[half_mal_indices]
    X_tr_aug_50 = np.concatenate([X_train, X_train_mal_50], axis=0)
    y_tr_aug_50 = np.concatenate([y_train, y_train_mal_50], axis=0)
    X_tr_prep_aug_50 = preprocess_mnv2(X_tr_aug_50.copy())
    
    # Balanced Augmentation: 100% Malignant + 30% Benign replication (Used in V11-D)
    ben_train_mask = (y_train == 0)
    X_train_ben = X_train[ben_train_mask]
    y_train_ben = y_train[ben_train_mask]
    ben_30_indices = np.random.choice(len(X_train_ben), size=int(len(X_train_ben)*0.30), replace=False)
    X_train_ben_30 = X_train_ben[ben_30_indices]
    y_train_ben_30 = y_train_ben[ben_30_indices]
    X_tr_aug_bal = np.concatenate([X_train, X_train_mal, X_train_ben_30], axis=0)
    y_tr_aug_bal = np.concatenate([y_train, y_train_mal, y_train_ben_30], axis=0)
    X_tr_prep_aug_bal = preprocess_mnv2(X_tr_aug_bal.copy())
    
    v11_candidates_def = [
        {
            "id": "v11_mobilenetv2_a",
            "name": "V11-A: V10-D Control Reproduction (100% Malignant Aug, Unweighted)",
            "X_tr_prep": X_tr_prep_aug_100,
            "y_tr": y_tr_aug_100,
            "weights": None,
            "dropout": 0.20,
            "v10_source": "v10_mobilenetv2_d.keras"
        },
        {
            "id": "v11_mobilenetv2_b",
            "name": "V11-B: Reduced Malignant Augmentation (50% Malignant Replication)",
            "X_tr_prep": X_tr_prep_aug_50,
            "y_tr": y_tr_aug_50,
            "weights": None,
            "dropout": 0.20,
            "v10_source": None
        },
        {
            "id": "v11_mobilenetv2_c",
            "name": "V11-C: Malignant Augmentation + Milder Class Emphasis {0: 1.0, 1: 1.25, 2: 1.05}",
            "X_tr_prep": X_tr_prep_aug_100,
            "y_tr": y_tr_aug_100,
            "weights": {0: 1.0, 1: 1.25, 2: 1.05},
            "dropout": 0.20,
            "v10_source": None
        },
        {
            "id": "v11_mobilenetv2_d",
            "name": "V11-D: Malignant Augmentation + Regularization (Dropout=0.30 & Benign Aug)",
            "X_tr_prep": X_tr_prep_aug_bal,
            "y_tr": y_tr_aug_bal,
            "weights": None,
            "dropout": 0.30,
            "v10_source": None
        },
        {
            "id": "v11_mobilenetv2_e",
            "name": "V11-E: Composite Combination (50% Aug + Milder Class Weights + Dropout=0.25)",
            "X_tr_prep": X_tr_prep_aug_50,
            "y_tr": y_tr_aug_50,
            "weights": {0: 1.0, 1: 1.20, 2: 1.0},
            "dropout": 0.25,
            "v10_source": None
        }
    ]
    
    results = {}
    
    for cand in v11_candidates_def:
        cid = cand["id"]
        cname = cand["name"]
        save_path = os.path.join(candidates_dir, f"{cid}.keras")
        v10_src = cand.get("v10_source")
        
        if os.path.exists(save_path):
            print(f"\n=======================================================================")
            print(f"FOUND EXISTING CANDIDATE MODEL ({cid}): Loading {save_path}")
            print(f"=======================================================================")
            model = tf.keras.models.load_model(save_path)
        elif v10_src and os.path.exists(os.path.join(candidates_dir, v10_src)):
            src_path = os.path.join(candidates_dir, v10_src)
            print(f"\n=======================================================================")
            print(f"REUSING VERIFIED V10-D MODEL FOR V11-A CONTROL: Copying {src_path} -> {save_path}")
            print(f"=======================================================================")
            shutil.copyfile(src_path, save_path)
            model = tf.keras.models.load_model(save_path)
        else:
            print(f"\n=======================================================================")
            print(f"TRAINING V11 CANDIDATE: {cname}")
            print(f"=======================================================================")
            
            model, base_model = build_v11_model(dropout_rate=cand["dropout"])
            
            print("  Stage 1: Warming up classifier head (8 epochs, LR=5e-4)...")
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
                loss="sparse_categorical_crossentropy",
                metrics=['accuracy']
            )
            model.fit(
                cand["X_tr_prep"], cand["y_tr"],
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
                loss="sparse_categorical_crossentropy",
                metrics=['accuracy']
            )
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, mode='min'),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, mode='min')
            ]
            model.fit(
                cand["X_tr_prep"], cand["y_tr"],
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
            # Composite score prioritizing: Malignant Recall >= 85% + Malignant F1 + 0.5 * Precision
            rec = eval_t["malignant"]["recall"]
            prec = eval_t["malignant"]["precision"]
            f1_m = eval_t["malignant"]["f1"]
            
            # Penalize heavily if recall drops below 0.85 on validation
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
        
        results[cid] = {
            "id": cid,
            "name": cname,
            "model_path": save_path,
            "weights": str(cand["weights"]),
            "dropout": cand["dropout"],
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
        "dropout": res["dropout"],
        "validation_standard": res["validation_standard"],
        "best_val_threshold": res["best_val_threshold"],
        "validation_calibrated": res["validation_calibrated"],
        "threshold_sweep": res["threshold_sweep"]
    } for cid, res in results.items()}
    
    val_json_out_path = os.path.join(reports_dir, "v11_validation_summary.json")
    with open(val_json_out_path, "w", encoding="utf-8") as f:
        json.dump(val_summary, f, indent=4)
    print(f"\nSaved V11 validation summary JSON to: {val_json_out_path}")
    
    json_out_path = os.path.join(reports_dir, "v11_evaluation_results.json")
    with open(json_out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"Saved V11 evaluation raw JSON to: {json_out_path}")
    
    generate_v11_markdown_report(reports_dir, results)

def generate_v11_markdown_report(reports_dir, results):
    report_path = os.path.join(reports_dir, "v11_final_evaluation_report.md")
    
    cand_ids = ["v11_mobilenetv2_a", "v11_mobilenetv2_b", "v11_mobilenetv2_c", "v11_mobilenetv2_d", "v11_mobilenetv2_e"]
    cands_present = [results[cid] for cid in cand_ids if cid in results]
    
    # Sort candidates strictly by validation calibrated composite score
    cands_val_sorted = sorted(cands_present, key=lambda c: (c["validation_calibrated"]["malignant"]["f1"] + 0.5 * c["validation_calibrated"]["malignant"]["precision"]), reverse=True)
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

    # Check promotion condition vs V5-B baseline (Grouped Malignant Recall >= 85% AND Malignant Precision >= 60% AND Grouped Acc >= 70%)
    v5b_grp_mal_rec = 0.8065
    top_grp_rec = best_val_cand["grouped_test"]["malignant"]["recall"]
    top_grp_prec = best_val_cand["grouped_test"]["malignant"]["precision"]
    top_grp_acc = best_val_cand["grouped_test"]["accuracy"]
    
    if top_grp_rec >= 0.85 and top_grp_prec >= 0.60 and top_grp_acc >= 0.70:
        promo_recommendation = f"**PROMOTABLE FOR PRODUCTION REVIEW**. `{best_val_cand['id'].upper()}` achieves >=85% malignant sensitivity ({top_grp_rec*100:.2f}%) and recovered precision ({top_grp_prec*100:.2f}%) with {top_grp_acc*100:.2f}% overall accuracy."
    else:
        promo_recommendation = f"**DO NOT PROMOTE**. Recommend retaining **V5-B** as active production model."

    report_md = fr"""PRODUCTION MODEL CHANGED: NO
PRODUCTION METADATA CHANGED: NO
LOCKED 117-IMAGE TEST SET MODIFIED: NO
DATASET IMAGES MODIFIED: NO
DEPLOYMENT PERFORMED: NO
GIT COMMIT PERFORMED: NO
GIT PUSH PERFORMED: NO

# V11 Final Evaluation & Precision/Recall Optimization Report

**Date**: September 02, 2026  
**Project**: Breast Cancer AI — Ultrasound Image Classifier  
**Dataset**: `dataset/BUSI` (776 Clean Trainable Scans)  
**Production Model Status**: `backend/models/breast_image_classifier.keras` remains **unmodified** and **active** (V5-B).  
**Candidate Models Location**: `backend/models/candidates/`

---

## 1. Executive Summary & Production Status
- **Baseline Benchmarks**:
  - **V5-B (Production)**: Grouped Accuracy 70.09%, Macro F1 69.46%, Malignant Recall 80.65% (6 FN, 15 FP), Malignant Precision 62.50%, Malignant F1 70.42%.
  - **V10-D (Prior Best Sensitivity)**: Grouped Accuracy 67.29%, Macro F1 67.23%, Malignant Recall 87.10% (4 FN, 21 FP), Malignant Precision 56.25%, Malignant F1 68.35%.
- **V11 Objective**: Recover precision while retaining V10-D's malignant sensitivity gain ($\ge 85\%$ Grouped Malignant Recall, $\ge 60\%$ Precision, $\ge 70\%$ F1, $\text{{FN}} \le 5$, substantially reduced FP below 21).

---

## 2. V11 Candidate Configurations
- **V11-A (V10-D Control)**: Direct reproduction of V10-D (100% malignant sample replication, unweighted).
- **V11-B (Reduced Augmentation)**: 50% malignant sample replication (milder augmentation ratio).
- **V11-C (Augmentation + Mild Class Emphasis)**: 100% malignant sample replication + class weights `{{0: 1.0, 1: 1.25, 2: 1.05}}`.
- **V11-D (Augmentation + Regularization)**: 100% malignant sample replication + 30% benign replication + Dropout(0.30).
- **V11-E (Composite Strategy)**: 50% malignant sample replication + class weights `{{0: 1.0, 1: 1.20, 2: 1.0}}` + Dropout(0.25).

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
| **V5-B (Active Prod)** | 80.34% | 79.75% | 77.78% | 87.50% | **82.35%** | 4 | 8 | 23 |
| **V6-B** | 81.20% | 80.54% | 78.57% | 87.50% | 82.76% | 4 | 7 | 22 |
| **V9-D (Dual-View Aux)**| 87.18% | 86.46% | 81.25% | 81.25% | 81.25% | 6 | 6 | 15 |
| **V10-D** | 76.92% | 76.57% | 69.77% | **93.75%** | 80.00% | **2** | 13 | 27 |
| **V11-A** | {results.get('v11_mobilenetv2_a', {}).get('historical_test_117', {}).get('accuracy', 0)*100:.2f}% | {results.get('v11_mobilenetv2_a', {}).get('historical_test_117', {}).get('macro_f1', 0)*100:.2f}% | {results.get('v11_mobilenetv2_a', {}).get('historical_test_117', {}).get('malignant', {}).get('precision', 0)*100:.2f}% | {results.get('v11_mobilenetv2_a', {}).get('historical_test_117', {}).get('malignant', {}).get('recall', 0)*100:.2f}% | {results.get('v11_mobilenetv2_a', {}).get('historical_test_117', {}).get('malignant', {}).get('f1', 0)*100:.2f}% | {results.get('v11_mobilenetv2_a', {}).get('historical_test_117', {}).get('false_negatives', '-')} | {results.get('v11_mobilenetv2_a', {}).get('historical_test_117', {}).get('false_positives', '-')} | {results.get('v11_mobilenetv2_a', {}).get('historical_test_117', {}).get('total_incorrect', '-')} |
| **V11-B** | {results.get('v11_mobilenetv2_b', {}).get('historical_test_117', {}).get('accuracy', 0)*100:.2f}% | {results.get('v11_mobilenetv2_b', {}).get('historical_test_117', {}).get('macro_f1', 0)*100:.2f}% | {results.get('v11_mobilenetv2_b', {}).get('historical_test_117', {}).get('malignant', {}).get('precision', 0)*100:.2f}% | {results.get('v11_mobilenetv2_b', {}).get('historical_test_117', {}).get('malignant', {}).get('recall', 0)*100:.2f}% | {results.get('v11_mobilenetv2_b', {}).get('historical_test_117', {}).get('malignant', {}).get('f1', 0)*100:.2f}% | {results.get('v11_mobilenetv2_b', {}).get('historical_test_117', {}).get('false_negatives', '-')} | {results.get('v11_mobilenetv2_b', {}).get('historical_test_117', {}).get('false_positives', '-')} | {results.get('v11_mobilenetv2_b', {}).get('historical_test_117', {}).get('total_incorrect', '-')} |
| **V11-C** | {results.get('v11_mobilenetv2_c', {}).get('historical_test_117', {}).get('accuracy', 0)*100:.2f}% | {results.get('v11_mobilenetv2_c', {}).get('historical_test_117', {}).get('macro_f1', 0)*100:.2f}% | {results.get('v11_mobilenetv2_c', {}).get('historical_test_117', {}).get('malignant', {}).get('precision', 0)*100:.2f}% | {results.get('v11_mobilenetv2_c', {}).get('historical_test_117', {}).get('malignant', {}).get('recall', 0)*100:.2f}% | {results.get('v11_mobilenetv2_c', {}).get('historical_test_117', {}).get('malignant', {}).get('f1', 0)*100:.2f}% | {results.get('v11_mobilenetv2_c', {}).get('historical_test_117', {}).get('false_negatives', '-')} | {results.get('v11_mobilenetv2_c', {}).get('historical_test_117', {}).get('false_positives', '-')} | {results.get('v11_mobilenetv2_c', {}).get('historical_test_117', {}).get('total_incorrect', '-')} |
| **V11-D** | {results.get('v11_mobilenetv2_d', {}).get('historical_test_117', {}).get('accuracy', 0)*100:.2f}% | {results.get('v11_mobilenetv2_d', {}).get('historical_test_117', {}).get('macro_f1', 0)*100:.2f}% | {results.get('v11_mobilenetv2_d', {}).get('historical_test_117', {}).get('malignant', {}).get('precision', 0)*100:.2f}% | {results.get('v11_mobilenetv2_d', {}).get('historical_test_117', {}).get('malignant', {}).get('recall', 0)*100:.2f}% | {results.get('v11_mobilenetv2_d', {}).get('historical_test_117', {}).get('malignant', {}).get('f1', 0)*100:.2f}% | {results.get('v11_mobilenetv2_d', {}).get('historical_test_117', {}).get('false_negatives', '-')} | {results.get('v11_mobilenetv2_d', {}).get('historical_test_117', {}).get('false_positives', '-')} | {results.get('v11_mobilenetv2_d', {}).get('historical_test_117', {}).get('total_incorrect', '-')} |
| **V11-E** | {results.get('v11_mobilenetv2_e', {}).get('historical_test_117', {}).get('accuracy', 0)*100:.2f}% | {results.get('v11_mobilenetv2_e', {}).get('historical_test_117', {}).get('macro_f1', 0)*100:.2f}% | {results.get('v11_mobilenetv2_e', {}).get('historical_test_117', {}).get('malignant', {}).get('precision', 0)*100:.2f}% | {results.get('v11_mobilenetv2_e', {}).get('historical_test_117', {}).get('malignant', {}).get('recall', 0)*100:.2f}% | {results.get('v11_mobilenetv2_e', {}).get('historical_test_117', {}).get('malignant', {}).get('f1', 0)*100:.2f}% | {results.get('v11_mobilenetv2_e', {}).get('historical_test_117', {}).get('false_negatives', '-')} | {results.get('v11_mobilenetv2_e', {}).get('historical_test_117', {}).get('false_positives', '-')} | {results.get('v11_mobilenetv2_e', {}).get('historical_test_117', {}).get('total_incorrect', '-')} |

---

## 6. Grouped Generalization Comparison Across Iterations

| Model Iteration | Accuracy | Macro F1 | Malignant Precision | Malignant Recall | Malignant F1 | Malignant FN | Malignant FP | Total Errors |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **V5-B (Active Prod)** | 70.09% | 69.46% | 62.50% | 80.65% | 70.42% | 6 | 15 | 32 |
| **V6-B** | 71.96% | 71.10% | 64.10% | 80.65% | 71.43% | 6 | 14 | 30 |
| **V9-D (Dual-View Aux)**| 77.57% | 77.84% | 63.64% | 67.74% | 65.62% | 10 | 12 | 24 |
| **V10-D** | 67.29% | 67.23% | 56.25% | **87.10%** | 68.35% | **4** | 21 | 35 |
| **V11-A** | {results.get('v11_mobilenetv2_a', {}).get('grouped_test', {}).get('accuracy', 0)*100:.2f}% | {results.get('v11_mobilenetv2_a', {}).get('grouped_test', {}).get('macro_f1', 0)*100:.2f}% | {results.get('v11_mobilenetv2_a', {}).get('grouped_test', {}).get('malignant', {}).get('precision', 0)*100:.2f}% | {results.get('v11_mobilenetv2_a', {}).get('grouped_test', {}).get('malignant', {}).get('recall', 0)*100:.2f}% | {results.get('v11_mobilenetv2_a', {}).get('grouped_test', {}).get('malignant', {}).get('f1', 0)*100:.2f}% | {results.get('v11_mobilenetv2_a', {}).get('grouped_test', {}).get('false_negatives', '-')} | {results.get('v11_mobilenetv2_a', {}).get('grouped_test', {}).get('false_positives', '-')} | {results.get('v11_mobilenetv2_a', {}).get('grouped_test', {}).get('total_incorrect', '-')} |
| **V11-B** | {results.get('v11_mobilenetv2_b', {}).get('grouped_test', {}).get('accuracy', 0)*100:.2f}% | {results.get('v11_mobilenetv2_b', {}).get('grouped_test', {}).get('macro_f1', 0)*100:.2f}% | {results.get('v11_mobilenetv2_b', {}).get('grouped_test', {}).get('malignant', {}).get('precision', 0)*100:.2f}% | {results.get('v11_mobilenetv2_b', {}).get('grouped_test', {}).get('malignant', {}).get('recall', 0)*100:.2f}% | {results.get('v11_mobilenetv2_b', {}).get('grouped_test', {}).get('malignant', {}).get('f1', 0)*100:.2f}% | {results.get('v11_mobilenetv2_b', {}).get('grouped_test', {}).get('false_negatives', '-')} | {results.get('v11_mobilenetv2_b', {}).get('grouped_test', {}).get('false_positives', '-')} | {results.get('v11_mobilenetv2_b', {}).get('grouped_test', {}).get('total_incorrect', '-')} |
| **V11-C** | {results.get('v11_mobilenetv2_c', {}).get('grouped_test', {}).get('accuracy', 0)*100:.2f}% | {results.get('v11_mobilenetv2_c', {}).get('grouped_test', {}).get('macro_f1', 0)*100:.2f}% | {results.get('v11_mobilenetv2_c', {}).get('grouped_test', {}).get('malignant', {}).get('precision', 0)*100:.2f}% | {results.get('v11_mobilenetv2_c', {}).get('grouped_test', {}).get('malignant', {}).get('recall', 0)*100:.2f}% | {results.get('v11_mobilenetv2_c', {}).get('grouped_test', {}).get('malignant', {}).get('f1', 0)*100:.2f}% | {results.get('v11_mobilenetv2_c', {}).get('grouped_test', {}).get('false_negatives', '-')} | {results.get('v11_mobilenetv2_c', {}).get('grouped_test', {}).get('false_positives', '-')} | {results.get('v11_mobilenetv2_c', {}).get('grouped_test', {}).get('total_incorrect', '-')} |
| **V11-D** | {results.get('v11_mobilenetv2_d', {}).get('grouped_test', {}).get('accuracy', 0)*100:.2f}% | {results.get('v11_mobilenetv2_d', {}).get('grouped_test', {}).get('macro_f1', 0)*100:.2f}% | {results.get('v11_mobilenetv2_d', {}).get('grouped_test', {}).get('malignant', {}).get('precision', 0)*100:.2f}% | {results.get('v11_mobilenetv2_d', {}).get('grouped_test', {}).get('malignant', {}).get('recall', 0)*100:.2f}% | {results.get('v11_mobilenetv2_d', {}).get('grouped_test', {}).get('malignant', {}).get('f1', 0)*100:.2f}% | {results.get('v11_mobilenetv2_d', {}).get('grouped_test', {}).get('false_negatives', '-')} | {results.get('v11_mobilenetv2_d', {}).get('grouped_test', {}).get('false_positives', '-')} | {results.get('v11_mobilenetv2_d', {}).get('grouped_test', {}).get('total_incorrect', '-')} |
| **V11-E** | {results.get('v11_mobilenetv2_e', {}).get('grouped_test', {}).get('accuracy', 0)*100:.2f}% | {results.get('v11_mobilenetv2_e', {}).get('grouped_test', {}).get('macro_f1', 0)*100:.2f}% | {results.get('v11_mobilenetv2_e', {}).get('grouped_test', {}).get('malignant', {}).get('precision', 0)*100:.2f}% | {results.get('v11_mobilenetv2_e', {}).get('grouped_test', {}).get('malignant', {}).get('recall', 0)*100:.2f}% | {results.get('v11_mobilenetv2_e', {}).get('grouped_test', {}).get('malignant', {}).get('f1', 0)*100:.2f}% | {results.get('v11_mobilenetv2_e', {}).get('grouped_test', {}).get('false_negatives', '-')} | {results.get('v11_mobilenetv2_e', {}).get('grouped_test', {}).get('false_positives', '-')} | {results.get('v11_mobilenetv2_e', {}).get('grouped_test', {}).get('total_incorrect', '-')} |

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

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"\nSuccessfully generated V11 final evaluation report at: {report_path}")

if __name__ == "__main__":
    main()
