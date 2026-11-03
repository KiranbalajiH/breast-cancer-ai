import os
import sys
import json
import cv2
import time
import random
import numpy as np
import tensorflow as tf
from datetime import datetime
from collections import Counter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as preprocess_mnv2
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input as preprocess_effnet
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import evaluate_rigorous as er

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def build_model(architecture="mobilenetv2", input_shape=(224, 224, 3), dropout_rate=0.20, aug_rotation=0.05, aug_zoom=0.03):
    inputs = Input(shape=input_shape)
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(aug_rotation),
        tf.keras.layers.RandomTranslation(height_factor=0.03, width_factor=0.03),
        tf.keras.layers.RandomZoom(aug_zoom)
    ])
    x = data_augmentation(inputs)
    
    if architecture == "mobilenetv2":
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    elif architecture == "efficientnetb0":
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
        
    base_model.trainable = False
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    if dropout_rate > 0:
        x = Dropout(dropout_rate)(x)
    outputs = Dense(3, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model, base_model

def evaluate_metrics(y_true, y_pred, y_probs=None):
    acc = float(accuracy_score(y_true, y_pred))
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0)
    _, _, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    
    tp = int(cm[1, 1])
    fn = int(cm[1, 0] + cm[1, 2])
    fp = int(cm[0, 1] + cm[2, 1])
    tn = int(cm[0, 0] + cm[0, 2] + cm[2, 0] + cm[2, 2])
    
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    
    return {
        "accuracy": float(round(acc * 100, 2)),
        "macro_f1": float(round(float(f1_macro) * 100, 2)),
        "malignant_sensitivity": float(round(sens * 100, 2)),
        "malignant_specificity": float(round(spec * 100, 2)),
        "malignant_ppv": float(round(ppv * 100, 2)),
        "malignant_npv": float(round(npv * 100, 2)),
        "malignant_fn": int(fn),
        "malignant_fp": int(fp),
        "confusion_matrix": cm.tolist()
    }

def main():
    start_time = time.time()
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    bcd_root = os.path.dirname(backend_dir)
    dataset_root = os.path.join(bcd_root, "dataset", "BUSI")
    
    output_candidate_dir = os.path.join(backend_dir, "models", "research_phase_candidates")
    reports_dir = os.path.join(backend_dir, "models", "evaluation_reports")
    os.makedirs(output_candidate_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    print("=" * 80)
    print("PHASE 3: TRAINING CONTROLLED RESEARCH CANDIDATES")
    print("=" * 80)
    
    # 1. Load data and clusters
    set_seed(42)
    scans, classes = er.load_and_clean_dataset(dataset_root, backend_dir)
    labels = np.array([s["class_idx"] for s in scans], dtype=np.int32)
    cluster_labels, total_clusters = er.group_scans_by_perceptual_cluster(scans)
    
    grp_train_idx, grp_val_idx, grp_test_idx = er.create_grouped_stratified_split(scans, cluster_labels, seed=42)
    
    print(f"Data Split Summary: Train={len(grp_train_idx)} scans, Val={len(grp_val_idx)} scans, Test={len(grp_test_idx)} scans")
    print(f"Class counts in Train: {Counter(labels[grp_train_idx])}")
    print(f"Class counts in Val  : {Counter(labels[grp_val_idx])}")
    print(f"Class counts in Test : {Counter(labels[grp_test_idx])}")
    
    # Pre-load image arrays
    print("Loading image arrays...")
    X_all = er.load_images(scans)
    X_train = X_all[grp_train_idx]
    y_train = labels[grp_train_idx]
    X_val = X_all[grp_val_idx]
    y_val = labels[grp_val_idx]
    X_test = X_all[grp_test_idx]
    y_test = labels[grp_test_idx]
    
    # Preprocessing inputs
    X_tr_mnv2 = preprocess_mnv2(X_train.copy())
    X_val_mnv2 = preprocess_mnv2(X_val.copy())
    X_te_mnv2 = preprocess_mnv2(X_test.copy())
    
    X_tr_eff = preprocess_effnet(X_train.copy())
    X_val_eff = preprocess_effnet(X_val.copy())
    X_te_eff = preprocess_effnet(X_test.copy())
    
    # Compute inverse class frequencies for balanced weighting experiment
    class_counts = Counter(y_train)
    total_samples = len(y_train)
    # Balanced weights: total / (n_classes * count)
    class_weights_balanced = {c: float(total_samples / (3.0 * count)) for c, count in class_counts.items()}
    # Mild malignant penalty: give malignant 1.35x weight to reduce false negatives
    class_weights_cost_sensitive = {0: 1.0, 1: 1.35, 2: 1.0}
    
    experiments = [
        {
            "id": "exp_v15_mnv2_balanced_weights",
            "name": "Exp 1: MobileNetV2 with Balanced Class Weights",
            "architecture": "mobilenetv2",
            "weights": class_weights_balanced,
            "dropout": 0.20,
            "epochs_stage1": 6,
            "epochs_stage2": 10,
            "seed": 42
        },
        {
            "id": "exp_v15_effnet_cost_sensitive",
            "name": "Exp 2: EfficientNetB0 with Cost-Sensitive Malignant Weight (1.35x)",
            "architecture": "efficientnetb0",
            "weights": class_weights_cost_sensitive,
            "dropout": 0.25,
            "epochs_stage1": 6,
            "epochs_stage2": 10,
            "seed": 42
        },
        {
            "id": "exp_v15_mnv2_seed101",
            "name": "Exp 3: MobileNetV2 with Seed 101 (Multi-Seed Stability Check)",
            "architecture": "mobilenetv2",
            "weights": None,
            "dropout": 0.20,
            "epochs_stage1": 6,
            "epochs_stage2": 10,
            "seed": 101
        }
    ]
    
    trained_models = {}
    experiment_log = {}
    
    for exp in experiments:
        exp_id = exp["id"]
        exp_name = exp["name"]
        print(f"\n=======================================================")
        print(f"RUNNING EXPERIMENT: {exp_name}")
        print(f"=======================================================")
        set_seed(exp["seed"])
        
        arch = exp["architecture"]
        model_save_path = os.path.join(output_candidate_dir, f"{exp_id}.keras")
        
        model, base_model = build_model(
            architecture=arch,
            dropout_rate=exp["dropout"]
        )
        
        # Select correct preprocessed data
        if arch == "mobilenetv2":
            X_tr_p = X_tr_mnv2
            X_val_p = X_val_mnv2
            X_te_p = X_te_mnv2
        else:
            X_tr_p = X_tr_eff
            X_val_p = X_val_eff
            X_te_p = X_te_eff
            
        if os.path.exists(model_save_path):
            print(f"  Found existing trained model at: {model_save_path}. Loading...")
            model = tf.keras.models.load_model(model_save_path)
            t_hist1 = None
            t_hist2 = None
        else:
            # Stage 1: Warmup classifier head
            print(f"  Stage 1: Frozen backbone warmup ({exp['epochs_stage1']} epochs, LR=5e-4)...")
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
                loss="sparse_categorical_crossentropy",
                metrics=['accuracy']
            )
            t_hist1 = model.fit(
                X_tr_p, y_train,
                validation_data=(X_val_p, y_val),
                epochs=exp["epochs_stage1"],
                batch_size=32,
                class_weight=exp["weights"],
                verbose=1
            )
            
            # Stage 2: Fine-tune top 15 backbone layers
            print(f"  Stage 2: Fine-tuning top 15 layers ({exp['epochs_stage2']} epochs, LR=1e-4)...")
            base_model.trainable = True
            for layer in base_model.layers[:-15]:
                layer.trainable = False
                
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                loss="sparse_categorical_crossentropy",
                metrics=['accuracy']
            )
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, mode='min'),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1, mode='min')
            ]
            t_hist2 = model.fit(
                X_tr_p, y_train,
                validation_data=(X_val_p, y_val),
                epochs=exp["epochs_stage2"],
                batch_size=32,
                class_weight=exp["weights"],
                callbacks=callbacks,
                verbose=1
            )
            
            model.save(model_save_path)
            print(f"  Saved trained candidate to: {model_save_path}")
            
        trained_models[exp_id] = model
        
        # Validation Evaluation strictly for model selection and threshold selection
        val_probs = model.predict(X_val_p, verbose=0)
        
        # Find best validation threshold for malignant class (t in [0.35, 0.65])
        best_t = 0.50
        best_val_score = -1.0
        best_val_metrics = None
        
        for t in np.arange(0.35, 0.66, 0.05):
            t_curr = round(float(t), 2)
            y_pred_t = np.zeros(len(val_probs), dtype=int)
            for i, p in enumerate(val_probs):
                if p[1] >= t_curr:
                    y_pred_t[i] = 1
                else:
                    y_pred_t[i] = 0 if p[0] >= p[2] else 2
            m = evaluate_metrics(y_val, y_pred_t)
            # Prioritize malignant sensitivity while maintaining macro F1
            composite_score = m["malignant_sensitivity"] * 1.5 + m["macro_f1"]
            if composite_score > best_val_score:
                best_val_score = composite_score
                best_t = t_curr
                best_val_metrics = m
                
        print(f"  Validation Selected Threshold: t*={best_t}")
        print(f"  Validation Metrics at t*: Acc={best_val_metrics['accuracy']}%, Sens={best_val_metrics['malignant_sensitivity']}%, FN={best_val_metrics['malignant_fn']}, FP={best_val_metrics['malignant_fp']}")
        
        experiment_log[exp_id] = {
            "name": exp_name,
            "config": exp,
            "model_path": model_save_path,
            "best_validation_threshold": best_t,
            "validation_metrics": best_val_metrics,
            "history_stage1": {k: [float(v) for v in vals] for k, vals in t_hist1.history.items()} if t_hist1 else None,
            "history_stage2": {k: [float(v) for v in vals] for k, vals in t_hist2.history.items()} if t_hist2 else None
        }
        
    # Phase 4: Construct Research Ensemble Candidate (V15 Ensemble: Exp 1 + Exp 2)
    print("\n=======================================================")
    print("CONSTRUCTING PHASE 4 CANDIDATE: V15 Ensemble (Exp 1 + Exp 2)")
    print("=======================================================")
    p_val_exp1 = trained_models["exp_v15_mnv2_balanced_weights"].predict(X_val_mnv2, verbose=0)
    p_val_exp2 = trained_models["exp_v15_effnet_cost_sensitive"].predict(X_val_eff, verbose=0)
    p_val_v15 = 0.5 * p_val_exp1 + 0.5 * p_val_exp2
    
    best_v15_t = 0.50
    best_v15_val_score = -1.0
    best_v15_val_metrics = None
    
    for t in np.arange(0.35, 0.66, 0.05):
        t_curr = round(float(t), 2)
        y_pred_t = np.zeros(len(p_val_v15), dtype=int)
        for i, p in enumerate(p_val_v15):
            if p[1] >= t_curr:
                y_pred_t[i] = 1
            else:
                y_pred_t[i] = 0 if p[0] >= p[2] else 2
        m = evaluate_metrics(y_val, y_pred_t)
        composite = m["malignant_sensitivity"] * 1.5 + m["macro_f1"]
        if composite > best_v15_val_score:
            best_v15_val_score = composite
            best_v15_t = t_curr
            best_v15_val_metrics = m
            
    print(f"V15 Ensemble Validation Selected Threshold: t*={best_v15_t}")
    print(f"V15 Ensemble Validation Metrics: Acc={best_v15_val_metrics['accuracy']}%, Sens={best_v15_val_metrics['malignant_sensitivity']}%, FN={best_v15_val_metrics['malignant_fn']}, FP={best_v15_val_metrics['malignant_fp']}")
    
    experiment_log["v15_ensemble"] = {
        "name": "V15 Research Ensemble (Exp1 + Exp2, 50/50)",
        "best_validation_threshold": best_v15_t,
        "validation_metrics": best_v15_val_metrics
    }
    
    # Phase 5: Final Locked Evaluation on Grouped Test Set (N=107)
    print("\n=================================================================")
    print("PHASE 5: LOCKED FINAL EVALUATION ON GROUPED TEST SPLIT (N=107)")
    print("=================================================================")
    # Load benchmarks
    v5b_path = os.path.join(backend_dir, "models", "breast_image_classifier.keras")
    v11b_path = os.path.join(backend_dir, "models", "candidates", "v11_mobilenetv2_b.keras")
    v13_path = os.path.join(backend_dir, "models", "candidates", "v13_efficientnet_b0.keras")
    m_v5b = tf.keras.models.load_model(v5b_path)
    m_v11b = tf.keras.models.load_model(v11b_path)
    m_v13 = tf.keras.models.load_model(v13_path)
    
    # Inferences on Test Set
    p_te_v5b = m_v5b.predict(X_te_mnv2, verbose=0)
    p_te_v11b = m_v11b.predict(X_te_mnv2, verbose=0)
    p_te_v13 = m_v13.predict(X_te_eff, verbose=0)
    p_te_v14 = 0.5 * p_te_v11b + 0.5 * p_te_v13
    
    # V5-B Prediction (Argmax)
    pred_te_v5b = np.argmax(p_te_v5b, axis=1)
    
    # V14 Benchmark Prediction (t*=0.58)
    pred_te_v14 = np.zeros(len(p_te_v14), dtype=int)
    for i, p in enumerate(p_te_v14):
        if p[1] >= 0.58:
            pred_te_v14[i] = 1
        else:
            pred_te_v14[i] = 0 if p[0] >= p[2] else 2
            
    # New Trained Candidate: Exp 1 Test Prediction
    p_te_exp1 = trained_models["exp_v15_mnv2_balanced_weights"].predict(X_te_mnv2, verbose=0)
    t_exp1 = experiment_log["exp_v15_mnv2_balanced_weights"]["best_validation_threshold"]
    pred_te_exp1 = np.zeros(len(p_te_exp1), dtype=int)
    for i, p in enumerate(p_te_exp1):
        if p[1] >= t_exp1:
            pred_te_exp1[i] = 1
        else:
            pred_te_exp1[i] = 0 if p[0] >= p[2] else 2
            
    # New Trained Candidate: Exp 2 Test Prediction
    p_te_exp2 = trained_models["exp_v15_effnet_cost_sensitive"].predict(X_te_eff, verbose=0)
    t_exp2 = experiment_log["exp_v15_effnet_cost_sensitive"]["best_validation_threshold"]
    pred_te_exp2 = np.zeros(len(p_te_exp2), dtype=int)
    for i, p in enumerate(p_te_exp2):
        if p[1] >= t_exp2:
            pred_te_exp2[i] = 1
        else:
            pred_te_exp2[i] = 0 if p[0] >= p[2] else 2
            
    # V15 Ensemble Test Prediction
    p_te_v15 = 0.5 * p_te_exp1 + 0.5 * p_te_exp2
    t_v15 = experiment_log["v15_ensemble"]["best_validation_threshold"]
    pred_te_v15 = np.zeros(len(p_te_v15), dtype=int)
    for i, p in enumerate(p_te_v15):
        if p[1] >= t_v15:
            pred_te_v15[i] = 1
        else:
            pred_te_v15[i] = 0 if p[0] >= p[2] else 2
            
    test_evaluations = {
        "V5-B Production (MobileNetV2)": {
            "metrics": evaluate_metrics(y_test, pred_te_v5b),
            "threshold_used": "argmax"
        },
        "V14 Research Benchmark (50/50, t*=0.58)": {
            "metrics": evaluate_metrics(y_test, pred_te_v14),
            "threshold_used": 0.58
        },
        "Exp 1 (MobileNetV2 Balanced)": {
            "metrics": evaluate_metrics(y_test, pred_te_exp1),
            "threshold_used": t_exp1
        },
        "Exp 2 (EfficientNetB0 Cost-Sensitive)": {
            "metrics": evaluate_metrics(y_test, pred_te_exp2),
            "threshold_used": t_exp2
        },
        "V15 Research Ensemble (Exp1+Exp2)": {
            "metrics": evaluate_metrics(y_test, pred_te_v15),
            "threshold_used": t_v15
        }
    }
    
    # Statistical tests: V14 vs V15
    mcnemar_v14_v15 = er.mcnemar_test(y_test, pred_te_v14, pred_te_v15)
    
    def sanitize(obj):
        if isinstance(obj, dict):
            return {str(k): sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sanitize(v) for v in obj]
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return sanitize(obj.tolist())
        return obj

    final_output = {
        "timestamp": datetime.now().isoformat(),
        "runtime_seconds": round(time.time() - start_time, 2),
        "dataset": "BUSI",
        "split_counts": {
            "train": len(grp_train_idx),
            "val": len(grp_val_idx),
            "test": len(grp_test_idx)
        },
        "experiments": experiment_log,
        "test_evaluations": test_evaluations,
        "statistical_comparison_v14_vs_v15": mcnemar_v14_v15
    }

    final_output = sanitize(final_output)

    out_json_path = os.path.join(reports_dir, "phase3_research_candidates_audit.json")
    with open(out_json_path, "w") as f:
        json.dump(final_output, f, indent=2)
    print(f"\nFinal Phase 3-5 Report saved to: {out_json_path}")
    
    print("\nSUMMARY OF LOCKED TEST RESULTS:")
    for mname, mdata in test_evaluations.items():
        met = mdata["metrics"]
        print(f"  {mname}: Acc={met['accuracy']}%, Sens={met['malignant_sensitivity']}%, Spec={met['malignant_specificity']}%, Macro F1={met['macro_f1']}%, FN={met['malignant_fn']}, FP={met['malignant_fp']}")
        
    print(f"\nMcNemar Test (V14 Benchmark vs V15 Candidate): Discordant: V14-only={mcnemar_v14_v15['n10_a_only']}, V15-only={mcnemar_v14_v15['n01_b_only']}, Exact Binomial p={mcnemar_v14_v15['exact_binomial_p_value']}")

if __name__ == "__main__":
    main()
