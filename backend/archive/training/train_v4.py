import os
import sys
import json
import cv2
import random
import numpy as np
import tensorflow as tf
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
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
    image_paths = []
    labels = []
    
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
            image_paths.append(full_path)
            labels.append(class_idx)
            
    print(f"Loaded {len(image_paths)} clean images.")
    return image_paths, labels, classes

def load_images_to_numpy(image_paths, target_size=(224, 224)):
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Could not read image: {path}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, target_size)
        images.append(img_resized)
    return np.array(images, dtype=np.float32)

def build_v4_model(base_model_fn, input_shape=(224, 224, 3), dropout_rate=0.2, augment=True):
    inputs = Input(shape=input_shape)
    if augment:
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomTranslation(height_factor=0.03, width_factor=0.03),
            tf.keras.layers.RandomZoom(0.03)
        ])
        x = data_augmentation(inputs)
    else:
        x = inputs
    
    base_model = base_model_fn(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False
    
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    if dropout_rate > 0:
        x = Dropout(dropout_rate)(x)
    
    outputs = Dense(3, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    
    return model, base_model

def evaluate_metrics(model, X_data, y_data, classes, prep_fn):
    X_prep = prep_fn(X_data.copy())
    predictions = model.predict(X_prep, verbose=0)
    pred_classes = np.argmax(predictions, axis=1)
    
    acc = float(accuracy_score(y_data, pred_classes))
    prec, rec, f1, support = precision_recall_fscore_support(y_data, pred_classes, labels=[0, 1, 2], zero_division=0)
    cm = confusion_matrix(y_data, pred_classes, labels=[0, 1, 2]).tolist()
    
    mal_indices = np.where(y_data == 1)[0]
    fn_count = int(np.sum(pred_classes[mal_indices] != 1))
    
    non_mal_indices = np.where(y_data != 1)[0]
    fp_count = int(np.sum(pred_classes[non_mal_indices] == 1))
    
    total_incorrect = int(np.sum(pred_classes != y_data))
    
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
        "confusion_matrix": cm
    }

def main():
    set_seed(42)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.dirname(script_dir)
    bcd_root = os.path.dirname(backend_dir)
    dataset_root = os.path.join(bcd_root, "dataset", "BUSI")
    if not os.path.exists(dataset_root):
        dataset_root = os.path.join(bcd_root, "dataset", "Dataset_BUSI_with_GT")
        
    candidates_dir = os.path.join(backend_dir, "models", "candidates")
    os.makedirs(candidates_dir, exist_ok=True)
    
    print("="*70)
    print("V4 MODEL TRAINING PIPELINE — BALANCED MALIGNANT SENSITIVITY")
    print("="*70)
    
    # 1. Load Clean Dataset
    image_paths, labels, classes = load_and_clean_dataset(dataset_root)
    labels = np.array(labels, dtype=np.int32)
    
    # 2. Load Raw Images into Memory
    print("\nLoading images into memory...")
    X_raw = load_images_to_numpy(image_paths)
    print(f"Loaded raw image array shape: {X_raw.shape}")
    
    # 3. Create Stratified Splits (70% Train, 15% Val, 15% Test) with Seed 42
    X_train, X_val_test, y_train, y_val_test = train_test_split(
        X_raw, labels, test_size=0.30, stratify=labels, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_val_test, y_val_test, test_size=0.50, stratify=y_val_test, random_state=42
    )
    
    train_counts = {classes[i]: int(np.sum(y_train == i)) for i in range(len(classes))}
    val_counts = {classes[i]: int(np.sum(y_val == i)) for i in range(len(classes))}
    test_counts = {classes[i]: int(np.sum(y_test == i)) for i in range(len(classes))}
    
    print(f"\nDataset Split Sizes:")
    print(f"  Train ({len(X_train)}): {train_counts}")
    print(f"  Val   ({len(X_val)}): {val_counts}")
    print(f"  Test  ({len(X_test)}): {test_counts} [UNTOUCHED]")
    
    # Calculate class weighting options with explicit int keys
    balanced_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    balanced_dict = {int(i): float(balanced_weights[i]) for i in range(len(balanced_weights))}
    
    # Mild malignant class weighting with explicit integer keys 0, 1, 2:
    mild_malignant_dict = {0: 1.0, 1: 1.25, 2: 1.1}
    
    print(f"\nWeight Configurations for V4:")
    print(f"  No Weights (Standard)     : {{0: 1.0, 1: 1.0, 2: 1.0}}")
    print(f"  Mild Malignant Weighting  : {mild_malignant_dict}")
    print(f"  Balanced Weighting        : {balanced_dict}")
    
    # Controlled V4 Variants
    v4_configs = [
        {
            "id": "v4_mobilenetv2_a",
            "name": "V4-A: MobileNetV2 (No Weights / Conservative Fine-Tuning)",
            "base_fn": MobileNetV2,
            "prep_fn": preprocess_mnv2,
            "weights": None,
            "dropout": 0.2,
            "unfreeze_layers": 15,
            "stage1_epochs": 8,
            "stage2_epochs": 15
        },
        {
            "id": "v4_mobilenetv2_b",
            "name": "V4-B: MobileNetV2 (Mild Malignant Weighting 1.25x / Fine-Tuned)",
            "base_fn": MobileNetV2,
            "prep_fn": preprocess_mnv2,
            "weights": mild_malignant_dict,
            "dropout": 0.2,
            "unfreeze_layers": 15,
            "stage1_epochs": 8,
            "stage2_epochs": 15
        },
        {
            "id": "v4_mobilenetv2_c",
            "name": "V4-C: MobileNetV2 (Balanced Loss Weighting / Fine-Tuned)",
            "base_fn": MobileNetV2,
            "prep_fn": preprocess_mnv2,
            "weights": balanced_dict,
            "dropout": 0.25,
            "unfreeze_layers": 15,
            "stage1_epochs": 8,
            "stage2_epochs": 15
        },
        {
            "id": "v4_efficientnet_d",
            "name": "V4-D: EfficientNetB0 (Mild Weighting / Lightweight Alt)",
            "base_fn": EfficientNetB0,
            "prep_fn": preprocess_eff,
            "weights": mild_malignant_dict,
            "dropout": 0.2,
            "unfreeze_layers": 15,
            "stage1_epochs": 8,
            "stage2_epochs": 15
        },
        {
            "id": "v4_mobilenetv2_headonly",
            "name": "V4-E: MobileNetV2 (Frozen Backbone / Mild Weighting 1.25x)",
            "base_fn": MobileNetV2,
            "prep_fn": preprocess_mnv2,
            "weights": mild_malignant_dict,
            "dropout": 0.2,
            "unfreeze_layers": 0,
            "stage1_epochs": 25,
            "stage2_epochs": 0
        }
    ]
    
    val_results = {}
    
    print("\n" + "="*70)
    print("STEP 1: TRAINING V4 VARIANTS & HYPERPARAMETER SELECTION ON VALIDATION SET")
    print("="*70)
    
    for cfg in v4_configs:
        cid = cfg["id"]
        cname = cfg["name"]
        print(f"\n---> Training Variant: {cname} ({cid})")
        
        X_train_prep = cfg["prep_fn"](X_train.copy())
        X_val_prep = cfg["prep_fn"](X_val.copy())
        
        model, base_model = build_v4_model(cfg["base_fn"], dropout_rate=cfg["dropout"])
        
        # STAGE 1: Head Warmup / Training
        print(f"  Stage 1: Head training for {cfg['stage1_epochs']} epochs...")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4 if cfg["stage2_epochs"] > 0 else 1e-3),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        early_stopping_stage1 = EarlyStopping(
            monitor='val_loss',
            patience=6,
            restore_best_weights=True
        )
        
        callbacks_stage1 = [early_stopping_stage1] if cfg["stage2_epochs"] == 0 else []
        
        model.fit(
            X_train_prep, y_train,
            validation_data=(X_val_prep, y_val),
            epochs=cfg["stage1_epochs"],
            batch_size=32,
            class_weight=cfg["weights"],
            callbacks=callbacks_stage1,
            verbose=1
        )
        
        # STAGE 2: Fine-tuning (if unfreeze_layers > 0)
        if cfg["unfreeze_layers"] > 0 and cfg["stage2_epochs"] > 0:
            print(f"  Stage 2: Conservative fine-tuning top {cfg['unfreeze_layers']} layers of backbone...")
            base_model.trainable = True
            
            for layer in base_model.layers[:-cfg["unfreeze_layers"]]:
                layer.trainable = False
                
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
            
            lr_scheduler = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1
            )
            
            model.fit(
                X_train_prep, y_train,
                validation_data=(X_val_prep, y_val),
                epochs=cfg["stage2_epochs"],
                batch_size=32,
                class_weight=cfg["weights"],
                callbacks=[early_stopping, lr_scheduler],
                verbose=1
            )
        
        # Evaluate on VALIDATION split ONLY for selection
        v_metrics = evaluate_metrics(model, X_val, y_val, classes, cfg["prep_fn"])
        val_results[cid] = v_metrics
        
        print(f"\n  Validation Metrics for {cid}:")
        print(f"    Val Accuracy           : {v_metrics['accuracy']*100:.2f}%")
        print(f"    Val Macro F1           : {v_metrics['macro_f1']*100:.2f}%")
        print(f"    Val Malignant Precision: {v_metrics['malignant']['precision']*100:.2f}%")
        print(f"    Val Malignant Recall   : {v_metrics['malignant']['recall']*100:.2f}%")
        print(f"    Val Malignant F1       : {v_metrics['malignant']['f1']*100:.2f}%")
        print(f"    Val Malignant FN / FP  : {v_metrics['false_negatives']} / {v_metrics['false_positives']}")
        
        # Save model and metadata
        m_path = os.path.join(candidates_dir, f"{cid}.keras")
        meta_path = os.path.join(candidates_dir, f"{cid}_metadata.json")
        
        model.save(m_path)
        metadata = {
            "model_version": "4.0.0",
            "variant_id": cid,
            "variant_name": cname,
            "architecture": cfg["base_fn"].__name__,
            "class_mapping": classes,
            "image_size": [224, 224],
            "split_counts": {
                "train": train_counts,
                "validation": val_counts,
                "test": test_counts
            },
            "weight_configuration": str(cfg["weights"]),
            "validation_metrics": v_metrics,
            "training_date": datetime.now().isoformat()
        }
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"  Saved candidate model to: {m_path}")

    # Save validation summary JSON
    val_summary_path = os.path.join(backend_dir, "v4_validation_summary.json")
    with open(val_summary_path, 'w') as f:
        json.dump({
            "split_counts": {"train": train_counts, "val": val_counts, "test": test_counts},
            "validation_results": val_results
        }, f, indent=4)
    print(f"\nAll V4 candidates trained and validation results recorded to: {val_summary_path}")

if __name__ == "__main__":
    main()
