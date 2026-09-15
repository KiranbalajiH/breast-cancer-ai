import os
import cv2
import numpy as np
import json
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from datetime import datetime

# Import applications & preprocess_inputs
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as preprocess_mnv2
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input as preprocess_eff
from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input as preprocess_dense

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
    
    # Files to exclude (exact duplicates with label conflict found in Phase B)
    exclude_files = {
        "malignant (145).png",
        "benign (433).png"
    }
    
    excluded_count = 0
    
    for class_idx, class_name in enumerate(classes):
        class_dir = os.path.join(dataset_root, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        class_files = os.listdir(class_dir)
        for f in class_files:
            if not f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                continue
            
            # Exclude mask files
            if '_mask' in f.lower() or 'mask' in f.lower():
                continue
                
            # Exclude exact duplicates
            if f in exclude_files:
                print(f"Excluding duplicate/conflicted file: {f}")
                excluded_count += 1
                continue
                
            full_path = os.path.join(class_dir, f)
            image_paths.append(full_path)
            labels.append(class_idx)
            
    print(f"Loaded {len(image_paths)} clean original images (excluded {excluded_count} duplicate files).")
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

def build_transfer_learning_model(base_model_fn, input_shape=(224, 224, 3), dropout_rate=0.3):
    # Medically reasonable data augmentation
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05), # max ~10 degrees
        tf.keras.layers.RandomTranslation(height_factor=0.05, width_factor=0.05), # max ~11 pixels
        tf.keras.layers.RandomZoom(0.05) # max 5% zoom
    ])
    
    inputs = Input(shape=input_shape)
    x = data_augmentation(inputs)
    
    # Load base model with frozen weights
    base_model = base_model_fn(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False
    
    # x = base_model(x, training=False) is crucial to keep BatchNormalization layers in inference mode
    # during fine-tuning (stage 2), which prevents ruining pre-trained BatchNorm statistics.
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    if dropout_rate > 0:
        x = Dropout(dropout_rate)(x)
    
    outputs = Dense(3, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model, base_model

def evaluate_model_on_test(model, X_test, y_test, classes, test_paths=None):
    predictions = model.predict(X_test, verbose=0)
    pred_classes = np.argmax(predictions, axis=1)
    
    acc = accuracy_score(y_test, pred_classes)
    prec, rec, f1, support = precision_recall_fscore_support(y_test, pred_classes, labels=[0, 1, 2], zero_division=0)
    cm = confusion_matrix(y_test, pred_classes, labels=[0, 1, 2])
    
    # Calculate false negatives and false positives for malignant (class 1)
    mal_indices = np.where(y_test == 1)[0]
    fn_count = np.sum(pred_classes[mal_indices] != 1)
    
    non_mal_indices = np.where(y_test != 1)[0]
    fp_count = np.sum(pred_classes[non_mal_indices] == 1)
    
    incorrect_count = int(np.sum(pred_classes != y_test))
    
    misclassifications = []
    if test_paths is not None:
        for idx in range(len(y_test)):
            if pred_classes[idx] != y_test[idx]:
                misclassifications.append({
                    "filename": os.path.basename(test_paths[idx]),
                    "true_class": classes[y_test[idx]],
                    "predicted_class": classes[pred_classes[idx]],
                    "confidence": float(predictions[idx][pred_classes[idx]])
                })
                
    return {
        "accuracy": float(acc),
        "precision": [float(p) for p in prec],
        "recall": [float(r) for r in rec],
        "f1": [float(f) for f in f1],
        "confusion_matrix": cm.tolist(),
        "false_negatives": int(fn_count),
        "false_positives": int(fp_count),
        "incorrect_count": incorrect_count,
        "misclassifications": misclassifications
    }

def main():
    set_seed(42)
    
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_root = os.path.join(os.path.dirname(backend_dir), "dataset", "BUSI")
    candidates_dir = os.path.join(backend_dir, "models", "candidates")
    os.makedirs(candidates_dir, exist_ok=True)
    
    print("="*60)
    print("PHASE C: V3 TRAINING PIPELINE - STAGED FINE-TUNING")
    print("="*60)
    
    # 1. Load Clean Dataset
    image_paths, labels, classes = load_and_clean_dataset(dataset_root)
    labels = np.array(labels, dtype=np.int32)
    
    # 2. Load Raw Images into Memory
    print("\nLoading images into memory...")
    X_raw = load_images_to_numpy(image_paths)
    print(f"Loaded image array shape: {X_raw.shape}")
    
    # 3. Create Stratified Splits (70/15/15)
    X_train, X_val_test, y_train, y_val_test = train_test_split(
        X_raw, labels, test_size=0.30, stratify=labels, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_val_test, y_val_test, test_size=0.50, stratify=y_val_test, random_state=42
    )
    
    paths_train, paths_val_test = train_test_split(
        image_paths, test_size=0.30, stratify=labels, random_state=42
    )
    paths_val, paths_test = train_test_split(
        paths_val_test, test_size=0.50, stratify=y_val_test, random_state=42
    )
    
    print(f"\nDataset Split Sizes:")
    print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    
    # 4. Class Balancing Configuration
    # Standard balanced weights:
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    # Mild class weights (raised to power of 0.5 to damp extreme class weighting values):
    class_weights_mild = class_weights ** 0.5
    class_weights_mild_dict = {i: float(class_weights_mild[i]) for i in range(len(class_weights_mild))}
    
    print("\nClass weights:")
    for idx, c_name in enumerate(classes):
        print(f"  - {c_name:<10}: Balanced = {class_weights[idx]:.4f} | Mild (Dampened) = {class_weights_mild_dict[idx]:.4f}")
        
    candidates = {
        "MobileNetV2": {
            "base_fn": MobileNetV2,
            "preprocess_fn": preprocess_mnv2,
        },
        "EfficientNetB0": {
            "base_fn": EfficientNetB0,
            "preprocess_fn": preprocess_eff,
        },
        "DenseNet121": {
            "base_fn": DenseNet121,
            "preprocess_fn": preprocess_dense,
        }
    }
    
    weight_configs = {
        "noweights": None,
        "mildweights": class_weights_mild_dict
    }
    
    for name, config in candidates.items():
        for weight_name, weight_dict in weight_configs.items():
            candidate_id = f"{name.lower()}_v3_{weight_name}"
            print(f"\n" + "="*50)
            print(f"TRAINING CANDIDATE: {name} ({weight_name.upper()})")
            print("="*50)
            
            # Preprocess splits specifically for this backbone architecture
            X_train_prep = config["preprocess_fn"](X_train.copy())
            X_val_prep = config["preprocess_fn"](X_val.copy())
            X_test_prep = config["preprocess_fn"](X_test.copy())
            
            # Build transfer learning model with Dropout=0.3
            model, base_model = build_transfer_learning_model(config["base_fn"], dropout_rate=0.3)
            
            # ----------------- STAGE 1: HEAD WARMUP -----------------
            print("\n--- Stage 1: Warmup classification head (backbone frozen) ---")
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            model.fit(
                X_train_prep, y_train,
                validation_data=(X_val_prep, y_val),
                epochs=4,
                batch_size=32,
                class_weight=weight_dict,
                verbose=1
            )
            
            # ----------------- STAGE 2: BACKBONE FINE-TUNING -----------------
            print("\n--- Stage 2: Fine-tuning entire backbone (with learning rate scheduling) ---")
            # Unfreeze the backbone
            base_model.trainable = True
            
            # Compile with very low learning rate for fine-tuning
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=4,
                restore_best_weights=True
            )
            
            lr_scheduler = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                min_lr=1e-6,
                verbose=1
            )
            
            model.fit(
                X_train_prep, y_train,
                validation_data=(X_val_prep, y_val),
                epochs=12,
                batch_size=32,
                class_weight=weight_dict,
                callbacks=[early_stopping, lr_scheduler],
                verbose=1
            )
            
            # 5. Evaluate on Validation split
            eval_metrics = evaluate_model_on_test(model, X_val_prep, y_val, classes, test_paths=paths_val)
            print(f"\nValidation metrics for {name} ({weight_name}):")
            print(f"  Accuracy: {eval_metrics['accuracy']:.4f}")
            print(f"  Malignant Recall: {eval_metrics['recall'][1]:.4f}")
            print(f"  Malignant Precision: {eval_metrics['precision'][1]:.4f}")
            print(f"  False Negatives: {eval_metrics['false_negatives']}")
            print(f"  False Positives: {eval_metrics['false_positives']}")
            
            # 6. Save model and metadata
            model_path = os.path.join(candidates_dir, f"{candidate_id}.keras")
            metadata_path = os.path.join(candidates_dir, f"{candidate_id}_metadata.json")
            
            model.save(model_path)
            print(f"Saved model to: {model_path}")
            
            metadata = {
                "model_version": "3.0.0",
                "architecture": name,
                "weight_configuration": weight_name,
                "class_mapping": classes,
                "image_size": [224, 224],
                "preprocessing_method": f"Keras {name} Native Preprocessing",
                "dataset_sources_used": ["BUSI"],
                "split_counts": {
                    "train": int(len(y_train)),
                    "validation": int(len(y_val)),
                    "test": int(len(y_test))
                },
                "augmentation_summary": {
                    "horizontal_flip": True,
                    "max_rotation": 0.05,
                    "max_translation": 0.05,
                    "max_zoom": 0.05
                },
                "dropout_rate": 0.3,
                "staged_training": {
                    "stage1_warmup_epochs": 8,
                    "stage1_lr": 5e-4,
                    "stage2_finetune_epochs": 30,
                    "stage2_lr": 1e-5,
                    "early_stopping_patience": 7,
                    "lr_reduction_patience": 3
                },
                "class_balancing_method": "Mild/None Loss Weighting" if weight_name == "mildweights" else "None",
                "validation_evaluation": eval_metrics,
                "training_date": datetime.now().isoformat()
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            print(f"Saved metadata to: {metadata_path}")
            
    print("\nAll candidate models have been trained and saved successfully.")

if __name__ == "__main__":
    main()
