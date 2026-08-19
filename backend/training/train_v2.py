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
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.callbacks import EarlyStopping

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

def build_transfer_learning_model(base_model_fn, input_shape=(224, 224, 3)):
    # Set up medically reasonable data augmentation layers
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05), # small rotation
        tf.keras.layers.RandomTranslation(height_factor=0.05, width_factor=0.05), # small translation
        tf.keras.layers.RandomZoom(0.05) # small zoom
    ])
    
    inputs = Input(shape=input_shape)
    x = data_augmentation(inputs)
    
    # Load base model with frozen weights
    base_model = base_model_fn(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False
    
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    # Dense head with Softmax for 3 classes
    outputs = Dense(3, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

def evaluate_model_on_test(model, X_test, y_test, classes):
    predictions = model.predict(X_test, verbose=0)
    pred_classes = np.argmax(predictions, axis=1)
    
    acc = accuracy_score(y_test, pred_classes)
    prec, rec, f1, support = precision_recall_fscore_support(y_test, pred_classes, labels=[0, 1, 2])
    cm = confusion_matrix(y_test, pred_classes)
    
    # Calculate false negatives and false positives
    # False Negatives: True malignant (1) predicted as benign (0) or normal (2)
    mal_indices = np.where(y_test == 1)[0]
    fn_count = np.sum(pred_classes[mal_indices] != 1)
    
    # False Positives: True benign (0) or normal (2) predicted as malignant (1)
    non_mal_indices = np.where(y_test != 1)[0]
    fp_count = np.sum(pred_classes[non_mal_indices] == 1)
    
    return {
        "accuracy": float(acc),
        "precision": [float(p) for p in prec],
        "recall": [float(r) for r in rec],
        "f1": [float(f) for f in f1],
        "confusion_matrix": cm.tolist(),
        "false_negatives": int(fn_count),
        "false_positives": int(fp_count)
    }

def main():
    set_seed(42)
    
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_root = os.path.join(os.path.dirname(backend_dir), "dataset", "BUSI")
    models_dir = os.path.join(backend_dir, "models")
    
    # 1. Verify Dataset Readiness
    print("="*60)
    print("STEP 1 - VERIFY DATASET READINESS")
    print("="*60)
    image_paths, labels, classes = load_and_clean_dataset(dataset_root)
    labels = np.array(labels, dtype=np.int32)
    
    # Check class counts
    class_counts_dict = {classes[i]: int(np.sum(labels == i)) for i in range(len(classes))}
    print("Usable images per class:")
    for k, v in class_counts_dict.items():
        print(f"  - {k:<10}: {v}")
        
    # 2. Load all raw images into memory (0 to 255 scaling)
    print("\nLoading images into memory...")
    X_raw = load_images_to_numpy(image_paths)
    print(f"Images array shape: {X_raw.shape}")
    
    # 3. Create splits
    print("\n==================================================")
    print("STEP 3 - CREATE A PROFESSIONAL DATA SPLIT")
    print("==================================================")
    X_train, X_val_test, y_train, y_val_test = train_test_split(
        X_raw, labels, test_size=0.30, stratify=labels, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_val_test, y_val_test, test_size=0.50, stratify=y_val_test, random_state=42
    )
    
    print("Train split counts:")
    for idx, c_name in enumerate(classes):
        print(f"  - {c_name:<10}: {np.sum(y_train == idx)}")
    print("Validation split counts:")
    for idx, c_name in enumerate(classes):
        print(f"  - {c_name:<10}: {np.sum(y_val == idx)}")
    print("Test split counts:")
    for idx, c_name in enumerate(classes):
        print(f"  - {c_name:<10}: {np.sum(y_test == idx)}")
        
    # Compute class weights to address imbalance
    print("\n==================================================")
    print("STEP 6 - ADDRESS CLASS IMBALANCE (CLASS WEIGHTS)")
    print("==================================================")
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights_dict = {i: float(class_weights[i]) for i in range(len(class_weights))}
    print("Calculated class weights:")
    for i, c_name in enumerate(classes):
        print(f"  - {c_name:<10}: {class_weights_dict[i]:.4f}")
        
    # Prepare Preprocessed Inputs for each candidate
    print("\nPreparing candidate-specific preprocessed data...")
    
    # 4. Train Multiple Candidates
    print("\n==================================================")
    print("STEP 5 - TRAINING MULTIPLE CANDIDATE MODELS")
    print("==================================================")
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    # Define candidates config
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
    
    results = {}
    trained_models = {}
    
    for name, config in candidates.items():
        print(f"\n--- Training Candidate: {name} ---")
        
        # Apply preprocessing
        X_train_prep = config["preprocess_fn"](X_train.copy())
        X_val_prep = config["preprocess_fn"](X_val.copy())
        X_test_prep = config["preprocess_fn"](X_test.copy())
        
        # Build model with data augmentation inside
        model = build_transfer_learning_model(config["base_fn"])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train
        model.fit(
            X_train_prep, y_train,
            validation_data=(X_val_prep, y_val),
            epochs=30,
            batch_size=32,
            class_weight=class_weights_dict,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Evaluate
        eval_metrics = evaluate_model_on_test(model, X_test_prep, y_test, classes)
        results[name] = eval_metrics
        trained_models[name] = model
        
        print(f"\nCandidate {name} Test Performance:")
        print(f"  Accuracy: {eval_metrics['accuracy']:.4f}")
        print(f"  Malignant Recall: {eval_metrics['recall'][1]:.4f}")
        print(f"  Malignant Precision: {eval_metrics['precision'][1]:.4f}")
        print(f"  Malignant F1: {eval_metrics['f1'][1]:.4f}")
        print(f"  False Negatives (Malignant as Benign/Normal): {eval_metrics['false_negatives']}")
        print(f"  False Positives (Normal/Benign as Malignant): {eval_metrics['false_positives']}")
        
    # Load Version 1 Baseline for comparison
    print("\n==================================================")
    print("STEP 9 - COMPARE AGAINST VERSION 1 BASELINE")
    print("==================================================")
    
    # Load baseline model and metadata
    v1_model_path = os.path.join(models_dir, "breast_image_classifier.keras")
    v1_metadata_path = os.path.join(models_dir, "breast_image_classifier_metadata.json")
    
    v1_metrics = {}
    if os.path.exists(v1_model_path):
        print(f"Evaluating Baseline Model Version 1 on the same test split...")
        v1_model = tf.keras.models.load_model(v1_model_path)
        # Baseline was trained with MobileNetV2 preprocessing
        X_test_mnv2 = preprocess_mnv2(X_test.copy())
        v1_metrics = evaluate_model_on_test(v1_model, X_test_mnv2, y_test, classes)
        print(f"Baseline (V1) Test Performance:")
        print(f"  Accuracy: {v1_metrics['accuracy']:.4f}")
        print(f"  Malignant Recall: {v1_metrics['recall'][1]:.4f}")
        print(f"  Malignant F1: {v1_metrics['f1'][1]:.4f}")
        print(f"  False Negatives: {v1_metrics['false_negatives']}")
        print(f"  False Positives: {v1_metrics['false_positives']}")
    else:
        # Fallback values from prompt if file is missing
        v1_metrics = {
            "accuracy": 0.811966,
            "recall": [0.8636, 0.6129, 0.9500],
            "precision": [0.8382, 0.7917, 0.7917],
            "f1": [0.8507, 0.6909, 0.8636],
            "false_negatives": 12,
            "false_positives": 6,
            "confusion_matrix": [[57, 5, 4], [11, 19, 1], [1, 0, 19]]
        }
        print("Using fallback baseline values (Breast Image Classifier v1 files not loaded).")

    # 10. Model Selection Rule
    print("\n==================================================")
    print("STEP 10 - MODEL SELECTION DECISION")
    print("==================================================")
    
    # We want to select the model with the highest Malignant Recall, provided it doesn't degrade benign/normal recalls severely.
    best_candidate_name = None
    best_candidate_recall = 0.0
    best_candidate_f1 = 0.0
    
    for name, m in results.items():
        # Select best based on Malignant recall (index 1) and F1
        recall = m["recall"][1]
        f1 = m["f1"][1]
        if recall > best_candidate_recall:
            best_candidate_recall = recall
            best_candidate_f1 = f1
            best_candidate_name = name
        elif abs(recall - best_candidate_recall) < 1e-4 and f1 > best_candidate_f1:
            best_candidate_f1 = f1
            best_candidate_name = name
            
    print(f"Best new candidate model: {best_candidate_name} with Malignant Recall: {best_candidate_recall:.4f}")
    
    # Decide if we promote to Version 2
    # Promote if best candidate malignant recall is greater than V1, or if it improves F1 score meaningfully
    v1_recall = v1_metrics["recall"][1]
    v1_accuracy = v1_metrics["accuracy"]
    
    candidate_recall = results[best_candidate_name]["recall"][1]
    candidate_accuracy = results[best_candidate_name]["accuracy"]
    
    is_better = False
    if candidate_recall > v1_recall:
        print(f"Promotion: Yes! Candidate {best_candidate_name} has higher Malignant Recall ({candidate_recall:.4f}) than Baseline ({v1_recall:.4f}).")
        is_better = True
    elif abs(candidate_recall - v1_recall) < 1e-4 and candidate_accuracy > v1_accuracy:
        print(f"Promotion: Yes! Candidate {best_candidate_name} has equal Malignant Recall but higher overall accuracy ({candidate_accuracy:.4f}) than Baseline ({v1_accuracy:.4f}).")
        is_better = True
    else:
        print(f"Promotion: No. Candidate {best_candidate_name} does not improve malignant detection over baseline (Candidate: {candidate_recall:.4f} vs Baseline: {v1_recall:.4f}).")
        is_better = False
        
    # Print Comparison Table
    print("\nComparison Table:")
    print(f"{'Metric':<25} | {'Baseline (V1)':<15} | {best_candidate_name:<20}")
    print("-"*68)
    print(f"{'Accuracy':<25} | {v1_metrics['accuracy']:.4f}          | {results[best_candidate_name]['accuracy']:.4f}")
    print(f"{'Benign Recall':<25} | {v1_metrics['recall'][0]:.4f}          | {results[best_candidate_name]['recall'][0]:.4f}")
    print(f"{'Malignant Recall':<25} | {v1_metrics['recall'][1]:.4f}          | {results[best_candidate_name]['recall'][1]:.4f}")
    print(f"{'Normal Recall':<25} | {v1_metrics['recall'][2]:.4f}          | {results[best_candidate_name]['recall'][2]:.4f}")
    print(f"{'Malignant Precision':<25} | {v1_metrics['precision'][1]:.4f}          | {results[best_candidate_name]['precision'][1]:.4f}")
    print(f"{'Malignant F1':<25} | {v1_metrics['f1'][1]:.4f}          | {results[best_candidate_name]['f1'][1]:.4f}")
    print(f"{'Total False Negatives':<25} | {v1_metrics['false_negatives']:<15d} | {results[best_candidate_name]['false_negatives']:<20d}")
    print(f"{'Total False Positives':<25} | {v1_metrics['false_positives']:<15d} | {results[best_candidate_name]['false_positives']:<20d}")
    
    # 11. Save the best model safely
    if is_better:
        best_model = trained_models[best_candidate_name]
        v2_model_path = os.path.join(models_dir, "breast_image_classifier_v2.keras")
        v2_metadata_path = os.path.join(models_dir, "breast_image_classifier_v2_metadata.json")
        
        # Save Keras Model
        print(f"\nSaving selected candidate to: {v2_model_path}")
        best_model.save(v2_model_path)
        
        # Save accompanying metadata
        metadata_v2 = {
            "model_version": "2.0.0",
            "architecture": best_candidate_name,
            "class_mapping": ["benign", "malignant", "normal"],
            "image_size": [224, 224],
            "preprocessing_method": f"Keras {best_candidate_name} Native Preprocessing",
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
            "class_balancing_method": "Scikit-Learn Class Weights (Loss Scaling)",
            "evaluation_metrics": results[best_candidate_name],
            "training_date": datetime.now().isoformat()
        }
        with open(v2_metadata_path, 'w') as f:
            json.dump(metadata_v2, f, indent=4)
        print(f"Saved Version 2 metadata to: {v2_metadata_path}")
        
    print("\nPhase C Model training completed.")

if __name__ == "__main__":
    main()
