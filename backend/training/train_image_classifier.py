import os
import cv2
import numpy as np
import json
import random
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def inspect_and_load_dataset(dataset_root):
    print("==================================================")
    print("STEP 1 — INSPECTING BUSI DATASET STRUCTURE")
    print("==================================================")
    
    classes = ['benign', 'malignant', 'normal']
    image_paths = []
    labels = []
    mask_count = 0
    
    for class_idx, class_name in enumerate(classes):
        class_dir = os.path.join(dataset_root, class_name)
        if not os.path.isdir(class_dir):
            print(f"Warning: Class directory {class_dir} not found.")
            continue
            
        class_files = os.listdir(class_dir)
        class_image_count = 0
        for f in class_files:
            if not f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                continue
            
            # Identify and exclude mask files
            if '_mask' in f.lower() or 'mask' in f.lower():
                mask_count += 1
                continue
                
            full_path = os.path.join(class_dir, f)
            image_paths.append(full_path)
            labels.append(class_idx)
            class_image_count += 1
            
        print(f"Class: {class_name:<10} | Original Images: {class_image_count}")
        
    print("-" * 50)
    print(f"Total Original Images: {len(image_paths)}")
    print(f"Total Mask Files Excluded: {mask_count}")
    print("==================================================")
    
    return image_paths, labels, classes, mask_count

def preprocess_images(image_paths, target_size=(224, 224)):
    processed_images = []
    for path in image_paths:
        # Load image (OpenCV loads as BGR)
        img = cv2.imread(path)
        if img is None:
            print(f"Warning: Could not read image {path}. Skipping.")
            continue
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Resize to 224x224
        img_resized = cv2.resize(img_rgb, target_size)
        processed_images.append(img_resized)
        
    # Convert list to numpy array
    x = np.array(processed_images, dtype=np.float32)
    # Apply MobileNetV2 preprocessing
    x = preprocess_input(x)
    return x

def main():
    set_seed(42)
    
    dataset_root = r"C:\Users\kiran\BCD\dataset\Dataset_BUSI_with_GT"
    image_paths, labels, classes, mask_count = inspect_and_load_dataset(dataset_root)
    
    if len(image_paths) == 0:
        print("Error: No images found to train on. Aborting.")
        return
        
    labels = np.array(labels, dtype=np.int32)
    
    # Load and preprocess all original images
    print("\nLoading and preprocessing images...")
    X = preprocess_images(image_paths)
    
    # Stratified Splits: 70% Train, 15% Validation, 15% Test
    # First split off 30% for validation + test
    X_train, X_val_test, y_train, y_val_test = train_test_split(
        X, labels, test_size=0.30, stratify=labels, random_state=42
    )
    # Then split the 30% into validation (50%) and test (50%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_val_test, y_val_test, test_size=0.50, stratify=y_val_test, random_state=42
    )
    
    print("\n==================================================")
    print("STEP 4 — TRAIN/VALIDATION/TEST SPLIT")
    print("==================================================")
    for split_name, split_y in [("TRAIN", y_train), ("VALIDATION", y_val), ("TEST", y_test)]:
        print(f"{split_name}:")
        for idx, class_name in enumerate(classes):
            count = np.sum(split_y == idx)
            print(f"  {class_name}: {count}")
    print("==================================================")
    
    # Build MobileNetV2 Transfer Learning Model
    print("\nBuilding model...")
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  # Freeze pretrained base
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(3, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train the Model
    print("\nTraining model...")
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )
    
    epochs_trained = len(history.history['loss'])
    print(f"\nTraining completed after {epochs_trained} epochs.")
    
    # Save the Best Model
    models_dir = r"C:\Users\kiran\BCD\backend\models"
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "breast_image_classifier.keras")
    model.save(model_path)
    print(f"Model saved to: {model_path}")
    
    # Evaluate on Held-Out Test Split
    print("\n==================================================")
    print("STEP 6 — EVALUATING THE IMAGE MODEL ON TEST SPLIT")
    print("==================================================")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_acc:.4f}")
    
    predictions = model.predict(X_test, verbose=0)
    pred_classes = np.argmax(predictions, axis=1)
    
    # Per-class report
    print("\nClassification Report:")
    print(classification_report(y_test, pred_classes, target_names=classes))
    
    # Confusion matrix
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, pred_classes)
    print(cm)
    print("==================================================")
    
    # Save Metadata
    metadata_path = os.path.join(models_dir, "breast_image_classifier_metadata.json")
    class_counts = {
        classes[0]: int(np.sum(labels == 0)),
        classes[1]: int(np.sum(labels == 1)),
        classes[2]: int(np.sum(labels == 2))
    }
    metadata = {
        "model_type": "MobileNetV2 Transfer Learning",
        "classes": classes,
        "image_size": [224, 224],
        "test_accuracy": float(test_acc),
        "class_counts": class_counts
    }
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata saved to: {metadata_path}")
    
    # Test Real Predictions
    print("\n==================================================")
    print("STEP 7 — TEST REAL PREDICTIONS (5 HELD-OUT IMAGES)")
    print("==================================================")
    
    # Find test image paths (map X_test back to original paths by index in split)
    # We can split the paths array using the same logic or split a tuple of (X, path)
    indices = np.arange(len(image_paths))
    _, indices_val_test = train_test_split(indices, test_size=0.30, stratify=labels, random_state=42)
    _, indices_test = train_test_split(indices_val_test, test_size=0.50, stratify=y_val_test, random_state=42)
    
    test_paths = [image_paths[idx] for idx in indices_test]
    
    # Select 5 random test images
    sample_indices = random.sample(range(len(X_test)), 5)
    for idx in sample_indices:
        filepath = test_paths[idx]
        filename = os.path.basename(filepath)
        true_lbl = classes[y_test[idx]]
        pred_probs = predictions[idx]
        pred_lbl = classes[pred_classes[idx]]
        conf = float(pred_probs[pred_classes[idx]])
        
        print(f"Image: {filename}")
        print(f"True: {true_lbl}")
        print(f"Predicted: {pred_lbl}")
        print(f"Confidence: {conf:.4f}")
        print("Probabilities:")
        for c_idx, c_name in enumerate(classes):
            print(f"  {c_name}: {pred_probs[c_idx]:.4f}")
        print("-" * 40)
    print("==================================================")

if __name__ == "__main__":
    main()
