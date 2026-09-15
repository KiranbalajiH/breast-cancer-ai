import os
import cv2
import numpy as np
import json
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

def load_dataset(dataset_path, img_size=(128, 128)):
    X = []
    y = []
    classes = ["benign", "malignant", "normal"]
    
    for label, class_name in enumerate(classes):
        class_dir = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        print(f"Loading {class_name} images...")
        for filename in os.listdir(class_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')) and "_mask" not in filename.lower():
                img_path = os.path.join(class_dir, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    # Resize and flatten
                    img_resized = cv2.resize(img, img_size)
                    X.append(img_resized.flatten())
                    y.append(label)
                    
    return np.array(X) / 255.0, np.array(y), classes

def main():
    dataset_path = r"C:\Users\kiran\BCD\dataset\Dataset_BUSI_with_GT"
    img_size = (128, 128)
    
    print("="*60)
    print("TRAINING BREAST ULTRASOUND IMAGE CLASSIFIER")
    print("="*60)
    
    X, y, classes = load_dataset(dataset_path, img_size=img_size)
    print(f"Loaded {len(X)} images of size {img_size[0]}x{img_size[1]}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Compare simple models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "SVM (RBF)": SVC(probability=True, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    best_model = None
    best_acc = 0.0
    best_name = ""
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"{name} Accuracy: {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_name = name
            
    print(f"\nSelected best model: {best_name} with {best_acc:.4f} accuracy.")
    
    # Train on full dataset
    print(f"\nRetraining {best_name} on the full dataset...")
    full_model = models[best_name]
    full_model.fit(X, y)
    
    # Save the model
    models_dir = r"C:\Users\kiran\BCD\backend\models"
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "breast_ultrasound_model.joblib")
    joblib.dump(full_model, model_path)
    
    # Save metadata
    metadata = {
        "model_name": best_name,
        "model_type": "ultrasound_image_classifier",
        "model_version": "1.0.0",
        "dataset_name": "Dataset_BUSI_with_GT",
        "training_date": datetime.now().isoformat(),
        "img_size": img_size,
        "target_names": classes,
        "test_accuracy": float(best_acc)
    }
    
    metadata_path = os.path.join(models_dir, "ultrasound_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"Model saved to: {model_path}")
    print(f"Metadata saved to: {metadata_path}")

if __name__ == "__main__":
    main()
