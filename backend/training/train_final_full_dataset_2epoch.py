import os
import sys
import json
import cv2
import hashlib
import random
import numpy as np
import tensorflow as tf

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as preprocess_mnv2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def compute_sha256(filepath):
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest().lower()

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
    if img is None:
        raise ValueError(f"Could not read image at {filepath}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return letterbox_resize(img_rgb, (224, 224))

def load_and_clean_dataset(bcd_root):
    manifest_path = os.path.join(bcd_root, "backend", "data", "dataset_manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found at {manifest_path}")
        
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
        
    exclude_files = {"malignant (145).png", "benign (433).png"}
    classes = ['benign', 'malignant', 'normal']
    class_to_idx = {c: i for i, c in enumerate(classes)}
    
    scans = []
    class_counts = {c: 0 for c in classes}
    
    for item in manifest:
        rel_path = item["image_path"]
        fname = os.path.basename(rel_path)
        c_name = item["class_label"]
        
        if fname in exclude_files:
            continue
            
        full_path = os.path.join(bcd_root, rel_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Image missing: {full_path}")
            
        scans.append({
            "filename": fname,
            "filepath": full_path,
            "class_name": c_name,
            "class_idx": class_to_idx[c_name]
        })
        class_counts[c_name] += 1
        
    return scans, classes, class_counts

def build_mobilenetv2_model(input_shape=(224, 224, 3), dropout_rate=0.20):
    inputs = Input(shape=input_shape)
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomTranslation(height_factor=0.03, width_factor=0.03),
        tf.keras.layers.RandomZoom(0.03)
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

def main():
    set_seed(42)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.dirname(script_dir)
    bcd_root = os.path.dirname(backend_dir)
    
    prod_model_path = os.path.join(backend_dir, "models", "breast_image_classifier.keras")
    v11_path = os.path.join(backend_dir, "models", "candidates", "v11_mobilenetv2_b.keras")
    v13_path = os.path.join(backend_dir, "models", "candidates", "v13_efficientnet_b0.keras")
    
    candidate_model_path = os.path.join(backend_dir, "models", "candidates", "final_full_dataset_2epoch_mobilenetv2.keras")
    candidate_meta_path = os.path.join(backend_dir, "models", "candidates", "final_full_dataset_2epoch_mobilenetv2_metadata.json")
    report_path = os.path.join(backend_dir, "models", "evaluation_reports", "final_full_dataset_2epoch_report.md")
    
    EXPECTED_V5B_SHA256 = "93b3c106cff51993b605220c98cfaf54c8fa404bb581656fa11337a26b32e03c"
    
    print("=" * 70)
    print("STEP 1: PRE-TRAINING AUDIT & PROTECTION VERIFICATION")
    print("=" * 70)
    
    v5b_sha_pre = compute_sha256(prod_model_path)
    if v5b_sha_pre != EXPECTED_V5B_SHA256:
        print(f"CRITICAL ERROR: Production V5-B SHA256 mismatch! Found: {v5b_sha_pre}, Expected: {EXPECTED_V5B_SHA256}")
        sys.exit(1)
    print(f"[VERIFIED] Production V5-B SHA256: {v5b_sha_pre}")
    
    v11_exists = os.path.exists(v11_path)
    v13_exists = os.path.exists(v13_path)
    v11_sha_pre = compute_sha256(v11_path) if v11_exists else "N/A"
    v13_sha_pre = compute_sha256(v13_path) if v13_exists else "N/A"
    print(f"[VERIFIED] V11 Candidate exists: {v11_exists} (SHA256: {v11_sha_pre})")
    print(f"[VERIFIED] V13 Candidate exists: {v13_exists} (SHA256: {v13_sha_pre})")
    
    scans, classes, class_counts = load_and_clean_dataset(bcd_root)
    total_scans = len(scans)
    print(f"[AUDIT] Total Clean Scans: {total_scans}")
    print(f"[AUDIT] Class Distribution: {class_counts}")
    
    if total_scans != 776:
        print(f"CRITICAL ERROR: Expected exactly 776 clean scans, but found {total_scans}. STOPPING.")
        sys.exit(1)
    print("[VERIFIED] Exactly 776 eligible scans confirmed.")
    
    print("\n" + "=" * 70)
    print("STEP 2: PREPROCESSING ALL 776 TRAINABLE IMAGES")
    print("=" * 70)
    
    images_raw = []
    labels_raw = []
    for s in scans:
        img = preprocess_letterbox(s["filepath"])
        images_raw.append(img)
        labels_raw.append(s["class_idx"])
        
    X_all = np.array(images_raw, dtype=np.float32)
    y_all = np.array(labels_raw, dtype=np.int32)
    
    X_prep = preprocess_mnv2(X_all.copy())
    print(f"Processed feature shape: {X_prep.shape}, label shape: {y_all.shape}")
    
    print("\n" + "=" * 70)
    print("STEP 3: EXECUTING 2 COMPLETE EPOCHS OF MOBILENETV2 TRAINING")
    print("=" * 70)
    
    batch_size = 32
    steps_per_epoch = int(np.ceil(total_scans / batch_size))
    print(f"Total training samples: {total_scans}")
    print(f"Batch size: {batch_size}")
    print(f"Steps per epoch: {steps_per_epoch}")
    
    model, base_model = build_mobilenetv2_model()
    
    # Epoch 1: Warmup Dense Head
    print("\n--- [Epoch 1/2] Training Dense Head (LR=5e-4) ---")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss="sparse_categorical_crossentropy",
        metrics=['accuracy']
    )
    h1 = model.fit(
        X_prep, y_all,
        epochs=1,
        batch_size=batch_size,
        verbose=1
    )
    e1_loss = float(h1.history['loss'][0])
    e1_acc = float(h1.history['accuracy'][0])
    print(f"Epoch 1 Results -> Loss: {e1_loss:.4f}, Accuracy: {e1_acc:.4f} ({e1_acc*100:.2f}%)")
    
    # Epoch 2: Fine-Tuning Top 15 Layers of Backbone
    print("\n--- [Epoch 2/2] Fine-Tuning Top 15 Backbone Layers (LR=1e-4) ---")
    base_model.trainable = True
    for layer in base_model.layers[:-15]:
        layer.trainable = False
        
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=['accuracy']
    )
    h2 = model.fit(
        X_prep, y_all,
        epochs=1,
        batch_size=batch_size,
        verbose=1
    )
    e2_loss = float(h2.history['loss'][0])
    e2_acc = float(h2.history['accuracy'][0])
    print(f"Epoch 2 Results -> Loss: {e2_loss:.4f}, Accuracy: {e2_acc:.4f} ({e2_acc*100:.2f}%)")
    
    print("\n" + "=" * 70)
    print("STEP 4: SAVING CANDIDATE MODEL AND METADATA")
    print("=" * 70)
    
    os.makedirs(os.path.dirname(candidate_model_path), exist_ok=True)
    model.save(candidate_model_path)
    print(f"Saved candidate model to: {candidate_model_path}")
    
    cand_size_bytes = os.path.getsize(candidate_model_path)
    cand_sha256 = compute_sha256(candidate_model_path)
    
    metadata = {
        "model_name": "final_full_dataset_2epoch_mobilenetv2",
        "candidate_file": "backend/models/candidates/final_full_dataset_2epoch_mobilenetv2.keras",
        "architecture": "MobileNetV2 Transfer Learning",
        "input_shape": [224, 224, 3],
        "output_classes": classes,
        "class_mapping": {"benign": 0, "malignant": 1, "normal": 2},
        "class_counts": class_counts,
        "total_training_samples": total_scans,
        "epochs_trained": 2,
        "steps_per_epoch": steps_per_epoch,
        "batch_size": batch_size,
        "preprocessing": "aspect_ratio_letterbox_224x224_mnv2",
        "epoch_metrics": [
            {"epoch": 1, "loss": e1_loss, "accuracy": e1_acc},
            {"epoch": 2, "loss": e2_loss, "accuracy": e2_acc}
        ],
        "file_size_bytes": cand_size_bytes,
        "sha256": cand_sha256,
        "status": "FULL-DATASET EXPERIMENTAL CANDIDATE",
        "held_out_evaluation": "NONE - Trained on 100% of 776 eligible BUSI scans. Historical and grouped test sets cannot be used as independent benchmarks."
    }
    
    with open(candidate_meta_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved metadata to: {candidate_meta_path}")
    
    print("\n" + "=" * 70)
    print("STEP 5: RUNTIME VERIFICATION & SMOKE TEST")
    print("=" * 70)
    
    # 1. Model Loading Test
    print("[RUN_CHECK 1/4] Model Loading Test...")
    loaded_model = tf.keras.models.load_model(candidate_model_path)
    print("Successfully loaded candidate model.")
    
    # 2. Output Shape Test
    print("[RUN_CHECK 2/4] Output Shape Test...")
    dummy_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
    dummy_output = loaded_model.predict(dummy_input, verbose=0)
    output_shape = dummy_output.shape
    print(f"Dummy output shape: {output_shape} (Expected: (1, 3))")
    assert output_shape == (1, 3), f"Expected (1, 3), got {output_shape}"
    
    # 3. Probability Validity Test
    print("[RUN_CHECK 3/4] Probability Validity Test...")
    prob_min = float(np.min(dummy_output))
    prob_max = float(np.max(dummy_output))
    prob_sum = float(np.sum(dummy_output))
    print(f"Probabilities min: {prob_min:.6f}, max: {prob_max:.6f}, sum: {prob_sum:.6f}")
    assert prob_min >= 0.0, "Negative probability detected"
    assert abs(prob_sum - 1.0) < 1e-4, f"Probabilities do not sum to 1.0: sum={prob_sum}"
    
    # 4. Representative Inference Smoke Test
    print("[RUN_CHECK 4/4] Representative Inference Smoke Test...")
    sample_indices = [0, 10, 450, 500, 700]
    sample_preds = []
    for idx in sample_indices:
        s = scans[idx]
        sample_img = preprocess_mnv2(np.expand_dims(preprocess_letterbox(s["filepath"]), axis=0).astype(np.float32))
        p = loaded_model.predict(sample_img, verbose=0)[0]
        pred_class = classes[int(np.argmax(p))]
        sample_preds.append({
            "filename": s["filename"],
            "true_class": s["class_name"],
            "predicted_class": pred_class,
            "probabilities": {c: float(p[i]) for i, c in enumerate(classes)}
        })
        print(f"  Image: {s['filename']} ({s['class_name']}) -> Predicted: {pred_class} (Probs: {np.round(p, 4)})")
        
    print("\n" + "=" * 70)
    print("STEP 6: POST-TRAINING PROTECTION RE-VERIFICATION")
    print("=" * 70)
    
    v5b_sha_post = compute_sha256(prod_model_path)
    if v5b_sha_post != EXPECTED_V5B_SHA256:
        print(f"CRITICAL ERROR: Production V5-B SHA256 altered! Pre: {v5b_sha_pre}, Post: {v5b_sha_post}")
        sys.exit(1)
    print(f"[RE-VERIFIED] Production V5-B SHA256 unchanged: {v5b_sha_post}")
    
    v11_sha_post = compute_sha256(v11_path) if v11_exists else "N/A"
    v13_sha_post = compute_sha256(v13_path) if v13_exists else "N/A"
    if v11_sha_pre != v11_sha_post or v13_sha_pre != v13_sha_post:
        print("CRITICAL ERROR: V14 candidate files were altered!")
        sys.exit(1)
    print(f"[RE-VERIFIED] V11 SHA256 unchanged: {v11_sha_post}")
    print(f"[RE-VERIFIED] V13 SHA256 unchanged: {v13_sha_post}")
    
    print("\n" + "=" * 70)
    print("STEP 7: GENERATING FINAL REPORT")
    print("=" * 70)
    
    report_content = f"""# Final Full-Dataset 2-Epoch MobileNetV2

**Date**: September 16, 2026  
**Project**: Breast Cancer Detection (BCD) Platform  
**Candidate Model File**: `backend/models/candidates/final_full_dataset_2epoch_mobilenetv2.keras`  
**Candidate Metadata**: `backend/models/candidates/final_full_dataset_2epoch_mobilenetv2_metadata.json`  

---

## Dataset
- **Total Eligible Scans**: 776 clean, eligible BUSI scans
- **Class Distribution**:
  - `benign`: {class_counts['benign']} ({class_counts['benign']/776*100:.2f}%)
  - `malignant`: {class_counts['malignant']} ({class_counts['malignant']/776*100:.2f}%)
  - `normal`: {class_counts['normal']} ({class_counts['normal']/776*100:.2f}%)
- **Preprocessing**: Aspect-Ratio Letterbox resize to $224 \\times 224 \\times 3$, RGB conversion, MobileNetV2 `preprocess_input`.

---

## Training
- **Architecture**: MobileNetV2 Transfer Learning (ImageNet backbone + GAP + Dropout(0.2) + Softmax Dense Head)
- **Input Size**: $224 \\times 224 \\times 3$
- **Epochs Completed**: 2 Complete Epochs
  - *Epoch 1 (Dense Head Warmup)*: LR = `5e-4`, Adam, Loss = `{e1_loss:.4f}`, Accuracy = `{e1_acc*100:.2f}%`
  - *Epoch 2 (Top 15 Backbone Layers Fine-Tuning)*: LR = `1e-4`, Adam, Loss = `{e2_loss:.4f}`, Accuracy = `{e2_acc*100:.2f}%`
- **Total Training Samples**: 776 scans (100% of dataset used for training weight updates)
- **Batch Size**: {batch_size}
- **Steps per Epoch**: {steps_per_epoch} steps

---

## Evaluation Limitation
> [!WARNING]
> **Evaluation Limitation Statement**:
> "This model was trained using all 776 eligible BUSI scans. Therefore the historical 117-image and grouped 107-image frozen sets cannot be used as independent held-out evaluation sets for this model."

---

## Runtime Verification
- **Model Loading Test**: PASSED (Loaded cleanly via `tf.keras.models.load_model`)
- **Output Shape Test**: PASSED (Dummy output shape: `(1, 3)`)
- **Probability Validity Test**: PASSED (Non-negative softmax values, sum = `{prob_sum:.6f}`)
- **Inference Smoke Test**: PASSED (5/5 sample images produced valid categorical probability distributions)

---

## Model Integrity
- **Candidate Path**: `backend/models/candidates/final_full_dataset_2epoch_mobilenetv2.keras`
- **File Size**: {cand_size_bytes:,} bytes ({cand_size_bytes/(1024*1024):.2f} MB)
- **Candidate SHA256**: `{cand_sha256}`
- **Candidate Metadata Path**: `backend/models/candidates/final_full_dataset_2epoch_mobilenetv2_metadata.json`

---

## Production Protection
- **Production V5-B Status**: UNTOUCHED
  - Path: `backend/models/breast_image_classifier.keras`
  - SHA256 Pre-Training: `{v5b_sha_pre}`
  - SHA256 Post-Training: `{v5b_sha_post}`
  - Match: **VERIFIED MATCH**
- **V14 Research Components Status**: UNTOUCHED
  - `v11_mobilenetv2_b.keras` SHA256: `{v11_sha_post}`
  - `v13_efficientnet_b0.keras` SHA256: `{v13_sha_post}`

---

## Status
- **Classification**: `FULL-DATASET EXPERIMENTAL CANDIDATE`
- **Deployment Status**: Not promoted to production; production inference remains unchanged.
- **Generalization Claim**: No held-out performance claims are made because all 776 eligible scans were included in the training dataset.
"""

    with open(report_path, "w") as f:
        f.write(report_content)
    print(f"Saved evaluation report to: {report_path}")
    print("\nTRAINING AND VERIFICATION COMPLETE SUCCESSFULLY.")

if __name__ == "__main__":
    main()
