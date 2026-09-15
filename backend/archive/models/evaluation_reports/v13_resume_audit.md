# V13 State Audit & Experiment Design Report

**Date**: September 02, 2026  
**Project**: Breast Cancer AI — Ultrasound Image Classifier (`c:\Users\kiran\BCD`)  
**Audit Purpose**: Complete state audit and V13 hypothesis design prior to executing V13 experiment.  

---

## 1. Production Model State Verification — FIRST

Production model state verified. **No production files were modified, overwritten, or promoted.**

| Artifact Path | Status | File Size | SHA256 Hash | Match |
| :--- | :--- | :---: | :---: | :---: |
| `backend/models/breast_image_classifier.keras` | **ACTIVE PROD (V5-B)** | 18,015,816 bytes | `93b3c106cff51993b605220c98cfaf54c8fa404bb581656fa11337a26b32e03c` | **EXACT** |
| `backend/models/breast_image_classifier_metadata.json` | **INTACT PROD METADATA** | 585 bytes | `ef4e91d5eb859636adfd2db78ee0c7b543384d2047ccc6a39a056796daf65059` | **EXACT** |

> [!IMPORTANT]  
> `breast_image_classifier.keras` remains active V5-B production state. File size and SHA256 hash match the locked verification hash `93b3c106cff51993b605220c98cfaf54c8fa404bb581656fa11337a26b32e03c` exactly. Zero model promotions have taken place.

---

## 2. Dataset & Frozen Test Set Integrity

- **Dataset Directory**: `dataset/BUSI` is **100% intact**. No scan or mask images were created, altered, or deleted.
  - Benign: 435 scans, 451 masks
  - Malignant: 210 scans, 211 masks
  - Normal: 133 scans, 133 masks
- **Clean Trainable Dataset**: 776 scans. Conflicting duplicate scans `benign (433).png` and `malignant (145).png` remain excluded.
- **Locked Historical 117-Image Test Set**: 65 benign, 32 malignant, 20 normal (`random_state=42`). Completely isolated and un-modified.
- **Grouped Generalization Split**: 680 unique perceptual lesion clusters (`seed=42`).

---

## 3. Reusable Infrastructure & Utilities

The project contains established utilities in `backend/training/train_v11.py` and `backend/training/train_v12.py`:
- `load_and_clean_dataset()`: Loads 776 clean scans and applies standard exclusion rules.
- `group_scans_by_perceptual_cluster()` & `create_grouped_stratified_split()`: Generates deterministic grouped train/val/test splits (`seed=42`).
- `preprocess_letterbox()`: Aspect-ratio preserving 224x224 letterbox resize.
- `evaluate_model()`: Computes accuracy, per-class metrics, macro F1, FN, FP, and confusion matrices.
- Threshold calibration sweep logic: Fits decision thresholds strictly on validation data (`grp_val_idx`).

---

## 4. Single Backbone Selection & Technical Rationale

### Selected Backbone: **`EfficientNetB0`**

### Why EfficientNetB0 over MobileNetV2 Variations?
1. **Architecture & Capacity**: MobileNetV2 uses standard inverted residual blocks with depthwise separable convolutions (2.26M parameters). EfficientNetB0 (4.05M parameters) employs compound scaling (balancing depth, width, and resolution) and includes **Squeeze-and-Excitation (SE) channel attention modules** within its MBConv blocks.
2. **Medical Image Feature Representation**: Channel attention mechanism allows the network to dynamically weight subtle texture and boundary features characteristic of malignant vs. benign ultrasound lesions.
3. **Efficiency**: EfficientNetB0 offers a lightweight, practical option that trains fast on CPU without the parameter bloat of larger models (e.g. ResNet50V2 at 23.5M parameters).
4. **Environment Compatibility**: Pretrained ImageNet weights (`tf.keras.applications.EfficientNetB0`) were verified available and successfully loaded in the current environment.

---

## 5. V13 Training & Evaluation Plan

### Training Protocol
1. **Model Building**:
   - Backbone: Pretrained `EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))`.
   - Classification Head: GlobalAveragePooling2D + Dropout(0.20) + Dense(3, activation='softmax').
   - Preprocessing: `tf.keras.applications.efficientnet.preprocess_input` (or standard $[0, 255]$ scale since EfficientNetB0 includes scaling layer).
2. **Two-Stage Transfer Learning**:
   - **Stage 1 (Warmup)**: Freeze backbone (`base_model.trainable = False`), train classification head for 8 epochs with Adam optimizer ($\text{LR}=5\times 10^{-4}$).
   - **Stage 2 (Controlled Fine-Tuning)**: Unfreeze top 20 layers of EfficientNetB0, fine-tune for max 15 epochs with Adam ($\text{LR}=1\times 10^{-4}$), EarlyStopping ($\text{patience}=5$), and ReduceLROnPlateau ($\text{patience}=3$).
3. **Validation Threshold Calibration**:
   - Perform malignant threshold sweep $t \in [0.20, 0.70]$ on validation predictions ONLY to select $t^*$.
4. **Frozen Benchmark Evaluation**:
   - Single-pass evaluation of candidate on frozen 117-image historical test set and grouped test set using $t^*$.

---

## 6. Success Criteria & Benchmark Baselines

### Primary Success Criteria:
- **Grouped Malignant Recall**: $\ge 85\%$
- **Grouped Malignant Precision**: $\ge 60\%$
- **Grouped Overall Accuracy**: $\ge 70\%$

### Benchmark Baselines for Comparison:
- **V5-B (Active Production)**: Grouped Accuracy 70.09%, Macro F1 69.46%, Malignant Recall 80.65% (6 FN, 15 FP), Malignant Precision 62.50%.
- **V11-B (Research Candidate)**: Grouped Accuracy 69.16%, Macro F1 70.12%, Malignant Recall 90.32% (3 FN, 21 FP), Malignant Precision 57.14%.

---

## 7. Artifact Paths to be Created

- Candidate Model: `backend/models/candidates/v13_efficientnet_b0.keras`
- Validation Summary: `backend/models/evaluation_reports/v13_validation_summary.json`
- Raw Evaluation Results: `backend/models/evaluation_reports/v13_evaluation_results.json`
- Final Markdown Report: `backend/models/evaluation_reports/v13_final_evaluation_report.md`
- Script: `backend/training/train_v13.py`

---

## 8. Safety & Non-Promotion Guarantee

- Production model `breast_image_classifier.keras` will **NOT** be modified or overwritten.
- Locked historical test set (117 images) will **NOT** be used for training or threshold tuning.
- **NO automatic promotion** will occur even if V13 beats V5-B.

---

> [!NOTE]  
> Phase 1 State Audit complete. Training script creation and execution will begin ONLY upon user approval.
