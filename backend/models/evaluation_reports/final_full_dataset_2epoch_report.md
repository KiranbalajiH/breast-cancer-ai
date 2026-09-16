# Final Full-Dataset 2-Epoch MobileNetV2

**Date**: September 16, 2026  
**Project**: Breast Cancer Detection (BCD) Platform  
**Candidate Model File**: `backend/models/candidates/final_full_dataset_2epoch_mobilenetv2.keras`  
**Candidate Metadata**: `backend/models/candidates/final_full_dataset_2epoch_mobilenetv2_metadata.json`  

---

## Dataset
- **Total Eligible Scans**: 776 clean, eligible BUSI scans
- **Class Distribution**:
  - `benign`: 434 (55.93%)
  - `malignant`: 209 (26.93%)
  - `normal`: 133 (17.14%)
- **Preprocessing**: Aspect-Ratio Letterbox resize to $224 \times 224 \times 3$, RGB conversion, MobileNetV2 `preprocess_input`.

---

## Training
- **Architecture**: MobileNetV2 Transfer Learning (ImageNet backbone + GAP + Dropout(0.2) + Softmax Dense Head)
- **Input Size**: $224 \times 224 \times 3$
- **Epochs Completed**: 2 Complete Epochs
  - *Epoch 1 (Dense Head Warmup)*: LR = `5e-4`, Adam, Loss = `1.1464`, Accuracy = `49.61%`
  - *Epoch 2 (Top 15 Backbone Layers Fine-Tuning)*: LR = `1e-4`, Adam, Loss = `0.7213`, Accuracy = `66.49%`
- **Total Training Samples**: 776 scans (100% of dataset used for training weight updates)
- **Batch Size**: 32
- **Steps per Epoch**: 25 steps

---

## Evaluation Limitation
> [!WARNING]
> **Evaluation Limitation Statement**:
> "This model was trained using all 776 eligible BUSI scans. Therefore the historical 117-image and grouped 107-image frozen sets cannot be used as independent held-out evaluation sets for this model."

---

## Runtime Verification
- **Model Loading Test**: PASSED (Loaded cleanly via `tf.keras.models.load_model`)
- **Output Shape Test**: PASSED (Dummy output shape: `(1, 3)`)
- **Probability Validity Test**: PASSED (Non-negative softmax values, sum = `1.000000`)
- **Inference Smoke Test**: PASSED (5/5 sample images produced valid categorical probability distributions)

---

## Model Integrity
- **Candidate Path**: `backend/models/candidates/final_full_dataset_2epoch_mobilenetv2.keras`
- **File Size**: 18,015,776 bytes (17.18 MB)
- **Candidate SHA256**: `a1a5ad9ffe063eeb83db7f5a9d889b266e504f1fb2ff89e7206cec5a4d70a994`
- **Candidate Metadata Path**: `backend/models/candidates/final_full_dataset_2epoch_mobilenetv2_metadata.json`

---

## Production Protection
- **Production V5-B Status**: UNTOUCHED
  - Path: `backend/models/breast_image_classifier.keras`
  - SHA256 Pre-Training: `93b3c106cff51993b605220c98cfaf54c8fa404bb581656fa11337a26b32e03c`
  - SHA256 Post-Training: `93b3c106cff51993b605220c98cfaf54c8fa404bb581656fa11337a26b32e03c`
  - Match: **VERIFIED MATCH**
- **V14 Research Components Status**: UNTOUCHED
  - `v11_mobilenetv2_b.keras` SHA256: `ac1bceff1483e87903921196b1f8a5077d7d1d30f98dc65d6b9cb88e80d1380a`
  - `v13_efficientnet_b0.keras` SHA256: `85382fcbb2d6c494e85bcf449e0a2f253bc3d7f82dd836f6cce63bed744b4741`

---

## Status
- **Classification**: `FULL-DATASET EXPERIMENTAL CANDIDATE`
- **Deployment Status**: Not promoted to production; production inference remains unchanged.
- **Generalization Claim**: No held-out performance claims are made because all 776 eligible scans were included in the training dataset.
