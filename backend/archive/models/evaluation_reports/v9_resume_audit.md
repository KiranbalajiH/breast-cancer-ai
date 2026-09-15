# V9 Resume & State Audit Report

**Date**: September 02, 2026  
**Project**: Breast Cancer AI — Ultrasound Image Classifier (`C:\Users\kiran\BCD`)  
**Audit Purpose**: Complete state inspection following mid-experiment cancellation of V9.  

---

## 1. Production Safety Audit — FIRST

Production model state verified. **No production files were modified, overwritten, or promoted.**

| Artifact Path | Status | File Size | Last Modified | MD5 Hash | SHA256 Hash |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `backend/models/breast_image_classifier.keras` | **ACTIVE PROD (V5-B)** | 18,015,816 bytes | 2026-08-28T08:39:49 | `676dad9c993d04bac493a6ee50b1d344` | `93b3c106cff51993b605220c98cfaf54c8fa404bb581656fa11337a26b32e03c` |
| `backend/models/breast_image_classifier_v1_backup.keras` | **INTACT BACKUP** | 9,664,069 bytes | 2026-08-23T17:48:20 | `ede117b473539fe7866b060208a1ecbf` | `cd85617ac1de0e9c899443f62b47570624dad1fcf130f8744564cb1cf5def1ef` |

> [!IMPORTANT]  
> `breast_image_classifier.keras` remains the active V5-B production model. No changes were made to production metadata (`backend/models/breast_image_classifier_metadata.json`).

---

## 2. Dataset & Test Set Safety Audit

- **Dataset Directory**: `dataset/BUSI` is **100% intact**. No scan or mask images were created, altered, or deleted.
- **Clean Trainable Dataset**: 776 scans (434 benign scan files / 433 clean trainable; 209 malignant clean trainable; 133 normal clean trainable). Label conflict exclusions `benign (433).png` and `malignant (145).png` are active.
- **Locked Historical Test Set Verification**:
  - Total test scans: **117 images**
  - Benign: **65**
  - Malignant: **32**
  - Normal: **20**
  - Status: Completely isolated, un-regenerated, and excluded from all V9 candidate selection and hyperparameter tuning decisions.

---

## 3. V9 Candidate Model Audit & Load Verification

All files in `backend/models/candidates/` and `backend/models/evaluation_reports/` were searched and load-tested.

| Candidate ID | Model File | Status | Size (bytes) | Last Modified | Keras Load Test | Input Shape(s) | Total Params | Trainable Params |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **V9-A** | `v9_mobilenetv2_baseline.keras` | **COMPLETE** | 18,020,326 | 2026-09-01T21:04:05 | **SUCCESS** | `(None, 224, 224, 3)` | 2,261,827 | 1,043,843 |
| **V9-B** | `v9_mobilenetv2_cropaug.keras` | **COMPLETE** | 18,020,370 | 2026-09-01T21:10:48 | **SUCCESS** | `(None, 224, 224, 3)` | 2,261,827 | 1,043,843 |
| **V9-C** | `v9_mobilenetv2_dualview.keras` | **COMPLETE** | 18,074,424 | 2026-09-01T21:21:57 | **SUCCESS** | `[(None, 224, 224, 3), (None, 224, 224, 3)]` | 2,265,667 | 1,047,683 |
| **V9-D** | `v9_mobilenetv2_auxloss.keras` | **MISSING** | N/A | N/A | N/A | N/A | N/A | N/A |

### Evaluation Report Artifact Audit
- `v9_validation_summary.json`: **MISSING**
- `v9_evaluation_results.json`: **MISSING**
- `v9_final_evaluation_report.md`: **MISSING**

---

## 4. Inspection of Training Script (`backend/training/train_v9.py`)

- **Script Status**: Complete and fully functional (772 lines, 41,302 bytes).
- **Candidate Definitions**: Defines 4 candidate configurations:
  1. `v9_mobilenetv2_baseline`: Single MobileNetV2 whole-image control baseline.
  2. `v9_mobilenetv2_cropaug`: Single-input MobileNetV2 with whole-image + 30% margin lesion crop augmentation for abnormal scans + mild class weights `{0: 1.0, 1: 1.25, 2: 1.1}`.
  3. `v9_mobilenetv2_dualview`: Multi-scale Dual-View architecture processing `[Whole Image, Lesion Crop]` concatenated into twin 1280-dim feature embeddings (2560 total) + mild class weights.
  4. `v9_mobilenetv2_auxloss`: Multi-scale Dual-View architecture with auxiliary local crop classification loss ($\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{main}} + 0.2 \cdot \mathcal{L}_{\text{local\_aux}}$).
- **Resumability Logic**: Lines 354–359 implement explicit existence checks:
  ```python
  if os.path.exists(save_path):
      print(f"FOUND EXISTING CANDIDATE MODEL ({cid}): Loading {save_path}")
      model = tf.keras.models.load_model(save_path)
  else:
      # Train model stage 1 & stage 2...
  ```
- **Conclusion**: Candidates V9-A, V9-B, and V9-C are already completely saved and verified loadable. `train_v9.py` will **automatically skip retraining Candidates V9-A, V9-B, and V9-C**, train only the missing Candidate V9-D, and then proceed directly to validation threshold tuning and test set evaluation.

---

## 5. BUSI Lesion Mask & Crop Pipeline Verification

The mask extraction logic in `extract_whole_and_crop()` was audited for correctness:
- **Prefix Matching**: Correctly groups multiple mask files for scans (e.g. `benign (10)_mask.png`, `benign (10)_mask_1.png`).
- **Multiple Masks**: Combines mask files via bitwise OR (`cv2.bitwise_or`).
- **Bounding Box Extraction**: Calculates exact lesion extent with a **30% context margin** around the lesion boundary.
- **Aspect Ratio Preservation**: Both whole image and lesion crop are resized using `letterbox_resize` to $224 \times 224$ with constant padding, preventing spatial distortion.
- **Normal Class Handling**: Scans without mask pixels (Normal class) safely default to whole-image representations.

---

## 6. Audit Summary & Findings (A–I Checklist)

- **A. What V9 work was completed before quitting**:
  - Candidate V9-A (`v9_mobilenetv2_baseline.keras`) trained & saved.
  - Candidate V9-B (`v9_mobilenetv2_cropaug.keras`) trained & saved.
  - Candidate V9-C (`v9_mobilenetv2_dualview.keras`) trained & saved.
- **B. What artifacts exist**:
  - `v9_mobilenetv2_baseline.keras` (18.02 MB)
  - `v9_mobilenetv2_cropaug.keras` (18.02 MB)
  - `v9_mobilenetv2_dualview.keras` (18.07 MB)
  - `train_v9.py` (41.3 KB)
- **C. What artifacts are missing**:
  - Candidate V9-D model file (`v9_mobilenetv2_auxloss.keras`).
  - `v9_evaluation_results.json`
  - `v9_final_evaluation_report.md`
- **D. Which candidates are valid**:
  - V9-A, V9-B, V9-C are 100% valid and verified loadable.
- **E. Whether any candidate needs retraining**:
  - Candidates V9-A, V9-B, V9-C do **NOT** need retraining.
  - Candidate V9-D needs initial training.
- **F. Whether validation was completed**:
  - Validation metrics were not saved to disk before quitting.
- **G. Whether test evaluation was completed**:
  - Test evaluation was not run or saved before quitting.
- **H. Whether V9 can safely resume**:
  - **YES**. Running `train_v9.py` is safe, deterministic, non-destructive, and will reuse all 3 completed candidates.
- **I. Exact next command / action required**:
  Run the training script using the backend virtual environment:
  ```powershell
  backend\venv\Scripts\python.exe backend\training\train_v9.py
  ```

---

## 7. Next Action Status

> [!NOTE]  
> Per prompt instructions ("STOP after the resume audit and report the findings. DO NOT promote anything. DO NOT modify production."), no training execution or model promotion has been initiated. Awaiting user directive to execute the resume action.
