# V10 Resume & State Audit Report

**Date**: September 02, 2026  
**Project**: Breast Cancer AI — Ultrasound Image Classifier (`c:\Users\kiran\BCD`)  
**Audit Purpose**: Complete state inspection following mid-experiment cancellation of V10.  

---

## 1. Production Safety Audit — FIRST

Production model state verified. **No production files were modified, overwritten, or promoted.**

| Artifact Path | Status | File Size | Last Modified | MD5 Hash | SHA256 Hash |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `backend/models/breast_image_classifier.keras` | **ACTIVE PROD (V5-B)** | 18,015,816 bytes | 2026-08-28T08:39:49 | `676dad9c993d04bac493a6ee50b1d344` | `93b3c106cff51993b605220c98cfaf54c8fa404bb581656fa11337a26b32e03c` |
| `backend/models/breast_image_classifier_metadata.json` | **INTACT PROD METADATA** | 585 bytes | 2026-08-28T08:39:49 | `a5f22e70e9c6062fcd5c9527ec3bc398` | `f59bf0f5bf40df03b41d2f97aa2c7aaef8ed16efecddae1eeb805cf3ad1d53db` |

> [!IMPORTANT]  
> `breast_image_classifier.keras` remains the active V5-B production model. No changes were made to production metadata (`backend/models/breast_image_classifier_metadata.json` or `backend/models/metadata.json`). Zero V10 models have been promoted or copied into production.

---

## 2. Dataset & Test Set Safety Audit

- **Dataset Directory**: `dataset/BUSI` is **100% intact**. No scan or mask images were created, altered, or deleted.
- **Raw Scans**:
  - Benign: 435 scans, 451 masks
  - Malignant: 210 scans, 211 masks
  - Normal: 133 scans, 133 masks
- **Clean Trainable Dataset**: 776 scans (433 benign clean trainable; 209 malignant clean trainable; 133 normal clean trainable). Label conflict exclusions `benign (433).png` and `malignant (145).png` remain excluded in data pipelines.
- **Locked Historical Test Set Verification**:
  - Total test scans: **117 images**
  - Benign: **65**
  - Malignant: **32**
  - Normal: **20**
  - Seed: `random_state=42`
  - Status: Completely isolated, un-regenerated, and excluded from all V10 training iterations and hyperparameter tuning decisions.

---

## 3. V10 Artifact Audit & Candidate Load Verification

All files in `backend/training/`, `backend/models/candidates/`, and `backend/models/evaluation_reports/` were searched and load-tested.

| Candidate ID | Model File | Status | Size (bytes) | Last Modified | Keras Load Test | Input Shape | Output Shape | Total Params | Trainable Params |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **V10-A** | `v10_mobilenetv2_a.keras` | **COMPLETE** | 18,015,776 | 2026-09-02T10:17:37 | **SUCCESS** | `(None, 224, 224, 3)` | `(None, 3)` | 2,261,827 | 1,043,843 |
| **V10-B** | `v10_mobilenetv2_b.keras` | **NOT STARTED** | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| **V10-C** | `v10_mobilenetv2_c.keras` | **NOT STARTED** | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| **V10-D** | `v10_mobilenetv2_d.keras` | **NOT STARTED** | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| **V10-E** | `v10_mobilenetv2_e.keras` | **NOT STARTED** | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

### Evaluation Report Artifact Audit
- `v10_validation_summary.json`: **MISSING** (Not generated before quit)
- `v10_evaluation_results.json`: **MISSING** (Not generated before quit)
- `v10_final_evaluation_report.md`: **MISSING** (Not generated before quit)

---

## 4. Inspection of Training Script (`backend/training/train_v10.py`)

- **Script Status**: Complete and fully functional (665 lines, 36,098 bytes).
- **Candidate Definitions**: Defines 5 candidate configurations focused on malignant sensitivity optimization:
  1. `v10_mobilenetv2_a`: V5-B Control Baseline (Unweighted).
  2. `v10_mobilenetv2_b`: Moderate Class Weighting `{0: 1.0, 1: 1.6, 2: 1.1}`.
  3. `v10_mobilenetv2_c`: Sparse Categorical Focal Loss ($\gamma=2.0, \alpha=[1.0, 1.5, 1.1]$).
  4. `v10_mobilenetv2_d`: Controlled Malignant-Focused Augmentation.
  5. `v10_mobilenetv2_e`: Combined Moderate Weighting + Malignant-Focused Augmentation.
- **Resumability Logic**: Lines 366–370 implement explicit candidate existence checks:
  ```python
  if os.path.exists(save_path):
      print(f"FOUND EXISTING CANDIDATE MODEL ({cid}): Loading {save_path}")
      model = tf.keras.models.load_model(save_path, custom_objects={"SparseCategoricalFocalLoss": SparseCategoricalFocalLoss})
  else:
      # Stage 1 warmup & Stage 2 fine-tuning...
  ```
- **Conclusion**: Candidate V10-A is already completely saved and verified loadable. `train_v10.py` will **automatically skip retraining Candidate V10-A**, train Candidates V10-B, V10-C, V10-D, and V10-E, and then proceed directly to validation threshold sweeps and frozen test set evaluations.

---

## 5. Recent Modifications Check (Post-V10 Start)

The following files were created/modified during the V10 session:
- `backend/training/train_v10.py` (Created 2026-09-02T10:13:45)
- `backend/models/candidates/v10_mobilenetv2_a.keras` (Saved 2026-09-02T10:17:37)

No changes were made to production models, metadata, or dataset files.

---

## 6. V10 Resume Status & Recommendations

| Item | Status |
| :--- | :--- |
| **Production Status** | **INTACT & UNTOUCHED** (V5-B Active) |
| **Dataset Status** | **INTACT & UNTOUCHED** (776 Clean Trainable Scans) |
| **Locked Test Status** | **ISOLATED & UNCONTAMINATED** (117 Images, `random_state=42`) |
| **V10-A Status** | **COMPLETED & VALID** (18,015,776 bytes, Keras load verified) |
| **V10-B Status** | **NOT STARTED** |
| **V10-C Status** | **NOT STARTED** |
| **V10-D Status** | **NOT STARTED** |
| **V10-E Status** | **NOT STARTED** |
| **Missing Artifacts** | `v10_mobilenetv2_b.keras`, `v10_mobilenetv2_c.keras`, `v10_mobilenetv2_d.keras`, `v10_mobilenetv2_e.keras`, `v10_validation_summary.json`, `v10_evaluation_results.json`, `v10_final_evaluation_report.md` |
| **Partial Artifacts** | None |
| **Exact Next Safe Command** | `..\venv\Scripts\python.exe backend\training\train_v10.py` (from `backend/` directory) |

> [!NOTE]  
> Per explicit instructions, no training execution, test evaluations, or production changes have been initiated. The audit is complete. Awaiting user approval to resume V10 training.
