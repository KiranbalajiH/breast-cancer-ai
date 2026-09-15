# V11 Resume & State Audit Report

**Date**: September 02, 2026  
**Project**: Breast Cancer AI — Ultrasound Image Classifier (`c:\Users\kiran\BCD`)  
**Audit Purpose**: Complete state inspection following mid-experiment cancellation of V11.  

---

## 1. Production Safety Audit — FIRST

Production model state verified. **No production files were modified, overwritten, or promoted.**

| Artifact Path | Status | File Size | Last Modified | MD5 Hash | SHA256 Hash |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `backend/models/breast_image_classifier.keras` | **ACTIVE PROD (V5-B)** | 18,015,816 bytes | 2026-08-28T08:39:49 | `676dad9c993d04bac493a6ee50b1d344` | `93b3c106cff51993b605220c98cfaf54c8fa404bb581656fa11337a26b32e03c` |
| `backend/models/breast_image_classifier_metadata.json` | **INTACT PROD METADATA** | 585 bytes | 2026-08-28T09:07:56 | `2af1b24e0f609a04910ff7fba1545511` | `ef4e91d5eb859636adfd2db78ee0c7b543384d2047ccc6a39a056796daf65059` |
| `backend/models/metadata.json` | **INTACT TABULAR METADATA** | 4,261 bytes | 2026-08-15T11:45:19 | `4d1d37474833cbfb997e119713706c16` | `d16ae432df53285a93ba2757f54012bc2b68f46d8bdba9933e9feb5a4900ec38` |

> [!IMPORTANT]  
> `breast_image_classifier.keras` remains the active V5-B production model. File size, MD5 hash, and SHA256 hash match the previously verified V5-B state exactly. Zero V11 models have been promoted or copied into production.

---

## 2. Dataset & Test Set Safety Audit

- **Dataset Directory**: `dataset/BUSI` is **100% intact**. No scan or mask images were created, altered, or deleted.
- **Raw Scans**:
  - Benign: 435 scans, 451 masks
  - Malignant: 210 scans, 211 masks
  - Normal: 133 scans, 133 masks
- **Clean Trainable Dataset**: 776 scans (433 benign clean trainable; 209 malignant clean trainable; 133 normal clean trainable). Label conflict exclusions `benign (433).png` and `malignant (145).png` remain excluded in training pipelines.
- **Locked Historical Test Set Verification**:
  - Total test scans: **117 images**
  - Benign: **65**
  - Malignant: **32**
  - Normal: **20**
  - Seed: `random_state=42`
  - Status: **Completely isolated, un-regenerated, and excluded** from all training iterations, augmentation selection, class weight selection, and threshold tuning. Zero evaluations were performed on this set during this audit.

---

## 3. V11 Artifact Audit & Candidate Load Verification

All files in `backend/training/`, `backend/models/candidates/`, and `backend/models/evaluation_reports/` were searched and load-tested.

| Candidate ID | Model File | Status | Size (bytes) | Last Modified | Keras Load Test | Input Shape | Output Shape | Total Params | Trainable Params |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **V11-A** | `v11_mobilenetv2_a.keras` | **COMPLETED** | 18,015,816 | 2026-09-02T11:06:05 | **SUCCESS** | `(None, 224, 224, 3)` | `(None, 3)` | 2,261,827 | 1,043,843 |
| **V11-B** | `v11_mobilenetv2_b.keras` | **COMPLETED** | 18,015,776 | 2026-09-02T11:09:45 | **SUCCESS** | `(None, 224, 224, 3)` | `(None, 3)` | 2,261,827 | 1,043,843 |
| **V11-C** | `v11_mobilenetv2_c.keras` | **COMPLETED** | 18,015,816 | 2026-09-02T11:12:45 | **SUCCESS** | `(None, 224, 224, 3)` | `(None, 3)` | 2,261,827 | 1,043,843 |
| **V11-D** | `v11_mobilenetv2_d.keras` | **NOT STARTED** | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| **V11-E** | `v11_mobilenetv2_e.keras` | **NOT STARTED** | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

### Evaluation Report & Metadata Artifact Audit
- `v11_validation_summary.json`: **MISSING** (Not generated before user quit)
- `v11_evaluation_results.json`: **MISSING** (Not generated before user quit)
- `v11_final_evaluation_report.md`: **MISSING** (Not generated before user quit)

---

## 4. V10-D Reference Model Audit

Verify source candidate for V11-A control:
- **File**: `backend/models/candidates/v10_mobilenetv2_d.keras`
- **Status**: **EXISTS & VALID**
- **File Size**: 18,015,816 bytes
- **Last Modified**: 2026-09-02T10:59:13
- **Keras Load Test**: **SUCCESS**
- **Architecture**: MobileNetV2 Transfer Learning (`(None, 224, 224, 3)` -> `(None, 3)`), 2,261,827 Total Params, 1,043,843 Trainable Params.
- **Strategy Alignment**: V11-A successfully reproduced / reused this exact configuration (100% Malignant Augmentation, Unweighted).

---

## 5. Inspection of Training Script (`backend/training/train_v11.py`)

Complete verification of script logic:

- **Checkpoint & Resume Logic**: Lines 360–372 implement explicit existence checks:
  ```python
  if os.path.exists(save_path):
      print(f"FOUND EXISTING CANDIDATE MODEL ({cid}): Loading {save_path}")
      model = tf.keras.models.load_model(save_path)
  ```
  `v11_mobilenetv2_a.keras`, `v11_mobilenetv2_b.keras`, and `v11_mobilenetv2_c.keras` exist and load successfully. When executed, `train_v11.py` will **automatically skip retraining candidates V11-A, V11-B, and V11-C**, train missing candidates V11-D and V11-E, and then proceed directly to validation threshold sweeps and frozen test set evaluations.
- **Candidate Definitions**:
  1. `v11_mobilenetv2_a`: V10-D Control Reproduction (100% Malignant Augmentation, Unweighted).
  2. `v11_mobilenetv2_b`: Reduced Malignant Augmentation (50% Malignant Replication).
  3. `v11_mobilenetv2_c`: Malignant Augmentation + Milder Class Emphasis `{0: 1.0, 1: 1.25, 2: 1.05}`.
  4. `v11_mobilenetv2_d`: Malignant Augmentation + Regularization (Dropout=0.30 & 30% Benign Replication).
  5. `v11_mobilenetv2_e`: Composite Combination (50% Aug + Milder Class Weights `{0: 1.0, 1: 1.20, 2: 1.0}` + Dropout=0.25).
- **Isolation of Locked Test Set**:
  - Training (`model.fit`) receives ONLY `grp_train_idx` data.
  - Validation during training receives ONLY `grp_val_idx` data.
  - Threshold calibration (lines 429–453) optimizes composite metrics on `grp_val_idx` ONLY.
  - Historical 117-image test set is evaluated ONLY at the very end (lines 456–463) after model weights and thresholds are frozen.
  - Model selection in report generation ranks models strictly by validation performance.

---

## 6. Recent File Modifications Check

Files modified on September 02, 2026 prior to audit completion:

- `backend/models/candidates/v11_mobilenetv2_a.keras` (Saved 2026-09-02T11:06:05)
- `backend/models/candidates/v11_mobilenetv2_b.keras` (Saved 2026-09-02T11:09:45)
- `backend/models/candidates/v11_mobilenetv2_c.keras` (Saved 2026-09-02T11:12:45)
- `backend/training/train_v11.py` (Modified 2026-09-02T11:13:21)

No changes were made to production model files, production metadata, dataset files, or locked test sets.

---

## 7. State Summary & Recommendations

| Item | Audit Finding |
| :--- | :--- |
| **Production Model** | **INTACT & UNTOUCHED** (V5-B Active, SHA256 matches) |
| **Production Metadata** | **INTACT & UNTOUCHED** |
| **Dataset (BUSI)** | **INTACT & UNTOUCHED** (776 Clean Trainable Scans) |
| **Locked Test Set** | **ISOLATED & UNCONTAMINATED** (117 Images, `random_state=42`) |
| **V10-D Reference Model** | **VALID & LOADABLE** |
| **V11-A Status** | **COMPLETED & VALID** (18,015,816 bytes, Keras load verified) |
| **V11-B Status** | **COMPLETED & VALID** (18,015,776 bytes, Keras load verified) |
| **V11-C Status** | **COMPLETED & VALID** (18,015,816 bytes, Keras load verified) |
| **V11-D Status** | **NOT STARTED** |
| **V11-E Status** | **NOT STARTED** |
| **Existing Artifacts** | `train_v11.py`, `v11_mobilenetv2_a.keras`, `v11_mobilenetv2_b.keras`, `v11_mobilenetv2_c.keras` |
| **Missing Artifacts** | `v11_mobilenetv2_d.keras`, `v11_mobilenetv2_e.keras`, `v11_validation_summary.json`, `v11_evaluation_results.json`, `v11_final_evaluation_report.md` |
| **Partial Artifacts** | **None** |
| **Exact Stage Quit** | Immediately after Candidate **V11-C** completed training and saved (`11:12:45`), prior to V11-D training start. |
| **Exact Next Safe Action** | `backend\venv\Scripts\python.exe backend\training\train_v11.py` (Resumes safely without retraining V11-A, V11-B, V11-C) |

> [!NOTE]  
> Per explicit instructions, no training execution, test evaluations, or production changes have been initiated. The audit is complete. Awaiting user approval to resume V11 training.
