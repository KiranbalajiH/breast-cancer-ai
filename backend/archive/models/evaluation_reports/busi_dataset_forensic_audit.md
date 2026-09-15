# BUSI Dataset Forensic Audit Report

**Date**: August 27, 2026  
**Project**: Breast Cancer AI — Ultrasound Image Classification  
**Dataset**: `dataset/BUSI` (Dataset of Breast Ultrasound Images)  
**Audit Purpose**: Forensic investigation of dataset quality, image properties, class distributions, split integrity, preprocessing, and error patterns to identify factors limiting model generalization and malignant sensitivity.  
**Audit Scope**: AUDIT ONLY. No training executed, no models promoted, no production artifacts overwritten, no files deleted, locked test set strictly preserved.

---

## Executive Summary

A forensic audit of `dataset/BUSI` was conducted across 1,573 total files (778 trainable ultrasound scans and 795 ground-truth masks). The investigation revealed key structural, statistical, and visual factors limiting model generalization:

1. **Patient Leakage Risk**: BUSI contains 778 scans from 600 female patients (~1.3 scans/patient). Because dataset splitting is currently **image-level**, sequential slice images of the same lesion/patient could appear across both training and validation/test splits.
2. **Aspect Ratio Distortion**: Image aspect ratios vary widely from $0.57$ to $2.05$. Standard isotropic resizing to $224 \times 224$ distorts lesion margins, length-to-width ratios, and acoustic shadowing features critical for distinguishing malignant from benign tumors.
3. **Systematic Class-Specific Background Artifacts**: The `normal` tissue class has a significantly lower average blur variance (113.8) and larger average frame dimensions ($651 \times 532$) compared to `benign` (191.7 blur, $613 \times 495$) and `malignant` (175.2 blur, $598 \times 494$). Models risk learning background texture shortcuts rather than pathological features.
4. **Color & Caliper Noise**: 97.9% of scans are grayscale ultrasound images saved as 3-channel RGB. However, 2.1% of scans contain burned-in software color calipers, measurement lines, or Doppler color maps.
5. **Exact Duplicate Label Conflict**: `benign (433).png` and `malignant (145).png` are 100% identical pixel duplicates (`MD5: aed81b43`) with conflicting labels in the raw dataset.

---

## 1. Dataset Structure & File Inventory

- **Dataset Root Directory**: `dataset/BUSI`
- **Class Subdirectories**:
  1. `benign`
  2. `malignant`
  3. `normal`

### File Counts Breakdown

| Class | Total Files | Trainable Scans | Ground-Truth Masks | Multi-Mask Images |
| :--- | :---: | :---: | :---: | :---: |
| **Benign** | 886 | 435 | 451 | 16 scans have 2+ mask files |
| **Malignant** | 421 | 210 | 211 | 1 scan has 2 mask files |
| **Normal** | 266 | 133 | 133 | 0 |
| **Total** | **1,573** | **778** | **795** | **17 scans** |

### File Properties
- **File Extensions**: 100% `.png`.
- **Image Dimensions**: Widths range from $190\text{px}$ to $1048\text{px}$; Heights range from $310\text{px}$ to $719\text{px}$.
- **Color Channels**: 100% of images are loaded as 3-channel RGB ($H \times W \times 3$).
- **Color Distribution**: **762 of 778 scans (97.9%)** are grayscale images where all R, G, B channel matrices are identical. Only 16 scans (2.1%) contain actual non-grayscale color content.
- **Corrupt / Zero-Byte Files**: 0 corrupt files, 0 zero-byte files. All 1,573 files are valid, readable PNGs.
- **Mask Separation**: Masks are cleanly demarcated by `_mask` or `_mask_1` suffixes in filenames and can be completely isolated from trainable image arrays.

---

## 2. Duplicate Analysis

A complete cryptographic MD5 hash audit of raw file bytes and pixel matrices was performed:

| Duplicate Category | Group Count | Affected Files | Status / Resolution |
| :--- | :---: | :--- | :--- |
| **Exact File MD5 Duplicates** | 1 group | `benign (433).png` and `malignant (145).png` (`MD5: 88f5ac30`) | **Label Conflict** (Excluded in `train_v4.py`) |
| **Exact Pixel MD5 Duplicates** | 1 group | `benign (433).png` and `malignant (145).png` (`MD5: aed81b43`) | **Label Conflict** (Excluded in `train_v4.py`) |
| **Near-Duplicate / Sequential Scans** | Multiple | Adjacent index pairs (e.g. `benign (10)` & `benign (11)`) | Sequential slice views of same lesion |

> [!WARNING]
> **Label Conflict Identified**: `benign (433).png` is categorized under `benign` while `malignant (145).png` is categorized under `malignant` despite being the exact same ultrasound image file. Excluding both during dataset initialization is required.

---

## 3. Patient / Study Leakage Analysis

### Findings
- **Patient Population**: According to the BUSI dataset specifications (Al-Dhabyani et al., 2020), the dataset comprises images from **600 female patients**.
- **Scans per Patient**: Total clean scans (778) / Total patients (600) $\approx 1.3$ scans per patient.
- **Filename Scheme**: Scans are anonymized as `benign (1).png` ... `benign (435).png`. Patient ID tags were stripped from DICOM headers prior to public release.
- **Current Split Method**: The current splitting strategy (`train_test_split(random_state=42, stratify=labels)`) performs an **image-level split**.

### Risk Assessment
- **What CAN be established**: Filenames do not contain explicit patient ID strings.
- **What CANNOT be established**: Without original DICOM metadata, exact multi-image patient groupings cannot be proven algorithmically from filenames alone.
- **Leakage Implication**: Because multiple images belong to single patients, random image-level splitting means sequential slices or multi-angle scans of the same patient lesion likely exist across Train, Validation, and Test sets, leading to potential optimistic test bias.

---

## 4. Current Split Audit

The dataset split logic in `train_v4.py` / `evaluate_v4.py` was audited:

- **Random Seed**: `42`
- **Stratification**: Class-stratified sampling
- **Data Exclusions**: 2 files (`malignant (145).png`, `benign (433).png`)
- **Total Clean Scans**: 776 images

### Split Distribution

| Split | Benign | Malignant | Normal | Total Images | Percentage |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Train Set** | 304 | 146 | 93 | **543** | 70.0% |
| **Validation Set** | 65 | 31 | 20 | **116** | 14.9% |
| **Test Set (Locked)** | 65 | 32 | 20 | **117** | 15.1% |
| **Total** | **434** | **209** | **133** | **776** | 100.0% |

> [!NOTE]
> **Test Set Verification**: Executing `train_test_split(random_state=42)` reproduces the **exact 117-image locked test set**. The test set has remained 100% untouched and unmodified.

---

## 5. Image Quality Analysis

| Metric / Artifact | Count | Percentage | Description / Impact |
| :--- | :---: | :---: | :--- |
| **Excessively Dark Scans** (`mean_intensity` < 35) | 4 | 0.5% | Deep shadow attenuation or acoustic shadowing |
| **Excessively Bright Scans** (`mean_intensity` > 140) | 1 | 0.1% | High gain setting or superficial tissue reflection |
| **Low Contrast Scans** (`std_intensity` < 30) | 3 | 0.4% | Homogeneous tissue texture, low signal dynamic range |
| **Blurry Scans** (`laplacian_var` < 100) | 239 | 30.7% | Out-of-focus focal zone or smooth normal tissue |
| **Color Calipers / Annotations** | 15 | 1.9% | Software measurement lines (yellow/red crosses) burned into scan |
| **Wide Aspect Ratio Variation** ($AR < 0.8$ or $AR > 1.5$) | 42 | 5.4% | Non-square ultrasound acquisition frames |

---

## 6. Class Visual & Distribution Analysis

Statistical analysis across classes reveals significant systematic differences in image properties:

| Class | Height (Mean) | Width (Mean) | Aspect Ratio (Mean) | Mean Intensity | Laplacian Blur Variance |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Benign** | $495.1\text{px}$ | $613.1\text{px}$ | 1.24 | 86.3 | 191.7 |
| **Malignant** | $494.0\text{px}$ | $598.0\text{px}$ | 1.21 | 79.8 | 175.2 |
| **Normal** | $532.8\text{px}$ | $651.2\text{px}$ | 1.22 | 81.0 | **113.8** |

### Key Observations
1. **Normal Tissue Smoothness**: The `normal` class has a dramatically lower average blur variance (113.8) than `benign` (191.7) or `malignant` (175.2). Normal scans lack focal mass lesions, producing uniform parenchyma that appears "blurry" to Laplacian filters.
2. **Frame Dimension Bias**: `normal` scans are on average $38\text{px}$ taller and $38\text{px}$ wider than pathological scans. Models can easily exploit background dimensions as a shortcut.

---

## 7. Difficult Case Analysis (V1 Baseline Errors)

Evaluating the 18 misclassifications of V1 Baseline on the 117-image test set:

### A. Malignant False Negatives (10 cases where True=Malignant, Pred=Benign)
- **Affected Files**: `malignant (138)`, `malignant (12)`, `malignant (36)`, `malignant (140)`, `malignant (112)`, `malignant (53)`, `malignant (13)`, `malignant (107)`, `malignant (91)`, `malignant (1)`.
- **Common Characteristics**:
  - `malignant (1)`: Vertical orientation ($AR = 0.75$), high intensity (118.2). Resizing to $224 \times 224$ horizontally stretched the lesion, making it appear wider-than-tall (a classic benign feature!).
  - `malignant (36)`: Low intensity (52.0), low blur variance (45.8). Dark and smooth presentation confused with benign cyst.
  - `malignant (140)` & `malignant (112)`: Well-circumscribed anterior margins mimicking benign fibroadenoma boundaries.

### B. Benign False Positives (4 cases where True=Benign, Pred=Malignant)
- **Affected Files**: `benign (316)`, `benign (397)`, `benign (341)`, `benign (141)`.
- **Common Characteristics**:
  - `benign (141)`: High sharpness (`blur_var` = 556.9) with intense posterior acoustic shadowing, causing V1 to predict Malignant with 89.0% confidence.
  - `benign (341)`: Extremely low contrast (`std_dev` = 29.6), blurry margins.

### C. Normal / Benign Confusion (4 cases)
- `benign (256)` & `benign (383)` predicted Normal (383 has 99.5% Normal confidence due to diffuse parenchymal background).
- `normal (85)` & `normal (80)` predicted Benign (contain localized glandular tissue structures mimicking small cysts).

---

## 8. Preprocessing Audit

A line-by-line audit comparing training preprocessing (`train_v4.py`) against live API inference preprocessing (`backend/app/image_model.py`) was performed:

| Preprocessing Step | Training Pipeline (`train_v4.py`) | API Inference Pipeline (`image_model.py`) | Status |
| :--- | :--- | :--- | :---: |
| **Image Decoding** | `cv2.imread(path)` (BGR) | `cv2.imdecode(bytes)` (BGR) | **MATCH** |
| **Color Conversion** | `cv2.cvtColor(img, COLOR_BGR2RGB)` | `cv2.cvtColor(img, COLOR_BGR2RGB)` | **MATCH** |
| **Resizing** | `cv2.resize(img_rgb, (224, 224))` | `cv2.resize(img_rgb, (224, 224))` | **MATCH** |
| **Scaling Function** | `mobilenet_v2.preprocess_input` | `mobilenet_v2.preprocess_input` | **MATCH** |
| **Value Range** | $[-1.0, +1.0]$ | $[-1.0, +1.0]$ | **MATCH** |
| **Data Type** | `np.float32` | `np.float32` | **MATCH** |
| **Tensor Shape** | `(1, 224, 224, 3)` | `(1, 224, 224, 3)` | **MATCH** |

> [!NOTE]
> **Preprocessing Alignment**: Training and inference pipelines match 100%. Model performance discrepancies are NOT caused by preprocessing mismatch between backend API and training scripts.

---

## 9. Dataset Risk Assessment Matrix

| Issue | Evidence | Severity | Potential Effect |
| :--- | :--- | :---: | :--- |
| **Aspect Ratio Distortion** | Aspect ratios vary from $0.57$ to $2.05$; non-aspect-preserved resize stretches lesions | **HIGH** | Distorts lesion margins & length-width ratios; misclassifies vertical malignant lesions as benign |
| **Image-Level Split Leakage** | 778 scans from 600 patients; random split without patient grouping | **HIGH** | Overoptimistic validation/test scores; poor generalization on external patient data |
| **Class Imbalance** | Benign: 435, Malignant: 210, Normal: 133 ($2.07 : 1 : 0.63$) | **HIGH** | Bias toward Benign predictions; suppressed Malignant recall |
| **Systematic Class Background Shortcuts** | Normal class has lower blur variance (113.8 vs 191.7) and larger frame dimensions | **MEDIUM** | Model learns background image texture/size shortcuts rather than tissue pathology |
| **Software Caliper / Color Annotations** | 15 scans contain burned-in colored measurement markers | **MEDIUM** | Model relies on presence of doctor calipers as a diagnostic shortcut |
| **Label Conflict Noise** | `benign (433).png` & `malignant (145).png` identical pixel duplicate | **LOW** | Handled in V4 cleaning, but indicates underlying annotation noise in BUSI |

---

## 10. Evidence-Based Recommendations

Based strictly on empirical evidence gathered during this forensic audit, the following changes are recommended **BEFORE** any future model training cycle:

### 1. Aspect-Ratio Preserving Resize with Padding (Letterboxing)
- **Evidence**: `malignant (1).png` has $AR=0.75$ (taller-than-wide, a key clinical sign of malignancy). Resizing to a square $224 \times 224$ squashes the height, causing V1 to classify it as Benign.
- **Action**: Implement aspect-ratio-preserving resize with reflection or black padding (letterboxing) to maintain true tumor geometry.

### 2. Single-Channel Grayscale Pipeline & Color Marking Removal
- **Evidence**: 97.9% of scans are grayscale images stored as RGB, while 2.1% contain colored software calipers.
- **Action**: Explicitly convert all input scans to 1-channel grayscale (or duplicate grayscale across 3 channels) to strip spurious color caliper lines.

### 3. Cost-Sensitive / Focal Loss Formulation
- **Evidence**: Standard cross-entropy loss with 2:1 class imbalance causes the model to favor Benign predictions to maximize overall accuracy.
- **Action**: Use Focal Loss ($\gamma = 2.0$) or class-weighted margin loss to heavily penalize malignant false negatives without exploding false positives.

### 4. Perceptual Grouping for Patient Leakage Reduction
- **Evidence**: BUSI contains 778 scans from 600 patients. Sequential scans share near-identical background noise.
- **Action**: Perform structural feature similarity clustering on un-split data to ensure near-identical sequential scans belong to the same split.

---

## Next-Step Training Strategy

1. **Do NOT train immediately**. Keep V1 MobileNetV2 Baseline (`backend/models/breast_image_classifier.keras`) active in production.
2. **Implement Aspect-Ratio Preserved Letterboxing** in dataset loader and API service.
3. **Validate letterboxed preprocessing** against the locked 117-image test set once prepared.
4. **Obtain user approval** before executing any V5 training pipeline.
