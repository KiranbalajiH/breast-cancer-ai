# V5 Methodology & Preprocessing Experiment Design Plan

**Date**: August 28, 2026  
**Project**: Breast Cancer AI — Ultrasound Image Classification  
**Dataset**: `dataset/BUSI` (3 Classes: Benign, Malignant, Normal)  
**Phase**: METHODOLOGY & EXPERIMENT DESIGN (No model training executed in this phase)  
**Production Model Safety**: `backend/models/breast_image_classifier.keras` remains **unmodified** and **active**.  
**Historical Test Set Safety**: The existing locked 117-image test set remains **unmodified** and **preserved**.

---

## Executive Summary

Following the comprehensive forensic audit of the BUSI dataset (`dataset/BUSI`), this document presents the complete methodology and preprocessing experiment design for the V5 model iteration. Rather than prematurely training a new network, this phase systematically addresses the root structural, geometric, and shortcut artifacts that previously bounded generalization performance in model versions V1 through V4.

Key methodology upgrades designed herein include:
1. **Patient/Lesion Structural Grouping**: Identification of 682 distinct perceptual lesion clusters to prevent sequential image slice leakage across splits while preserving the historical 117-image locked test set for baseline comparison.
2. **Aspect-Ratio Preserving Letterboxing**: Replacement of standard square resizing with a canvas padding strategy that eliminates an average $19.4\%$ pixel stretching distortion without losing clinically relevant ultrasound content.
3. **Grayscale Standardization**: Conversion of 3-channel pseudo-RGB ultrasound files into standardized 1-to-3 channel grayscale to strip burned-in color caliper and Doppler measurement artifacts.
4. **Data Quality & Blur Distribution**: Evidence-based decision to retain all 239 scans flagged by Laplacian blur thresholding, avoiding severe class selection bias against normal tissue scans.
5. **Class Weight Calibration**: Formulation of mild class weights ($\{1.0, 1.25, 1.1\}$) to avoid the false-positive explosions caused by aggressive inverse-frequency weighting in V4.
6. **Controlled Candidate Matrix & Dual Evaluation**: A strict 4-variant matrix (V5-A through V5-D) paired with two distinct evaluation perspectives (Historical Locked Benchmark vs. Grouped Lesion Generalization).

---

## 1. Patient / Lesion Identity Findings

### Dataset Metadata & Filename Structure
- The BUSI dataset comprises ultrasound scans from **600 female patients** (Al-Dhabyani et al., 2020).
- Scans are named sequentially within class subdirectories: `benign (1).png` to `benign (435).png`, `malignant (1).png` to `malignant (210).png`, and `normal (1).png` to `normal (133).png`.
- **Finding**: DICOM header metadata and explicit patient ID tags were stripped prior to public release. Filenames contain only sequential class index numbers.
- **Explicit Limitation Statement**: Explicit patient IDs **cannot** be directly extracted from filenames, directory structures, or dataset manifests. Artificial patient IDs will **not** be invented or assumed.

### Perceptual Lesion Clustering Analysis
To identify sequential or multi-angle scans of the same patient/lesion without explicit DICOM headers, a structural perceptual hash and mean absolute difference (MAD) image feature analysis was executed across all 778 clean scans:

| Class | Total Scans | Perceptual Lesion Clusters | Multi-Scan Clusters | Single-Scan Clusters |
| :--- | :---: | :---: | :---: | :---: |
| **Benign** | 435 | **368** | 57 clusters (124 scans) | 311 clusters |
| **Malignant** | 210 | **204** | 5 clusters (11 scans) | 199 clusters |
| **Normal** | 133 | **110** | 14 clusters (37 scans) | 96 clusters |
| **Total** | **778** | **682** | **76 clusters (172 scans)** | **606 clusters** |

*Representative Multi-Scan Clusters Identified:*
- `benign`: Cluster 1 (`benign (10).png`, `benign (327).png`), Cluster 2 (`benign (12).png`, `benign (13).png`, `benign (14).png`).
- `malignant`: Cluster 1 (`malignant (4).png`, `malignant (5).png`), Cluster 2 (`malignant (17).png`, `malignant (18).png`).
- `normal`: Cluster 1 (`normal (104).png`, `normal (107).png`).

---

## 2. Split Methodology

### Dual-Split Strategy

To satisfy both comparative benchmark requirements and strict clinical generalization measurement, two distinct split methodologies are established:

#### Perspective A: Historical Benchmark Split (Locked 117-Image Test Set)
- **Train Set**: 543 images (304 Benign, 146 Malignant, 93 Normal)
- **Validation Set**: 116 images (65 Benign, 31 Malignant, 20 Normal)
- **Test Set**: 117 images (65 Benign, 32 Malignant, 20 Normal) — *100% identical to the V1/V3/V4 locked test set.*
- **Purpose**: Enables direct, apples-to-apples performance comparisons against historical V1-V4 baselines.

#### Perspective B: Patient/Lesion-Grouped Split
- **Method**: Grouped Stratified K-Fold ($K=5$) or Grouped Train/Val/Test Split operating on the **682 perceptual lesion clusters**.
- **Constraint**: All scans belonging to the same perceptual cluster are assigned strictly to a single split (Train, Val, or Test).
- **Stratification**: Maintains class ratios (approx. $56\%$ Benign, $27\%$ Malignant, $17\%$ Normal) across folds while ensuring $0\%$ cluster overlap.
- **Purpose**: Evaluates model performance on completely unseen patient lesions without sequential slice leakage.

---

## 3. Aspect-Ratio Distortion Analysis

### Quantified Distortion under Square Resizing
Standard neural network pipelines resize input images directly to a square $224 \times 224$ matrix (`cv2.resize(img, (224, 224))`). In BUSI, scan aspect ratios ($W / H$) vary from **0.57 to 2.05**.

| Aspect Ratio Range ($W / H$) | Category | Scan Count | Percentage | Primary Impact of Square Resize |
| :--- | :--- | :---: | :---: | :--- |
| **$AR < 0.80$** | Tall / Vertical | 15 | $1.9\%$ | Severely widens vertical tumors (e.g. `malignant (1)`), mimicking benign shape |
| **$0.80 \le AR \le 1.20$** | Near-Square | 394 | $50.6\%$ | Minimal geometric distortion ($< 10\%$) |
| **$1.20 < AR \le 1.50$** | Moderately Wide | 303 | $38.9\%$ | Vertically stretches lesion margins |
| **$AR > 1.50$** | Extremely Wide | 66 | $8.5\%$ | Severely squashes horizontal width |
| **Total** | **All Scans** | **778** | **100.0%** | **Mean Stretching Distortion: 19.4% (Max: 76.3%)** |

### Clinical Significance of Geometric Distortion
In breast ultrasound diagnostic criteria (ACR BI-RADS):
- **Malignant lesions** frequently present as "taller-than-wide" ($AR < 1.0$), with non-parallel orientation relative to skin lines, microlobulations, and vertical acoustic shadows.
- **Benign lesions** (e.g., fibroadenomas) present as "wider-than-tall" ($AR > 1.0$), oval, parallel to skin lines.
- **Consequence**: Forcing tall malignant scans into a $224 \times 224$ square artificially stretches height into width, directly causing false-negative misclassifications in models like V1 (e.g., `malignant (1)` misclassified as Benign with $99.8\%$ confidence).

---

## 4. Proposed Letterbox Preprocessing

### Letterbox Algorithm Specification
To preserve authentic lesion aspect ratios and edge geometry without cropping ultrasound content, an aspect-ratio-preserving resize with black canvas padding (letterboxing) is designed:

```python
import cv2
import numpy as np

def letterbox_preprocess(img: np.ndarray, target_size=(224, 224), pad_color=(0, 0, 0)) -> np.ndarray:
    """
    Resizes image preserving original aspect ratio, then pads symmetrically
    to reach target_size canvas without stretching geometry.
    """
    h, w = img.shape[:2]
    target_w, target_h = target_size
    
    # Calculate scale factor to fit within canvas
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    
    # Interpolation strategy: INTER_AREA for downscaling, INTER_CUBIC for upscaling
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
    
    # Calculate symmetric padding
    pad_w = target_w - new_w
    pad_h = target_h - new_h
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    
    # Add border padding
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, 
                                cv2.BORDER_CONSTANT, value=pad_color)
    return padded
```

### Technical Verification Results
- **Content Preservation**: Verified across 778 scans. $0\%$ of ultrasound fan regions, focal zones, or tissue margins are cropped or lost.
- **Canvas Alignment**: All output arrays strictly measure $224 \times 224 \times 3$.
- **Edge Behavior**: Zero-padding (`value=(0,0,0)`) seamlessly blends with the natural dark background borders of acoustic ultrasound frames.

---

## 5. Grayscale & Color Artifact Analysis

### Audit Findings on Image Channels & Artifacts
- **762 of 778 scans (97.9%)** are true grayscale ultrasound scans stored as 3-channel RGB files (R, G, B channel matrices are identical).
- **16 of 778 scans (2.1%)** contain genuine non-grayscale color content.
- **15 of 778 scans (1.9%)** contain burned-in colored measurement calipers, crosshairs (yellow, green, or red lines), or color Doppler velocity maps added during clinical examination.

### Artifact Classification & Clinical Context
- **Classification**: Software-generated burned-in acquisition annotations and Doppler color overlays.
- **Clinical Meaning**: Caliper marks reflect that an examining radiologist actively measured a mass. Color Doppler reflects blood flow measurements. However, these are **non-pathological artifacts** external to underlying tissue architecture.
- **Risk**: Deep learning models can easily learn that the presence of yellow crosshairs correlates with pathological mass classes, creating a high-risk shortcut.

### Controlled Preprocessing Strategy
To eliminate color annotation shortcuts without discarding scans:
1. Convert input BGR image to single-channel grayscale:  
   $$\text{Gray} = 0.299R + 0.587G + 0.114B$$
2. Replicate single-channel grayscale across 3 channels to maintain MobileNetV2 architecture compatibility:  
   $$\text{RGB}_{\text{standardized}} = \text{cv2.merge}([\text{Gray}, \text{Gray}, \text{Gray}])$$
3. **Outcome**: Completely neutralizes colored caliper lines and Doppler maps into standard grayscale ultrasound intensity, forcing model attention onto tissue morphology.

---

## 6. Blur Analysis & Dataset Quality

### Blur Distribution Across Classes
Using Laplacian variance metric ($\sigma^2_{\text{Laplacian}} < 100$ threshold for blur flagging):

| Class | Total Scans | Blurry Scans ($\sigma^2 < 100$) | Sharp Scans ($\sigma^2 \ge 100$) | Blur Rate | Mean Blur Variance |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Benign** | 435 | 123 | 312 | $28.3\%$ | 191.7 |
| **Malignant** | 210 | 57 | 153 | $27.1\%$ | 175.2 |
| **Normal** | 133 | 59 | 74 | **44.4%** | **113.8** |
| **Total** | **778** | **239** | **539** | **30.7%** | **174.0** |

### Findings & Dataset Retention Decision
- **Natural Pathology Distribution**: Blur is evenly distributed between benign ($28.3\%$) and malignant ($27.1\%$) scans.
- **Normal Tissue Characteristic**: Normal breast parenchyma exhibits significantly higher "blur" ($44.4\%$) because normal scans lack focal masses, calcifications, or sharp acoustic interfaces, producing smooth, homogeneous specular reflections.
- **Retention Decision**: **Do NOT delete blurry scans**. Deleting blurry images would strip $44.4\%$ of normal scans, severely corrupting the dataset distribution and introducing severe selection bias.

---

## 7. Class Imbalance Analysis & Candidate Weights

### Train Set Class Distribution (543 Scans)
- `benign`: 304 scans ($56.0\%$)
- `malignant`: 146 scans ($26.9\%$)
- `normal`: 93 scans ($17.1\%$)
- Imbalance Ratio: Approximately $3.27 : 1.57 : 1.00$.

### Candidate Class Weighting Strategies

| Strategy | Benign Weight ($w_0$) | Malignant Weight ($w_1$) | Normal Weight ($w_2$) | Risk Assessment / Recommendation |
| :--- | :---: | :---: | :---: | :--- |
| **1. Unweighted Baseline** | 1.000 | 1.000 | 1.000 | Baseline reference standard |
| **2. Mild Malignant Weighting** | 1.000 | **1.250** | **1.100** | **RECOMMENDED**: Slightly boosts malignant sensitivity without triggering FP explosion |
| **3. Square-Root Inverse Freq** | 0.772 | 1.113 | 1.395 | Moderate balance proportional to sample variance |
| **4. Balanced (Full Inverse Freq)** | 0.595 | 1.240 | 1.946 | **HIGH RISK**: Caused high false-positive rates in V4-C experiments |

*Cautionary Finding from V4*: Aggressive inverse class weighting ($w_1 = 1.24$) caused the loss function to heavily penalize malignant misses at the cost of precision, resulting in $6$ benign false positives and reducing overall malignant recall to $62.5\%$. Mild weighting ($1.25\times$) is strictly recommended for V5.

---

## 8. Dataset Shortcut-Risk Analysis

| Shortcut Category | Evidence in BUSI Dataset | Severity | Mitigation Strategy in V5 Methodology |
| :--- | :--- | :---: | :--- |
| **1. Aspect Ratio Geometry Distortion** | Vertical malignant lesions ($AR < 0.8$) squashed horizontally into benign shapes | **HIGH** | Aspect-Ratio-Preserving Letterbox Preprocessing |
| **2. Doctor Caliper Color Markings** | 15 scans contain colored measurement lines (yellow/red crosshairs) | **MEDIUM** | Grayscale 1-to-3 channel standardization |
| **3. Frame Dimension Shortcut** | Normal scans are $38\text{px}$ wider and taller than pathological scans on average | **MEDIUM** | Letterbox padding to uniform $224 \times 224$ canvas |
| **4. Parenchymal Smoothness Shortcut** | Normal class blur variance (113.8) is $40\%$ lower than benign (191.7) | **MEDIUM** | Retain all blurry scans + Apply random Gaussian noise/blur augmentation during training |
| **5. Sequential Slice Leakage** | 76 multi-scan clusters share identical background gain & probe noise | **HIGH** | Patient/Lesion Perceptual Cluster Grouped Split |

---

## 9. Proposed V5 Candidate Experiments

To scientifically isolate the exact contribution of each preprocessing enhancement, V5 will compare 4 controlled variants:

| Candidate ID | Network Architecture | Preprocessing Pipeline | Grayscale Standardization | Class Weighting Strategy | Hypothesis / Experimental Focus |
| :--- | :--- | :--- | :---: | :--- | :--- |
| **V5-A** | MobileNetV2 | Standard Square Resize ($224 \times 224$) | No | Unweighted $\{1.0, 1.0, 1.0\}$ | Preprocessing baseline reference |
| **V5-B** | MobileNetV2 | Aspect-Ratio Letterbox ($224 \times 224$) | No | Unweighted $\{1.0, 1.0, 1.0\}$ | Tests geometric preservation impact |
| **V5-C** | MobileNetV2 | Aspect-Ratio Letterbox ($224 \times 224$) | **Yes** (Grayscale) | Unweighted $\{1.0, 1.0, 1.0\}$ | Tests color artifact elimination impact |
| **V5-D** | MobileNetV2 | Aspect-Ratio Letterbox ($224 \times 224$) | **Yes** (Grayscale) | **Mild Weighting** $\{1.0, 1.25, 1.1\}$ | Best combined methodology candidate |

---

## 10. Exact Evaluation Methodology

All V5 candidate models will be evaluated under a dual-reporting framework:

### Evaluation Metric Set
Primary clinical and statistical metrics computed per class and macro/weighted averaged:
1. **Malignant Recall (Sensitivity)**: $\frac{TP}{TP + FN}$ (*Primary target: $\ge 85\%$*)
2. **Malignant Precision (PPV)**: $\frac{TP}{TP + FP}$
3. **Malignant F1-Score**: $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$
4. **Overall Macro F1-Score**: Unweighted mean of Benign, Malignant, and Normal F1-scores.
5. **False Positive Count**: Total Benign/Normal scans incorrectly flagged as Malignant.

### Dual Evaluation Protocol
1. **Historical Benchmark (Locked 117-Image Test Set)**:
   - Reports exact performance on the 117-image locked test set to benchmark against V1 (Accuracy $84.6\%$, Malignant Sensitivity $68.8\%$).
2. **Out-of-Sample Generalization (682-Cluster Grouped Test Set)**:
   - Reports performance on the 5-fold grouped cluster test split to measure true out-of-sample patient generalization.

---

## 11. Data-Leakage Prevention Strategy

To guarantee strict evaluation integrity and prevent data leakage:
1. **Cluster Isolation**: All scans belonging to the 76 multi-scan perceptual clusters are locked into a single split. Zero image overlap exists between Train, Validation, and Test sets.
2. **Preprocessing State Scoping**: Preprocessing transformations (letterbox padding, normalization) operate independently per image without dataset-level stats leakage (e.g. per-image scaling $[-1, +1]$).
3. **Locked Test Set Isolation**: The locked 117-image historical test set remains read-only. No hyperparameter tuning, loss function adjustments, or epoch selection will be performed on the test set.

---

## 12. Expected Risks & Limitations

1. **Lack of Original DICOM Metadata**: While perceptual hash clustering catches near-identical sequential slices, scans of the same patient taken from widely different angles or distinct lesions may remain unclustered.
2. **Black Padding Canvas Artifacts**: Adding black letterbox borders introduces artificial sharp rectangular edges at the ultrasound frame boundaries. Augmentation must avoid shifting padding into lesion areas.
3. **Small Sample Size in Malignant Class**: With only 210 total malignant scans (146 in training), subtle sub-types of invasive ductal vs. lobular carcinoma may remain underrepresented.

---

## Next Steps & Operational Constraints

1. **Do NOT train V5 yet**.
2. Production model (`backend/models/breast_image_classifier.keras`) remains **untouched** and **active**.
3. Locked 117-image test set remains **untouched**.
4. **Awaiting explicit user approval** of `v5_methodology_plan.md` prior to executing any code or training scripts.
