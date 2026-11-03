import os
import sys
import json
import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

import evaluate_rigorous as er
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_mnv2
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_effnet

def apply_artifact_masking(img_rgb, mask_type="border_crop"):
    """
    Masks typical ultrasound shortcut artifacts:
    1. border_crop: Insets 8% from top, bottom, left, right to eliminate edge markers, dates, machine brands, calipers.
    2. top_bottom_mask: Zeroes out top 12% (patient name, machine header) and bottom 10% (scale bars, brand logos).
    3. high_intensity_text_mask: Inpainting/zeroing out small saturated white text overlays (intensity > 240).
    """
    h, w, c = img_rgb.shape
    masked = img_rgb.copy()
    
    if mask_type == "border_crop":
        # Crop 8% margin and resize back to original shape
        dh = int(0.08 * h)
        dw = int(0.08 * w)
        cropped = img_rgb[dh:h-dh, dw:w-dw]
        masked = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        
    elif mask_type == "top_bottom_mask":
        # Zero out typical header (12%) and footer (10%)
        top_h = int(0.12 * h)
        bot_h = int(0.10 * h)
        masked[:top_h, :] = 0
        masked[h-bot_h:, :] = 0
        
    elif mask_type == "high_intensity_overlay_suppression":
        # Detect small isolated saturated pixels (calipers, text characters)
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        # Text/caliper markers in ultrasound are almost pure white (>245)
        text_mask = (gray > 240).astype(np.uint8)
        # Remove large contiguous white areas so we only target small text/crosshair artifacts
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # Keep components that are small
        num_labels, labels, stats_cc, _ = cv2.connectedComponentsWithStats(text_mask)
        small_text_mask = np.zeros_like(text_mask)
        for i in range(1, num_labels):
            area = stats_cc[i, cv2.CC_STAT_AREA]
            if 3 < area < 500: # Typical text glyph or caliper crosshair
                small_text_mask[labels == i] = 1
                
        # Inpaint or replace with local median
        masked[small_text_mask == 1] = 0
        
    return masked

def run_shortcut_audit():
    er.set_seed(42)
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    bcd_root = os.path.dirname(backend_dir)
    dataset_root = os.path.join(bcd_root, "dataset", "BUSI")
    
    scans, classes = er.load_and_clean_dataset(dataset_root, backend_dir)
    labels = np.array([s["class_idx"] for s in scans], dtype=np.int32)
    cluster_labels, total_clusters = er.group_scans_by_perceptual_cluster(scans)
    
    grp_train_idx, grp_val_idx, grp_test_idx = er.create_grouped_stratified_split(scans, cluster_labels, seed=42)
    
    # Load Models
    v5b_path = os.path.join(backend_dir, "models", "breast_image_classifier.keras")
    v11b_path = os.path.join(backend_dir, "models", "candidates", "v11_mobilenetv2_b.keras")
    v13_path = os.path.join(backend_dir, "models", "candidates", "v13_efficientnet_b0.keras")
    
    m_v5b = tf.keras.models.load_model(v5b_path)
    m_v11b = tf.keras.models.load_model(v11b_path)
    m_v13 = tf.keras.models.load_model(v13_path)
    
    test_scans = [scans[i] for i in grp_test_idx]
    y_test = labels[grp_test_idx]
    
    mask_variants = ["unmodified", "border_crop", "top_bottom_mask", "high_intensity_overlay_suppression"]
    variant_results = {}
    
    for variant in mask_variants:
        images = []
        for s in test_scans:
            img = cv2.imread(s["filepath"])
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if variant != "unmodified":
                img_rgb = apply_artifact_masking(img_rgb, mask_type=variant)
            img_letterbox = er.letterbox_resize(img_rgb, (224, 224))
            images.append(img_letterbox)
        X = np.array(images, dtype=np.float32)
        
        # Inference
        p_v5b = m_v5b.predict(preprocess_mnv2(X.copy()), verbose=0)
        p_v11b = m_v11b.predict(preprocess_mnv2(X.copy()), verbose=0)
        p_v13 = m_v13.predict(preprocess_effnet(X.copy()), verbose=0)
        p_v14 = 0.5 * p_v11b + 0.5 * p_v13
        
        pred_v5b = np.argmax(p_v5b, axis=1)
        pred_v14 = np.zeros(len(p_v14), dtype=int)
        for i, p in enumerate(p_v14):
            if p[1] >= 0.58:
                pred_v14[i] = 1
            else:
                pred_v14[i] = 0 if p[0] >= p[2] else 2
                
        def calc_diag(y_t, y_p):
            acc = float(accuracy_score(y_t, y_p))
            _, _, f1_macro, _ = precision_recall_fscore_support(y_t, y_p, average="macro", zero_division=0)
            cm = confusion_matrix(y_t, y_p, labels=[0, 1, 2])
            # Malignant (idx 1)
            tp = int(cm[1, 1])
            fn = int(cm[1, 0] + cm[1, 2])
            fp = int(cm[0, 1] + cm[2, 1])
            tn = int(cm[0, 0] + cm[0, 2] + cm[2, 0] + cm[2, 2])
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
            return {
                "accuracy": round(acc * 100, 2),
                "macro_f1": round(float(f1_macro) * 100, 2),
                "malignant_sensitivity": round(sens * 100, 2),
                "malignant_specificity": round(spec * 100, 2),
                "malignant_ppv": round(ppv * 100, 2),
                "malignant_npv": round(npv * 100, 2),
                "malignant_fn": fn,
                "malignant_fp": fp,
                "confusion_matrix": cm.tolist()
            }
            
        variant_results[variant] = {
            "v5b": calc_diag(y_test, pred_v5b),
            "v14": calc_diag(y_test, pred_v14)
        }
        
        print(f"\nVariant [{variant}]:")
        print(f"  V5-B : Acc={variant_results[variant]['v5b']['accuracy']}%, Mal Sens={variant_results[variant]['v5b']['malignant_sensitivity']}%, FN={variant_results[variant]['v5b']['malignant_fn']}, FP={variant_results[variant]['v5b']['malignant_fp']}")
        print(f"  V14  : Acc={variant_results[variant]['v14']['accuracy']}%, Mal Sens={variant_results[variant]['v14']['malignant_sensitivity']}%, FN={variant_results[variant]['v14']['malignant_fn']}, FP={variant_results[variant]['v14']['malignant_fp']}")
        
    # Save Report
    out_path = os.path.join(backend_dir, "models", "evaluation_reports", "shortcut_learning_audit.json")
    with open(out_path, "w") as f:
        json.dump(variant_results, f, indent=2)
    print(f"\nShortcut audit saved to {out_path}")

if __name__ == "__main__":
    run_shortcut_audit()
