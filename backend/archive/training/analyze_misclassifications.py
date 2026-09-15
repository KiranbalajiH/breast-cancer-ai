import os
import sys
import json
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

script_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(script_dir)
bcd_root = os.path.dirname(backend_dir)
dataset_root = os.path.join(bcd_root, "dataset", "BUSI")

sys.path.insert(0, backend_dir)
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

def set_seed(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)

def load_clean_dataset(dataset_root):
    classes = ['benign', 'malignant', 'normal']
    exclude_files = {"malignant (145).png", "benign (433).png"}
    scans = []
    
    for class_idx, class_name in enumerate(classes):
        class_dir = os.path.join(dataset_root, class_name)
        if not os.path.isdir(class_dir):
            continue
        for f in sorted(os.listdir(class_dir)):
            if not f.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            if '_mask' in f.lower() or 'mask' in f.lower():
                continue
            if f in exclude_files:
                continue
            
            full_path = os.path.join(class_dir, f)
            scans.append({
                "filename": f,
                "filepath": full_path,
                "class_name": class_name,
                "class_idx": class_idx
            })
    return scans, classes

def analyze_scan_properties(filepath):
    img = cv2.imread(filepath)
    if img is None:
        return {}
    h, w, c = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean_val = float(np.mean(gray))
    std_val = float(np.std(gray))
    
    # Check associated mask if available
    dir_name, base_name = os.path.split(filepath)
    name_no_ext, ext = os.path.splitext(base_name)
    mask_path = os.path.join(dir_name, f"{name_no_ext}_mask{ext}")
    mask_area_pct = 0.0
    if os.path.exists(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            mask_area_pct = float(np.count_nonzero(mask) / (h * w) * 100)
            
    return {
        "width": w,
        "height": h,
        "aspect_ratio": round(w / h, 2),
        "mean_intensity": round(mean_val, 1),
        "std_contrast": round(std_val, 1),
        "blur_laplacian_var": round(laplacian_var, 1),
        "mask_area_pct": round(mask_area_pct, 2)
    }

def main():
    set_seed(42)
    scans, classes = load_clean_dataset(dataset_root)
    labels = np.array([s["class_idx"] for s in scans], dtype=np.int32)
    
    # 30% Test split seed=42
    indices = np.arange(len(scans))
    train_idx, val_test_idx = train_test_split(indices, test_size=0.30, stratify=labels, random_state=42)
    val_idx, test_idx = train_test_split(val_test_idx, test_size=0.50, stratify=labels[val_test_idx], random_state=42)
    
    test_scans = [scans[i] for i in test_idx]
    
    print(f"Total Test Set Scans: {len(test_scans)}")
    print(f"  Benign: {sum(1 for s in test_scans if s['class_name'] == 'benign')}")
    print(f"  Malignant: {sum(1 for s in test_scans if s['class_name'] == 'malignant')}")
    print(f"  Normal: {sum(1 for s in test_scans if s['class_name'] == 'normal')}")
    
    # Load V1 baseline model or active model
    model_path = os.path.join(backend_dir, "models", "breast_image_classifier.keras")
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return
        
    print(f"\nLoading Active Model: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)
    
    images = []
    for s in test_scans:
        img = cv2.imread(s["filepath"])
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Letterbox to 224x224
        img_prep = letterbox_resize(img_rgb, (224, 224))
        images.append(img_prep)
    images = np.array(images, dtype=np.float32)
    images_prep = tf.keras.applications.mobilenet_v2.preprocess_input(images.copy())
    
    preds = model.predict(images_prep, verbose=0)
    pred_classes = np.argmax(preds, axis=1)
    
    misclassifications = []
    
    for i, s in enumerate(test_scans):
        true_c = s["class_idx"]
        pred_c = pred_classes[i]
        
        if true_c != pred_c:
            props = analyze_scan_properties(s["filepath"])
            misclassifications.append({
                "filename": s["filename"],
                "true_class": s["class_name"],
                "pred_class": classes[pred_c],
                "confidence": float(preds[i][pred_c]),
                "probabilities": {
                    "benign": float(preds[i][0]),
                    "malignant": float(preds[i][1]),
                    "normal": float(preds[i][2])
                },
                "properties": props
            })
            
    print(f"\nTotal Misclassifications on Test Set: {len(misclassifications)} / {len(test_scans)}")
    
    # Categorize errors
    fn_list = [m for m in misclassifications if m["true_class"] == "malignant"]
    fp_list = [m for m in misclassifications if m["true_class"] != "malignant" and m["pred_class"] == "malignant"]
    other_list = [m for m in misclassifications if m not in fn_list and m not in fp_list]
    
    print("\n" + "="*80)
    print("1. MALIGNANT FALSE NEGATIVES (CRITICAL ERRORS: True Malignant -> Pred Benign/Normal)")
    print("="*80)
    for m in fn_list:
        p = m["properties"]
        print(f"File: {m['filename']:<22} | True: {m['true_class']:<10} -> Pred: {m['pred_class']:<10} (Conf: {m['confidence']*100:5.1f}%)")
        print(f"   Probs -> Ben: {m['probabilities']['benign']*100:5.1f}% | Mal: {m['probabilities']['malignant']*100:5.1f}% | Nor: {m['probabilities']['normal']*100:5.1f}%")
        print(f"   Properties -> Size: {p.get('width')}x{p.get('height')} (AR: {p.get('aspect_ratio')}) | BlurVar: {p.get('blur_laplacian_var'):6.1f} | MeanInt: {p.get('mean_intensity'):5.1f} | MaskArea: {p.get('mask_area_pct'):4.1f}%\n")
        
    print("\n" + "="*80)
    print("2. MALIGNANT FALSE POSITIVES (OVER-DIAGNOSIS ERRORS: True Benign/Normal -> Pred Malignant)")
    print("="*80)
    for m in fp_list:
        p = m["properties"]
        print(f"File: {m['filename']:<22} | True: {m['true_class']:<10} -> Pred: {m['pred_class']:<10} (Conf: {m['confidence']*100:5.1f}%)")
        print(f"   Probs -> Ben: {m['probabilities']['benign']*100:5.1f}% | Mal: {m['probabilities']['malignant']*100:5.1f}% | Nor: {m['probabilities']['normal']*100:5.1f}%")
        print(f"   Properties -> Size: {p.get('width')}x{p.get('height')} (AR: {p.get('aspect_ratio')}) | BlurVar: {p.get('blur_laplacian_var'):6.1f} | MeanInt: {p.get('mean_intensity'):5.1f} | MaskArea: {p.get('mask_area_pct'):4.1f}%\n")

    print("\n" + "="*80)
    print("3. OTHER MISCLASSIFICATIONS (BENIGN <-> NORMAL)")
    print("="*80)
    for m in other_list:
        p = m["properties"]
        print(f"File: {m['filename']:<22} | True: {m['true_class']:<10} -> Pred: {m['pred_class']:<10} (Conf: {m['confidence']*100:5.1f}%)")
        print(f"   Probs -> Ben: {m['probabilities']['benign']*100:5.1f}% | Mal: {m['probabilities']['malignant']*100:5.1f}% | Nor: {m['probabilities']['normal']*100:5.1f}%")
        print(f"   Properties -> Size: {p.get('width')}x{p.get('height')} (AR: {p.get('aspect_ratio')}) | BlurVar: {p.get('blur_laplacian_var'):6.1f} | MeanInt: {p.get('mean_intensity'):5.1f} | MaskArea: {p.get('mask_area_pct'):4.1f}%\n")

    # Aggregate metric comparisons between Correct vs Misclassified
    all_fn_blur = [m["properties"].get("blur_laplacian_var", 0) for m in fn_list]
    all_fn_ar = [m["properties"].get("aspect_ratio", 0) for m in fn_list]
    all_fn_mask = [m["properties"].get("mask_area_pct", 0) for m in fn_list]
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICAL PATTERNS IN MISCLASSIFICATIONS")
    print("="*80)
    if fn_list:
        print(f"Malignant False Negatives ({len(fn_list)} cases):")
        print(f"  - Avg Blur Variance (Laplacian): {np.mean(all_fn_blur):.1f} (Lower variance = blurrier)")
        print(f"  - Avg Aspect Ratio (W/H): {np.mean(all_fn_ar):.2f}")
        print(f"  - Avg Lesion Area (% of Scan): {np.mean(all_fn_mask):.2f}%")

if __name__ == "__main__":
    main()
