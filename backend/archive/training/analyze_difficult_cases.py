import os
import json
import cv2
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(script_dir)
bcd_root = os.path.dirname(backend_dir)
dataset_root = os.path.join(bcd_root, "dataset", "BUSI")

v1_eval_json = os.path.join(backend_dir, "models", "candidates", "v4_evaluation_summary.json")

with open(v1_eval_json) as f:
    eval_data = json.load(f)

v1_mis = eval_data["v1_baseline"]["misclassifications"]

mal_fn_files = [x["filename"] for x in v1_mis if x["true_class"] == "malignant" and x["predicted_class"] == "benign"]
ben_fp_files = [x["filename"] for x in v1_mis if x["true_class"] == "benign" and x["predicted_class"] == "malignant"]
ben_norm_files = [x["filename"] for x in v1_mis if x["true_class"] == "benign" and x["predicted_class"] == "normal"]
norm_ben_files = [x["filename"] for x in v1_mis if x["true_class"] == "normal" and x["predicted_class"] == "benign"]

print("--- DIFFICULT CASE DETAILED FORENSIC METRICS ---")

def get_image_metrics(filename, class_folder):
    path = os.path.join(dataset_root, class_folder, filename)
    if not os.path.exists(path):
        return None
    img = cv2.imread(path)
    h, w, c = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return {
        "filename": filename,
        "width": w,
        "height": h,
        "aspect_ratio": round(w/h, 2),
        "mean_intensity": round(float(np.mean(gray)), 1),
        "std_intensity": round(float(np.std(gray)), 1),
        "blur_var": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 1)
    }

print("\n1. Malignant False Negatives (True: Malignant -> Pred: Benign) [10 cases]:")
for fn in mal_fn_files:
    m = get_image_metrics(fn, "malignant")
    conf = [x["confidence"] for x in v1_mis if x["filename"] == fn][0]
    print(f"  - {fn:<20}: {m['width']}x{m['height']} (AR: {m['aspect_ratio']}), MeanInt: {m['mean_intensity']}, StdInt: {m['std_intensity']}, BlurVar: {m['blur_var']}, Pred Benign Conf: {conf*100:.1f}%")

print("\n2. Benign False Positives (True: Benign -> Pred: Malignant) [4 cases]:")
for fn in ben_fp_files:
    m = get_image_metrics(fn, "benign")
    conf = [x["confidence"] for x in v1_mis if x["filename"] == fn][0]
    print(f"  - {fn:<20}: {m['width']}x{m['height']} (AR: {m['aspect_ratio']}), MeanInt: {m['mean_intensity']}, StdInt: {m['std_intensity']}, BlurVar: {m['blur_var']}, Pred Malignant Conf: {conf*100:.1f}%")

print("\n3. Benign -> Normal Errors [2 cases]:")
for fn in ben_norm_files:
    m = get_image_metrics(fn, "benign")
    conf = [x["confidence"] for x in v1_mis if x["filename"] == fn][0]
    print(f"  - {fn:<20}: {m['width']}x{m['height']} (AR: {m['aspect_ratio']}), MeanInt: {m['mean_intensity']}, StdInt: {m['std_intensity']}, BlurVar: {m['blur_var']}, Pred Normal Conf: {conf*100:.1f}%")

print("\n4. Normal -> Benign Errors [2 cases]:")
for fn in norm_ben_files:
    m = get_image_metrics(fn, "normal")
    conf = [x["confidence"] for x in v1_mis if x["filename"] == fn][0]
    print(f"  - {fn:<20}: {m['width']}x{m['height']} (AR: {m['aspect_ratio']}), MeanInt: {m['mean_intensity']}, StdInt: {m['std_intensity']}, BlurVar: {m['blur_var']}, Pred Benign Conf: {conf*100:.1f}%")

