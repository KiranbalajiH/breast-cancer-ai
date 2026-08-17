import os
import sys
import random
import json
import shutil
import cv2
import numpy as np
import pandas as pd
import joblib

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from image_processing.preprocessing import preprocess_image
from image_processing.segmentation import segment_nuclei
from image_processing.feature_extraction import extract_all_nuclei_features, smooth_contour
from image_processing.aggregation import aggregate_features, FEATURE_NAMES_ORDERED
from image_processing.compatibility import validate_compatibility

# Old pipeline emulation helper
def old_pipeline_segmentation(gray_original, preprocessed, min_nucleus_area=50, max_nucleus_area_ratio=0.25):
    h, w = preprocessed.shape[:2]
    total_area = h * w
    max_nucleus_area = int(total_area * max_nucleus_area_ratio)
    
    _, binary_otsu = cv2.threshold(preprocessed, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    fg_ratio = np.count_nonzero(binary_otsu) / total_area
    if fg_ratio > 0.6:
        binary_otsu = cv2.bitwise_not(binary_otsu)
        
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(binary_otsu, cv2.MORPH_OPEN, kernel_open, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    sure_bg = cv2.dilate(cleaned, kernel_dilate, iterations=3)
    
    dist_transform = cv2.distanceTransform(cleaned, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    
    unknown = cv2.subtract(sure_bg, sure_fg)
    num_labels, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    
    color_for_watershed = cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(color_for_watershed, markers)
    
    contours_list = []
    labeled_mask = np.zeros((h, w), dtype=np.int32)
    nucleus_id = 1
    for label_id in range(2, num_labels + 1):
        region_mask = np.uint8(markers == label_id) * 255
        region_contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in region_contours:
            area = cv2.contourArea(cnt)
            if min_nucleus_area <= area <= max_nucleus_area:
                contours_list.append(cnt)
                cv2.drawContours(labeled_mask, [cnt], -1, nucleus_id, -1)
                nucleus_id += 1
                
    return contours_list, cleaned

def main():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    
    print(f"Using fixed random seed: {seed}")
    
    dataset_root = r"C:\Users\kiran\BCD\dataset"
    supported_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
    
    all_images = []
    for root, dirs, files in os.walk(dataset_root):
        path_parts = root.lower().split(os.sep)
        if any("mask" in part for part in path_parts):
            continue
        for f in files:
            if f.lower().endswith(supported_extensions):
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, dataset_root)
                parts = rel_path.split(os.sep)
                label = parts[0] if len(parts) > 1 else "Unknown"
                if len(parts) > 2:
                    label = f"{parts[0]}/{parts[1]}"
                all_images.append({
                    "full_path": full_path,
                    "filename": f,
                    "label": label,
                    "relative_path": rel_path
                })
                
    # Sort and sample
    all_images.sort(key=lambda x: x["relative_path"])
    selected_images = random.sample(all_images, min(5, len(all_images)))
    
    # Load model
    model_path = os.path.join(os.path.dirname(__file__), "models", "breast_cancer_model.joblib")
    metadata_path = os.path.join(os.path.dirname(__file__), "models", "metadata.json")
    model = joblib.load(model_path)
    with open(metadata_path, 'r') as f:
        model_metadata = json.load(f)
        
    diagnostics_dir = os.path.join(os.path.dirname(__file__), "diagnostics_test_outputs")
    if os.path.exists(diagnostics_dir):
        shutil.rmtree(diagnostics_dir)
    os.makedirs(diagnostics_dir)
    
    comparison_data = []
    
    for idx, img_info in enumerate(selected_images):
        img_path = img_info["full_path"]
        filename = img_info["filename"]
        label = img_info["label"]
        
        print("\n" + "="*50)
        print(f"Processing Image {idx+1}/5: {filename}")
        
        img = cv2.imread(img_path)
        if img is None:
            print("Error reading image.")
            continue
            
        gray_original, preprocessed = preprocess_image(img)
        prefix = f"img_{idx+1}_{os.path.splitext(filename)[0]}"
        
        # Save steps 1 and 2
        cv2.imwrite(os.path.join(diagnostics_dir, f"{prefix}_1_original.png"), img)
        cv2.imwrite(os.path.join(diagnostics_dir, f"{prefix}_2_preprocessed.png"), preprocessed)
        
        # --- Run INITIAL pipeline ---
        old_contours, old_cleaned = old_pipeline_segmentation(gray_original, preprocessed)
        old_features = []
        for c in old_contours:
            # Unsmoothed feature extraction, no QC filtering
            f = extract_all_nuclei_features([c], gray_original, return_qc=False, sigma=0.0, 
                                            min_area=0, max_area=999999, min_circularity=0.0, 
                                            min_solidity=0.0, min_aspect_ratio=0.0, exclude_border=False)
            if len(f) > 0:
                old_features.append(f[0])
                
        old_agg = aggregate_features(old_features)
        
        # --- Run IMPROVED pipeline ---
        seg_res = segment_nuclei(gray_original, preprocessed)
        new_features, qc = extract_all_nuclei_features(
            seg_res["contours"], 
            gray_original,
            return_qc=True,
            sigma=1.0,
            min_area=50,
            max_area=20000,
            min_circularity=0.65,
            min_solidity=0.85,
            min_aspect_ratio=0.4,
            exclude_border=True
        )
        new_agg = aggregate_features(new_features)
        
        # Save step 3: Raw segmentation mask (before watershed)
        cv2.imwrite(os.path.join(diagnostics_dir, f"{prefix}_3_raw_mask.png"), seg_res["binary_mask"])
        
        # Save step 4: Watershed-separated nuclei
        cv2.imwrite(os.path.join(diagnostics_dir, f"{prefix}_4_watershed_separated.png"), seg_res["diagnostics"]["watershed_separated"])
        
        # Save step 5: Accepted nuclei overlay
        accepted_overlay = cv2.cvtColor(gray_original, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(accepted_overlay, qc["accepted_contours"], -1, (0, 255, 0), 1)
        cv2.imwrite(os.path.join(diagnostics_dir, f"{prefix}_5_accepted_overlay.png"), accepted_overlay)
        
        # Save step 6: Rejected object overlay
        rejected_overlay = cv2.cvtColor(gray_original, cv2.COLOR_GRAY2BGR)
        for rej in qc["rejected_info"]:
            cv2.drawContours(rejected_overlay, [rej["contour"].astype(np.int32)], -1, (0, 0, 255), 1)
            # Add text label for reason
            M = cv2.moments(rej["contour"].astype(np.float32))
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(rejected_overlay, rej["reason"][:4], (cx-10, cy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 255), 1)
        cv2.imwrite(os.path.join(diagnostics_dir, f"{prefix}_6_rejected_overlay.png"), rejected_overlay)
        
        # Validate compatibility for both
        old_compat_verdict = "N/A"
        new_compat_verdict = "N/A"
        old_compat_fails = []
        new_compat_fails = []
        
        if old_agg:
            old_compat = validate_compatibility(old_agg[0])
            old_compat_verdict = old_compat["overall_verdict"]
            old_compat_fails = [f['name'] for f in old_compat['per_feature'] if f['verdict'] == 'Incompatible']
            
        if new_agg:
            new_compat = validate_compatibility(new_agg[0])
            new_compat_verdict = new_compat["overall_verdict"]
            new_compat_fails = [f['name'] for f in new_compat['per_feature'] if f['verdict'] == 'Incompatible']
            
        # Before/after metric averages
        old_mean_area = np.mean([f["area"] for f in old_features]) if old_features else 0.0
        old_mean_perimeter = np.mean([f["perimeter"] for f in old_features]) if old_features else 0.0
        old_mean_compactness = np.mean([f["compactness"] for f in old_features]) if old_features else 0.0
        
        new_mean_area = np.mean([f["area"] for f in new_features]) if new_features else 0.0
        new_mean_perimeter = np.mean([f["perimeter"] for f in new_features]) if new_features else 0.0
        new_mean_compactness = np.mean([f["compactness"] for f in new_features]) if new_features else 0.0
        
        print(f"Initial: {len(old_contours)} raw, mean area: {old_mean_area:.2f}, mean perimeter: {old_mean_perimeter:.2f}, mean compactness: {old_mean_compactness:.2f}")
        print(f"Improved: {qc['raw_count']} raw, {qc['accepted_count']} accepted, {qc['rejected_count']} rejected")
        print(f"          mean area: {new_mean_area:.2f}, mean perimeter: {new_mean_perimeter:.2f}, mean compactness: {new_mean_compactness:.2f}")
        print(f"Improved Verdict: {new_compat_verdict}")
        
        comparison_data.append({
            "filename": filename,
            "label": label,
            "initial": {
                "detected": len(old_contours),
                "mean_area": float(old_mean_area),
                "mean_perimeter": float(old_mean_perimeter),
                "mean_compactness": float(old_mean_compactness),
                "compatibility": old_compat_verdict,
                "incompat_count": len(old_compat_fails)
            },
            "improved": {
                "detected": qc["raw_count"],
                "accepted": qc["accepted_count"],
                "rejected": qc["rejected_count"],
                "mean_area": float(new_mean_area),
                "mean_perimeter": float(new_mean_perimeter),
                "mean_compactness": float(new_mean_compactness),
                "compatibility": new_compat_verdict,
                "incompat_count": len(new_compat_fails),
                "incompat_features": new_compat_fails
            }
        })
        
    with open(os.path.join(os.path.dirname(__file__), "diagnostic_comparison_results.json"), 'w') as f:
        json.dump({"seed": seed, "comparisons": comparison_data}, f, indent=2)
        
    print("\nDiagnostic comparison run completed.")

if __name__ == "__main__":
    main()
