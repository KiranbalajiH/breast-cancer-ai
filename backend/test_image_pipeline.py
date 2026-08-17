"""
Test script for the experimental image analysis pipeline.
Creates a synthetic microscopy-like image with dark circles (simulated nuclei)
on a light background, then runs the full pipeline.
"""

import cv2
import numpy as np
import sys
import os
import json

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from image_processing.preprocessing import preprocess_image
from image_processing.segmentation import segment_nuclei
from image_processing.feature_extraction import extract_all_nuclei_features
from image_processing.aggregation import aggregate_features, FEATURE_NAMES_ORDERED
from image_processing.compatibility import validate_compatibility


def create_synthetic_nuclei_image(width=512, height=512, num_nuclei=15):
    """Create a synthetic image with dark blob-like nuclei on a light background."""
    # Light gray background with some texture noise
    img = np.ones((height, width, 3), dtype=np.uint8) * 220
    noise = np.random.normal(0, 8, (height, width, 3)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    rng = np.random.RandomState(42)
    nuclei_info = []
    
    for _ in range(num_nuclei):
        # Random position (avoid edges)
        cx = rng.randint(50, width - 50)
        cy = rng.randint(50, height - 50)
        
        # Random size
        rx = rng.randint(12, 30)
        ry = rng.randint(12, 30)
        
        # Dark nucleus color with variation
        color_val = rng.randint(40, 100)
        color = (color_val, color_val + rng.randint(-10, 10), color_val + rng.randint(-10, 10))
        
        # Draw filled ellipse
        angle = rng.randint(0, 180)
        cv2.ellipse(img, (cx, cy), (rx, ry), angle, 0, 360, color, -1)
        
        # Add some internal texture
        for _ in range(rng.randint(2, 5)):
            dx = rng.randint(-rx//2, rx//2)
            dy = rng.randint(-ry//2, ry//2)
            r = rng.randint(2, 5)
            cv2.circle(img, (cx + dx, cy + dy), r, 
                      (color_val + rng.randint(-20, 20),) * 3, -1)
        
        nuclei_info.append({"cx": cx, "cy": cy, "rx": rx, "ry": ry})
    
    return img, nuclei_info


def main():
    print("=" * 60)
    print("EXPERIMENTAL IMAGE ANALYSIS PIPELINE TEST")
    print("=" * 60)
    
    # Step 1: Create synthetic image
    print("\n1. Creating synthetic cell nuclei image...")
    image, nuclei_info = create_synthetic_nuclei_image(num_nuclei=15)
    print(f"   Image shape: {image.shape}")
    print(f"   Synthetic nuclei placed: {len(nuclei_info)}")
    
    # Save synthetic image for inspection
    test_img_path = os.path.join(os.path.dirname(__file__), "test_synthetic_nuclei.png")
    cv2.imwrite(test_img_path, image)
    print(f"   Saved to: {test_img_path}")
    
    # Step 2: Preprocess
    print("\n2. Preprocessing...")
    gray_original, preprocessed = preprocess_image(image)
    print(f"   Gray shape: {gray_original.shape}")
    print(f"   Preprocessed shape: {preprocessed.shape}")
    
    # Step 3: Segment
    print("\n3. Segmenting nuclei...")
    seg_result = segment_nuclei(gray_original, preprocessed)
    print(f"   Nuclei detected: {seg_result['num_nuclei']}")
    
    # Save overlay for inspection
    overlay_path = os.path.join(os.path.dirname(__file__), "test_nuclei_overlay.png")
    cv2.imwrite(overlay_path, seg_result["overlay"])
    print(f"   Overlay saved: {overlay_path}")
    
    if seg_result["num_nuclei"] == 0:
        print("   ERROR: No nuclei detected! Pipeline cannot continue.")
        return
    
    # Step 4: Extract features
    print("\n4. Extracting per-nucleus features...")
    nuclei_features = extract_all_nuclei_features(seg_result["contours"], gray_original)
    print(f"   Successfully measured: {len(nuclei_features)} nuclei")
    
    if len(nuclei_features) > 0:
        sample = nuclei_features[0]
        print(f"   Sample nucleus features:")
        for k, v in sample.items():
            print(f"     {k}: {v:.6f}")
    
    # Step 5: Aggregate
    print("\n5. Aggregating to 30-feature vector...")
    agg_result = aggregate_features(nuclei_features)
    
    if agg_result is None:
        print("   ERROR: Aggregation failed (insufficient nuclei).")
        return
    
    features_30, agg_metadata = agg_result
    print(f"   Feature vector generated ({len(features_30)} features)")
    print(f"   Features in correct order:")
    for name in FEATURE_NAMES_ORDERED:
        val = features_30.get(name, "MISSING")
        print(f"     {name:30s}: {val:.6f}" if isinstance(val, float) else f"     {name:30s}: {val}")
    
    # Step 6: Compatibility validation
    print("\n6. Running compatibility validation...")
    compat = validate_compatibility(features_30)
    print(f"   Overall verdict: {compat['overall_verdict']}")
    print(f"   Compatible: {compat['num_compatible']}, Marginal: {compat['num_marginal']}, Incompatible: {compat['num_incompatible']}")
    print(f"   Prediction allowed: {compat['prediction_allowed']}")
    print(f"   Message: {compat['message']}")
    
    print("\n   Per-feature report:")
    for f in compat["per_feature"]:
        val_str = f"{f['extracted']:.6f}" if f['extracted'] is not None else "MISSING"
        z_str = f"z={f['z_score']:.1f}" if f['z_score'] is not None else "z=N/A"
        print(f"     {f['name']:30s}: {val_str:>14s}  {z_str:>8s}  [{f['verdict']}]")
    
    # Step 7: Test existing prediction endpoint still works
    print("\n7. Verifying existing manual prediction still works...")
    import urllib.request
    benign_data = json.dumps({
        "mean radius": 13.54, "mean texture": 14.36, "mean perimeter": 87.46,
        "mean area": 566.3, "mean smoothness": 0.09779, "mean compactness": 0.08129,
        "mean concavity": 0.06664, "mean concave points": 0.04781,
        "mean symmetry": 0.1885, "mean fractal dimension": 0.05766,
        "radius error": 0.2699, "texture error": 0.7886, "perimeter error": 2.058,
        "area error": 23.56, "smoothness error": 0.008462, "compactness error": 0.0146,
        "concavity error": 0.02387, "concave points error": 0.01315,
        "symmetry error": 0.0198, "fractal dimension error": 0.0023,
        "worst radius": 15.11, "worst texture": 19.26, "worst perimeter": 99.7,
        "worst area": 711.2, "worst smoothness": 0.144, "worst compactness": 0.1773,
        "worst concavity": 0.239, "worst concave points": 0.1288,
        "worst symmetry": 0.2977, "worst fractal dimension": 0.07259
    }).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:8000/api/predict",
        data=benign_data,
        headers={"Content-Type": "application/json"}
    )
    try:
        resp = urllib.request.urlopen(req)
        result = json.loads(resp.read())
        print(f"   Manual prediction: {result['prediction']} (confidence: {result['confidence']:.4f})")
        print(f"   [OK] Existing pipeline is UNAFFECTED")
    except Exception as e:
        print(f"   [FAIL] Existing pipeline test FAILED: {e}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
