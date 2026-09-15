import os
import sys
import cv2
import json
import hashlib
import numpy as np
from collections import Counter, defaultdict

script_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(script_dir)
bcd_root = os.path.dirname(backend_dir)
dataset_root = os.path.join(bcd_root, "dataset", "BUSI")

classes = ['benign', 'malignant', 'normal']

print("=======================================================================")
print("V5 METHODOLOGY & FORENSIC INVESTIGATION SCRIPT")
print("=======================================================================\n")

# 1. TASK 1: PATIENT / LESION GROUPING INVESTIGATION
print("--- TASK 1: PATIENT / LESION GROUPING & IDENTIFIER INVESTIGATION ---")
# Check image filenames and EXIF/metadata
all_scans = []
for c in classes:
    cdir = os.path.join(dataset_root, c)
    files = sorted([f for f in os.listdir(cdir) if f.endswith('.png') and 'mask' not in f.lower()])
    for f in files:
        fpath = os.path.join(cdir, f)
        img = cv2.imread(fpath)
        if img is None:
            continue
        h, w, ch = img.shape
        all_scans.append({
            "class": c,
            "filename": f,
            "filepath": fpath,
            "height": h,
            "width": w,
            "aspect_ratio": round(w / h, 3),
            "mean_int": float(np.mean(img)),
            "std_int": float(np.std(img)),
            "blur_var": float(cv2.Laplacian(img, cv2.CV_64F).var())
        })

print(f"Total Clean Scans Loaded: {len(all_scans)}")

# Let's test perceptual hash / structural similarity to group sequential scans of the same lesion
# For images in the same class, check if resizing to 32x32 grayscale and computing mean absolute difference (MAD) < threshold identifies near-identical slice cuts.
def compute_phash(img_gray):
    resized = cv2.resize(img_gray, (32, 32))
    return resized

grouped_clusters = defaultdict(list)
visited = set()

# Perceptual similarity clustering per class
class_scans = defaultdict(list)
for s in all_scans:
    class_scans[s['class']].append(s)

clusters_by_class = {}

for c, scans in class_scans.items():
    phashes = [compute_phash(cv2.imread(s['filepath'], cv2.IMREAD_GRAYSCALE)) for s in scans]
    adj = defaultdict(list)
    n = len(scans)
    
    for i in range(n):
        for j in range(i + 1, n):
            diff = np.mean(np.abs(phashes[i].astype(float) - phashes[j].astype(float)))
            # If difference is very low (< 12.0 out of 255), they are near-identical scans/slices of the same lesion
            if diff < 12.0:
                adj[i].append(j)
                adj[j].append(i)
                
    # Connected components to find patient/lesion clusters
    visited_idx = set()
    c_clusters = []
    for i in range(n):
        if i not in visited_idx:
            component = []
            queue = [i]
            visited_idx.add(i)
            while queue:
                curr = queue.pop(0)
                component.append(scans[curr]['filename'])
                for nbr in adj[curr]:
                    if nbr not in visited_idx:
                        visited_idx.add(nbr)
                        queue.append(nbr)
            c_clusters.append(component)
            
    clusters_by_class[c] = c_clusters
    multi_clusters = [cl for cl in c_clusters if len(cl) > 1]
    print(f"  Class '{c:<9}': {len(scans)} scans grouped into {len(c_clusters)} patient/lesion clusters ({len(multi_clusters)} clusters have multiple scans, total {sum(len(cl) for cl in multi_clusters)} scans)")

print("\nSample Multi-Image Patient/Lesion Clusters Identified:")
for c in classes:
    multi = [cl for cl in clusters_by_class[c] if len(cl) > 1][:3]
    for idx, cl in enumerate(multi):
        print(f"  [{c} Cluster {idx+1}]: {cl}")


# 2. TASK 2: ASPECT RATIO DISTORTION & LETTERBOX ANALYSIS
print("\n--- TASK 2: ASPECT RATIO DISTORTION & LETTERBOX ANALYSIS ---")
ars = [s['aspect_ratio'] for s in all_scans]

ar_extreme_low = [s for s in all_scans if s['aspect_ratio'] < 0.8]  # Tall/Vertical (e.g. malignant 1)
ar_square = [s for s in all_scans if 0.8 <= s['aspect_ratio'] <= 1.2] # Roughly square
ar_wide = [s for s in all_scans if 1.2 < s['aspect_ratio'] <= 1.5]   # Moderately wide
ar_extreme_high = [s for s in all_scans if s['aspect_ratio'] > 1.5] # Extremely wide

print(f"Aspect Ratio Distribution:")
print(f"  - Vertical / Tall (AR < 0.8)       : {len(ar_extreme_low)} scans ({len(ar_extreme_low)/len(all_scans)*100:.1f}%)")
print(f"  - Near-Square (0.8 <= AR <= 1.2)   : {len(ar_square)} scans ({len(ar_square)/len(all_scans)*100:.1f}%)")
print(f"  - Moderately Wide (1.2 < AR <= 1.5): {len(ar_wide)} scans ({len(ar_wide)/len(all_scans)*100:.1f}%)")
print(f"  - Extremely Wide (AR > 1.5)        : {len(ar_extreme_high)} scans ({len(ar_extreme_high)/len(all_scans)*100:.1f}%)")

# Calculate distortion metric for square resize vs letterbox
# When squashing W x H to 224 x 224:
# Vertical scaling factor S_v = 224 / H
# Horizontal scaling factor S_h = 224 / W
# Distortion ratio = S_h / S_v = H / W = 1 / Aspect Ratio
distortions = [abs(1.0 - (s['height'] / s['width'])) * 100 for s in all_scans]
print(f"Average Pixel Stretching Distortion under Square Resize: {np.mean(distortions):.1f}% (Max: {max(distortions):.1f}%)")

def letterbox_resize(img, target_size=(224, 224), pad_color=(0,0,0)):
    h, w = img.shape[:2]
    target_w, target_h = target_size
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    
    pad_w = target_w - new_w
    pad_h = target_h - new_h
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color)
    return padded, scale, (left, top)

# Test letterbox on sample image
sample_img = cv2.imread(all_scans[0]['filepath'])
lb_img, scale, pad = letterbox_resize(sample_img)
print(f"Letterbox verification: Input {sample_img.shape} -> Scale {scale:.3f}, Padding {pad} -> Output {lb_img.shape}")


# 3. TASK 3 & TASK 5: GRAYSCALE & BLUR ANALYSIS BY CLASS
print("\n--- TASK 3 & TASK 5: GRAYSCALE & BLUR DISTRIBUTION BY CLASS ---")
for c in classes:
    c_scans = [s for s in all_scans if s['class'] == c]
    tot = len(c_scans)
    
    # Blurry scans
    blurry = [s for s in c_scans if s['blur_var'] < 100]
    
    # Non-grayscale
    non_gray = []
    for s in c_scans:
        img = cv2.imread(s['filepath'])
        b, g, r = img[:,:,0], img[:,:,1], img[:,:,2]
        if np.max(np.abs(b.astype(int) - g.astype(int))) > 15 or np.max(np.abs(g.astype(int) - r.astype(int))) > 15:
            non_gray.append(s)
            
    print(f"Class '{c:<9}' ({tot} total scans):")
    print(f"  - Blurry Scans (Laplacian < 100) : {len(blurry):2d} / {tot:3d} ({len(blurry)/tot*100:4.1f}%)")
    print(f"  - Scans with Color Annotations   : {len(non_gray):2d} / {tot:3d} ({len(non_gray)/tot*100:4.1f}%)")


# 4. TASK 4: CLASS IMBALANCE & WEIGHT CALCULATIONS
print("\n--- TASK 4: CLASS IMBALANCE & WEIGHT CANDIDATES ---")

exclude_files = {"malignant (145).png", "benign (433).png"}
train_counts = {"benign": 304, "malignant": 146, "normal": 93} # Train set counts
total_train = sum(train_counts.values())

n_classes = 3
balanced_weights = {
    c: total_train / (n_classes * count) for c, count in train_counts.items()
}

sqrt_weights = {
    c: np.sqrt(total_train / (n_classes * count)) for c, count in train_counts.items()
}

mild_weights = {
    "benign": 1.0,
    "malignant": 1.25,
    "normal": 1.1
}

print(f"Train Class Distribution (Total {total_train} scans):")
for c, count in train_counts.items():
    print(f"  - {c:<9}: {count:3d} scans ({count/total_train*100:4.1f}%)")

print("\nWeighting Strategies:")
print(f"  - Unweighted (Standard)           : {{'benign': 1.000, 'malignant': 1.000, 'normal': 1.000}}")
print(f"  - Mild Malignant Weighting        : {{'benign': {mild_weights['benign']:.3f}, 'malignant': {mild_weights['malignant']:.3f}, 'normal': {mild_weights['normal']:.3f}}}")
print(f"  - Square-Root Inverse Freq        : {{'benign': {sqrt_weights['benign']:.3f}, 'malignant': {sqrt_weights['malignant']:.3f}, 'normal': {sqrt_weights['normal']:.3f}}}")
print(f"  - Inverse Class Freq (Balanced)   : {{'benign': {balanced_weights['benign']:.3f}, 'malignant': {balanced_weights['malignant']:.3f}, 'normal': {balanced_weights['normal']:.3f}}}")

print("\nInvestigation complete. Writing findings to busi_methodology_findings.json...")
findings = {
    "total_scans": len(all_scans),
    "clusters_by_class": {c: len(clusters_by_class[c]) for c in classes},
    "ar_distribution": {
        "tall": len(ar_extreme_low),
        "square": len(ar_square),
        "mod_wide": len(ar_wide),
        "ext_wide": len(ar_extreme_high)
    },
    "avg_stretching_distortion_pct": float(np.mean(distortions)),
    "train_counts": train_counts,
    "balanced_weights": balanced_weights,
    "sqrt_weights": sqrt_weights,
    "mild_weights": mild_weights
}

with open(os.path.join(backend_dir, "busi_methodology_findings.json"), "w") as f:
    json.dump(findings, f, indent=4)

print("Findings saved successfully.")
