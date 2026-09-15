import os
import sys
import glob
import json
import hashlib
import cv2
import numpy as np
from collections import Counter, defaultdict

script_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(script_dir)
bcd_root = os.path.dirname(backend_dir)
dataset_root = os.path.join(bcd_root, "dataset", "BUSI")

print("=======================================================================")
print("BUSI DATASET FORENSIC AUDIT SCRIPT")
print("=======================================================================")
print(f"Dataset Path: {dataset_root}\n")

# TASK 1: Dataset Structure & File Inventory
classes = ['benign', 'malignant', 'normal']
inventory = {}
all_images = [] # trainable images
all_masks = []  # mask images
corrupt_files = []
zero_byte_files = []
suspicious_files = []

for c in classes:
    cdir = os.path.join(dataset_root, c)
    if not os.path.exists(cdir):
        print(f"Directory missing: {cdir}")
        continue
    
    files = sorted(os.listdir(cdir))
    c_images = []
    c_masks = []
    
    for f in files:
        fpath = os.path.join(cdir, f)
        if not os.path.isfile(fpath):
            continue
            
        size = os.path.getsize(fpath)
        if size == 0:
            zero_byte_files.append((c, f))
            continue
            
        ext = os.path.splitext(f)[1].lower()
        
        # Distinguish mask vs original
        is_mask = ('_mask' in f.lower()) or ('mask' in f.lower())
        
        # Try loading image
        img = cv2.imread(fpath)
        if img is None:
            corrupt_files.append((c, f))
            continue
            
        h, w, ch = img.shape
        # Check channels equality (grayscale saved as RGB)
        is_gray_rgb = False
        if ch == 3:
            if np.array_equal(img[:,:,0], img[:,:,1]) and np.array_equal(img[:,:,1], img[:,:,2]):
                is_gray_rgb = True
                
        info = {
            "class": c,
            "filename": f,
            "filepath": fpath,
            "size": size,
            "ext": ext,
            "height": h,
            "width": w,
            "channels": ch,
            "is_gray_rgb": is_gray_rgb,
            "is_mask": is_mask,
            "aspect_ratio": round(w / h, 3),
            "mean_intensity": float(np.mean(img)),
            "std_intensity": float(np.std(img)),
            "laplacian_var": float(cv2.Laplacian(img, cv2.CV_64F).var())
        }
        
        if is_mask:
            c_masks.append(info)
            all_masks.append(info)
        else:
            c_images.append(info)
            all_images.append(info)
            
    inventory[c] = {
        "images": c_images,
        "masks": c_masks,
        "total_files": len(files)
    }

print("--- TASK 1: DATASET STRUCTURE & INVENTORY ---")
total_trainable = len(all_images)
total_mask_count = len(all_masks)
print(f"Total Trainable Ultrasound Scans: {total_trainable}")
print(f"Total Mask Annotation Files    : {total_mask_count}")
print(f"Corrupt Files                  : {len(corrupt_files)}")
print(f"Zero-Byte Files                : {len(zero_byte_files)}")

for c in classes:
    n_img = len(inventory[c]["images"])
    n_mask = len(inventory[c]["masks"])
    print(f"  Class '{c:<9}': {n_img:3d} scans | {n_mask:3d} masks | Total files: {inventory[c]['total_files']}")

# Channel / Color Distribution
ch_dist = Counter([x['channels'] for x in all_images])
gray_rgb_count = sum(1 for x in all_images if x['is_gray_rgb'])
ext_dist = Counter([x['ext'] for x in all_images])

print(f"\nExtensions  : {dict(ext_dist)}")
print(f"Channels    : {dict(ch_dist)}")
print(f"Grayscale disguised as 3-channel RGB: {gray_rgb_count} / {total_trainable} ({gray_rgb_count/total_trainable*100:.1f}%)")


# TASK 2: DUPLICATE ANALYSIS
md5_raw = defaultdict(list)
md5_pixel = defaultdict(list)

for item in all_images:
    # Raw file hash
    with open(item['filepath'], 'rb') as f:
        h_raw = hashlib.md5(f.read()).hexdigest()
    md5_raw[h_raw].append(item)
    
    # Pixel content hash (read raw pixels)
    img = cv2.imread(item['filepath'], cv2.IMREAD_GRAYSCALE)
    h_pix = hashlib.md5(img.tobytes()).hexdigest()
    md5_pixel[h_pix].append(item)

exact_raw_dups = {k: v for k, v in md5_raw.items() if len(v) > 1}
exact_pixel_dups = {k: v for k, v in md5_pixel.items() if len(v) > 1}

print("\n--- TASK 2: DUPLICATE & LABEL CONFLICT ANALYSIS ---")
print(f"Exact File MD5 Duplicates (Groups): {len(exact_raw_dups)}")
for k, v in exact_raw_dups.items():
    fns = [f"{x['class']}/{x['filename']}" for x in v]
    print(f"  Raw Hash {k[:8]}: {fns}")

print(f"Exact Pixel MD5 Duplicates (Groups): {len(exact_pixel_dups)}")
label_conflicts = []
for k, v in exact_pixel_dups.items():
    classes_in_group = set(x['class'] for x in v)
    fns = [f"{x['class']}/{x['filename']}" for x in v]
    print(f"  Pixel Hash {k[:8]}: {fns}")
    if len(classes_in_group) > 1:
        label_conflicts.append((k, v))

print(f"Label Conflicts (Identical image in multiple classes): {len(label_conflicts)}")


# TASK 3: PATIENT / STUDY LEAKAGE
# In BUSI dataset, filenames follow 'benign (X).png'.
# Let's inspect if numbers correspond to patients, and whether images with multiple masks or related scans share patient IDs.
# Also inspect if benign (X) and malignant (Y) or sequential numbers come from same acquisition sessions.
print("\n--- TASK 3: PATIENT / STUDY LEAKAGE ANALYSIS ---")
# Check how masks map to images
# E.g. benign (1).png -> benign (1)_mask.png, benign (1)_mask_1.png
image_prefixes = defaultdict(list)
for item in all_images:
    # Extract base number: e.g. benign (123)
    base = item['filename'].split('.')[0]
    image_prefixes[item['class']].append(base)

print("Filename pattern analysis:")
for c in classes:
    print(f"  {c}: Image index range 1 to {len(inventory[c]['images'])}")

# Check if train/val/test splits contain potential patient clusters
# Let's check how train_test_split divides these images.


# TASK 4: CURRENT SPLIT AUDIT
from sklearn.model_selection import train_test_split

exclude_files = {"malignant (145).png", "benign (433).png"}
clean_image_paths = []
clean_labels = []

for class_idx, class_name in enumerate(classes):
    class_dir = os.path.join(dataset_root, class_name)
    for f in sorted(os.listdir(class_dir)):
        if not f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
            continue
        if '_mask' in f.lower() or 'mask' in f.lower():
            continue
        if f in exclude_files:
            continue
        clean_image_paths.append(os.path.join(class_dir, f))
        clean_labels.append(class_idx)

clean_labels = np.array(clean_labels)
X_train, X_val_test, y_train, y_val_test = train_test_split(
    clean_image_paths, clean_labels, test_size=0.30, stratify=clean_labels, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_val_test, y_val_test, test_size=0.50, stratify=y_val_test, random_state=42
)

print("\n--- TASK 4: CURRENT SPLIT AUDIT ---")
print(f"Total Clean Scans Analyzed: {len(clean_image_paths)}")
print(f"Train Count      : {len(X_train)} (Benign: {np.sum(y_train==0)}, Malignant: {np.sum(y_train==1)}, Normal: {np.sum(y_train==2)})")
print(f"Val Count        : {len(X_val)} (Benign: {np.sum(y_val==0)}, Malignant: {np.sum(y_val==1)}, Normal: {np.sum(y_val==2)})")
print(f"Test Count       : {len(X_test)} (Benign: {np.sum(y_test==0)}, Malignant: {np.sum(y_test==1)}, Normal: {np.sum(y_test==2)})")


# TASK 5 & 6: IMAGE QUALITY & CLASS VISUAL DISTRIBUTION
print("\n--- TASK 5 & 6: IMAGE QUALITY & CLASS VISUAL DISTRIBUTION ---")
# Statistics per class
for c in classes:
    c_imgs = [x for x in all_images if x['class'] == c]
    heights = [x['height'] for x in c_imgs]
    widths = [x['width'] for x in c_imgs]
    means = [x['mean_intensity'] for x in c_imgs]
    stds = [x['std_intensity'] for x in c_imgs]
    laps = [x['laplacian_var'] for x in c_imgs]
    ars = [x['aspect_ratio'] for x in c_imgs]
    
    print(f"Class '{c:<9}':")
    print(f"  - Height Range   : {min(heights)}px to {max(heights)}px (Mean: {np.mean(heights):.1f}px)")
    print(f"  - Width Range    : {min(widths)}px to {max(widths)}px (Mean: {np.mean(widths):.1f}px)")
    print(f"  - Aspect Ratio   : {min(ars):.2f} to {max(ars):.2f} (Mean: {np.mean(ars):.2f})")
    print(f"  - Mean Intensity : {min(means):.1f} to {max(means):.1f} (Average: {np.mean(means):.1f})")
    print(f"  - Intensity Std  : {min(stds):.1f} to {max(stds):.1f} (Average: {np.mean(stds):.1f})")
    print(f"  - Blur Variance  : {min(laps):.1f} to {max(laps):.1f} (Average: {np.mean(laps):.1f})")

# Check dark / bright / low contrast / blur thresholds across total dataset
dark_imgs = [x for x in all_images if x['mean_intensity'] < 35]
bright_imgs = [x for x in all_images if x['mean_intensity'] > 140]
low_contrast = [x for x in all_images if x['std_intensity'] < 30]
blurry_imgs = [x for x in all_images if x['laplacian_var'] < 100]

print(f"\nOverall Quality Metrics:")
print(f"  - Dark Images (mean < 35)       : {len(dark_imgs)} / {total_trainable} ({len(dark_imgs)/total_trainable*100:.1f}%)")
print(f"  - Bright Images (mean > 140)    : {len(bright_imgs)} / {total_trainable} ({len(bright_imgs)/total_trainable*100:.1f}%)")
print(f"  - Low Contrast (std < 30)       : {len(low_contrast)} / {total_trainable} ({len(low_contrast)/total_trainable*100:.1f}%)")
print(f"  - Blurry Images (Laplacian<100) : {len(blurry_imgs)} / {total_trainable} ({len(blurry_imgs)/total_trainable*100:.1f}%)")

# Save detailed json report of audit
audit_data = {
    "total_trainable": total_trainable,
    "total_masks": total_mask_count,
    "class_counts": {c: len(inventory[c]["images"]) for c in classes},
    "mask_counts": {c: len(inventory[c]["masks"]) for c in classes},
    "exact_raw_duplicates": len(exact_raw_dups),
    "exact_pixel_duplicates": len(exact_pixel_dups),
    "label_conflicts": len(label_conflicts),
    "quality_summary": {
        "dark": len(dark_imgs),
        "bright": len(bright_imgs),
        "low_contrast": len(low_contrast),
        "blurry": len(blurry_imgs)
    }
}

with open(os.path.join(backend_dir, "busi_audit_data.json"), "w") as f:
    json.dump(audit_data, f, indent=4)

print("\nAudit data written to busi_audit_data.json.")
