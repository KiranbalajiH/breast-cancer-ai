import os
import sys
import cv2
import numpy as np
import json

script_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(script_dir)
bcd_root = os.path.dirname(backend_dir)
dataset_root = os.path.join(bcd_root, "dataset", "BUSI")

classes = ['benign', 'malignant', 'normal']

color_annotations_count = 0
border_padded_count = 0
caliper_marker_count = 0

class_stats = {c: {"color_ann": 0, "border_pad": 0, "total": 0} for c in classes}

for c in classes:
    cdir = os.path.join(dataset_root, c)
    files = sorted([f for f in os.listdir(cdir) if f.endswith('.png') and 'mask' not in f.lower()])
    class_stats[c]["total"] = len(files)
    
    for f in files:
        fpath = os.path.join(cdir, f)
        img = cv2.imread(fpath)
        if img is None:
            continue
            
        # Check for color annotations (non-grayscale pixels)
        b, g, r = img[:,:,0], img[:,:,1], img[:,:,2]
        diff1 = np.abs(b.astype(int) - g.astype(int))
        diff2 = np.abs(g.astype(int) - r.astype(int))
        has_color = np.sum((diff1 > 15) | (diff2 > 15)) > 50
        
        if has_color:
            color_annotations_count += 1
            class_stats[c]["color_ann"] += 1
            
        # Check for heavy black border padding (outer 5% pixels being pure black)
        h, w, _ = img.shape
        top_strip = img[0:int(h*0.05), :]
        bottom_strip = img[int(h*0.95):, :]
        is_padded = (np.mean(top_strip) < 5) and (np.mean(bottom_strip) < 5)
        if is_padded:
            border_padded_count += 1
            class_stats[c]["border_pad"] += 1

print("--- TASK 5 & 6 DETAILED ARTIFACT AUDIT ---")
print(f"Total Scans Analyzed: {sum(class_stats[c]['total'] for c in classes)}")
print(f"Images with Color Calipers / Markings / Annotations: {color_annotations_count}")
print(f"Images with Heavy Top/Bottom Black Border Padding  : {border_padded_count}")
for c in classes:
    tot = class_stats[c]["total"]
    ca = class_stats[c]["color_ann"]
    bp = class_stats[c]["border_pad"]
    print(f"  Class '{c:<9}': Color Markings = {ca:2d}/{tot:3d} ({ca/tot*100:4.1f}%) | Black Padding = {bp:2d}/{tot:3d} ({bp/tot*100:4.1f}%)")

# TASK 8: PREPROCESSING AUDIT
print("\n--- TASK 8: PREPROCESSING AUDIT ---")

# Let's inspect backend inference code in app/ or image_processing/
inference_files = []
for root, _, files in os.walk(os.path.join(backend_dir, "app")):
    for f in files:
        if f.endswith('.py'):
            inference_files.append(os.path.join(root, f))
for root, _, files in os.walk(os.path.join(backend_dir, "image_processing")):
    for f in files:
        if f.endswith('.py'):
            inference_files.append(os.path.join(root, f))

print(f"Found {len(inference_files)} inference / backend python files:")
for f in inference_files:
    print(f"  - {os.path.relpath(f, backend_dir)}")

