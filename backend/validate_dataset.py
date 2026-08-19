import os
import cv2
import numpy as np
import json
import hashlib
from collections import Counter

def get_md5(file_path):
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        return f"Error: {e}"

def scan_dataset(dataset_path, dataset_name):
    supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')
    
    total_files = 0
    trainable_images = []
    masks = []
    corrupted_images = []
    unsupported_files = []
    unexpected_classes = []
    
    dimensions = Counter()
    formats = Counter()
    
    # Identify classes in the dataset folder
    if not os.path.exists(dataset_path):
        return None
        
    subdirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    
    valid_classes = {'benign', 'malignant', 'normal'}
    
    for subdir in subdirs:
        class_path = os.path.join(dataset_path, subdir)
        is_valid_class = subdir.lower() in valid_classes
        if not is_valid_class:
            unexpected_classes.append(subdir)
            
        class_label = subdir.lower() if is_valid_class else "unknown"
        
        for root, _, files in os.walk(class_path):
            for f in files:
                total_files += 1
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, dataset_path)
                
                # Check for support
                ext = os.path.splitext(f)[1].lower()
                if ext not in supported_extensions:
                    unsupported_files.append(rel_path)
                    continue
                    
                # Classify mask vs trainable original
                is_mask = any(x in f.lower() for x in ('mask', 'gt', 'ground_truth', 'groundtruth'))
                
                # Load image with cv2 to check corruption & dimensions
                img = cv2.imread(full_path)
                if img is None:
                    corrupted_images.append(rel_path)
                    continue
                    
                h, w, c = img.shape[0], img.shape[1], img.shape[2] if len(img.shape) == 3 else 1
                dim = (w, h, c)
                
                file_info = {
                    "filename": f,
                    "rel_path": rel_path,
                    "full_path": full_path,
                    "dataset": dataset_name,
                    "class_label": class_label,
                    "dimensions": dim,
                    "format": ext,
                    "size_bytes": os.path.getsize(full_path),
                    "md5": get_md5(full_path)
                }
                
                if is_mask:
                    masks.append(file_info)
                else:
                    trainable_images.append(file_info)
                    dimensions[dim] += 1
                    formats[ext] += 1
                    
    return {
        "dataset_name": dataset_name,
        "total_files": total_files,
        "trainable_images": trainable_images,
        "masks": masks,
        "corrupted_images": corrupted_images,
        "unsupported_files": unsupported_files,
        "unexpected_classes": unexpected_classes,
        "dimensions": dict(dimensions),
        "formats": dict(formats)
    }

def run_duplicate_checks(all_datasets):
    # Collect all file metadata
    all_trainables = []
    all_masks = []
    
    for d in all_datasets.values():
        all_trainables.extend(d["trainable_images"])
        all_masks.extend(d["masks"])
        
    md5_to_trainable = {}
    md5_to_mask = {}
    
    exact_duplicates = []
    leakage_masks_as_trainables = []
    duplicate_filenames_diff_content = []
    
    # Exact duplicate trainable images
    for img in all_trainables:
        h = img["md5"]
        if h in md5_to_trainable:
            exact_duplicates.append((img, md5_to_trainable[h]))
        else:
            md5_to_trainable[h] = img
            
    # Exact duplicate masks
    for mask in all_masks:
        h = mask["md5"]
        if h in md5_to_mask:
            # Mask duplicate, not critical but good to know
            pass
        else:
            md5_to_mask[h] = mask
            
    # Check if a mask is identical to a trainable image (leakage)
    for mask in all_masks:
        h = mask["md5"]
        if h in md5_to_trainable:
            leakage_masks_as_trainables.append((mask, md5_to_trainable[h]))
            
    # Check duplicate filenames with different content
    filename_to_img = {}
    for img in all_trainables:
        name = img["filename"]
        if name in filename_to_img:
            # If MD5 is different
            if img["md5"] != filename_to_img[name]["md5"]:
                duplicate_filenames_diff_content.append((img, filename_to_img[name]))
        else:
            filename_to_img[name] = img
            
    return {
        "exact_duplicates": exact_duplicates,
        "leakage_masks_as_trainables": leakage_masks_as_trainables,
        "duplicate_filenames_diff_content": duplicate_filenames_diff_content
    }

def main():
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(os.path.dirname(backend_dir), "dataset")
    
    print("="*60)
    print("PHASE B - BREAST ULTRASOUND DATASET INSPECTION & VALIDATION")
    print("="*60)
    
    # Find all dataset folders
    dataset_folders = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d)) and not d.startswith('.')]
    print(f"Datasets detected in directory: {dataset_folders}")
    
    all_datasets = {}
    for d_folder in dataset_folders:
        d_path = os.path.join(dataset_dir, d_folder)
        print(f"\nScanning dataset: {d_folder}...")
        res = scan_dataset(d_path, d_folder)
        if res:
            all_datasets[d_folder] = res
            print(f"  - Total trainable images: {len(res['trainable_images'])}")
            print(f"  - Total masks excluded: {len(res['masks'])}")
            print(f"  - Corrupt files found: {len(res['corrupted_images'])}")
            print(f"  - Unsupported files: {len(res['unsupported_files'])}")
            print(f"  - Unexpected class folders: {res['unexpected_classes']}")
            
    # Run duplicate and leakage checks
    print("\n" + "="*50)
    print("DUPLICATE AND LEAKAGE CHECKS")
    print("="*50)
    dup_res = run_duplicate_checks(all_datasets)
    
    print(f"1. Exact Duplicate Trainable Images: {len(dup_res['exact_duplicates'])}")
    for item1, item2 in dup_res['exact_duplicates']:
        print(f"   - File: '{item1['rel_path']}' in {item1['dataset']} is identical to '{item2['rel_path']}' in {item2['dataset']}")
        
    print(f"\n2. Leakage (Masks matching Original images): {len(dup_res['leakage_masks_as_trainables'])}")
    for mask, img in dup_res['leakage_masks_as_trainables']:
        print(f"   - Mask: '{mask['rel_path']}' in {mask['dataset']} matches original image '{img['rel_path']}' in {img['dataset']}!")
        
    print(f"\n3. Duplicate Filenames with Different Content: {len(dup_res['duplicate_filenames_diff_content'])}")
    for img1, img2 in dup_res['duplicate_filenames_diff_content']:
        print(f"   - Filename '{img1['filename']}' appears in different folders: '{img1['rel_path']}' ({img1['dataset']}) vs '{img2['rel_path']}' ({img2['dataset']}) but has different content.")
        
    # Class Distribution Analysis
    print("\n" + "="*50)
    print("CLASS DISTRIBUTION ANALYSIS")
    print("="*50)
    
    total_trainable = 0
    class_counts = Counter()
    
    for d_name, d_res in all_datasets.items():
        print(f"\nDataset: {d_name}")
        d_class_counts = Counter([img["class_label"] for img in d_res["trainable_images"]])
        d_total = len(d_res["trainable_images"])
        for cls, count in d_class_counts.items():
            pct = (count / d_total * 100) if d_total > 0 else 0
            print(f"  - {cls:<10}: {count:>4d} images ({pct:.2f}%)")
        total_trainable += d_total
        class_counts.update([img["class_label"] for img in d_res["trainable_images"]])
        
    print(f"\nCombined Dataset Distribution (All Trainable Images):")
    print(f"Total Trainable Images: {total_trainable}")
    for cls in ['benign', 'malignant', 'normal']:
        count = class_counts[cls]
        pct = (count / total_trainable * 100) if total_trainable > 0 else 0
        print(f"  - {cls:<10}: {count:>4d} images ({pct:.2f}%)")
        
    # Check for imbalance
    print("\nImbalance Assessment:")
    for cls in ['benign', 'malignant', 'normal']:
        pct = (class_counts[cls] / total_trainable * 100) if total_trainable > 0 else 0
        if pct < 20.0:
            print(f"  - WARNING: Class '{cls}' is under-represented ({pct:.2f}% of total).")
        elif pct > 60.0:
            print(f"  - WARNING: Class '{cls}' is over-represented ({pct:.2f}% of total).")
        else:
            print(f"  - Class '{cls}' is within normal range ({pct:.2f}% of total).")
            
    # Dataset Manifest Generation (for future training separation)
    manifest_list = []
    for d_res in all_datasets.values():
        for img in d_res["trainable_images"]:
            # rel_path relative to dataset/ directory
            rel_path_to_dataset = os.path.join(d_res["dataset_name"], img["rel_path"])
            manifest_list.append({
                "image_path": "dataset/" + rel_path_to_dataset.replace(os.sep, '/'),
                "source_dataset": d_res["dataset_name"],
                "class_label": img["class_label"]
            })
            
    data_dir = os.path.join(backend_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    manifest_path = os.path.join(data_dir, "dataset_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest_list, f, indent=4)
    print(f"\nGenerated source-aware dataset manifest at: {manifest_path} ({len(manifest_list)} records)")

if __name__ == "__main__":
    main()
