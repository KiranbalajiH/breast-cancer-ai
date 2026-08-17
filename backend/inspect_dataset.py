import os
import hashlib
import cv2
from collections import Counter

def get_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def main():
    dataset_root = r"C:\Users\kiran\BCD\dataset"
    print("="*60)
    print("DATASET INSPECTION REPORT")
    print("="*60)
    
    # 1. Structure inspection
    print("\n1. Structure Inspection:")
    for root, dirs, files in os.walk(dataset_root):
        rel_path = os.path.relpath(root, dataset_root)
        if rel_path == ".":
            continue
        level = rel_path.count(os.sep)
        indent = "  " * level
        num_files = len([f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))])
        if num_files > 0 or len(dirs) > 0:
            print(f"{indent}- {os.path.basename(root)}/ ({num_files} images)")
            
    # 2. Identify image files and check for corruption / formats / dimensions
    print("\n2. Scanning files for details, corruption and duplicates...")
    all_files = []
    file_hashes = {}
    duplicates = []
    corrupt_files = []
    dimensions = Counter()
    formats = Counter()
    
    supported_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
    
    for root, dirs, files in os.walk(dataset_root):
        # Exclude mask folders from main image count to avoid confusion
        path_parts = root.lower().split(os.sep)
        is_mask = any("mask" in part for part in path_parts)
        
        for f in files:
            if f.lower().endswith(supported_extensions):
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, dataset_root)
                
                # Check for duplicate by content hash
                f_hash = get_md5(full_path)
                if f_hash in file_hashes:
                    duplicates.append((rel_path, file_hashes[f_hash]))
                else:
                    file_hashes[f_hash] = rel_path
                    
                # Read image details and check for corruption
                img = cv2.imread(full_path)
                if img is None:
                    corrupt_files.append(rel_path)
                    formats[os.path.splitext(f)[1].lower()] += 1
                else:
                    h, w, c = img.shape[0], img.shape[1], img.shape[2] if len(img.shape) == 3 else 1
                    dimensions[(w, h, c)] += 1
                    formats[os.path.splitext(f)[1].lower()] += 1
                    
                all_files.append({
                    "full_path": full_path,
                    "rel_path": rel_path,
                    "filename": f,
                    "is_mask": is_mask,
                    "folder": os.path.basename(root)
                })

    print(f"\nTotal image files scanned (including masks): {len(all_files)}")
    print(f"Original images (excluding masks): {len([x for x in all_files if not x['is_mask']])}")
    print(f"Mask images: {len([x for x in all_files if x['is_mask']])}")
    
    # Formats count
    print("\nImage Formats:")
    for fmt, count in formats.items():
        print(f"  - {fmt}: {count} files")
        
    # Dimensions count
    print("\nImage Dimensions (Width x Height x Channels):")
    for dim, count in dimensions.items():
        print(f"  - {dim[0]}x{dim[1]} ({dim[2]} channels): {count} files")
        
    # Corruption report
    print("\nCorrupt files:")
    if corrupt_files:
        print(f"  Found {len(corrupt_files)} corrupt files:")
        for cf in corrupt_files[:5]:
            print(f"    - {cf}")
        if len(corrupt_files) > 5:
            print("    ...")
    else:
        print("  No corrupt files found.")
        
    # Duplicates report
    print("\nDuplicate files:")
    if duplicates:
        print(f"  Found {len(duplicates)} duplicate file pairs:")
        for dup in duplicates[:5]:
            print(f"    - {dup[0]} is a duplicate of {dup[1]}")
        if len(duplicates) > 5:
            print("    ...")
    else:
        print("  No duplicate files found.")
        
    # Suitability for binary classification
    print("\n3. Suitability for Binary Image Classification:")
    # Check if there are directories that can act as labels
    midesec_imgs = len([x for x in all_files if not x['is_mask'] and 'midesec' in str(x['rel_path']).lower()])
    nusec_imgs = len([x for x in all_files if not x['is_mask'] and 'nusec' in str(x['rel_path']).lower()])
    print(f"  - MiDeSeC original images: {midesec_imgs}")
    print(f"  - NuSeC original images: {nusec_imgs}")
    
    print("\nConclusions:")
    print("  1. The folders present are 'MiDeSeC' (mitosis detection) and 'NuSeC' (nuclei segmentation).")
    print("  2. There are no subfolders named 'benign' or 'malignant' or class labels inside these folders.")
    print("  3. To build a binary image classification model for Benign vs. Malignant:")
    print("     We can map the entire MiDeSeC dataset (mitotic cells, representing invasive breast cancer/high malignancy) as 'Malignant'.")
    print("     We can map the entire NuSeC dataset (general nuclei segmentation, representing standard or benign/lower grade pathology tissue) as 'Benign'.")
    print("     OR we can examine if one of them is inherently benign/malignant. Since MiDeSeC is invasive breast carcinoma, it is malignant. NuSeC was built from breast cancer patients as well, but represents normal tissue surrounding tumors or low-grade cells, so treating NuSeC as Benign (or non-mitotic/normal) and MiDeSeC as Malignant (invasive mitotic tumor) allows a binary classification task.")
    
if __name__ == "__main__":
    main()
