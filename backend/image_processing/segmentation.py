"""
Nuclei Segmentation Module
===========================
Segments individual cell nuclei from a preprocessed grayscale microscopy image.

Approach:
1. Otsu thresholding — automatic foreground/background separation
2. Morphological opening — remove small noise blobs
3. Morphological closing — fill small holes inside nuclei
4. Sure background via dilation
5. Distance transform — find nucleus centers
6. Watershed segmentation — separate touching nuclei
7. Connected component labeling — identify individual regions
8. Filter by area — remove artifacts too small or too large to be nuclei

Diagnostic outputs:
- Binary mask after thresholding
- Cleaned mask after morphology
- Labeled regions overlay
- Detected nuclei contours on original image
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Any


def segment_nuclei(
    gray_original: np.ndarray, 
    preprocessed: np.ndarray,
    min_nucleus_area: int = 50,
    max_nucleus_area_ratio: float = 0.25,
) -> Dict[str, Any]:
    """
    Segment individual nuclei from a preprocessed grayscale image.
    
    Args:
        gray_original: Original grayscale image for overlay generation.
        preprocessed: Preprocessed image (CLAHE enhanced).
        min_nucleus_area: Minimum pixel area for a valid nucleus.
        max_nucleus_area_ratio: Maximum ratio of image area for a single nucleus.
        
    Returns:
        Dictionary containing:
            - 'contours': List of numpy contours for each detected nucleus
            - 'labeled_mask': Integer-labeled mask (each nucleus gets a unique ID)
            - 'binary_mask': Binary segmentation mask
            - 'overlay': Color image with nuclei contours drawn
            - 'num_nuclei': Number of detected nuclei
            - 'diagnostics': Dict of intermediate images for visualization
    """
    h, w = preprocessed.shape[:2]
    total_area = h * w
    max_nucleus_area = int(total_area * max_nucleus_area_ratio)
    
    # Step 1: Otsu thresholding
    # Invert if nuclei are darker than background (typical in H&E stained images)
    _, binary_otsu = cv2.threshold(preprocessed, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Check if we got more foreground than background — if so, invert
    fg_ratio = np.count_nonzero(binary_otsu) / total_area
    if fg_ratio > 0.6:
        binary_otsu = cv2.bitwise_not(binary_otsu)
    
    # Step 2: Morphological operations
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    # Opening removes small noise
    cleaned = cv2.morphologyEx(binary_otsu, cv2.MORPH_OPEN, kernel_open, iterations=2)
    # Closing fills small holes
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    
    # Step 3: Sure background (dilated region)
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    sure_bg = cv2.dilate(cleaned, kernel_dilate, iterations=3)
    
    # Step 4: Distance transform and local maxima detection for seeds
    dist_transform = cv2.distanceTransform(cleaned, cv2.DIST_L2, 5)
    max_dist = dist_transform.max()
    
    if max_dist > 0:
        # Dilate distance transform to find local peaks
        dilated = cv2.dilate(dist_transform, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        # A pixel is a peak if it matches the dilated value and is above 15% of the max distance
        peaks = (dist_transform == dilated) & (dist_transform > 0.15 * max_dist) & (cleaned > 0)
        sure_fg = np.uint8(peaks) * 255
    else:
        sure_fg = np.zeros_like(cleaned)
    
    # Step 5: Unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Step 6: Connected components for markers
    num_labels, markers = cv2.connectedComponents(sure_fg)
    
    # Increment all labels so that background is not 0 but 1
    markers = markers + 1
    # Mark unknown region as 0
    markers[unknown == 255] = 0
    
    # Step 7: Watershed
    color_for_watershed = cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(color_for_watershed, markers)
    
    # Step 8: Extract contours for each labeled region
    contours_list: List[np.ndarray] = []
    labeled_mask = np.zeros((h, w), dtype=np.int32)
    
    # Generate a mask of the watershed boundaries (markers == -1) to visualize separation
    watershed_boundaries = np.uint8(markers == -1) * 255
    # The separated mask is the cleaned mask with watershed lines subtracted
    separated_mask = cv2.subtract(cleaned, watershed_boundaries)
    
    nucleus_id = 1
    for label_id in range(2, num_labels + 1):  # Skip background (1) and boundary (-1)
        region_mask = np.uint8(markers == label_id) * 255
        region_contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in region_contours:
            area = cv2.contourArea(cnt)
            if min_nucleus_area <= area <= max_nucleus_area:
                contours_list.append(cnt)
                cv2.drawContours(labeled_mask, [cnt], -1, nucleus_id, -1)
                nucleus_id += 1
    
    # Generate overlay image
    overlay = cv2.cvtColor(gray_original, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(overlay, contours_list, -1, (0, 255, 0), 1)
    
    # Number each nucleus
    for i, cnt in enumerate(contours_list):
        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(overlay, str(i + 1), (cx - 5, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
    
    # Diagnostic images
    diagnostics = {
        "binary_mask": binary_otsu,
        "cleaned_mask": cleaned,
        "distance_transform": cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U),
        "sure_fg": sure_fg,
        "watershed_separated": separated_mask
    }
    
    return {
        "contours": contours_list,
        "labeled_mask": labeled_mask,
        "binary_mask": cleaned,
        "overlay": overlay,
        "num_nuclei": len(contours_list),
        "diagnostics": diagnostics,
    }

