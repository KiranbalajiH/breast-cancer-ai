"""
Per-Nucleus Feature Extraction Module
======================================
Extracts 10 morphological and texture measurements from each individual
detected nucleus, using documented and reproducible formulas.

Feature Definitions (per nucleus):
-----------------------------------
1. Radius:     sqrt(area / π) — equivalent circular radius
2. Texture:    Standard deviation of grayscale intensity values within the nucleus
3. Perimeter:  cv2.arcLength of the contour (closed=True)
4. Area:       cv2.contourArea of the contour
5. Smoothness: 1 - 1/(1 + var(local_radii)) where local_radii are distances
               from centroid to each contour point, normalized by mean radius.
               Approximates local radius variation.
6. Compactness: (perimeter^2 / area) - 1.0
7. Concavity:  (convex_hull_area - contour_area) / convex_hull_area
               Fraction of the convex hull not covered by the contour.
8. Concave Points: Number of significant convexity defects (depth > threshold).
               Normalized by dividing by total contour points for scale invariance.
9. Symmetry:   minor_axis / major_axis from cv2.fitEllipse.
               1.0 = perfect circle, lower = more asymmetric.
10. Fractal Dimension: Box-counting method on the binary mask of the nucleus.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Any


def _box_counting_fractal_dimension(binary_mask: np.ndarray) -> float:
    """
    Compute fractal dimension of a binary shape using box-counting method.
    
    The box-counting dimension D is estimated from the slope of
    log(N(s)) vs log(1/s) where N(s) is the number of boxes of size s
    needed to cover the shape boundary.
    """
    # Extract edge pixels
    edges = cv2.Canny(binary_mask, 100, 200)
    points = np.argwhere(edges > 0)
    
    if len(points) < 10:
        return 1.0  # Degenerate case — treat as a line
    
    # Determine the range of scales
    min_dim = min(binary_mask.shape)
    if min_dim < 4:
        return 1.0
    
    # Box sizes: powers of 2 from 2 up to half the image size
    box_sizes = []
    s = 2
    while s <= min_dim // 2:
        box_sizes.append(s)
        s *= 2
    
    if len(box_sizes) < 2:
        return 1.0
    
    counts = []
    for s in box_sizes:
        # Count non-empty boxes
        # Divide the image into grid of size s x s
        n_boxes_y = (binary_mask.shape[0] + s - 1) // s
        n_boxes_x = (binary_mask.shape[1] + s - 1) // s
        count = 0
        for i in range(n_boxes_y):
            for j in range(n_boxes_x):
                box = edges[i*s:(i+1)*s, j*s:(j+1)*s]
                if np.any(box > 0):
                    count += 1
        counts.append(count)
    
    # Linear regression of log(count) vs log(1/size)
    log_sizes = np.log(1.0 / np.array(box_sizes, dtype=float))
    log_counts = np.log(np.array(counts, dtype=float))
    
    # Least squares fit
    if len(log_sizes) >= 2:
        coeffs = np.polyfit(log_sizes, log_counts, 1)
        return float(coeffs[0])  # Slope = fractal dimension
    
    return 1.0


def extract_single_nucleus_features(
    contour: np.ndarray, 
    gray_image: np.ndarray
) -> Optional[Dict[str, float]]:
    """
    Extract 10 morphological and texture features from a single nucleus.
    
    Args:
        contour: Contour of the nucleus (from cv2.findContours).
        gray_image: Grayscale image for texture measurement.
        
    Returns:
        Dictionary with 10 feature values, or None if extraction fails.
    """
    try:
        # Basic geometric measurements
        area = cv2.contourArea(contour)
        if area < 10:
            return None
        
        perimeter = cv2.arcLength(contour, True)
        if perimeter < 1:
            return None
        
        # 1. Radius: equivalent circular radius
        radius = np.sqrt(area / np.pi)
        
        # 2. Texture: std dev of grayscale pixel values within the nucleus
        mask = np.zeros(gray_image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        pixels = gray_image[mask == 255]
        texture = float(np.std(pixels)) if len(pixels) > 0 else 0.0
        
        # 3. Perimeter (already computed)
        
        # 4. Area (already computed)
        
        # 5. Smoothness: local radius variation
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return None
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        
        # Compute distances from centroid to each contour point
        contour_points = contour.reshape(-1, 2).astype(float)
        distances = np.sqrt((contour_points[:, 0] - cx)**2 + (contour_points[:, 1] - cy)**2)
        
        if len(distances) > 1 and np.mean(distances) > 0:
            # Normalize distances by mean radius
            normalized_distances = distances / np.mean(distances)
            var_radii = np.var(normalized_distances)
            smoothness = 1.0 - (1.0 / (1.0 + var_radii))
        else:
            smoothness = 0.0
        
        # 6. Compactness: perimeter^2 / (4 * pi * area) - 1
        compactness = (perimeter ** 2 / (4 * np.pi * area)) - 1.0
        
        # 7. Concavity: fraction of convex hull area not covered by contour
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            concavity = (hull_area - area) / hull_area
        else:
            concavity = 0.0
        
        # 8. Concave Points: count of significant convexity defects
        hull_indices = cv2.convexHull(contour, returnPoints=False)
        concave_points_count = 0
        
        if len(hull_indices) > 3 and len(contour) > 3:
            try:
                defects = cv2.convexityDefects(contour, hull_indices)
                if defects is not None:
                    # Threshold: depth must be > 10% of the equivalent radius
                    depth_threshold = radius * 0.1
                    for d in defects:
                        depth = d.ravel()[3] / 256.0  # Convert fixpoint to float
                        if depth > depth_threshold:
                            concave_points_count += 1
            except cv2.error:
                concave_points_count = 0
        
        # Normalize by perimeter to get a fraction-like value
        concave_points = concave_points_count / max(len(contour), 1)
        
        # 9. Symmetry: minor_axis / major_axis from fitted ellipse
        if len(contour) >= 5:
            try:
                ellipse = cv2.fitEllipse(contour)
                (_, (axis_a, axis_b), _) = ellipse
                if max(axis_a, axis_b) > 0:
                    # Measured as deviation from circular symmetry: 1.0 - (minor / major)
                    symmetry = 1.0 - (min(axis_a, axis_b) / max(axis_a, axis_b))
                else:
                    symmetry = 0.0
            except cv2.error:
                symmetry = 0.0
        else:
            symmetry = 0.0
        
        # 10. Fractal Dimension: box-counting on nucleus boundary
        # Create a tight bounding box mask for efficiency
        x, y, bw, bh = cv2.boundingRect(contour)
        pad = 5
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(gray_image.shape[1], x + bw + pad)
        y2 = min(gray_image.shape[0], y + bh + pad)
        
        nucleus_mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
        shifted_contour = contour.copy()
        shifted_contour[:, :, 0] -= x1
        shifted_contour[:, :, 1] -= y1
        cv2.drawContours(nucleus_mask, [shifted_contour], -1, 255, -1)
        
        # In WDBC dataset, fractal dimension is recorded as box dimension minus 1.0
        fractal_dim = max(float(_box_counting_fractal_dimension(nucleus_mask)) - 1.0, 0.0)
        
        return {
            "radius": float(radius),
            "texture": float(texture),
            "perimeter": float(perimeter),
            "area": float(area),
            "smoothness": float(smoothness),
            "compactness": float(compactness),
            "concavity": float(concavity),
            "concave_points": float(concave_points),
            "symmetry": float(symmetry),
            "fractal_dimension": float(fractal_dim),
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None


def smooth_contour(contour: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Smooth a 2D closed contour using a 1D Gaussian kernel applied to its coordinates.
    Circular padding is used to maintain smoothness at the closed boundaries.
    """
    if len(contour) < 5:
        return contour.copy()
        
    coords = contour.reshape(-1, 2).astype(float)
    x = coords[:, 0]
    y = coords[:, 1]
    
    ksize = int(6 * sigma + 1)
    if ksize % 2 == 0:
        ksize += 1
        
    kernel = cv2.getGaussianKernel(ksize, sigma).ravel()
    pad_len = ksize // 2
    
    # Circular padding to make it smooth at connections
    x_padded = np.concatenate([x[-pad_len:], x, x[:pad_len]])
    y_padded = np.concatenate([y[-pad_len:], y, y[:pad_len]])
    
    x_smoothed = np.convolve(x_padded, kernel, mode='valid')
    y_smoothed = np.convolve(y_padded, kernel, mode='valid')
    
    smoothed_coords = np.column_stack((x_smoothed, y_smoothed))
    return smoothed_coords.reshape(-1, 1, 2).astype(np.int32)


def extract_all_nuclei_features(
    contours: List[np.ndarray], 
    gray_image: np.ndarray,
    return_qc: bool = False,
    sigma: float = 1.0,
    min_area: float = 50,
    max_area: float = 20000,
    min_circularity: float = 0.65,
    min_solidity: float = 0.85,
    min_aspect_ratio: float = 0.4,
    exclude_border: bool = True
) -> Any:
    """
    Extract features from all detected nuclei, applying contour smoothing and QC filters.
    
    Args:
        contours: List of contours from segmentation.
        gray_image: Grayscale image for texture measurement.
        return_qc: If True, returns a tuple (accepted_features, qc_metadata).
                   If False, returns only accepted_features (backward compatibility).
        sigma: Standard deviation for Gaussian contour smoothing.
        min_area: Minimum area in pixels.
        max_area: Maximum area in pixels.
        min_circularity: Minimum circularity (4 * pi * area / perimeter^2).
        min_solidity: Minimum solidity (area / convex_hull_area).
        min_aspect_ratio: Minimum aspect ratio (minor_axis / major_axis).
        exclude_border: If True, rejects objects touching image boundaries.
        
    Returns:
        If return_qc is False: List[Dict[str, float]] of features for accepted nuclei.
        If return_qc is True: Tuple of (List[Dict[str, float]], Dict[str, Any] metadata).
    """
    accepted_features = []
    accepted_contours = []
    rejected_info = [] # List of dicts: {"contour": c, "reason": r, "metrics": m}
    
    h_img, w_img = gray_image.shape[:2]
    
    for idx, raw_cnt in enumerate(contours):
        # 1. Apply Contour Smoothing
        smoothed = smooth_contour(raw_cnt, sigma=sigma)
        
        # Calculate intermediate metrics for QC
        area = float(cv2.contourArea(smoothed))
        perimeter = float(cv2.arcLength(smoothed, True))
        
        # Circularity
        circularity = (4.0 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0
        
        # Solidity
        hull = cv2.convexHull(smoothed.astype(np.float32))
        hull_area = float(cv2.contourArea(hull))
        solidity = (area / hull_area) if hull_area > 0 else 0.0
        
        # Aspect Ratio (minor / major)
        if len(smoothed) >= 5:
            try:
                _, (axis_a, axis_b), _ = cv2.fitEllipse(smoothed)
                aspect_ratio = min(axis_a, axis_b) / max(axis_a, axis_b) if max(axis_a, axis_b) > 0 else 1.0
            except:
                _, _, w, h = cv2.boundingRect(smoothed)
                aspect_ratio = min(w, h) / max(w, h) if max(w, h) > 0 else 1.0
        else:
            _, _, w, h = cv2.boundingRect(smoothed)
            aspect_ratio = min(w, h) / max(w, h) if max(w, h) > 0 else 1.0
            
        # Border contact check
        coords = smoothed.reshape(-1, 2)
        touches_border = bool(np.any((coords[:, 0] <= 1) | (coords[:, 0] >= w_img - 2) | 
                                     (coords[:, 1] <= 1) | (coords[:, 1] >= h_img - 2)))
        
        # Perform QC Evaluations
        qc_metrics = {
            "area": area,
            "perimeter": perimeter,
            "circularity": circularity,
            "solidity": solidity,
            "aspect_ratio": aspect_ratio,
            "touches_border": touches_border
        }
        
        rejection_reason = None
        if area < min_area:
            rejection_reason = "Debris / Too Small"
        elif area > max_area:
            rejection_reason = "Cell Clump / Too Large"
        elif exclude_border and touches_border:
            rejection_reason = "Touches Border"
        elif circularity < min_circularity:
            rejection_reason = "Low Circularity"
        elif solidity < min_solidity:
            rejection_reason = "Low Solidity / Concave Clump"
        elif aspect_ratio < min_aspect_ratio:
            rejection_reason = "Elongated / Low Aspect Ratio"
            
        if rejection_reason is not None:
            rejected_info.append({
                "contour": smoothed,
                "raw_contour": raw_cnt,
                "reason": rejection_reason,
                "metrics": qc_metrics
            })
            continue
            
        # Extract features for accepted nucleus
        features = extract_single_nucleus_features(smoothed, gray_image)
        if features is not None:
            accepted_features.append(features)
            accepted_contours.append(smoothed)
        else:
            rejected_info.append({
                "contour": smoothed,
                "raw_contour": raw_cnt,
                "reason": "Feature extraction failed",
                "metrics": qc_metrics
            })
            
    qc_metadata = {
        "raw_count": len(contours),
        "accepted_count": len(accepted_features),
        "rejected_count": len(rejected_info),
        "accepted_contours": accepted_contours,
        "rejected_info": rejected_info
    }
    
    if return_qc:
        return accepted_features, qc_metadata
    else:
        return accepted_features

