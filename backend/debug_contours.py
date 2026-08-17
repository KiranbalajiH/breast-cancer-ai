import cv2
import numpy as np
from image_processing.preprocessing import preprocess_image
from image_processing.segmentation import segment_nuclei

import os
img = cv2.imread(os.path.join(os.path.dirname(__file__), "test_synthetic_nuclei.png"))
gray, pp = preprocess_image(img)
seg = segment_nuclei(gray, pp)

c = seg["contours"][0]
area = cv2.contourArea(c)
print(f"Area: {area}")
perimeter = cv2.arcLength(c, True)
print(f"Perimeter: {perimeter}")

# Radius
radius = np.sqrt(area / np.pi)
print(f"Radius: {radius}")

# Texture
mask = np.zeros(gray.shape[:2], dtype=np.uint8)
cv2.drawContours(mask, [c], -1, 255, -1)
pixels = gray[mask == 255]
print(f"Pixel count: {len(pixels)}")
print(f"Pixel std: {np.std(pixels)}")

# Moments
M = cv2.moments(c)
print(f"m00: {M['m00']}")

# Smoothness
cx = M["m10"] / M["m00"]
cy = M["m01"] / M["m00"]
contour_points = c.reshape(-1, 2).astype(float)
distances = np.sqrt((contour_points[:, 0] - cx)**2 + (contour_points[:, 1] - cy)**2)
print(f"Distances: mean={np.mean(distances):.2f}, std={np.std(distances):.2f}")

# Compactness
compactness = (perimeter ** 2 / area) - 1.0
print(f"Compactness: {compactness}")

# Concavity
hull = cv2.convexHull(c)
hull_area = cv2.contourArea(hull)
print(f"Hull area: {hull_area}, Concavity: {(hull_area - area) / hull_area}")

# Concave points
hull_indices = cv2.convexHull(c, returnPoints=False)
print(f"Hull indices shape: {hull_indices.shape}, contour length: {len(c)}")

try:
    defects = cv2.convexityDefects(c, hull_indices)
    print(f"Defects: {defects is not None}, count: {len(defects) if defects is not None else 0}")
except Exception as e:
    print(f"Defects error: {e}")

# Symmetry
if len(c) >= 5:
    try:
        ellipse = cv2.fitEllipse(c)
        print(f"Ellipse: {ellipse}")
    except Exception as e:
        print(f"Ellipse error: {e}")

# Fractal dimension
from image_processing.feature_extraction import _box_counting_fractal_dimension
x, y, bw, bh = cv2.boundingRect(c)
pad = 5
x1 = max(0, x - pad)
y1 = max(0, y - pad)
x2 = min(gray.shape[1], x + bw + pad)
y2 = min(gray.shape[0], y + bh + pad)
nucleus_mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
shifted = c.copy()
shifted[:, :, 0] -= x1
shifted[:, :, 1] -= y1
cv2.drawContours(nucleus_mask, [shifted], -1, 255, -1)
print(f"Mask shape: {nucleus_mask.shape}")
fd = _box_counting_fractal_dimension(nucleus_mask)
print(f"Fractal dim: {fd}")

print("\n--- All checks passed individually ---")

# Now run with exception catching
from image_processing.feature_extraction import extract_single_nucleus_features
import traceback

try:
    result = extract_single_nucleus_features(c, gray)
    print(f"Result: {result}")
except Exception as e:
    traceback.print_exc()
