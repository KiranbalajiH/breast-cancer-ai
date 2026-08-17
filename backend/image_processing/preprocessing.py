"""
Image Preprocessing Module
==========================
Prepares cell microscopy images for nuclei segmentation.

Pipeline:
1. Grayscale conversion (if RGB/BGR input)
2. Gaussian blur denoising (sigma=1.0) — reduces high-frequency noise
3. CLAHE contrast enhancement — improves nuclei/background separation
4. Normalization to uint8 [0, 255]
"""

import cv2
import numpy as np
from typing import Tuple


def preprocess_image(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess a cell microscopy image for nuclei segmentation.
    
    Args:
        image: Input image as numpy array (BGR or grayscale).
        
    Returns:
        Tuple of (grayscale_original, preprocessed_image), both uint8.
    """
    # Step 1: Convert to grayscale if color
    if len(image.shape) == 3 and image.shape[2] >= 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif len(image.shape) == 2:
        gray = image.copy()
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")
    
    gray_original = gray.copy()
    
    # Step 2: Gaussian blur for denoising
    # sigma=1.0 is conservative — removes sensor noise without destroying nucleus edges
    denoised = cv2.GaussianBlur(gray, (5, 5), sigmaX=1.0)
    
    # Step 3: CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # clipLimit=2.0 prevents over-amplification of noise
    # tileGridSize=(8,8) gives local contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    
    # Step 4: Ensure uint8 normalization
    preprocessed = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    return gray_original, preprocessed
