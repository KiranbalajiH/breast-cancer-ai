"""
Feature Aggregation Module
===========================
Aggregates per-nucleus measurements across all detected nuclei to produce
the 30-feature vector expected by the existing SVM model.

For each of the 10 base properties:
- Mean:           Average across all nuclei
- Standard Error: std(values) / sqrt(n) — standard error of the mean
- Worst:          Maximum value across all nuclei

The output feature names and order match exactly:
    model.feature_names_in_ from breast_cancer_model.joblib

Minimum 3 nuclei required for meaningful statistics.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple

# The 10 base measurement keys (in extraction output format)
BASE_PROPERTIES = [
    "radius", "texture", "perimeter", "area", "smoothness",
    "compactness", "concavity", "concave_points", "symmetry", "fractal_dimension"
]

# Mapping from extraction key to training dataset feature name components
# The training dataset uses spaces and "concave points" / "fractal dimension"
KEY_TO_NAME = {
    "radius": "radius",
    "texture": "texture",
    "perimeter": "perimeter",
    "area": "area",
    "smoothness": "smoothness",
    "compactness": "compactness",
    "concavity": "concavity",
    "concave_points": "concave points",
    "symmetry": "symmetry",
    "fractal_dimension": "fractal dimension",
}

# The exact 30 feature names in training order
FEATURE_NAMES_ORDERED = [
    "mean radius", "mean texture", "mean perimeter", "mean area",
    "mean smoothness", "mean compactness", "mean concavity", "mean concave points",
    "mean symmetry", "mean fractal dimension",
    "radius error", "texture error", "perimeter error", "area error",
    "smoothness error", "compactness error", "concavity error", "concave points error",
    "symmetry error", "fractal dimension error",
    "worst radius", "worst texture", "worst perimeter", "worst area",
    "worst smoothness", "worst compactness", "worst concavity", "worst concave points",
    "worst symmetry", "worst fractal dimension",
]


def aggregate_features(
    nuclei_features: List[Dict[str, float]],
    min_nuclei: int = 3,
) -> Optional[Tuple[Dict[str, float], Dict[str, any]]]:
    """
    Aggregate per-nucleus features into the 30-feature model input vector.
    
    Args:
        nuclei_features: List of per-nucleus feature dicts from extraction.
        min_nuclei: Minimum number of nuclei required.
        
    Returns:
        Tuple of (features_dict, metadata) where features_dict has keys matching
        the training feature names, or None if insufficient nuclei.
    """
    n = len(nuclei_features)
    
    if n < min_nuclei:
        return None
    
    features_30 = {}
    per_feature_stats = {}
    
    for prop_key in BASE_PROPERTIES:
        name = KEY_TO_NAME[prop_key]
        
        # Collect this property across all nuclei
        values = np.array([nf[prop_key] for nf in nuclei_features])
        
        mean_val = float(np.mean(values))
        std_val = float(np.std(values, ddof=1)) if n > 1 else 0.0  # sample std
        se_val = std_val / np.sqrt(n)  # standard error
        worst_val = float(np.max(values))
        
        # Map to the exact training feature names
        features_30[f"mean {name}"] = mean_val
        features_30[f"{name} error"] = se_val
        features_30[f"worst {name}"] = worst_val
        
        per_feature_stats[name] = {
            "values": values.tolist(),
            "mean": mean_val,
            "std": std_val,
            "se": se_val,
            "min": float(np.min(values)),
            "max": worst_val,
            "n": n,
        }
    
    # Order features to match model schema exactly
    ordered_features = {name: features_30[name] for name in FEATURE_NAMES_ORDERED}
    
    metadata = {
        "num_nuclei": n,
        "per_feature_stats": per_feature_stats,
    }
    
    return ordered_features, metadata
