"""
Feature Compatibility Validation Module
========================================
Validates whether image-extracted features are compatible with the
training distribution of the existing SVM model.

Methodology:
- For each of the 30 features, compute a z-score against the training
  dataset distribution: z = |extracted_value - training_mean| / training_std
- Per-feature verdict:
    - Compatible:     z < 3    (within 3 standard deviations)
    - Marginal:       3 ≤ z < 5
    - Incompatible:   z ≥ 5
- Overall verdict:
    - Compatible:                ≤3 Marginal AND 0 Incompatible
    - Potentially Incompatible:  >3 Marginal OR 1-2 Incompatible
    - Incompatible:              ≥3 Incompatible

Training statistics are embedded from the actual breast_cancer.csv dataset.
"""

from typing import Dict, List, Any


# Training dataset statistics (computed from backend/data/breast_cancer.csv)
TRAINING_STATS: Dict[str, Dict[str, float]] = {
    "mean radius": {"min": 6.981, "max": 28.11, "mean": 14.1273, "std": 3.5240},
    "mean texture": {"min": 9.71, "max": 39.28, "mean": 19.2896, "std": 4.3010},
    "mean perimeter": {"min": 43.79, "max": 188.5, "mean": 91.9690, "std": 24.2990},
    "mean area": {"min": 143.5, "max": 2501.0, "mean": 654.8891, "std": 351.9141},
    "mean smoothness": {"min": 0.05263, "max": 0.1634, "mean": 0.09636, "std": 0.01406},
    "mean compactness": {"min": 0.01938, "max": 0.3454, "mean": 0.10434, "std": 0.05281},
    "mean concavity": {"min": 0.0, "max": 0.4268, "mean": 0.08880, "std": 0.07972},
    "mean concave points": {"min": 0.0, "max": 0.2012, "mean": 0.04892, "std": 0.03880},
    "mean symmetry": {"min": 0.106, "max": 0.304, "mean": 0.18116, "std": 0.02741},
    "mean fractal dimension": {"min": 0.04996, "max": 0.09744, "mean": 0.06280, "std": 0.00706},
    "radius error": {"min": 0.1115, "max": 2.873, "mean": 0.40517, "std": 0.27731},
    "texture error": {"min": 0.3602, "max": 4.885, "mean": 1.21685, "std": 0.55165},
    "perimeter error": {"min": 0.757, "max": 21.98, "mean": 2.86606, "std": 2.02185},
    "area error": {"min": 6.802, "max": 542.2, "mean": 40.3371, "std": 45.4910},
    "smoothness error": {"min": 0.001713, "max": 0.03113, "mean": 0.00704, "std": 0.00300},
    "compactness error": {"min": 0.002252, "max": 0.1354, "mean": 0.02548, "std": 0.01791},
    "concavity error": {"min": 0.0, "max": 0.396, "mean": 0.03189, "std": 0.03019},
    "concave points error": {"min": 0.0, "max": 0.05279, "mean": 0.01180, "std": 0.00617},
    "symmetry error": {"min": 0.007882, "max": 0.07895, "mean": 0.02054, "std": 0.00827},
    "fractal dimension error": {"min": 0.0008948, "max": 0.02984, "mean": 0.00379, "std": 0.00265},
    "worst radius": {"min": 7.93, "max": 36.04, "mean": 16.2692, "std": 4.8332},
    "worst texture": {"min": 12.02, "max": 49.54, "mean": 25.6772, "std": 6.1463},
    "worst perimeter": {"min": 50.41, "max": 251.2, "mean": 107.2612, "std": 33.6025},
    "worst area": {"min": 185.2, "max": 4254.0, "mean": 880.5831, "std": 569.3570},
    "worst smoothness": {"min": 0.07117, "max": 0.2226, "mean": 0.13237, "std": 0.02283},
    "worst compactness": {"min": 0.02729, "max": 1.058, "mean": 0.25427, "std": 0.15734},
    "worst concavity": {"min": 0.0, "max": 1.252, "mean": 0.27219, "std": 0.20862},
    "worst concave points": {"min": 0.0, "max": 0.291, "mean": 0.11461, "std": 0.06573},
    "worst symmetry": {"min": 0.1565, "max": 0.6638, "mean": 0.29008, "std": 0.06187},
    "worst fractal dimension": {"min": 0.05504, "max": 0.2075, "mean": 0.08395, "std": 0.01806},
}


def validate_compatibility(
    extracted_features: Dict[str, float],
) -> Dict[str, Any]:
    """
    Validate extracted features against the training dataset distribution.
    
    Args:
        extracted_features: Dictionary of 30 features (name -> value).
        
    Returns:
        Dictionary with:
            - per_feature: list of per-feature reports
            - num_compatible: count
            - num_marginal: count
            - num_incompatible: count
            - overall_verdict: "Compatible" | "Potentially Incompatible" | "Incompatible"
            - prediction_allowed: bool
            - message: human-readable summary
    """
    per_feature: List[Dict[str, Any]] = []
    num_compatible = 0
    num_marginal = 0
    num_incompatible = 0
    
    for feature_name, stats in TRAINING_STATS.items():
        extracted_val = extracted_features.get(feature_name)
        
        if extracted_val is None:
            per_feature.append({
                "name": feature_name,
                "extracted": None,
                "z_score": None,
                "verdict": "Missing",
                "training_range": [stats["min"], stats["max"]],
                "training_mean": stats["mean"],
            })
            num_incompatible += 1
            continue
        
        # Compute z-score
        if stats["std"] > 0:
            z = abs(extracted_val - stats["mean"]) / stats["std"]
        else:
            z = 0.0 if extracted_val == stats["mean"] else 999.0
        
        if z < 3:
            verdict = "Compatible"
            num_compatible += 1
        elif z < 5:
            verdict = "Marginal"
            num_marginal += 1
        else:
            verdict = "Incompatible"
            num_incompatible += 1
        
        per_feature.append({
            "name": feature_name,
            "extracted": float(extracted_val),
            "z_score": round(float(z), 2),
            "verdict": verdict,
            "training_range": [stats["min"], stats["max"]],
            "training_mean": stats["mean"],
        })
    
    # Overall verdict
    if num_incompatible >= 3:
        overall = "Incompatible"
        prediction_allowed = False
        message = (
            "The extracted image measurements are outside the validated feature "
            "distribution of the current model. A reliable prediction cannot be generated."
        )
    elif num_incompatible >= 1 or num_marginal > 3:
        overall = "Potentially Incompatible"
        prediction_allowed = False
        message = (
            "Several extracted measurements fall outside the expected training "
            "distribution. Prediction is blocked to avoid unreliable results."
        )
    else:
        overall = "Compatible"
        prediction_allowed = True
        message = (
            "Extracted features are within the expected distribution. "
            "Prediction may proceed, but results remain experimental and unvalidated."
        )
    
    return {
        "per_feature": per_feature,
        "num_compatible": num_compatible,
        "num_marginal": num_marginal,
        "num_incompatible": num_incompatible,
        "overall_verdict": overall,
        "prediction_allowed": prediction_allowed,
        "message": message,
    }
