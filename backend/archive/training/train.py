import os
import json
import joblib
import pandas as pd
from datetime import datetime
from sklearn.inspection import permutation_importance

from compare_models import get_models, compare_models, select_best_model
from evaluate import evaluate_model

# Ensure models directory exists
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

def train_and_save():
    print("Loading local Breast Cancer Wisconsin Diagnostic dataset...")
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'breast_cancer.csv')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Please create it first.")
        
    df = pd.read_csv(data_path)
    # The last column 'target' is the target. The rest are features.
    X = df.drop(columns=['target'])
    y = df['target']
    
    feature_names = X.columns.tolist()
    target_names = ["malignant", "benign"] # standard mapping: 0=malignant, 1=benign
    
    print(f"Dataset shape: {X.shape}")
    print(f"Target classes: {target_names} (0: {target_names[0]}, 1: {target_names[1]})")
    
    print("\nStarting model comparison with Stratified 5-Fold CV...")
    results = compare_models(X, y)
    
    print("\nModel Comparison Results:")
    for name, metrics in results.items():
        print(f"--- {name} ---")
        for k, v in metrics.items():
            if isinstance(v, dict):
                print(f"  {k}: {v}")
            else:
                print(f"  {k}: {v:.4f}")
            
    best_model_name = select_best_model(results)
    best_metrics = results[best_model_name]
    print(f"\n=> Selected Best Model: {best_model_name}")
    
    print(f"Training {best_model_name} on the full dataset...")
    models = get_models()
    final_model = models[best_model_name]
    final_model.fit(X, y)
    
    # Save model comparison
    comparison_path = os.path.join(MODELS_DIR, 'model_comparison.json')
    with open(comparison_path, 'w') as f:
        json.dump(results, f, indent=2)
        
    # Save the final model
    model_path = os.path.join(MODELS_DIR, 'breast_cancer_model.joblib')
    joblib.dump(final_model, model_path)
    
    print(f"Calculating feature importance for {best_model_name}...")
    # Calculate permutation importance on the full dataset (as an approximation)
    perm_importance = permutation_importance(final_model, X, y, n_repeats=10, random_state=42)
    
    # Create sorted feature importance list
    feature_importances = []
    for i in perm_importance.importances_mean.argsort()[::-1]:
        feature_importances.append({
            "feature": feature_names[i],
            "importance": float(perm_importance.importances_mean[i])
        })
    
    # Save metadata
    metadata = {
        "model_name": best_model_name,
        "model_version": "1.1.0",
        "dataset_name": "Local Breast Cancer Wisconsin Diagnostic",
        "training_date": datetime.now().isoformat(),
        "random_seed": 42,
        "feature_count": X.shape[1],
        "feature_names": feature_names,
        "target_names": target_names, # ['malignant', 'benign'] -> [0, 1]
        "metrics": best_metrics,
        "feature_importance": feature_importances
    }
    
    metadata_path = os.path.join(MODELS_DIR, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
        
    print(f"\nTraining complete. Model saved to {model_path}")
    print(f"Metadata saved to {metadata_path}")
    print(f"Comparison saved to {comparison_path}")

if __name__ == "__main__":
    train_and_save()
