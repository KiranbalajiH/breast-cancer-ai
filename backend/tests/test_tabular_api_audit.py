import sys
import os
import json
import joblib
import pandas as pd

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)

from fastapi.testclient import TestClient
from app.main import app

def run_audit():
    csv_path = r"C:\Users\kiran\Downloads\data.csv"
    model_path = os.path.join(base_dir, "models", "breast_cancer_model.joblib")
    
    df = pd.read_csv(csv_path)

    feature_mapping = {
        'radius_mean': 'mean radius', 'texture_mean': 'mean texture', 'perimeter_mean': 'mean perimeter',
        'area_mean': 'mean area', 'smoothness_mean': 'mean smoothness', 'compactness_mean': 'mean compactness',
        'concavity_mean': 'mean concavity', 'concave points_mean': 'mean concave points', 'symmetry_mean': 'mean symmetry',
        'fractal_dimension_mean': 'mean fractal dimension', 'radius_se': 'radius error', 'texture_se': 'texture error',
        'perimeter_se': 'perimeter error', 'area_se': 'area error', 'smoothness_se': 'smoothness error',
        'compactness_se': 'compactness error', 'concavity_se': 'concavity error', 'concave points_se': 'concave points error',
        'symmetry_se': 'symmetry error', 'fractal_dimension_se': 'fractal dimension error', 'radius_worst': 'worst radius',
        'texture_worst': 'worst texture', 'perimeter_worst': 'worst perimeter', 'area_worst': 'worst area',
        'smoothness_worst': 'worst smoothness', 'compactness_worst': 'worst compactness', 'concavity_worst': 'worst concavity',
        'concave points_worst': 'worst concave points', 'symmetry_worst': 'worst symmetry', 'fractal_dimension_worst': 'worst fractal dimension'
    }

    benign_row = df[df['diagnosis'] == 'B'].iloc[0]
    malignant_row = df[df['diagnosis'] == 'M'].iloc[0]

    pipeline = joblib.load(model_path)

    results = []

    with TestClient(app) as client:
        for name, row, expected_code in [("Benign Sample", benign_row, "B"), ("Malignant Sample", malignant_row, "M")]:
            row_id = row['id']
            features_dict = {feature_mapping[k]: float(v) for k, v in row.items() if k in feature_mapping}

            # Direct Pipeline prediction
            single_df = pd.DataFrame([features_dict])
            direct_pred_class = pipeline.predict(single_df)[0]
            direct_probs = pipeline.predict_proba(single_df)[0]

            # API prediction
            response = client.post('/api/predict', json=features_dict)
            res_json = response.json()

            status_ok = (response.status_code == 200)
            code_match = (res_json.get('prediction_code') == expected_code)
            
            # Check prob agreement
            api_ben_prob = res_json.get('probabilities', {}).get('benign', 0.0)
            api_mal_prob = res_json.get('probabilities', {}).get('malignant', 0.0)
            
            prob_match = (abs(api_ben_prob - direct_probs[0]) < 1e-4) and (abs(api_mal_prob - direct_probs[1]) < 1e-4)

            result_summary = {
                "sample_type": name,
                "patient_id": int(row_id),
                "http_status": response.status_code,
                "expected_code": expected_code,
                "returned_code": res_json.get('prediction_code'),
                "returned_prediction": res_json.get('prediction'),
                "confidence": res_json.get('confidence'),
                "api_probabilities": res_json.get('probabilities'),
                "direct_probabilities": {"benign": float(direct_probs[0]), "malignant": float(direct_probs[1])},
                "model_metadata": res_json.get('model'),
                "status_ok": status_ok,
                "code_match": code_match,
                "prob_match": prob_match
            }
            results.append(result_summary)

            print(f"=== {name} (ID: {row_id}) ===")
            print(f"HTTP Status: {response.status_code}")
            print(f"Prediction: {res_json.get('prediction')} (Code: {res_json.get('prediction_code')})")
            print(f"Confidence: {res_json.get('confidence')}")
            print(f"API Probabilities: {res_json.get('probabilities')}")
            print(f"Direct Model Probabilities: benign={direct_probs[0]:.6f}, malignant={direct_probs[1]:.6f}")
            print(f"Model Metadata: {res_json.get('model')}")
            print(f"Match Verdict: {'PASS' if (status_ok and code_match and prob_match) else 'FAIL'}\n")

    all_passed = all(r['status_ok'] and r['code_match'] and r['prob_match'] for r in results)
    print(f"Overall Integration Audit Result: {'PASS' if all_passed else 'FAIL'}")

if __name__ == "__main__":
    run_audit()
