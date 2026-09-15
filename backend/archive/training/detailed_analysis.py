import json
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(script_dir)
json_path = os.path.join(backend_dir, "models", "candidates", "v4_evaluation_summary.json")

with open(json_path) as f:
    data = json.load(f)

classes = ['benign', 'malignant', 'normal']

print("=======================================================================")
print("COMPLETE MODEL EVALUATION BREAKDOWN (TEST SET: 117 IMAGES)")
print("=======================================================================\n")

for k, v in data.items():
    name = v.get('name', k)
    print(f"--- {name} ({k}) ---")
    print(f"Accuracy           : {v['accuracy']*100:.2f}%")
    print(f"Macro Precision    : {v['macro_precision']*100:.2f}%")
    print(f"Macro Recall       : {v['macro_recall']*100:.2f}%")
    print(f"Macro F1           : {v['macro_f1']*100:.2f}%")
    print("Class Metrics:")
    for c in classes:
        cm = v[c]
        print(f"  - {c.capitalize():<10}: Precision={cm['precision']*100:6.2f}%, Recall={cm['recall']*100:6.2f}%, F1={cm['f1']*100:6.2f}% (Support: {cm['support']})")
    print(f"Malignant False Negatives (FN): {v['false_negatives']}")
    print(f"Malignant False Positives (FP): {v['false_positives']}")
    print(f"Total Incorrect Predictions   : {v['total_incorrect']}")
    print("Confusion Matrix (Row=True, Col=Pred):")
    cm = v['confusion_matrix']
    print(f"  True Benign   : Benign={cm[0][0]:2d}, Malignant={cm[0][1]:2d}, Normal={cm[0][2]:2d}")
    print(f"  True Malignant: Benign={cm[1][0]:2d}, Malignant={cm[1][1]:2d}, Normal={cm[1][2]:2d}")
    print(f"  True Normal   : Benign={cm[2][0]:2d}, Malignant={cm[2][1]:2d}, Normal={cm[2][2]:2d}")
    print()
