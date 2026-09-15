import json
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(script_dir)
json_path = os.path.join(backend_dir, "models", "candidates", "v4_evaluation_summary.json")

with open(json_path) as f:
    data = json.load(f)

print(f"{'Model Name':<58} | {'Acc':<7} | {'MacroF1':<7} | {'Mal Prec':<8} | {'Mal Rec':<7} | {'Mal F1':<6} | {'FN':<2} | {'FP':<2} | {'Err':<3}")
print("-" * 118)
for k, v in data.items():
    name = v.get('name', k)
    acc = f"{v['accuracy']*100:.2f}%"
    mf1 = f"{v['macro_f1']*100:.2f}%"
    mp = f"{v['malignant']['precision']*100:.2f}%"
    mr = f"{v['malignant']['recall']*100:.2f}%"
    mf1_c = f"{v['malignant']['f1']*100:.2f}%"
    fn = v['false_negatives']
    fp = v['false_positives']
    err = v['total_incorrect']
    print(f"{name:<58} | {acc:<7} | {mf1:<7} | {mp:<8} | {mr:<7} | {mf1_c:<6} | {fn:<2} | {fp:<2} | {err:<3}")

val_json_path = os.path.join(backend_dir, "v4_validation_summary.json")

if os.path.exists(val_json_path):
    print("\nValidation Summary:")
    with open(val_json_path) as f:
        val_data = json.load(f)
    val_res = val_data.get("validation_results", {})
    print(f"{'Variant ID':<25} | {'Val Acc':<7} | {'Val MacroF1':<11} | {'Val Mal Prec':<12} | {'Val Mal Rec':<11} | {'Val Mal F1':<10} | {'FN':<2} | {'FP':<2}")
    print("-" * 105)
    for cid, v in val_res.items():
        acc = f"{v['accuracy']*100:.2f}%"
        mf1 = f"{v['macro_f1']*100:.2f}%"
        mp = f"{v['malignant']['precision']*100:.2f}%"
        mr = f"{v['malignant']['recall']*100:.2f}%"
        mf1_c = f"{v['malignant']['f1']*100:.2f}%"
        fn = v['false_negatives']
        fp = v['false_positives']
        print(f"{cid:<25} | {acc:<7} | {mf1:<11} | {mp:<12} | {mr:<11} | {mf1_c:<10} | {fn:<2} | {fp:<2}")
