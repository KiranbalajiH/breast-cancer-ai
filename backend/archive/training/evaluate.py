import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

def evaluate_model(y_true, y_pred, y_prob):
    """
    Evaluates a model using various metrics.
    Malignant is assumed to be the positive class (usually 0 in sklearn breast cancer dataset).
    We will ensure that positive_label is correctly configured to represent malignant.
    """
    # Assuming malignant is 0 and benign is 1 from scikit-learn dataset
    # We invert it for calculation so Malignant is positive (1)
    y_true_m = 1 - y_true
    y_pred_m = 1 - y_pred
    
    # If y_prob is for class 1 (benign), probability of malignant is 1 - y_prob
    # But usually predict_proba returns [prob_0, prob_1], we want prob_0 (malignant)
    y_prob_m = y_prob[:, 0] if len(y_prob.shape) > 1 else 1 - y_prob
    
    acc = accuracy_score(y_true_m, y_pred_m)
    prec = precision_score(y_true_m, y_pred_m, zero_division=0)
    rec = recall_score(y_true_m, y_pred_m, zero_division=0)
    
    # Specificity = TN / (TN + FP) -> Recall of the negative class
    # Here negative class is Benign (1 in original, 0 in mapped)
    # y_true == 1 and y_pred == 1 means TN for malignant
    # Or just use confusion matrix
    cm = confusion_matrix(y_true_m, y_pred_m, labels=[0, 1])
    # cm[0,0]: TN, cm[0,1]: FP
    # cm[1,0]: FN, cm[1,1]: TP
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    else:
        spec = 0.0

    f1 = f1_score(y_true_m, y_pred_m, zero_division=0)
    roc = roc_auc_score(y_true_m, y_prob_m)

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "specificity": float(spec),
        "f1_score": float(f1),
        "roc_auc": float(roc),
        "confusion_matrix": {
            "tn": int(tn) if 'tn' in locals() else 0,
            "fp": int(fp) if 'fp' in locals() else 0,
            "fn": int(fn) if 'fn' in locals() else 0,
            "tp": int(tp) if 'tp' in locals() else 0
        }
    }
