import pytest
import numpy as np
from scipy import stats

def compute_diagnostic_metrics(y_true, y_pred, target_class=1):
    """
    Computes Sensitivity, Specificity, PPV, and NPV for a target class.
    """
    tp = int(np.sum((y_true == target_class) & (y_pred == target_class)))
    fn = int(np.sum((y_true == target_class) & (y_pred != target_class)))
    fp = int(np.sum((y_true != target_class) & (y_pred == target_class)))
    tn = int(np.sum((y_true != target_class) & (y_pred != target_class)))
    
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    
    return {
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "sensitivity": sens,
        "specificity": spec,
        "ppv": ppv,
        "npv": npv
    }

def mcnemar_exact_and_asymptotic(y_true, pred_a, pred_b):
    correct_a = (pred_a == y_true)
    correct_b = (pred_b == y_true)
    
    n00 = int(np.sum(~correct_a & ~correct_b))
    n01 = int(np.sum(~correct_a & correct_b))
    n10 = int(np.sum(correct_a & ~correct_b))
    n11 = int(np.sum(correct_a & correct_b))
    
    b, c = n10, n01
    
    # Asymptotic Chi2 with continuity correction
    if b + c == 0:
        chi2_stat = 0.0
        chi2_pval = 1.0
    else:
        chi2_stat = (abs(b - c) - 1.0)**2 / (b + c)
        chi2_pval = 1.0 - stats.chi2.cdf(chi2_stat, 1)
        
    # Exact Binomial test
    if b + c == 0:
        exact_pval = 1.0
    else:
        exact_pval = stats.binomtest(min(b, c), b + c, 0.5, alternative='two-sided').pvalue
        
    return {
        "n00": n00, "n01_b_only": n01, "n10_a_only": n10, "n11": n11,
        "chi2_stat": float(chi2_stat),
        "chi2_p_value": float(chi2_pval),
        "exact_binomial_p_value": float(exact_pval)
    }

def test_mcnemar_exact_vs_asymptotic():
    # Known 17 discordant samples (3 vs 14)
    y_true = np.array([1]*17 + [0]*10)
    # Model A: gets first 3 right, fails next 14, gets last 10 right
    pred_a = np.array([1]*3 + [0]*14 + [0]*10)
    # Model B: fails first 3, gets next 14 right, gets last 10 right
    pred_b = np.array([0]*3 + [1]*14 + [0]*10)
    
    res = mcnemar_exact_and_asymptotic(y_true, pred_a, pred_b)
    assert res["n10_a_only"] == 3
    assert res["n01_b_only"] == 14
    assert round(res["chi2_stat"], 4) == 5.8824
    assert round(res["chi2_p_value"], 4) == 0.0153
    assert round(res["exact_binomial_p_value"], 4) == 0.0127

def test_diagnostic_metrics_known_cm():
    # Test on V5-B Grouped CM:
    # True Benign: 36 Benign, 15 Malignant, 8 Normal
    # True Malignant: 3 Benign, 25 Malignant, 3 Normal
    # True Normal: 3 Benign, 0 Malignant, 14 Normal
    # Class 1 (Malignant): TP = 25, FN = 6, FP = 15, TN = 61
    y_true = np.array([0]*59 + [1]*31 + [2]*17)
    y_pred = np.array(
        [0]*36 + [1]*15 + [2]*8 + # Benign true
        [0]*3 + [1]*25 + [2]*3 +  # Malignant true
        [0]*3 + [1]*0 + [2]*14    # Normal true
    )
    
    m = compute_diagnostic_metrics(y_true, y_pred, target_class=1)
    assert m["tp"] == 25
    assert m["fn"] == 6
    assert m["fp"] == 15
    assert m["tn"] == 61
    assert round(m["sensitivity"], 4) == round(25 / 31, 4)
    assert round(m["specificity"], 4) == round(61 / 76, 4)
    assert round(m["ppv"], 4) == round(25 / 40, 4)
    assert round(m["npv"], 4) == round(61 / 67, 4)
