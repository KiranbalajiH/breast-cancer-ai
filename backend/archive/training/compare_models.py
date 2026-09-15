import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from evaluate import evaluate_model

def get_models():
    """
    Returns a dictionary of model pipelines to compare.
    All pipelines include standard scaling to prevent data leakage.
    """
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(random_state=42, max_iter=1000))
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(random_state=42, n_estimators=100))
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=5))
        ]),
        "SVM (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True, random_state=42))
        ]),
        "Naive Bayes": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GaussianNB())
        ]),
        "Neural Network": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(random_state=42, max_iter=1000, hidden_layer_sizes=(100,)))
        ]),
        "LDA + Neural Network": Pipeline([
            ("scaler", StandardScaler()),
            ("lda", LinearDiscriminantAnalysis()),
            ("clf", MLPClassifier(random_state=42, max_iter=1000, hidden_layer_sizes=(100,)))
        ]),
        "SVM (RBF) + SMOTE": ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=42)),
            ("clf", SVC(kernel="rbf", probability=True, random_state=42))
        ]),
        "Random Forest + SMOTE": ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=42)),
            ("clf", RandomForestClassifier(random_state=42, n_estimators=100))
        ])
    }
    return models

def compare_models(X, y):
    """
    Evaluates all models using Stratified 5-Fold Cross-Validation.
    """
    models = get_models()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    results = {name: [] for name in models.keys()}
    
    for name, model in models.items():
        fold_metrics = []
        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)
            
            metrics = evaluate_model(y_test, y_pred, y_prob)
            fold_metrics.append(metrics)
            
        # Average the metrics across folds
        avg_metrics = {}
        for k in fold_metrics[0].keys():
            if isinstance(fold_metrics[0][k], dict):
                avg_metrics[k] = {}
                for sub_k in fold_metrics[0][k].keys():
                    avg_metrics[k][sub_k] = int(np.sum([m[k][sub_k] for m in fold_metrics]))
            else:
                avg_metrics[k] = float(np.mean([m[k] for m in fold_metrics]))
            
        results[name] = avg_metrics
        
    return results

def select_best_model(results):
    """
    Selects the best model based on:
    1. Strong malignant-case recall (recall)
    2. Strong F1 score
    3. Good ROC-AUC
    """
    best_model_name = None
    best_score = -1
    
    for name, metrics in results.items():
        # Custom scoring formula: prioritize recall, then f1, then roc_auc
        score = (metrics["recall"] * 0.5) + (metrics["f1_score"] * 0.3) + (metrics["roc_auc"] * 0.2)
        if score > best_score:
            best_score = score
            best_model_name = name
            
    return best_model_name
