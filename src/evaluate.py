
"""
Heart Disease Risk Prediction
Model Evaluation Module
"""

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


def evaluate_classifier(y_true, y_pred, y_score=None):
    """Calculate binary classification evaluation metrics."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("Targets and predictions must be 1D arrays.")
    if len(y_true) == 0 or len(y_true) != len(y_pred):
        raise ValueError("Inputs must be non-empty and have equal lengths.")
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("Actual labels must contain only 0 and 1.")
    if not np.isin(y_pred, [0, 1]).all():
        raise ValueError("Predictions must contain only 0 and 1.")
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    results = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "Specificity": specificity,
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": np.nan,
        "PR-AUC": np.nan,
        "Confusion Matrix": np.array([[tn, fp], [fn, tp]]),
    }

    if y_score is not None:
        y_score = np.asarray(y_score)
        if y_score.ndim != 1 or len(y_score) != len(y_true):
            raise ValueError("Scores must be a 1D array matching y_true.")
        if not np.isfinite(y_score).all():
            raise ValueError("Scores must contain only finite values.")
        if not np.isin(y_true, [0, 1]).all():
            raise ValueError("Actual labels must contain only 0 and 1.")
        if len(np.unique(y_true)) == 2:
            results["ROC-AUC"] = roc_auc_score(
                y_true, y_score
            )

            results["Average Precision"] = average_precision_score(
                y_true, y_score
            )
    return results

def print_evaluation_report(results, model_name="Classification Model"):
    """Display evaluation results in a clean terminal report."""
    width = 54
    line = "=" * width
    print("\n" + line)
    print("             MODEL EVALUATION REPORT")
    print(line)
    print(f"  Model: {model_name}")
    print("-" * width)
    print("\n  PERFORMANCE METRICS\n")
    for metric, value in results.items():
        if metric == "Confusion Matrix":
            continue
        if value is None or not np.isfinite(value):
            display_value = "N/A"
        else:
            display_value = f"{value:.4f}"
        print(f"  {metric:<22} : {display_value}")
    if "Confusion Matrix" in results:
        matrix = np.asarray(results["Confusion Matrix"])
        print("\n  CONFUSION MATRIX")
        print("  " + "-" * 34)
        print("                  Predicted")
        print("                  No     Yes")
        print(f"  Actual No      {matrix[0, 0]:>4}   {matrix[0, 1]:>4}")
        print(f"  Actual Yes     {matrix[1, 0]:>4}   {matrix[1, 1]:>4}")
        print("\n  TN = True Negative   FP = False Positive")
        print("  FN = False Negative  TP = True Positive")
    print("\n" + line)
    print("             END OF EVALUATION")
    print(line + "\n")
