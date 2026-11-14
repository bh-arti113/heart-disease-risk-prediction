
"""
Heart Disease Risk Prediction
Model Comparison Using Cross-Validation
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, confusion_matrix
from .train import build_model_pipelines
from sklearn.metrics import make_scorer, confusion_matrix, precision_score


def calculate_specificity(y_true, y_pred):
    """Calculate the specificity score."""
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()
    return tn / (tn + fp) if (tn + fp) else 0.0


def compare_models_cv(
    X_train,
    y_train,
    preprocessor,
    random_state=42
):
    """
    Compare candidate models using stratified
    cross-validation on training data only.
    """

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state
    )
    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
        "specificity": make_scorer(calculate_specificity),
    }
    pipelines = build_model_pipelines(
        preprocessor,
        random_state=random_state
    )
    results = []

    for model_name, pipeline in pipelines.items():
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
            error_score="raise"
        )
        result = {"Model": model_name}
        for metric in scoring:
            metric_scores = scores[f"test_{metric}"]
            result[f"{metric.upper()} Mean"] = np.mean(
                metric_scores
            )
            result[f"{metric.upper()} Std"] = np.std(
                metric_scores
            )
        results.append(result)
    results_df = pd.DataFrame(results)
    return results_df