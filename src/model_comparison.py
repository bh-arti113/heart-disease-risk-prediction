"""
Heart Disease Risk Prediction
Model Comparison Using Cross-Validation
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from .train import build_model_pipelines
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score
)


def calculate_specificity(y_true, y_pred):
    """Calculate the specificity score."""
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()
    return tn / (tn + fp) if (tn + fp) else 0.0


def compare_models_cv(
    X,
    y,
    preprocessor,
    random_state=42,
    n_split=5
):
    """
    Compare candidate models using stratified
    cross-validation on training data only.
    """

    cv = StratifiedKFold(
        n_splits = 5,
        shuffle=True,
        random_state=random_state
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(
            recall_score,
            zero_division=0
        ),
        "f1": make_scorer(
            f1_score,
            zero_division=0
            ),
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
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
            X,
            y,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
            error_score="raise",
            n_jobs=-1
        )
        result = {"Model": model_name}
        for metric in scoring:
            metric_scores = scores[f"test_{metric}"]
            result[f"{metric.upper()} Mean"] = np.mean(
                metric_scores
            )
            result[f"{metric.upper()} Std"] = np.std(
                metric_scores,
                ddof=1
            )
        results.append(result)
    
    return (
        pd.DataFrame(results)
        .sort_values("ROC_AUC Mean", ascending=False)
        .reset_index(drop=True)
    )

