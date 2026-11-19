
"""
Heart Disease Risk Prediction
Hyperparameter Tuning Module
"""
from math import prod

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from .train import build_model_pipelines


# ------------------------------------------------------------
# 1. Hyperparameter search spaces
# ------------------------------------------------------------

def get_parameter_grids():

    return {
        "Logistic Regression": {
            "classifier__C": [0.01, 0.1, 1, 10, 100],
            "classifier__solver": ["lbfgs"],
            "classifier__class_weight": [None, "balanced"]
        },

        "Random Forest": {
            "classifier__n_estimators": [200, 300, 500, 700],
            "classifier__max_depth": [None, 5, 10, 15, 20],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__min_samples_leaf": [1, 2, 4],
            "classifier__max_features": ["sqrt", "log2"],
            "classifier__class_weight": [None, "balanced"]
        },

        "Support Vector Machine": {
            "classifier__C": [0.1, 1, 10, 100],
            "classifier__kernel": ["linear", "rbf"],
            "classifier__gamma": ["scale", "auto"],
            "classifier__class_weight": [None, "balanced"]
        },

        "K-Nearest Neighbors": {
            "classifier__n_neighbors": [3, 5, 7, 9, 11, 13, 15],
            "classifier__weights": ["uniform", "distance"],
            "classifier__p": [1, 2]
        }
    }


# ------------------------------------------------------------
# 2. Reusable randomized search
# ------------------------------------------------------------

def _run_search(
    estimator,
    parameter_grid,
    X,
    y,
    cv,
    random_state=42,
    n_iter=20
):
    """
    Run randomized search without requesting more
    iterations than the available parameter combinations.
    """

    total_combinations = prod(
        len(values) for values in parameter_grid.values()
    )

    actual_iterations = min(n_iter, total_combinations)

    search = RandomizedSearchCV(
        estimator=clone(estimator),
        param_distributions=parameter_grid,
        n_iter=actual_iterations,
        scoring="roc_auc",
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        refit=True,
        error_score="raise"
    )

    search.fit(X, y)

    return search


# ------------------------------------------------------------
# 3. Nested cross-validation
# ------------------------------------------------------------

def nested_cv_model_selection(
    X,
    y,
    preprocessor,
    random_state=42,
    outer_splits=5,
    inner_splits=4,
    n_iter=20
):
    """
    Estimate the performance of the complete model-selection
    procedure using nested stratified cross-validation.

    Inner CV:
        Selects hyperparameters and the candidate model.

    Outer CV:
        Evaluates the selected model on unseen fold data.
    """

    parameter_grids = get_parameter_grids()

    outer_cv = StratifiedKFold(
        n_splits=outer_splits,
        shuffle=True,
        random_state=random_state
    )

    candidate_models = build_model_pipelines(
        preprocessor,
        random_state=random_state
    )

    candidate_names = list(parameter_grids.keys())

    fold_results = []
    oof_records = []

    for fold_number, (train_idx, valid_idx) in enumerate(
        outer_cv.split(X, y),
        start=1
    ):

        print(f"\nOUTER FOLD {fold_number}/{outer_splits}")
        print("-" * 55)

        X_outer_train = X.iloc[train_idx]
        X_outer_valid = X.iloc[valid_idx]

        y_outer_train = y.iloc[train_idx]
        y_outer_valid = y.iloc[valid_idx]

        inner_cv = StratifiedKFold(
            n_splits=inner_splits,
            shuffle=True,
            random_state=random_state + fold_number
        )

        fold_searches = {}

        # Tune each candidate using only outer training data.
        for model_name in candidate_names:

            search = _run_search(
                estimator=candidate_models[model_name],
                parameter_grid=parameter_grids[model_name],
                X=X_outer_train,
                y=y_outer_train,
                cv=inner_cv,
                random_state=random_state + fold_number,
                n_iter=n_iter
            )

            fold_searches[model_name] = search

            print(
                f"{model_name}: "
                f"Inner ROC-AUC = {search.best_score_:.4f}"
            )

        # Select winner using INNER CV only.
        selected_name = max(
            fold_searches,
            key=lambda name: fold_searches[name].best_score_
        )

        selected_search = fold_searches[selected_name]
        selected_model = selected_search.best_estimator_

        # Evaluate the selected model on the untouched outer fold.
        y_pred = selected_model.predict(X_outer_valid)
        y_score = selected_model.predict_proba(
            X_outer_valid
        )[:, 1]

        tn, fp, fn, tp = confusion_matrix(
            y_outer_valid,
            y_pred,
            labels=[0, 1]
        ).ravel()

        specificity = (
            tn / (tn + fp) if (tn + fp) else 0.0
        )

        metrics = {
            "Fold": fold_number,
            "Selected Model": selected_name,
            "Inner CV ROC-AUC": selected_search.best_score_,
            "Accuracy": accuracy_score(y_outer_valid, y_pred),
            "Precision": precision_score(
                y_outer_valid, y_pred, zero_division=0
            ),
            "Recall": recall_score(
                y_outer_valid, y_pred, zero_division=0
            ),
            "Specificity": specificity,
            "F1 Score": f1_score(
                y_outer_valid, y_pred, zero_division=0
            ),
            "ROC-AUC": roc_auc_score(
                y_outer_valid, y_score
            ),
            "Average Precision": average_precision_score(
                y_outer_valid, y_score
            )
        }

        fold_results.append(metrics)

        print(f"\nSelected model: {selected_name}")
        print(f"Outer ROC-AUC: {metrics['ROC-AUC']:.4f}")

        for position, (index, actual, predicted, probability) in enumerate(
            zip(
                valid_idx,
                y_outer_valid.to_numpy(),
                y_pred,
                y_score
            )
        ):

            oof_records.append({
                "Index": index,
                "Fold": fold_number,
                "Actual": actual,
                "Predicted": predicted,
                "Probability": probability,
                "Selected Model": selected_name
            })

    fold_results_df = pd.DataFrame(fold_results)

    oof_predictions = (
        pd.DataFrame(oof_records)
        .sort_values("Index")
        .reset_index(drop=True)
    )

    return fold_results_df, oof_predictions



# ------------------------------------------------------------
# 4. Final hyperparameter tuning
# ------------------------------------------------------------

def tune_models(
    models,
    parameter_grids,
    X_train,
    y_train,
    random_state=42,
    n_iter=30,
    n_splits=5
):
    """
    Tune selected model pipelines using randomized
    search and stratified cross-validation.
    """

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state
    )

    tuning_results = {}
    best_models = {}

    for model_name, model_pipeline in models.items():

        print(f"\nTuning: {model_name}")
        print("-" * 45)
        search = _run_search(
            estimator=model_pipeline,
            parameter_grid=parameter_grids[model_name],
            X=X_train,
            y=y_train,
            cv=cv,
            random_state=random_state,
            n_iter=n_iter
        )

        best_models[model_name] = search.best_estimator_

        tuning_results[model_name] = {
            "Best ROC-AUC": search.best_score_,
            "Best Parameters": search.best_params_,
            "Best Estimator": search.best_estimator_
        }

        print(f"Best ROC-AUC: {search.best_score_:.4f}")
        print(f"Best Parameters: {search.best_params_}")

    return best_models, tuning_results
