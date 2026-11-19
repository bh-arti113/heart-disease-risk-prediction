
"""
Heart Disease Risk Prediction
Model Explainability Module
"""

import numpy as np
import pandas as pd

from sklearn.inspection import permutation_importance


# ------------------------------------------------------------
# 1. Global permutation feature importance
# ------------------------------------------------------------

def calculate_permutation_importance(
    model,
    X,
    y,
    n_repeats=30,
    random_state=42,
    scoring="roc_auc"
):
    """
    Calculate permutation importance on original input features.

    The complete fitted pipeline is used, so the importance
    is calculated at the original feature level.
    
    Higher importance means that shuffling the feature
    reduces the selected evaluation score more substantially.
    """

    if not hasattr(model, "predict_proba"):
        raise ValueError(
            "Model must support predict_proba."
        )

    result = permutation_importance(
        estimator=model,
        X=X,
        y=y,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1
    )

    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance Mean": result.importances_mean,
        "Importance Std": result.importances_std
    })

    importance_df = importance_df.sort_values(
        "Importance Mean",
        ascending=False
    ).reset_index(drop=True)

    return importance_df


# ------------------------------------------------------------
# 2. Get transformed feature names
# ------------------------------------------------------------

def get_transformed_feature_names(model):
    """
    Retrieve feature names after preprocessing.

    This is useful for interpreting model-specific
    feature importance and SHAP explanations.
    """

    if not hasattr(model, "named_steps"):
        raise ValueError("Expected a fitted sklearn Pipeline.")

    preprocessor = model.named_steps["preprocessing"]

    return preprocessor.get_feature_names_out()


# ------------------------------------------------------------
# 3. Transform input for model-specific explanations
# ------------------------------------------------------------

def transform_features(model, X):
    """
    Apply the fitted preprocessing stage to input features.

    Does not refit or modify the fitted preprocessor.
    """

    if not hasattr(model, "named_steps"):
        raise ValueError("Expected a fitted sklearn Pipeline.")

    preprocessor = model.named_steps["preprocessing"]

    transformed = preprocessor.transform(X)

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    feature_names = get_transformed_feature_names(model)

    return pd.DataFrame(
        transformed,
        columns=feature_names,
        index=X.index
    )
