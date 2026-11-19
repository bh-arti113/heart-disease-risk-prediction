
"""
Heart Disease Risk Prediction
Data Validation and Dataset Preparation Module
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ------------------------------------------------------------
# Dataset configuration
# ------------------------------------------------------------

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal", "num"
]

TARGET_COLUMN = "target"

NUMERICAL_FEATURES = [
    "age", "trestbps", "chol", "thalach", "oldpeak"
]

CATEGORICAL_FEATURES = [
    "sex", "cp", "fbs", "restecg",
    "exang", "slope", "ca", "thal"
]

FEATURE_COLUMNS = [
    column for column in COLUMNS if column != "num"
]

EXPECTED_MODEL_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]

ALLOWED_CATEGORIES = {
    "sex": {0, 1},
    "cp": {1, 2, 3, 4},
    "fbs": {0, 1},
    "restecg": {0, 1, 2},
    "exang": {0, 1},
    "slope": {1, 2, 3},
    "ca": {0, 1, 2, 3},
    "thal": {3, 6, 7}
}


# ------------------------------------------------------------
# 1. Load raw dataset
# ------------------------------------------------------------

def load_raw_dataset(file_path):
    """
    Load the Cleveland dataset and assign column names.
    The '?' marker is interpreted as a missing value.
    """

    df = pd.read_csv(
        file_path,
        header=None,
        na_values="?"
    )

    if df.shape[1] != len(COLUMNS):
        raise ValueError(
            f"Expected {len(COLUMNS)} columns, "
            f"but found {df.shape[1]}."
        )

    df.columns = COLUMNS

    return df


# ------------------------------------------------------------
# 2. Validate raw dataset
# ------------------------------------------------------------

def validate_raw_dataset(df):
    """
    Validate the structure and values of the raw dataset.

    Missing feature values are allowed.
    Missing diagnosis values are not allowed.
    """

    errors = []

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Raw dataset is empty.")

    if df.columns.tolist() != COLUMNS:
        errors.append("Unexpected column names or column order.")

    if df["num"].isna().any():
        errors.append("Diagnosis column 'num' contains missing values.")

    if not df["num"].dropna().isin([0, 1, 2, 3, 4]).all():
        errors.append("Unexpected values found in diagnosis column.")

    for col in FEATURE_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(f"Feature '{col}' must be numeric.")

        elif np.isinf(df[col].dropna()).any():
            errors.append(f"Feature '{col}' contains infinite values.")

    for col, allowed_values in ALLOWED_CATEGORIES.items():
        observed = set(df[col].dropna().unique())
        invalid = observed - allowed_values

        if invalid:
            errors.append(
                f"Invalid categories in '{col}': {invalid}"
            )

    if errors:
        raise ValueError(
            "Raw dataset validation failed:\n- "
            + "\n- ".join(errors)
        )

    return {
        "records": len(df),
        "columns": len(df.columns),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_values": int(df.isna().sum().sum()),
        "diagnosis_classes": sorted(df["num"].unique().tolist())
    }


# ------------------------------------------------------------
# 3. Prepare modelling dataset
# ------------------------------------------------------------

def prepare_model_dataset(raw_df):
    """
    Convert the original diagnosis into a binary target.

    Missing feature values are intentionally retained.
    The original 'num' column is removed to prevent leakage.
    """

    validate_raw_dataset(raw_df)

    df = raw_df.copy()

    df[TARGET_COLUMN] = (df["num"] > 0).astype(int)

    df = df.drop(columns=["num"])

    validate_model_dataset(df)

    return df


# ------------------------------------------------------------
# 4. Validate modelling dataset
# ------------------------------------------------------------

def validate_model_dataset(df):
    """
    Validate the final modelling input.

    Missing feature values are allowed.
    Missing target values are not allowed.
    """

    errors = []

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Modelling dataset is empty.")

    if df.columns.tolist() != EXPECTED_MODEL_COLUMNS:
        errors.append("Unexpected modelling dataset columns or order.")

    if TARGET_COLUMN not in df.columns:
        errors.append("Target column is missing.")
    else:
        if df[TARGET_COLUMN].isna().any():
            errors.append("Target contains missing values.")

        elif not df[TARGET_COLUMN].isin([0, 1]).all():
            errors.append("Target must contain only 0 and 1.")

    for col in FEATURE_COLUMNS:
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                errors.append(f"Feature '{col}' must be numeric.")

            elif np.isinf(df[col].dropna()).any():
                errors.append(
                    f"Feature '{col}' contains infinite values."
                )

    for col, allowed_values in ALLOWED_CATEGORIES.items():
        if col in df.columns:
            observed = set(df[col].dropna().unique())
            invalid = observed - allowed_values

            if invalid:
                errors.append(
                    f"Invalid categories in '{col}': {invalid}"
                )

    if errors:
        raise ValueError(
            "Model dataset validation failed:\n- "
            + "\n- ".join(errors)
        )

    return {
        "records": len(df),
        "features": len(FEATURE_COLUMNS),
        "missing_feature_cells": int(
            df[FEATURE_COLUMNS].isna().sum().sum()
        ),
        "target_distribution": (
            df[TARGET_COLUMN].value_counts().to_dict()
        ),
        "duplicate_rows": int(df.duplicated().sum())
    }
