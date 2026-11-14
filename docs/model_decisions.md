# Model Decisions and Experimentation Plan

# 1. Objective

The objective of this phase is to develop, compare, and evaluate multiple machine learning classification models for predicting the presence of heart disease.
The goal is to identify a model that provides a suitable balance between predictive performance, reliability, interpretability, and generalization.
This project is intended for educational purposes and is not a clinical diagnostic system.

# 2. Candidate Models

We plan to experiment with the following classification algorithms:
| Model | Purpose |
|---|---|
| Dummy Classifier | Establish a baseline performance |
| Logistic Regression | Provide an interpretable linear model |
| K-Nearest Neighbors (KNN) | Explore distance-based classification |
| Decision Tree | Explore rule-based classification |
| Random Forest | Evaluate ensemble learning |
| Support Vector Machine (SVM) | Explore margin-based classification |

# 3. Evaluation Metrics

Models will be evaluated using the following metrics:
- **Accuracy:** Overall proportion of correct predictions.
- **Precision:** Proportion of predicted positive cases that are actually positive.
- **Recall (Sensitivity):** Proportion of actual positive cases correctly identified.
- **Specificity:** Proportion of actual negative cases correctly identified.
- **F1-score:** Harmonic mean of precision and recall.
- **ROC-AUC:** Ability to distinguish between positive and negative classes across classification thresholds.
- **PR-AUC:** Performance across precision and recall trade-offs.
Confusion matrices and classification reports will also be used to understand model performance.
Accuracy alone will not determine the final model selection.

# 4. Experimental Strategy

The planned workflow is:
1. Obtain the validated and processed dataset from the data-processing phase.
2. Verify the dataset's feature columns and target variable.
3. Separate input features from the target.
4. Create training and testing sets using a stratified split where appropriate.
5. Build preprocessing and model pipelines to prevent data leakage.
6. Establish a baseline using Dummy Classifier.
7. Train and evaluate all candidate models.
8. Compare their performance using consistent evaluation metrics.
9. Perform hyperparameter tuning using cross-validation.
10. Evaluate the selected model on the held-out test set.
All preprocessing steps that learn from data must be fitted only on the training data or within the relevant cross-validation folds.

# 5. Model Selection Criteria

The final model will be selected based on:
- Generalization performance on unseen data.
- Recall and specificity balance.
- ROC-AUC and PR-AUC where appropriate.
- Consistency across cross-validation folds.
- Interpretability and computational requirements.
The selected model will not necessarily be the model with the highest accuracy.

# 6. Explainability

The selected model will be examined using appropriate explainability techniques.
Planned techniques include:
- Logistic Regression coefficients.
- Permutation feature importance.
- SHAP explanations, where supported.
These techniques will help us understand how input features influence model predictions.

# 7. Model Saving and Integration

After model selection and final evaluation:
- Save the trained preprocessing and classification pipeline.
- Store relevant model metadata and evaluation results.
- Make the saved pipeline available for integration with the Streamlit application.

# 8. Current Status

**Status:** Planning
The actual experiments will begin after the processed dataset, feature definitions, and target encoding have been confirmed by the data-processing phase.
No model has been selected or trained at this stage.

---

*This document will be updated as experiments are conducted and model-selection decisions are finalized.*


# Model Selection Decisions

## Hyperparameter Tuning Results

Four machine learning models were tuned using 5-fold Stratified Cross-Validation with ROC-AUC as the optimization metric.

The tuned models were evaluated on the held-out test dataset containing 61 records.

## Final Test Performance

| Model | Accuracy | Precision | Recall | Specificity | F1 Score | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8852 | 0.8387 | 0.9286 | 0.8485 | 0.8814 | 0.9654 | 0.9622 |
| Random Forest | 0.8689 | 0.8125 | 0.9286 | 0.8182 | 0.8667 | 0.9491 | 0.9246 |
| Support Vector Machine | 0.8361 | 0.8000 | 0.8571 | 0.8182 | 0.8276 | 0.9426 | 0.9382 |
| K-Nearest Neighbors | 0.8852 | 0.8621 | 0.8929 | 0.8788 | 0.8772 | 0.9621 | 0.9518 |

## Preliminary Model Selection

Logistic Regression is selected as the leading candidate based on its highest test ROC-AUC (0.9654), highest F1 score (0.8814), and high recall (0.9286).

K-Nearest Neighbors is retained as a strong alternative because it achieved the highest precision (0.8621) and specificity (0.8788), while matching Logistic Regression's accuracy.

Random Forest and Support Vector Machine are retained for comparative analysis.

## Important Considerations

- The test dataset contains only 61 records, so performance estimates have uncertainty.
- Test-set results are used for final evaluation, not further hyperparameter tuning.
- The selected model is a candidate for further explainability and integration, not a clinically validated diagnostic system.
- Final conclusions should acknowledge the limitations of the dataset and evaluation sample size.