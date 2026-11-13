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