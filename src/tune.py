
"""
Heart Disease Risk Prediction
Hyperparameter Tuning Module
"""

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold


def tune_models(
    models,
    parameter_grids,
    X_train,
    y_train,
    random_state=42,
    n_iter=30
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
        search = RandomizedSearchCV(
            estimator=model_pipeline,
            param_distributions=parameter_grids[model_name],
            n_iter=n_iter,
            scoring="roc_auc",
            cv=cv,
            random_state=random_state,
            n_jobs=-1,
            refit=True,
            error_score="raise"
        )
        search.fit(X_train, y_train)
        best_models[model_name] = search.best_estimator_
        tuning_results[model_name] = {
            "Best ROC-AUC": search.best_score_,
            "Best Parameters": search.best_params_,
            "Best Estimator": search.best_estimator_,
        }
        print(f"Best ROC-AUC: {search.best_score_:.4f}")
        print(f"Best Parameters: {search.best_params_}")
    return best_models, tuning_results