"""
Heart Disease Risk Prediction
Model Training Module
"""

from sklearn.base import clone
from sklearn.pipeline import Pipeline

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


def build_model_pipelines(preprocessor, random_state=42):
    """
    Create preprocessing and classification pipelines
    for all candidate models.
    """

    classifiers = {
        "Dummy Classifier": DummyClassifier(
            strategy="most_frequent"
        ),

        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            random_state=random_state
        ),

        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=5
        ),

        "Decision Tree": DecisionTreeClassifier(
            random_state=random_state
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=random_state
        ),

        "Support Vector Machine": SVC(
            probability=True,
            random_state=random_state
        )
    }

    pipelines = {}
    for name, classifier in classifiers.items():
        pipelines[name] = Pipeline(
            steps=[
                ("preprocessing", clone(preprocessor)),
                ("classifier", classifier)
            ]
        )
    return pipelines


def train_models(X_train, y_train, preprocessor, random_state=42):
    """
    Train all candidate models using the training dataset.
    """

    pipelines = build_model_pipelines(
        preprocessor,
        random_state=random_state
    )

    trained_models = {}
    for name, pipeline in pipelines.items():
        pipeline.fit(X_train, y_train)
        trained_models[name] = pipeline
    return trained_models