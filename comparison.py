"""TabPFN and baseline classifier evaluation."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from sap_ai_core import predict


def _columns(frame: pd.DataFrame) -> dict[str, list]:
    clean = frame.astype(object).where(pd.notna(frame), None)
    return {str(column): clean[column].tolist() for column in clean.columns}


def _tabpfn_payload(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, target: str) -> dict:
    categorical = [
        index for index, column in enumerate(X_train.columns)
        if pd.api.types.is_object_dtype(X_train[column]) or pd.api.types.is_bool_dtype(X_train[column])
    ]
    return {
        "task_config": {
            "task": "classification",
            "tabpfn_config": {
                "n_estimators": 8,
                "softmax_temperature": 0.9,
                "average_before_softmax": False,
                "balance_probabilities": True,
                "categorical_features_indices": categorical,
                "random_state": 0,
                "inference_precision": "auto",
                "fit_mode": "fit_preprocessors",
                "memory_saving_mode": False,
            },
            "predict_params": {"output_type": "preds"},
        },
        "x_train": _columns(X_train),
        "y_train": {target: y_train.tolist()},
        "x_test": _columns(X_test),
    }


def _baseline_scores(X_train, y_train, X_test, y_test, tabpfn_predictions: list) -> dict[str, float]:
    numeric = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [column for column in X_train.columns if column not in numeric]
    transforms = []
    if numeric:
        transforms.append(("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric))
    if categorical:
        transforms.append(("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))]), categorical))
    preprocessor = ColumnTransformer(transforms)
    models = {
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
        "svm": SVC(kernel="rbf", random_state=42),
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=42),
    }
    scores = {"tabpfn": float(accuracy_score(y_test, tabpfn_predictions))}
    for name, model in models.items():
        pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
        pipeline.fit(X_train, y_train)
        scores[name] = float(accuracy_score(y_test, pipeline.predict(X_test)))
    return scores


def compare(data: pd.DataFrame, target_column: str, test_size: float = 0.2) -> dict:
    """Evaluate TabPFN and baselines using one reproducible holdout split."""
    if target_column not in data.columns:
        raise ValueError(f"Target column not found: {target_column}")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be greater than 0 and less than 1.")
    X, y = data.drop(columns=[target_column]), data[target_column]
    if X.empty or y.nunique(dropna=False) < 2:
        raise ValueError("Data needs feature columns and a target with at least two classes.")
    stratify = y if y.value_counts(dropna=False).min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=stratify)
    response = predict(_tabpfn_payload(X_train, y_train, X_test, target_column))
    predictions = response.get("prediction", response.get("predictions"))
    if not isinstance(predictions, list) or len(predictions) != len(y_test):
        raise RuntimeError("TabPFN returned an unexpected prediction response.")
    scores = _baseline_scores(X_train, y_train, X_test, y_test.reset_index(drop=True), predictions)
    return {"predictions": predictions, "accuracy": scores["tabpfn"], "comparison": scores}
