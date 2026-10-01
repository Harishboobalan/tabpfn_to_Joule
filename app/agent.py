"""Business logic for the HANA-to-TabPFN A2A agent."""

import json
import os
import re
import time
from pathlib import Path
from threading import Lock

import pandas as pd
import requests
from dotenv import load_dotenv
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

load_dotenv(Path(__file__).with_name(".env"))
load_dotenv(Path(__file__).parent.parent / ".env")

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")
_lock = Lock()
_token: str | None = None
_expires_at = 0.0
_deployment_url: str | None = None


def _required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required setting: {name}")
    return value


def _boolean(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes"}


def _quoted_identifier(value: str) -> str:
    if not _IDENTIFIER.fullmatch(value):
        raise ValueError("Schema and table names may contain only letters, digits, _, and $.")
    return f'"{value}"'


def _load_table(table_name: str) -> pd.DataFrame:
    try:
        from hdbcli import dbapi
    except ImportError as exc:
        raise RuntimeError("Install the SAP HANA Python driver: pip install hdbcli") from exc
    connection = dbapi.connect(
        address=_required("HANA_HOST"), port=int(_required("HANA_PORT")),
        user=_required("HANA_USER"), password=_required("HANA_PASSWORD"),
        encrypt=_boolean("HANA_ENCRYPT", True),
        sslValidateCertificate=_boolean("HANA_SSL_VALIDATE_CERTIFICATE", True),
    )
    cursor = connection.cursor()
    try:
        query = f"SELECT * FROM {_quoted_identifier(_required('HANA_SCHEMA'))}.{_quoted_identifier(table_name)}"
        cursor.execute(query)
        return pd.DataFrame(cursor.fetchall(), columns=[column[0] for column in cursor.description])
    finally:
        cursor.close()
        connection.close()


def _access_token() -> str:
    global _token, _expires_at
    if _token and time.time() < _expires_at:
        return _token
    with _lock:
        if _token and time.time() < _expires_at:
            return _token
        auth_url = _required("AICORE_AUTH_URL").rstrip("/")
        token_url = auth_url if auth_url.endswith("/oauth/token") else f"{auth_url}/oauth/token"
        response = requests.post(token_url, data={"grant_type": "client_credentials"}, auth=(_required("AICORE_CLIENT_ID"), _required("AICORE_CLIENT_SECRET")), timeout=30)
        response.raise_for_status()
        body = response.json()
        _token = body["access_token"]
        _expires_at = time.time() + max(int(body.get("expires_in", 300)) - 60, 30)
        return _token


def _deployment_url_for_tabpfn() -> str:
    global _deployment_url
    if _deployment_url:
        return _deployment_url
    api_url = _required("AICORE_API_URL").rstrip("/")
    api_url = api_url if api_url.endswith("/v2") else f"{api_url}/v2"
    response = requests.get(
        f"{api_url}/lm/deployments/{_required('TABPFN_DEPLOYMENT_ID')}",
        headers={"Authorization": f"Bearer {_access_token()}", "AI-Resource-Group": os.getenv("AICORE_RESOURCE_GROUP", "default")}, timeout=30,
    )
    response.raise_for_status()
    _deployment_url = response.json()["deploymentUrl"].rstrip("/")
    return _deployment_url


def _predict(payload: dict) -> dict:
    response = requests.post(
        f"{_deployment_url_for_tabpfn()}/predict",
        headers={"Authorization": f"Bearer {_access_token()}", "AI-Resource-Group": os.getenv("AICORE_RESOURCE_GROUP", "default"), "Content-Type": "application/json"},
        json=payload, timeout=180,
    )
    response.raise_for_status()
    return response.json()


def _columns(frame: pd.DataFrame) -> dict[str, list]:
    clean = frame.astype(object).where(pd.notna(frame), None)
    return {str(column): clean[column].tolist() for column in clean.columns}


def _tabpfn_payload(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, target: str) -> dict:
    categorical = [index for index, column in enumerate(X_train.columns) if pd.api.types.is_object_dtype(X_train[column]) or pd.api.types.is_bool_dtype(X_train[column])]
    return {"task_config": {"task": "classification", "tabpfn_config": {"n_estimators": 8, "softmax_temperature": 0.9, "average_before_softmax": False, "balance_probabilities": True, "categorical_features_indices": categorical, "random_state": 0, "inference_precision": "auto", "fit_mode": "fit_preprocessors", "memory_saving_mode": False}, "predict_params": {"output_type": "preds"}}, "x_train": _columns(X_train), "y_train": {target: y_train.tolist()}, "x_test": _columns(X_test)}


def _baseline_scores(X_train, y_train, X_test, y_test, tabpfn_predictions: list) -> dict[str, float]:
    numeric = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [column for column in X_train.columns if column not in numeric]
    transforms = []
    if numeric:
        transforms.append(("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric))
    if categorical:
        transforms.append(("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))]), categorical))
    preprocessor = ColumnTransformer(transforms)
    models = {"random_forest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1), "svm": SVC(kernel="rbf", random_state=42), "logistic_regression": LogisticRegression(max_iter=2000, random_state=42)}
    scores = {"tabpfn": float(accuracy_score(y_test, tabpfn_predictions))}
    for name, model in models.items():
        pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
        pipeline.fit(X_train, y_train)
        scores[name] = float(accuracy_score(y_test, pipeline.predict(X_test)))
    return scores


class TabPFNComparisonAgent:
    """Runs a HANA table comparison requested through an A2A text message."""

    def invoke(self, user_input: str) -> dict:
        """Accept JSON: target_column (required), table_name and test_size (optional)."""
        try:
            request = json.loads(user_input)
        except json.JSONDecodeError as exc:
            raise ValueError("Send a JSON object with target_column, optional table_name, and optional test_size.") from exc
        if not isinstance(request, dict):
            raise ValueError("The A2A message must contain a JSON object.")
        target_column = request.get("target_column")
        if not isinstance(target_column, str) or not target_column.strip():
            raise ValueError("target_column is required.")
        table_name = request.get("table_name") or _required("HANA_TABLE")
        if not isinstance(table_name, str):
            raise ValueError("table_name must be a string.")
        test_size = request.get("test_size", 0.2)
        if not isinstance(test_size, (int, float)) or isinstance(test_size, bool) or not 0 < test_size < 1:
            raise ValueError("test_size must be a number greater than 0 and less than 1.")
        data = _load_table(table_name)
        if target_column not in data.columns:
            raise ValueError(f"Target column not found: {target_column}")
        X, y = data.drop(columns=[target_column]), data[target_column]
        if X.empty or y.nunique(dropna=False) < 2:
            raise ValueError("Data needs feature columns and a target with at least two classes.")
        stratify = y if y.value_counts(dropna=False).min() >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(test_size), random_state=42, stratify=stratify)
        response = _predict(_tabpfn_payload(X_train, y_train, X_test, target_column))
        predictions = response.get("prediction", response.get("predictions"))
        if not isinstance(predictions, list) or len(predictions) != len(y_test):
            raise RuntimeError("TabPFN returned an unexpected prediction response.")
        scores = _baseline_scores(X_train, y_train, X_test, y_test.reset_index(drop=True), predictions)
        return {"table_name": table_name, "target_column": target_column, "test_size": test_size, "predictions": predictions, "accuracy": scores["tabpfn"], "comparison": scores}
