"""Titanic data loading and TabPFN inference used by the A2A agent.

The dataset and TabPFN checkpoint are downloaded on first prediction and then
cached locally.  Set ``TITANIC_DATA_PATH`` to use a local CSV instead.
"""

from __future__ import annotations

import os
from pathlib import Path
from threading import RLock

import pandas as pd
import requests
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]
TARGET = "Survived"
DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
PROJECT_DIR = Path(__file__).parent
_bundled_dataset = PROJECT_DIR / "Titanic-Dataset.csv"
DATA_PATH = Path(
    os.getenv(
        "TITANIC_DATA_PATH",
        _bundled_dataset if _bundled_dataset.exists() else PROJECT_DIR / "data" / "titanic.csv",
    )
)

_lock = RLock()
_dataset: pd.DataFrame | None = None
_model = None


def _load_dataset() -> pd.DataFrame:
    """Load a cached Titanic CSV, downloading it once if necessary."""
    global _dataset
    if _dataset is not None:
        return _dataset

    with _lock:
        if _dataset is not None:
            return _dataset
        if not DATA_PATH.exists():
            try:
                response = requests.get(DATA_URL, timeout=30)
                response.raise_for_status()
            except requests.RequestException as exc:
                raise RuntimeError(
                    "Could not download the Titanic dataset. Check internet access or set "
                    f"TITANIC_DATA_PATH to a local CSV with {TARGET} and {', '.join(FEATURES)} columns."
                ) from exc
            DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
            DATA_PATH.write_bytes(response.content)

        raw = pd.read_csv(DATA_PATH)
        columns = {column.lower(): column for column in raw.columns}
        required = [TARGET, *FEATURES]
        missing = [name for name in required if name.lower() not in columns]
        if missing:
            raise RuntimeError(f"Titanic CSV is missing required columns: {', '.join(missing)}")

        data = raw[[columns[name.lower()] for name in required]].copy()
        data.columns = required
        data["Sex"] = data["Sex"].astype(str).str.lower().map({"female": 1, "male": 0})
        for column in ["Pclass", "Age", "SibSp", "Parch", "Fare", TARGET]:
            data[column] = pd.to_numeric(data[column], errors="coerce")
        data = data.dropna(subset=[TARGET, "Pclass", "Sex"])
        data[TARGET] = data[TARGET].astype(int)
        _dataset = data
        return _dataset


def _new_classifier():
    try:
        from tabpfn import TabPFNClassifier
    except ImportError as exc:
        raise RuntimeError("TabPFN is not installed. Run: pip install -r requirements.txt") from exc
    return TabPFNClassifier()


def _training_data() -> tuple[pd.DataFrame, pd.Series]:
    data = _load_dataset()
    X = data[FEATURES].copy()
    # TabPFN accepts missing values, but deterministic median imputation also
    # keeps this agent compatible with older TabPFN releases.
    for column in ["Age", "Fare"]:
        X[column] = X[column].fillna(X[column].median())
    X[["SibSp", "Parch"]] = X[["SibSp", "Parch"]].fillna(0)
    return X, data[TARGET]


def _input_frame(passengers: list[dict]) -> pd.DataFrame:
    if not passengers:
        raise ValueError("Provide at least one passenger.")

    rows = []
    training_X, _ = _training_data()
    defaults = training_X.median(numeric_only=True).to_dict()
    for index, passenger in enumerate(passengers, start=1):
        normalized = {str(key).lower(): value for key, value in passenger.items()}
        pclass = normalized.get("pclass")
        sex = normalized.get("sex")
        if pclass is None or sex is None:
            raise ValueError(f"Passenger {index} needs both Pclass (1-3) and Sex (male/female).")
        try:
            pclass = int(pclass)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Passenger {index} has an invalid Pclass.") from exc
        if pclass not in (1, 2, 3):
            raise ValueError(f"Passenger {index} Pclass must be 1, 2, or 3.")
        sex = str(sex).lower()
        if sex not in ("female", "male"):
            raise ValueError(f"Passenger {index} Sex must be male or female.")

        row = {"Pclass": pclass, "Sex": 1 if sex == "female" else 0}
        for column in ("Age", "SibSp", "Parch", "Fare"):
            value = normalized.get(column.lower(), defaults[column])
            try:
                row[column] = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Passenger {index} has an invalid {column}.") from exc
        rows.append(row)
    return pd.DataFrame(rows, columns=FEATURES)


def _get_model():
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                X, y = _training_data()
                classifier = _new_classifier()
                # First fit downloads the official TabPFN checkpoint if needed.
                classifier.fit(X, y)
                _model = classifier
    return _model


def predict_passengers(passengers: list[dict]) -> list[dict]:
    """Return TabPFN survival predictions and probabilities for passengers."""
    X = _input_frame(passengers)
    model = _get_model()
    probabilities = model.predict_proba(X)
    predictions = model.predict(X)
    positive_index = list(model.classes_).index(1)

    results = []
    for passenger, prediction, probability in zip(passengers, predictions, probabilities, strict=True):
        clean = _input_frame([passenger]).iloc[0]
        results.append(
            {
                "passenger": {
                    "Pclass": int(clean["Pclass"]),
                    "Sex": "female" if clean["Sex"] == 1 else "male",
                    "Age": float(clean["Age"]),
                    "SibSp": int(clean["SibSp"]),
                    "Parch": int(clean["Parch"]),
                    "Fare": float(clean["Fare"]),
                },
                "prediction": "Survived" if int(prediction) == 1 else "Did not survive",
                "survival_probability": round(float(probability[positive_index]), 4),
            }
        )
    return results


def evaluate_holdout() -> dict:
    """Run a reproducible hold-out evaluation without changing the live model."""
    X, y = _training_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = _new_classifier()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return {
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "predictions": [int(value) for value in predictions],
        "actual": [int(value) for value in y_test],
    }
