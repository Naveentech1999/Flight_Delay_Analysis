"""Train and evaluate a flight delay prediction model."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FEATURE_COLUMNS = [
    "airline",
    "origin",
    "destination",
    "scheduled_departure_hour",
    "day_of_week",
    "month",
    "distance",
]
TARGET_COLUMN = "delayed"
CATEGORICAL_COLUMNS = ["airline", "origin", "destination"]
NUMERIC_COLUMNS = [column for column in FEATURE_COLUMNS if column not in CATEGORICAL_COLUMNS]


def build_pipeline() -> Pipeline:
    """Create the preprocessing and classification pipeline."""
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
            ("numeric", numeric_pipeline, NUMERIC_COLUMNS),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")),
        ]
    )


def load_dataset(data_path: Path) -> pd.DataFrame:
    """Load a CSV dataset and validate the required columns."""
    dataset = pd.read_csv(data_path)
    missing_columns = [column for column in [*FEATURE_COLUMNS, TARGET_COLUMN] if column not in dataset.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return dataset


def train(data_path: Path, model_output: Path) -> None:
    """Train the model, print metrics, and save the fitted pipeline."""
    dataset = load_dataset(data_path)
    x = dataset[FEATURE_COLUMNS]
    y = dataset[TARGET_COLUMN]

    stratify = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=42,
        stratify=stratify,
    )

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)

    predictions = pipeline.predict(x_test)
    probabilities = pipeline.predict_proba(x_test)[:, 1]

    print("Model evaluation")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")
    print(f"Precision: {precision_score(y_test, predictions, zero_division=0):.2f}")
    print(f"Recall: {recall_score(y_test, predictions, zero_division=0):.2f}")
    print(f"F1 Score: {f1_score(y_test, predictions, zero_division=0):.2f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, probabilities):.2f}")

    model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_output)
    print(f"Saved model to {model_output}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a flight delay prediction model.")
    parser.add_argument("--data", type=Path, default=Path("data/sample_flights.csv"), help="Path to the input CSV dataset.")
    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("models/flight_delay_model.joblib"),
        help="Path where the trained model should be saved.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.data, args.model_output)
