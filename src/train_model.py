"""Train a scikit-learn multinomial logistic-regression baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TARGETS = ["H", "D", "A"]
EXCLUDED = {"season", "date", "home_team", "away_team", "home_goals", "away_goals", "result", "source"}


# creates the preprocessing and logistic regression pipeline
def build_pipeline(max_iter: int, regularization: float) -> Pipeline:
    """Create the complete preprocessing and classification pipeline."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=regularization,
                    max_iter=max_iter,
                    solver="lbfgs",
                ),
            ),
        ]
    )


# converts selected dataframe columns into a numeric feature matrix
def numeric_features(frame: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    """Coerce model inputs to numbers and leave missing values for the imputer."""
    return frame[feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)


# trains, evaluates, refits, and saves the model pipeline
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/processed/matches_with_features.csv"))
    parser.add_argument("--output", type=Path, default=Path("models/logistic_regression.joblib"))
    parser.add_argument("--test-season")
    parser.add_argument("--max-iter", type=int, default=2500)
    parser.add_argument(
        "--regularization",
        type=float,
        default=1.0,
        help="Inverse L2 regularization strength (the LogisticRegression C parameter)",
    )
    parser.add_argument(
        "--refit-all",
        action="store_true",
        help="Evaluate on the test season, then refit the saved model on every season",
    )
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    seasons = sorted(frame["season"].unique())
    test_season = args.test_season or seasons[-1]
    train = frame.loc[frame["season"] != test_season].copy()
    test = frame.loc[frame["season"] == test_season].copy()
    if train.empty or test.empty:
        raise ValueError("Training and test datasets must both contain matches")

    feature_columns = [column for column in frame.columns if column not in EXCLUDED]
    pipeline = build_pipeline(args.max_iter, args.regularization)
    pipeline.fit(numeric_features(train, feature_columns), train["result"])

    test_features = numeric_features(test, feature_columns)
    test_probabilities = pipeline.predict_proba(test_features)
    predictions = pipeline.predict(test_features)
    accuracy = float(accuracy_score(test["result"], predictions))
    loss = float(log_loss(test["result"], test_probabilities, labels=pipeline.classes_))

    saved_rows = len(train)
    if args.refit_all:
        pipeline = build_pipeline(args.max_iter, args.regularization)
        pipeline.fit(numeric_features(frame, feature_columns), frame["result"])
        saved_rows = len(frame)

    artifact = {
        "pipeline": pipeline,
        "feature_columns": feature_columns,
        "targets": pipeline.classes_.tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.output)

    metrics = {
        "model_type": "scikit-learn multinomial logistic regression",
        "train_rows": len(train),
        "test_rows": len(test),
        "test_season": test_season,
        "accuracy": accuracy,
        "log_loss": loss,
        "saved_model_rows": saved_rows,
        "refit_all": args.refit_all,
        "features": feature_columns,
        "max_iter": args.max_iter,
        "regularization": args.regularization,
    }
    args.output.with_suffix(".json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
