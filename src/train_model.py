"""Train a reproducible NumPy softmax-regression baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TARGETS = ["H", "D", "A"]
EXCLUDED = {"season", "date", "home_team", "away_team", "home_goals", "away_goals", "result", "source"}


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def prepare_matrix(frame: pd.DataFrame, feature_columns: list[str], medians: pd.Series, means: pd.Series, scales: pd.Series) -> np.ndarray:
    numeric = frame[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(medians)
    standardized = (numeric - means) / scales
    return np.column_stack([np.ones(len(standardized)), standardized.to_numpy(dtype=float)])


def fit_softmax(x: np.ndarray, y: np.ndarray, epochs: int, learning_rate: float, l2: float) -> np.ndarray:
    """Fit multiclass softmax regression with batch gradient descent."""
    weights = np.zeros((x.shape[1], len(TARGETS)))
    for _ in range(epochs):
        probabilities = softmax(x @ weights)
        gradient = x.T @ (probabilities - y) / len(x)
        gradient[1:] += l2 * weights[1:]
        weights -= learning_rate * gradient
    return weights


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/processed/matches_with_features.csv"))
    parser.add_argument("--output", type=Path, default=Path("models/softmax_baseline.npz"))
    parser.add_argument("--test-season")
    parser.add_argument("--epochs", type=int, default=2500)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=0.001)
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
    train_numeric = train[feature_columns].apply(pd.to_numeric, errors="coerce")
    medians = train_numeric.median().fillna(0.0)
    means = train_numeric.fillna(medians).mean()
    scales = train_numeric.fillna(medians).std().replace(0, 1).fillna(1)
    x_train = prepare_matrix(train, feature_columns, medians, means, scales)
    x_test = prepare_matrix(test, feature_columns, medians, means, scales)

    target_index = {target: index for index, target in enumerate(TARGETS)}
    y_train = np.eye(len(TARGETS))[[target_index[value] for value in train["result"]]]
    y_test_index = np.array([target_index[value] for value in test["result"]])

    weights = fit_softmax(x_train, y_train, args.epochs, args.learning_rate, args.l2)

    test_probabilities = softmax(x_test @ weights)
    predictions = test_probabilities.argmax(axis=1)
    accuracy = float((predictions == y_test_index).mean())
    log_loss = float(-np.log(test_probabilities[np.arange(len(test)), y_test_index].clip(1e-12)).mean())

    saved_rows = len(train)
    if args.refit_all:
        all_numeric = frame[feature_columns].apply(pd.to_numeric, errors="coerce")
        medians = all_numeric.median().fillna(0.0)
        means = all_numeric.fillna(medians).mean()
        scales = all_numeric.fillna(medians).std().replace(0, 1).fillna(1)
        x_all = prepare_matrix(frame, feature_columns, medians, means, scales)
        y_all = np.eye(len(TARGETS))[[target_index[value] for value in frame["result"]]]
        weights = fit_softmax(x_all, y_all, args.epochs, args.learning_rate, args.l2)
        saved_rows = len(frame)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        weights=weights,
        medians=medians.to_numpy(),
        means=means.to_numpy(),
        scales=scales.to_numpy(),
        feature_columns=np.array(feature_columns),
        targets=np.array(TARGETS),
    )
    metrics = {
        "train_rows": len(train),
        "test_rows": len(test),
        "test_season": test_season,
        "accuracy": accuracy,
        "log_loss": log_loss,
        "saved_model_rows": saved_rows,
        "refit_all": args.refit_all,
        "features": feature_columns,
    }
    args.output.with_suffix(".json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
