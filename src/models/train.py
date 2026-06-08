"""Temporal model training pipeline for artist success prediction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.synthetic import load_synthetic_dataset
from src.data.quality import save_data_quality_report
from src.graph.build_graph import build_investment_graph
from src.graph.features import ARTIST_GRAPH_FEATURE_COLUMNS, build_artist_graph_features
from src.models.baselines import get_baselines
from src.models.evaluate import evaluate_predictions
from src.models.targets import TARGET_COLUMNS, generate_artist_targets


DEFAULT_PREDICTION_DATES = (
    "2020-01-01",
    "2020-07-01",
    "2021-01-01",
    "2021-07-01",
    "2022-01-01",
    "2022-07-01",
    "2023-01-01",
)
MODEL_FEATURE_COLUMNS = [column for column in ARTIST_GRAPH_FEATURE_COLUMNS if column != "artist_id"]


def build_temporal_model_panel(
    dataset: dict[str, pd.DataFrame],
    prediction_dates: tuple[str, ...] = DEFAULT_PREDICTION_DATES,
) -> pd.DataFrame:
    """Build point-in-time features and future labels for prediction dates."""
    graph = build_investment_graph(dataset)
    frames = []

    for prediction_date in prediction_dates:
        features = build_artist_graph_features(graph, prediction_date)
        features["prediction_date"] = prediction_date
        targets = generate_artist_targets(dataset, prediction_date)
        targets["prediction_date"] = prediction_date
        frames.append(features.merge(targets, on=["artist_id", "prediction_date"], how="inner"))

    panel = pd.concat(frames, ignore_index=True)
    panel["prediction_date"] = pd.to_datetime(panel["prediction_date"])
    return panel


def temporal_train_test_split(
    panel: pd.DataFrame,
    train_end_date: str,
    test_start_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows by prediction date to prevent temporal leakage."""
    train_end = pd.Timestamp(train_end_date)
    test_start = pd.Timestamp(test_start_date)
    if train_end >= test_start:
        raise ValueError("train_end_date must be before test_start_date.")

    dated_panel = panel.copy()
    dated_panel["prediction_date"] = pd.to_datetime(dated_panel["prediction_date"])
    train = dated_panel[dated_panel["prediction_date"] <= train_end].copy()
    test = dated_panel[dated_panel["prediction_date"] >= test_start].copy()

    if train.empty:
        raise ValueError("Temporal split produced an empty training set.")
    if test.empty:
        raise ValueError("Temporal split produced an empty test set.")
    if train["prediction_date"].max() >= test["prediction_date"].min():
        raise ValueError("Temporal split leaked later test dates into training.")

    return train, test


def train_temporal_models(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    k: int = 5,
) -> tuple[dict[str, dict[str, float | None]], pd.DataFrame, dict[str, Pipeline]]:
    """Train supported classifiers and evaluate them on later dates."""
    models = get_model_registry()
    metrics = {}
    predictions = []
    fitted_models = {}

    x_train = train[MODEL_FEATURE_COLUMNS]
    y_train = train[target].astype(int)
    x_test = test[MODEL_FEATURE_COLUMNS]

    for model_name, model in models.items():
        model.fit(x_train, y_train)
        probabilities = model.predict_proba(x_test)[:, 1]
        model_predictions = _format_predictions(test, model_name, probabilities, target)
        metrics[model_name] = evaluate_predictions(model_predictions, test, target, k)
        predictions.append(model_predictions)
        fitted_models[model_name] = model

    return metrics, pd.concat(predictions, ignore_index=True), fitted_models


def evaluate_baselines_on_panel(
    test: pd.DataFrame,
    target: str,
    k: int = 5,
) -> tuple[dict[str, dict[str, float | None]], pd.DataFrame]:
    """Score all baselines on the temporal test panel."""
    metrics = {}
    predictions = []

    for baseline_name, baseline_fn in get_baselines().items():
        baseline_predictions = []
        for prediction_date, frame in test.groupby("prediction_date"):
            date_string = pd.Timestamp(prediction_date).strftime("%Y-%m-%d")
            scored = baseline_fn(frame.reset_index(drop=True), date_string)
            baseline_predictions.append(scored)
        scored_baseline = pd.concat(baseline_predictions, ignore_index=True)
        scored_baseline["model"] = baseline_name
        scored_baseline = scored_baseline.rename(columns={"probability_like_score": "probability_like_score"})
        scored_baseline = _attach_targets(scored_baseline, test, target)
        metrics[baseline_name] = evaluate_predictions(scored_baseline, test, target, k)
        predictions.append(scored_baseline)

    return metrics, pd.concat(predictions, ignore_index=True)


def get_model_registry() -> dict[str, Pipeline]:
    """Return supported temporal ML classifiers."""
    return {
        "logistic_regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("classifier", RandomForestClassifier(n_estimators=200, max_depth=4, random_state=42)),
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("classifier", GradientBoostingClassifier(random_state=42)),
            ]
        ),
    }


def run_training(
    target: str,
    train_end_date: str,
    test_start_date: str,
    prediction_dates: tuple[str, ...] = DEFAULT_PREDICTION_DATES,
    reports_dir: str | Path = "reports",
    artifacts_dir: str | Path = "artifacts",
    k: int = 5,
) -> dict[str, object]:
    """Run the full temporal training pipeline and save outputs."""
    if target not in TARGET_COLUMNS:
        valid_targets = ", ".join(TARGET_COLUMNS)
        raise ValueError(f"Unknown target '{target}'. Valid targets: {valid_targets}.")

    dataset = load_synthetic_dataset()
    graph = build_investment_graph(dataset)
    panel = build_temporal_model_panel(dataset, prediction_dates)
    train, test = temporal_train_test_split(panel, train_end_date, test_start_date)

    _validate_target_classes(train, target)
    model_metrics, model_predictions, fitted_models = train_temporal_models(train, test, target, k)
    baseline_metrics, baseline_predictions = evaluate_baselines_on_panel(test, target, k)

    metrics = {
        "target": target,
        "train_end_date": train_end_date,
        "test_start_date": test_start_date,
        "feature_columns": MODEL_FEATURE_COLUMNS,
        "models": {**model_metrics, **baseline_metrics},
    }
    predictions = pd.concat([model_predictions, baseline_predictions], ignore_index=True)

    reports_path = Path(reports_dir)
    artifacts_path = Path(artifacts_dir)
    reports_path.mkdir(parents=True, exist_ok=True)
    artifacts_path.mkdir(parents=True, exist_ok=True)

    (reports_path / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    predictions.to_csv(reports_path / "predictions.csv", index=False)
    save_data_quality_report(dataset, graph, reports_path / "data_quality.csv")

    best_model_name = _best_model_name(model_metrics)
    joblib.dump(
        {
            "model_name": best_model_name,
            "target": target,
            "feature_columns": MODEL_FEATURE_COLUMNS,
            "estimator": fitted_models[best_model_name],
        },
        artifacts_path / "model.joblib",
    )

    return {
        "metrics": metrics,
        "predictions": predictions,
        "best_model_name": best_model_name,
        "train_rows": len(train),
        "test_rows": len(test),
    }


def _format_predictions(
    test: pd.DataFrame,
    model_name: str,
    probabilities: pd.Series | list[float],
    target: str,
) -> pd.DataFrame:
    """Format model predictions for reports."""
    output = test[["artist_id", "prediction_date", target]].copy()
    output["prediction_date"] = pd.to_datetime(output["prediction_date"]).dt.strftime("%Y-%m-%d")
    output["model"] = model_name
    output["score"] = probabilities
    output["probability_like_score"] = probabilities
    return output[
        ["model", "artist_id", "prediction_date", "score", "probability_like_score", target]
    ]


def _attach_targets(predictions: pd.DataFrame, test: pd.DataFrame, target: str) -> pd.DataFrame:
    """Attach target labels to baseline predictions."""
    labels = test[["artist_id", "prediction_date", target]].copy()
    labels["prediction_date"] = pd.to_datetime(labels["prediction_date"]).dt.strftime("%Y-%m-%d")
    output = predictions.merge(labels, on=["artist_id", "prediction_date"], how="left")
    return output[["model", "artist_id", "prediction_date", "score", "probability_like_score", target]]


def _validate_target_classes(train: pd.DataFrame, target: str) -> None:
    """Ensure supervised models have both target classes in training data."""
    if train[target].nunique() < 2:
        raise ValueError(f"Training target '{target}' contains a single class.")


def _best_model_name(metrics: dict[str, dict[str, float | None]]) -> str:
    """Choose the ML model with the best average precision."""
    return max(
        metrics,
        key=lambda model_name: (
            metrics[model_name].get("average_precision")
            if metrics[model_name].get("average_precision") is not None
            else -1
        ),
    )


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Train temporal artist success models.")
    parser.add_argument("--target", required=True, help="Target column to train against.")
    parser.add_argument("--train-end-date", required=True, help="Latest prediction date allowed in training.")
    parser.add_argument("--test-start-date", required=True, help="Earliest prediction date allowed in testing.")
    parser.add_argument("--k", type=int, default=5, help="k for precision@k and recall@k.")
    return parser.parse_args()


def main() -> None:
    """Run the temporal training CLI."""
    args = _parse_args()
    result = run_training(
        target=args.target,
        train_end_date=args.train_end_date,
        test_start_date=args.test_start_date,
        k=args.k,
    )
    print(f"Saved metrics to reports/metrics.json")
    print(f"Saved predictions to reports/predictions.csv")
    print(f"Saved trained model to artifacts/model.joblib")
    print(f"Best ML model: {result['best_model_name']}")


if __name__ == "__main__":
    main()
