"""Tests for temporal model training pipeline logic."""

import json

import pandas as pd
import pytest

from src.data.synthetic import load_synthetic_dataset
from src.models.train import (
    build_temporal_model_panel,
    run_training,
    temporal_train_test_split,
)


def test_temporal_train_test_split_uses_earlier_dates_for_training() -> None:
    """Verify rows are split strictly by prediction date boundaries."""
    panel = pd.DataFrame(
        {
            "artist_id": ["a", "b", "c", "d"],
            "prediction_date": ["2020-01-01", "2021-12-31", "2022-01-01", "2023-01-01"],
            "institutional_success_3y": [0, 1, 0, 1],
        }
    )

    train, test = temporal_train_test_split(panel, "2021-12-31", "2022-01-01")

    assert train["prediction_date"].max() == pd.Timestamp("2021-12-31")
    assert test["prediction_date"].min() == pd.Timestamp("2022-01-01")
    assert set(train["artist_id"]) == {"a", "b"}
    assert set(test["artist_id"]) == {"c", "d"}


def test_temporal_train_test_split_rejects_overlapping_boundaries() -> None:
    """Verify invalid temporal split boundaries fail loudly."""
    panel = pd.DataFrame(
        {
            "artist_id": ["a", "b"],
            "prediction_date": ["2021-01-01", "2022-01-01"],
        }
    )

    with pytest.raises(ValueError, match="train_end_date must be before test_start_date"):
        temporal_train_test_split(panel, "2022-01-01", "2022-01-01")


def test_temporal_panel_contains_point_in_time_feature_rows_and_future_labels() -> None:
    """Verify panel generation creates artist-date rows without future feature leakage."""
    panel = build_temporal_model_panel(
        load_synthetic_dataset(),
        prediction_dates=("2021-01-01", "2022-01-01"),
    )
    ada = panel[panel["artist_id"] == "artist_ada_rios"].set_index("prediction_date")

    assert len(panel) == 16
    assert ada.loc[pd.Timestamp("2021-01-01"), "museum_acquisition_count"] == 0
    assert ada.loc[pd.Timestamp("2022-01-01"), "museum_acquisition_count"] == 0
    assert ada.loc[pd.Timestamp("2022-01-01"), "institutional_success_3y"] == 1


def test_run_training_writes_metrics_predictions_and_model(tmp_path) -> None:
    """Verify the temporal training pipeline writes requested artifacts."""
    reports_dir = tmp_path / "reports"
    artifacts_dir = tmp_path / "artifacts"

    result = run_training(
        target="institutional_success_3y",
        train_end_date="2021-12-31",
        test_start_date="2022-01-01",
        reports_dir=reports_dir,
        artifacts_dir=artifacts_dir,
    )

    metrics_path = reports_dir / "metrics.json"
    predictions_path = reports_dir / "predictions.csv"
    model_path = artifacts_dir / "model.joblib"

    assert metrics_path.exists()
    assert predictions_path.exists()
    assert model_path.exists()
    assert result["best_model_name"] in {"logistic_regression", "random_forest", "gradient_boosting"}

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    predictions = pd.read_csv(predictions_path)

    assert {"logistic_regression", "random_forest", "gradient_boosting"}.issubset(metrics["models"])
    assert "random_baseline" in metrics["models"]
    assert {"model", "artist_id", "prediction_date", "score", "probability_like_score"}.issubset(
        predictions.columns
    )
