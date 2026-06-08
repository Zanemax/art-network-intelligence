"""Tests for baseline ranking models and baseline evaluation."""

import sys
from subprocess import run

import pandas as pd
import pytest

from src.models.baselines import (
    BASELINE_OUTPUT_COLUMNS,
    evaluate_rankings,
    gallery_prestige_baseline,
    get_baselines,
    museum_count_baseline,
    precision_at_k,
    random_baseline,
    recall_at_k,
    simple_weighted_score_baseline,
)


def test_baselines_return_required_columns_and_probability_like_scores() -> None:
    """Verify every baseline emits the required ranking schema."""
    features = _baseline_features()

    for baseline_fn in get_baselines().values():
        predictions = baseline_fn(features, "2024-01-01")

        assert tuple(predictions.columns) == BASELINE_OUTPUT_COLUMNS
        assert predictions["artist_id"].tolist() == features["artist_id"].tolist()
        assert predictions["prediction_date"].eq("2024-01-01").all()
        assert predictions["probability_like_score"].between(0, 1).all()


def test_named_baselines_rank_expected_artists() -> None:
    """Verify simple signal baselines rank artists by their intended feature family."""
    features = _baseline_features()

    museum_predictions = museum_count_baseline(features, "2024-01-01")
    gallery_predictions = gallery_prestige_baseline(features, "2024-01-01")
    weighted_predictions = simple_weighted_score_baseline(features, "2024-01-01")

    assert _top_artist(museum_predictions) == "artist_a"
    assert _top_artist(gallery_predictions) == "artist_b"
    assert _top_artist(weighted_predictions) in {"artist_a", "artist_b"}


def test_random_baseline_is_deterministic_with_seed() -> None:
    """Verify seeded random scores are stable."""
    features = _baseline_features()

    first = random_baseline(features, "2024-01-01", random_state=7)
    second = random_baseline(features, "2024-01-01", random_state=7)

    pd.testing.assert_frame_equal(first, second)


def test_evaluation_metrics_include_auc_precision_recall_and_ap() -> None:
    """Verify ranking metric calculations on a small known example."""
    predictions = pd.DataFrame(
        {
            "artist_id": ["artist_a", "artist_b", "artist_c"],
            "prediction_date": "2024-01-01",
            "score": [0.9, 0.8, 0.1],
            "probability_like_score": [0.9, 0.8, 0.1],
        }
    )
    labels = pd.DataFrame(
        {
            "artist_id": ["artist_a", "artist_b", "artist_c"],
            "institutional_success_3y": [1, 0, 1],
        }
    )

    metrics = evaluate_rankings(predictions, labels, "institutional_success_3y", k=2)

    assert set(metrics) == {"roc_auc", "precision_at_k", "recall_at_k", "average_precision"}
    assert metrics["roc_auc"] == pytest.approx(0.5)
    assert metrics["precision_at_k"] == pytest.approx(0.5)
    assert metrics["recall_at_k"] == pytest.approx(0.5)
    assert metrics["average_precision"] == pytest.approx(0.8333333333)


def test_precision_and_recall_at_k_handle_empty_or_zero_positive_inputs() -> None:
    """Verify metric helpers handle degenerate rankings."""
    scored = pd.DataFrame(
        {
            "artist_id": ["artist_a"],
            "probability_like_score": [0.1],
            "target": [0],
        }
    )

    assert precision_at_k(scored, "target", 0) == 0.0
    assert recall_at_k(scored, "target", 5) == 0.0


def test_evaluate_baselines_cli_runs_for_institutional_target() -> None:
    """Verify the requested module command runs and prints baseline metrics."""
    result = run(
        [
            sys.executable,
            "-m",
            "src.models.evaluate_baselines",
            "--target",
            "institutional_success_3y",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "random_baseline" in result.stdout
    assert "roc_auc" in result.stdout
    assert "average_precision" in result.stdout


def _baseline_features() -> pd.DataFrame:
    """Create a compact artist feature fixture for baseline tests."""
    return pd.DataFrame(
        [
            {
                "artist_id": "artist_a",
                "museum_exhibition_count": 2,
                "museum_acquisition_count": 1,
                "major_museum_exhibition_count": 1,
                "major_museum_acquisition_count": 1,
                "gallery_prestige_score": 0.4,
                "gallery_tier": 2,
                "collector_centrality_score": 0.2,
                "curator_centrality_score": 0.3,
                "auction_price_growth_1y": 0.1,
                "press_mention_growth_1y": 0.2,
            },
            {
                "artist_id": "artist_b",
                "museum_exhibition_count": 0,
                "museum_acquisition_count": 0,
                "major_museum_exhibition_count": 0,
                "major_museum_acquisition_count": 0,
                "gallery_prestige_score": 0.95,
                "gallery_tier": 4,
                "collector_centrality_score": 0.1,
                "curator_centrality_score": 0.1,
                "auction_price_growth_1y": 0.3,
                "press_mention_growth_1y": 0.4,
            },
            {
                "artist_id": "artist_c",
                "museum_exhibition_count": 0,
                "museum_acquisition_count": 0,
                "major_museum_exhibition_count": 0,
                "major_museum_acquisition_count": 0,
                "gallery_prestige_score": 0.1,
                "gallery_tier": 1,
                "collector_centrality_score": 0.0,
                "curator_centrality_score": 0.0,
                "auction_price_growth_1y": 0.0,
                "press_mention_growth_1y": 0.0,
            },
        ]
    )


def _top_artist(predictions: pd.DataFrame) -> str:
    """Return the highest-ranked artist ID."""
    return str(predictions.sort_values("probability_like_score", ascending=False).iloc[0]["artist_id"])
