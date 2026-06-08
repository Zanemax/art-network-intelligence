"""Baseline ranking models and evaluation metrics for artist outcomes."""

from __future__ import annotations

import numpy as np
import pandas as pd
from collections.abc import Callable

from src.models.evaluate import evaluate_predictions, precision_at_k, recall_at_k


BASELINE_OUTPUT_COLUMNS = (
    "artist_id",
    "prediction_date",
    "score",
    "probability_like_score",
)


def random_baseline(
    features: pd.DataFrame,
    prediction_date: str,
    random_state: int = 42,
) -> pd.DataFrame:
    """Assign deterministic random scores to artists."""
    rng = np.random.default_rng(random_state)
    scores = pd.Series(rng.random(len(features)), index=features.index)
    return _format_baseline_output(features, prediction_date, scores)


def museum_count_baseline(features: pd.DataFrame, prediction_date: str) -> pd.DataFrame:
    """Rank artists by museum exhibition and acquisition count."""
    scores = (
        _feature(features, "museum_exhibition_count")
        + _feature(features, "museum_acquisition_count")
        + (2 * _feature(features, "major_museum_exhibition_count"))
        + (2 * _feature(features, "major_museum_acquisition_count"))
    )
    return _format_baseline_output(features, prediction_date, scores)


def gallery_prestige_baseline(features: pd.DataFrame, prediction_date: str) -> pd.DataFrame:
    """Rank artists by current gallery prestige and tier."""
    scores = _feature(features, "gallery_prestige_score") + (0.1 * _feature(features, "gallery_tier"))
    return _format_baseline_output(features, prediction_date, scores)


def simple_weighted_score_baseline(features: pd.DataFrame, prediction_date: str) -> pd.DataFrame:
    """Rank artists with a simple hand-weighted signal blend."""
    scores = (
        (2.0 * _feature(features, "major_museum_exhibition_count"))
        + (2.0 * _feature(features, "major_museum_acquisition_count"))
        + (1.5 * _feature(features, "gallery_prestige_score"))
        + (0.5 * _feature(features, "gallery_tier"))
        + (0.75 * _feature(features, "collector_centrality_score"))
        + (0.75 * _feature(features, "curator_centrality_score"))
        + (0.25 * _feature(features, "auction_price_growth_1y"))
        + (0.1 * _feature(features, "press_mention_growth_1y"))
    )
    return _format_baseline_output(features, prediction_date, scores)


def evaluate_rankings(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    target_column: str,
    k: int = 5,
) -> dict[str, float]:
    """Evaluate ranking scores against a binary target column."""
    return evaluate_predictions(predictions, labels, target_column, k)


def get_baselines() -> dict[str, Callable[[pd.DataFrame, str], pd.DataFrame]]:
    """Return registered baseline scorer functions."""
    return {
        "random_baseline": random_baseline,
        "museum_count_baseline": museum_count_baseline,
        "gallery_prestige_baseline": gallery_prestige_baseline,
        "simple_weighted_score_baseline": simple_weighted_score_baseline,
    }


def _format_baseline_output(
    features: pd.DataFrame,
    prediction_date: str,
    scores: pd.Series,
) -> pd.DataFrame:
    """Format raw baseline scores with a normalized ranking score."""
    output = pd.DataFrame(
        {
            "artist_id": features["artist_id"].values,
            "prediction_date": prediction_date,
            "score": scores.astype(float).values,
        }
    )
    output["probability_like_score"] = _min_max_scale(output["score"])
    return output.loc[:, BASELINE_OUTPUT_COLUMNS]


def _feature(features: pd.DataFrame, column: str) -> pd.Series:
    """Return a numeric feature column or zeros if unavailable."""
    if column not in features.columns:
        return pd.Series(0.0, index=features.index)
    return pd.to_numeric(features[column], errors="coerce").fillna(0.0)


def _min_max_scale(scores: pd.Series) -> pd.Series:
    """Scale scores into the 0-1 interval for ranking comparability."""
    minimum = scores.min()
    maximum = scores.max()
    if pd.isna(minimum) or pd.isna(maximum) or maximum == minimum:
        return pd.Series(0.5, index=scores.index)
    return (scores - minimum) / (maximum - minimum)

