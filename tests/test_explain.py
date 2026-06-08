"""Tests for plain-English artist prediction explanations."""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.models.explain import explain_artist_prediction, explain_baseline_score


def test_explain_baseline_score_returns_plain_english_drivers() -> None:
    """Verify baseline explanations include score, drivers, confidence, and warning."""
    row = _feature_frame().iloc[0]

    explanation = explain_baseline_score(
        artist_id="artist_a",
        score=0.82,
        feature_row=row,
        baseline_name="simple_weighted_score_baseline",
        comparable_artists=["Artist B", "Artist C"],
    )

    assert explanation.score == 0.82
    assert explanation.top_positive_drivers
    assert explanation.confidence_level == "medium"
    assert "Score increased because" in explanation.explanation_text
    assert "Comparable artists include Artist B and Artist C" in explanation.explanation_text


def test_explain_artist_prediction_uses_tree_importances_when_available() -> None:
    """Verify tree-based models can produce model-driven explanation drivers."""
    features = _feature_frame()
    labels = pd.Series([1, 0, 1, 0])
    model = Pipeline([("classifier", RandomForestClassifier(n_estimators=10, random_state=42))])
    model.fit(features, labels)

    explanation = explain_artist_prediction(
        artist_id="artist_a",
        score=0.7,
        feature_row=features.iloc[0],
        comparable_artists=["Artist B"],
        model=model,
        feature_frame=features,
        labels=labels,
    )

    assert explanation.top_positive_drivers
    assert explanation.confidence_level in {"medium", "high"}
    assert "Confidence is" in explanation.explanation_text


def test_explanation_warns_when_data_quality_is_low() -> None:
    """Verify sparse data produces a low-confidence warning."""
    row = pd.Series(
        {
            "auction_lot_count": 0,
            "press_mention_count_1y": 0,
            "collector_degree": 0,
            "museum_exhibition_count": 0,
            "museum_acquisition_count": 0,
            "gallery_prestige_score": 0,
        }
    )

    explanation = explain_artist_prediction(
        artist_id="artist_sparse",
        score=0.2,
        feature_row=row,
    )

    assert explanation.confidence_level == "low"
    assert "auction history is limited" in explanation.data_quality_warning
    assert "recent press data is sparse" in explanation.data_quality_warning
    assert "Confidence is low" in explanation.explanation_text


def _feature_frame() -> pd.DataFrame:
    """Create a compact feature fixture for explanations."""
    return pd.DataFrame(
        [
            {
                "major_museum_exhibition_count": 1,
                "major_museum_acquisition_count": 0,
                "gallery_prestige_score": 0.9,
                "gallery_tier": 4,
                "collector_centrality_score": 0.4,
                "curator_centrality_score": 0.2,
                "auction_price_growth_1y": 0.3,
                "press_mention_growth_1y": 0.1,
                "auction_lot_count": 2,
                "press_mention_count_1y": 5,
                "collector_degree": 1,
                "museum_exhibition_count": 1,
                "museum_acquisition_count": 0,
                "curator_degree": 1,
            },
            {
                "major_museum_exhibition_count": 0,
                "major_museum_acquisition_count": 0,
                "gallery_prestige_score": 0.2,
                "gallery_tier": 1,
                "collector_centrality_score": 0,
                "curator_centrality_score": 0,
                "auction_price_growth_1y": 0,
                "press_mention_growth_1y": 0,
                "auction_lot_count": 0,
                "press_mention_count_1y": 0,
                "collector_degree": 0,
                "museum_exhibition_count": 0,
                "museum_acquisition_count": 0,
                "curator_degree": 0,
            },
            {
                "major_museum_exhibition_count": 1,
                "major_museum_acquisition_count": 1,
                "gallery_prestige_score": 0.7,
                "gallery_tier": 3,
                "collector_centrality_score": 0.3,
                "curator_centrality_score": 0.4,
                "auction_price_growth_1y": 0.2,
                "press_mention_growth_1y": 0.4,
                "auction_lot_count": 3,
                "press_mention_count_1y": 6,
                "collector_degree": 2,
                "museum_exhibition_count": 2,
                "museum_acquisition_count": 1,
                "curator_degree": 2,
            },
            {
                "major_museum_exhibition_count": 0,
                "major_museum_acquisition_count": 0,
                "gallery_prestige_score": 0.1,
                "gallery_tier": 1,
                "collector_centrality_score": 0,
                "curator_centrality_score": 0,
                "auction_price_growth_1y": 0,
                "press_mention_growth_1y": 0,
                "auction_lot_count": 1,
                "press_mention_count_1y": 1,
                "collector_degree": 0,
                "museum_exhibition_count": 0,
                "museum_acquisition_count": 0,
                "curator_degree": 0,
            },
        ]
    )
