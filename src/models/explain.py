"""Plain-English explanations for artist prediction scores."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.inspection import permutation_importance

from src.data.quality import artist_quality_warning


BASELINE_WEIGHTS = {
    "museum_count_baseline": {
        "museum_exhibition_count": 1.0,
        "museum_acquisition_count": 1.0,
        "major_museum_exhibition_count": 2.0,
        "major_museum_acquisition_count": 2.0,
    },
    "gallery_prestige_baseline": {
        "gallery_prestige_score": 1.0,
        "gallery_tier": 0.1,
    },
    "simple_weighted_score_baseline": {
        "major_museum_exhibition_count": 2.0,
        "major_museum_acquisition_count": 2.0,
        "gallery_prestige_score": 1.5,
        "gallery_tier": 0.5,
        "collector_centrality_score": 0.75,
        "curator_centrality_score": 0.75,
        "auction_price_growth_1y": 0.25,
        "press_mention_growth_1y": 0.1,
    },
}

FEATURE_PHRASES = {
    "museum_exhibition_count": "museum exhibition history",
    "museum_acquisition_count": "museum acquisition history",
    "major_museum_exhibition_count": "recent major museum exposure",
    "major_museum_acquisition_count": "major museum acquisition signals",
    "gallery_prestige_score": "representation by a prestigious gallery",
    "gallery_tier": "representation by a higher-tier gallery",
    "collector_degree": "breadth of collector support",
    "collector_centrality_score": "proximity to influential collectors",
    "curator_degree": "breadth of curator support",
    "curator_centrality_score": "proximity to influential curators",
    "art_fair_count": "art fair and biennial exposure",
    "auction_lot_count": "auction history depth",
    "auction_median_price": "auction price level",
    "auction_price_growth_1y": "recent auction price momentum",
    "press_mention_count_1y": "recent press visibility",
    "press_mention_growth_1y": "press momentum",
    "graph_distance_to_major_institution": "network proximity to major institutions",
    "graph_distance_to_top_gallery": "network proximity to top galleries",
    "career_age_years": "career maturity",
}


@dataclass(frozen=True)
class PredictionExplanation:
    """Structured explanation for one artist prediction."""

    artist_id: str
    score: float
    top_positive_drivers: list[str]
    top_negative_drivers: list[str]
    confidence_level: str
    comparable_artists: list[str]
    data_quality_warning: str
    explanation_text: str


def explain_artist_prediction(
    artist_id: str,
    score: float,
    feature_row: pd.Series,
    comparable_artists: list[str] | None = None,
    model=None,
    feature_frame: pd.DataFrame | None = None,
    labels: pd.Series | None = None,
    baseline_name: str | None = None,
    quality_row: pd.Series | None = None,
) -> PredictionExplanation:
    """Explain a prediction using model importances or baseline contributions."""
    comparable_artists = comparable_artists or []
    importances = _feature_importances(model, feature_frame, labels, baseline_name)
    positive = _top_positive_drivers(feature_row, importances, feature_frame)
    negative = _top_negative_drivers(feature_row, importances, feature_frame)
    confidence = _confidence_level(feature_row)
    warning = artist_quality_warning(quality_row) or _data_quality_warning(feature_row)
    text = _plain_english_summary(score, positive, negative, confidence, comparable_artists, warning)

    return PredictionExplanation(
        artist_id=artist_id,
        score=float(score),
        top_positive_drivers=positive,
        top_negative_drivers=negative,
        confidence_level=confidence,
        comparable_artists=comparable_artists,
        data_quality_warning=warning,
        explanation_text=text,
    )


def explain_baseline_score(
    artist_id: str,
    score: float,
    feature_row: pd.Series,
    baseline_name: str,
    comparable_artists: list[str] | None = None,
    quality_row: pd.Series | None = None,
) -> PredictionExplanation:
    """Explain a baseline score from weighted feature contributions."""
    return explain_artist_prediction(
        artist_id=artist_id,
        score=score,
        feature_row=feature_row,
        comparable_artists=comparable_artists,
        baseline_name=baseline_name,
        quality_row=quality_row,
    )


def _feature_importances(
    model,
    feature_frame: pd.DataFrame | None,
    labels: pd.Series | None,
    baseline_name: str | None,
) -> dict[str, float]:
    """Return feature importances from baseline weights, permutation, or tree importances."""
    if baseline_name:
        return BASELINE_WEIGHTS.get(baseline_name, BASELINE_WEIGHTS["simple_weighted_score_baseline"])

    if model is not None and feature_frame is not None and labels is not None and labels.nunique() > 1:
        try:
            result = permutation_importance(model, feature_frame, labels, n_repeats=5, random_state=42)
            return dict(zip(feature_frame.columns, result.importances_mean))
        except Exception:
            pass

    classifier = getattr(model, "named_steps", {}).get("classifier") if model is not None else None
    if classifier is not None and hasattr(classifier, "feature_importances_") and feature_frame is not None:
        return dict(zip(feature_frame.columns, classifier.feature_importances_))

    return BASELINE_WEIGHTS["simple_weighted_score_baseline"]


def _top_positive_drivers(
    feature_row: pd.Series,
    importances: dict[str, float],
    feature_frame: pd.DataFrame | None,
    limit: int = 3,
) -> list[str]:
    """Return highest positive feature contribution phrases."""
    contributions = {}
    for feature, importance in importances.items():
        if feature not in feature_row:
            continue
        value = float(feature_row.get(feature, 0) or 0)
        contributions[feature] = value * max(float(importance), 0)
    return _phrases_from_ranked_features(contributions, descending=True, limit=limit)


def _top_negative_drivers(
    feature_row: pd.Series,
    importances: dict[str, float],
    feature_frame: pd.DataFrame | None,
    limit: int = 3,
) -> list[str]:
    """Return feature phrases likely holding the score back."""
    drag = {}
    for feature, importance in importances.items():
        if feature not in feature_row:
            continue
        value = float(feature_row.get(feature, 0) or 0)
        reference = _feature_reference_value(feature, feature_frame)
        drag[feature] = max(reference - value, 0) * abs(float(importance))
    phrases = _phrases_from_ranked_features(drag, descending=True, limit=limit)
    if phrases:
        return phrases
    low_signal_features = [
        feature
        for feature in ("auction_lot_count", "press_mention_count_1y", "collector_degree")
        if float(feature_row.get(feature, 0) or 0) <= 0
    ]
    return [_feature_phrase(feature) for feature in low_signal_features[:limit]]


def _feature_reference_value(feature: str, feature_frame: pd.DataFrame | None) -> float:
    """Return median feature value used to identify weak signals."""
    if feature_frame is None or feature not in feature_frame:
        return 1.0
    return float(pd.to_numeric(feature_frame[feature], errors="coerce").fillna(0).median())


def _phrases_from_ranked_features(
    values: dict[str, float],
    descending: bool,
    limit: int,
) -> list[str]:
    ranked = sorted(values.items(), key=lambda item: item[1], reverse=descending)
    filtered = [feature for feature, value in ranked if value > 0]
    return [_feature_phrase(feature) for feature in filtered[:limit]]


def _feature_phrase(feature: str) -> str:
    """Return a plain-English phrase for a feature."""
    return FEATURE_PHRASES.get(feature, feature.replace("_", " "))


def _confidence_level(feature_row: pd.Series) -> str:
    """Estimate explanation confidence from observable data depth."""
    auction_lots = float(feature_row.get("auction_lot_count", 0) or 0)
    institutional_signals = (
        float(feature_row.get("museum_exhibition_count", 0) or 0)
        + float(feature_row.get("museum_acquisition_count", 0) or 0)
    )
    network_signals = (
        float(feature_row.get("collector_degree", 0) or 0)
        + float(feature_row.get("curator_degree", 0) or 0)
    )
    if auction_lots >= 3 and institutional_signals >= 2 and network_signals >= 2:
        return "high"
    if auction_lots >= 1 and (institutional_signals >= 1 or network_signals >= 1):
        return "medium"
    return "low"


def _data_quality_warning(feature_row: pd.Series) -> str:
    """Return a warning when data depth is thin."""
    warnings = []
    if float(feature_row.get("auction_lot_count", 0) or 0) < 2:
        warnings.append("auction history is limited")
    if float(feature_row.get("press_mention_count_1y", 0) or 0) == 0:
        warnings.append("recent press data is sparse")
    if float(feature_row.get("collector_degree", 0) or 0) == 0:
        warnings.append("collector network data is sparse")
    if not warnings:
        return ""
    return "Data quality warning: " + "; ".join(warnings) + "."


def _plain_english_summary(
    score: float,
    positive: list[str],
    negative: list[str],
    confidence: str,
    comparable_artists: list[str],
    warning: str,
) -> str:
    """Build a concise plain-English explanation paragraph."""
    positive_text = _join_phrases(positive) if positive else "the available career signals are limited"
    text = f"Score increased because the artist has {positive_text}."
    if negative:
        text += f" The score is held back by {_join_phrases(negative)}."
    text += f" Confidence is {confidence}"
    if warning:
        text += " because " + warning.removeprefix("Data quality warning: ").rstrip(".")
    text += "."
    if comparable_artists:
        text += f" Comparable artists include {_join_phrases(comparable_artists)}."
    return text


def _join_phrases(phrases: list[str]) -> str:
    """Join phrases into readable English."""
    if len(phrases) <= 1:
        return phrases[0] if phrases else ""
    if len(phrases) == 2:
        return f"{phrases[0]} and {phrases[1]}"
    return ", ".join(phrases[:-1]) + f", and {phrases[-1]}"
