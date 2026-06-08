"""Tests for similar artist search."""

import sys
from subprocess import run

import pandas as pd
import pytest

from src.data.synthetic import load_synthetic_dataset
from src.graph.build_graph import build_investment_graph
from src.models.similarity import (
    SIMILARITY_FEATURE_COLUMNS,
    find_similar_artists,
    normalize_feature_vectors,
)


def test_find_similar_artists_returns_scores_outcomes_and_shared_signals() -> None:
    """Verify similar artist output includes ranking, later outcomes, and explanations."""
    dataset = load_synthetic_dataset()
    graph = build_investment_graph(dataset)

    similar = find_similar_artists(
        graph=graph,
        dataset=dataset,
        artist_id="artist_ada_rios",
        as_of_date="2021-12-31",
        top_n=3,
    )

    assert len(similar) == 3
    assert "artist_ada_rios" not in set(similar["artist_id"])
    assert similar["similarity_score"].between(-1, 1).all()
    assert similar["similarity_score"].is_monotonic_decreasing
    assert similar["later_outcome_summary"].str.len().gt(0).all()
    assert similar["shared_signals"].str.contains(",").all()
    assert {"institutional_success_3y", "market_success_3y", "gallery_success_3y"}.issubset(similar.columns)


def test_normalize_feature_vectors_scales_numeric_features() -> None:
    """Verify numeric similarity features are standardized."""
    features = pd.DataFrame(
        {
            "artist_id": ["a", "b", "c"],
            **{column: [0.0, 1.0, 2.0] for column in SIMILARITY_FEATURE_COLUMNS},
        }
    )

    normalized = normalize_feature_vectors(features)

    assert normalized["artist_id"].tolist() == ["a", "b", "c"]
    assert normalized[SIMILARITY_FEATURE_COLUMNS].mean().abs().max() < 1e-12


def test_find_similar_artists_rejects_unknown_artist() -> None:
    """Verify unknown source artists fail clearly."""
    dataset = load_synthetic_dataset()
    graph = build_investment_graph(dataset)

    with pytest.raises(ValueError, match="not present"):
        find_similar_artists(graph, dataset, "missing_artist", "2021-12-31")


def test_similarity_cli_runs() -> None:
    """Verify the requested similarity CLI prints similar artists."""
    result = run(
        [
            sys.executable,
            "-m",
            "src.models.similarity",
            "--artist-id",
            "artist_ada_rios",
            "--as-of-date",
            "2021-12-31",
            "--top-n",
            "2",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "similarity_score" in result.stdout
    assert "shared_signals" in result.stdout
