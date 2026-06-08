"""Similar artist search from temporal graph-derived career features."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from src.data.synthetic import load_synthetic_dataset
from src.graph.build_graph import build_investment_graph
from src.graph.features import ARTIST_GRAPH_FEATURE_COLUMNS, build_artist_graph_features
from src.models.targets import TARGET_COLUMNS, generate_artist_targets


SIMILARITY_FEATURE_COLUMNS = [
    column for column in ARTIST_GRAPH_FEATURE_COLUMNS if column != "artist_id"
]


def find_similar_artists(
    graph,
    dataset: dict[str, pd.DataFrame],
    artist_id: str,
    as_of_date: str,
    top_n: int = 5,
) -> pd.DataFrame:
    """Return top-N similar artists using normalized graph-derived features."""
    features = build_artist_feature_vectors(graph, as_of_date)
    if artist_id not in set(features["artist_id"]):
        raise ValueError(f"artist_id '{artist_id}' is not present in the graph features.")

    normalized = normalize_feature_vectors(features)
    similarities = _cosine_scores(normalized, artist_id)
    outcomes = generate_artist_targets(dataset, as_of_date)
    artist_names = _artist_names(dataset)

    rows = []
    for similar_artist_id, similarity_score in similarities.head(top_n).items():
        rows.append(
            {
                "artist_id": similar_artist_id,
                "artist_name": artist_names.get(similar_artist_id, similar_artist_id),
                "as_of_date": as_of_date,
                "similarity_score": float(similarity_score),
                "later_outcome_summary": _later_outcome_summary(outcomes, similar_artist_id),
                "shared_signals": explain_similarity(
                    normalized,
                    artist_id,
                    similar_artist_id,
                    limit=5,
                ),
            }
        )

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.merge(outcomes, on="artist_id", how="left")


def build_artist_feature_vectors(graph, as_of_date: str) -> pd.DataFrame:
    """Compute artist-level feature vectors as of a chosen date."""
    return build_artist_graph_features(graph, as_of_date)


def normalize_feature_vectors(features: pd.DataFrame) -> pd.DataFrame:
    """Normalize numeric artist feature vectors with standard scaling."""
    normalized = features[["artist_id"]].copy()
    scaler = StandardScaler()
    numeric = features[SIMILARITY_FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    normalized[SIMILARITY_FEATURE_COLUMNS] = scaler.fit_transform(numeric)
    return normalized


def explain_similarity(
    normalized_features: pd.DataFrame,
    source_artist_id: str,
    similar_artist_id: str,
    limit: int = 5,
) -> str:
    """Explain similarity by listing features with closest normalized values."""
    indexed = normalized_features.set_index("artist_id")
    source = indexed.loc[source_artist_id, SIMILARITY_FEATURE_COLUMNS]
    similar = indexed.loc[similar_artist_id, SIMILARITY_FEATURE_COLUMNS]
    differences = (source - similar).abs().sort_values()
    shared = differences.head(limit).index.tolist()
    return ", ".join(shared)


def _cosine_scores(normalized_features: pd.DataFrame, artist_id: str) -> pd.Series:
    """Compute cosine similarity scores from one artist to all others."""
    indexed = normalized_features.set_index("artist_id")
    matrix = indexed[SIMILARITY_FEATURE_COLUMNS]
    source_vector = matrix.loc[[artist_id]]
    scores = cosine_similarity(source_vector, matrix)[0]
    similarities = pd.Series(scores, index=matrix.index).drop(index=artist_id)
    return similarities.sort_values(ascending=False)


def _later_outcome_summary(outcomes: pd.DataFrame, artist_id: str) -> str:
    """Summarize available later outcomes for a similar artist."""
    row = outcomes.loc[outcomes["artist_id"] == artist_id]
    if row.empty:
        return "No later outcomes available"
    row = row.iloc[0]
    positive_targets = [target for target in TARGET_COLUMNS if int(row[target]) == 1]
    if not positive_targets:
        return "No positive 3-year outcomes"
    return ", ".join(positive_targets)


def _artist_names(dataset: dict[str, pd.DataFrame]) -> dict[str, str]:
    """Return artist display names keyed by ID."""
    artists = dataset.get("artists", pd.DataFrame())
    if artists.empty or "artist_id" not in artists.columns:
        return {}
    name_column = "name" if "name" in artists.columns else "canonical_name"
    if name_column not in artists.columns:
        return {}
    return artists.set_index("artist_id")[name_column].to_dict()


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Find similar artists from temporal graph features.")
    parser.add_argument("--artist-id", required=True, help="Source artist ID.")
    parser.add_argument("--as-of-date", required=True, help="Point-in-time feature cutoff date.")
    parser.add_argument("--top-n", type=int, default=5, help="Number of similar artists to return.")
    return parser.parse_args()


def main() -> None:
    """Run similar artist search from the command line."""
    args = _parse_args()
    dataset = load_synthetic_dataset()
    graph = build_investment_graph(dataset)
    similar = find_similar_artists(
        graph=graph,
        dataset=dataset,
        artist_id=args.artist_id,
        as_of_date=args.as_of_date,
        top_n=args.top_n,
    )
    print(similar.to_string(index=False))


if __name__ == "__main__":
    main()
