"""Artist-level feature extraction for the art investment model.

Features combine tabular market signals with graph-derived centrality and
distance metrics. The output is one row per artist and is designed to feed a
simple scikit-learn classifier.
"""

import networkx as nx
import pandas as pd


FEATURE_COLUMNS = [
    "museum_exhibitions",
    "museum_acquisitions",
    "gallery_prestige_score",
    "collector_centrality_score",
    "curator_centrality_score",
    "auction_price_growth",
    "press_mention_velocity",
    "distance_to_top_tier_institution",
]


def build_artist_features(
    dataset: dict[str, pd.DataFrame],
    graph: nx.MultiDiGraph,
) -> pd.DataFrame:
    """Return model-ready artist features and the three-year doubling target."""
    artists = dataset["artists"].copy()
    artist_ids = artists["artist_id"].tolist()
    undirected_graph = nx.Graph(graph)
    centrality = nx.degree_centrality(undirected_graph)

    features = artists[["artist_id", "name"]].copy()
    features["museum_exhibitions"] = _count_by_artist(dataset["exhibitions"], artist_ids)
    features["museum_acquisitions"] = _count_by_artist(dataset["acquisitions"], artist_ids)
    features["gallery_prestige_score"] = _gallery_prestige(dataset)
    features["collector_centrality_score"] = _neighbor_centrality(
        graph,
        artist_ids,
        centrality,
        "collector",
        "collects",
    )
    features["curator_centrality_score"] = _neighbor_centrality(
        graph,
        artist_ids,
        centrality,
        "curator",
        "curated_artist",
    )
    features["auction_price_growth"] = _auction_growth(dataset["auction_results"], artist_ids)
    features["press_mention_velocity"] = _press_velocity(dataset["press_mentions"], artist_ids)
    features["distance_to_top_tier_institution"] = _distance_to_top_tier(dataset, undirected_graph)
    features["doubled_in_3_years"] = (features["auction_price_growth"] >= 1.0).astype(int)

    return features.fillna(0)


def _count_by_artist(frame: pd.DataFrame, artist_ids: list[str]) -> pd.Series:
    """Count records per artist with zeros for missing artists."""
    return (
        frame.groupby("artist_id")
        .size()
        .reindex(artist_ids, fill_value=0)
        .reset_index(drop=True)
    )


def _gallery_prestige(dataset: dict[str, pd.DataFrame]) -> pd.Series:
    """Return each artist's highest gallery prestige score."""
    joined = dataset["representation"].merge(dataset["galleries"], on="gallery_id", how="left")
    artist_ids = dataset["artists"]["artist_id"]
    return (
        joined.groupby("artist_id")["prestige_score"]
        .max()
        .reindex(artist_ids, fill_value=0)
        .reset_index(drop=True)
    )


def _neighbor_centrality(
    graph: nx.MultiDiGraph,
    artist_ids: list[str],
    centrality: dict[str, float],
    node_type: str,
    relationship: str,
) -> pd.Series:
    """Average centrality of connected collectors or curators for each artist."""
    values: list[float] = []

    for artist_id in artist_ids:
        neighbor_scores = []
        for source, target, edge_data in graph.in_edges(artist_id, data=True):
            if target != artist_id or edge_data.get("relationship_type") != relationship:
                continue
            if graph.nodes[source].get("node_type") == node_type:
                neighbor_scores.append(centrality.get(source, 0.0))
        values.append(sum(neighbor_scores) / len(neighbor_scores) if neighbor_scores else 0.0)

    return pd.Series(values)


def _auction_growth(auction_results: pd.DataFrame, artist_ids: list[str]) -> pd.Series:
    """Calculate three-year auction price growth per artist."""
    ordered = auction_results.sort_values(["artist_id", "sale_date"])
    growth = ordered.groupby("artist_id").agg(
        first_price=("price_usd", "first"),
        latest_price=("price_usd", "last"),
    )
    growth["auction_price_growth"] = (
        growth["latest_price"] - growth["first_price"]
    ) / growth["first_price"]
    return growth["auction_price_growth"].reindex(artist_ids, fill_value=0).reset_index(drop=True)


def _press_velocity(press_mentions: pd.DataFrame, artist_ids: list[str]) -> pd.Series:
    """Calculate annualized press mention growth per artist."""
    ordered = press_mentions.sort_values(["artist_id", "year"])
    velocity = ordered.groupby("artist_id").agg(
        first_year=("year", "first"),
        latest_year=("year", "last"),
        first_mentions=("mentions", "first"),
        latest_mentions=("mentions", "last"),
    )
    years = (velocity["latest_year"] - velocity["first_year"]).clip(lower=1)
    velocity["press_mention_velocity"] = (
        velocity["latest_mentions"] - velocity["first_mentions"]
    ) / years
    return velocity["press_mention_velocity"].reindex(artist_ids, fill_value=0).reset_index(drop=True)


def _distance_to_top_tier(
    dataset: dict[str, pd.DataFrame],
    graph: nx.Graph,
) -> pd.Series:
    """Return shortest graph distance from each artist to any top-tier museum."""
    artist_ids = dataset["artists"]["artist_id"].tolist()
    top_tier_ids = dataset["museums"].loc[dataset["museums"]["tier"] == "top", "museum_id"].tolist()
    max_distance = graph.number_of_nodes()
    distances: list[int] = []

    for artist_id in artist_ids:
        artist_distances = []
        for institution_id in top_tier_ids:
            try:
                artist_distances.append(nx.shortest_path_length(graph, artist_id, institution_id))
            except nx.NetworkXNoPath:
                continue
        distances.append(min(artist_distances) if artist_distances else max_distance)

    return pd.Series(distances)
