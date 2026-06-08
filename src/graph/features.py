"""Artist-level point-in-time features from a temporal art graph."""

from __future__ import annotations

import math

import networkx as nx
import pandas as pd


ARTIST_GRAPH_FEATURE_COLUMNS = [
    "artist_id",
    "museum_exhibition_count",
    "museum_acquisition_count",
    "major_museum_exhibition_count",
    "major_museum_acquisition_count",
    "gallery_prestige_score",
    "gallery_tier",
    "collector_degree",
    "collector_centrality_score",
    "curator_degree",
    "curator_centrality_score",
    "art_fair_count",
    "auction_lot_count",
    "auction_median_price",
    "auction_price_growth_1y",
    "press_mention_count_1y",
    "press_mention_growth_1y",
    "graph_distance_to_major_institution",
    "graph_distance_to_top_gallery",
    "career_age_years",
]

MAJOR_INSTITUTION_TIERS = {"top", "major"}
TOP_GALLERY_TIERS = {"top", "mega"}
ART_FAIR_EVENT_TYPES = {"art_fair", "fair", "biennial", "biennial_inclusion"}
GALLERY_TIER_RANK = {
    "emerging": 1,
    "local": 1,
    "mid": 2,
    "regional": 2,
    "major": 3,
    "top": 4,
    "mega": 5,
}


def build_artist_graph_features(
    graph: nx.MultiDiGraph,
    prediction_date: str,
) -> pd.DataFrame:
    """Return one point-in-time feature row per artist in the graph."""
    snapshot = _snapshot_graph(graph, prediction_date)
    prediction_timestamp = pd.Timestamp(prediction_date)
    one_year_prior = prediction_timestamp - pd.DateOffset(years=1)
    two_year_prior = prediction_timestamp - pd.DateOffset(years=2)
    centrality = nx.degree_centrality(nx.Graph(snapshot)) if snapshot.number_of_nodes() > 1 else {}

    rows = []
    for artist_id, artist_data in sorted(_artist_nodes(snapshot)):
        rows.append(
            {
                "artist_id": artist_id,
                "museum_exhibition_count": _museum_exhibition_count(snapshot, artist_id),
                "museum_acquisition_count": _museum_acquisition_count(snapshot, artist_id),
                "major_museum_exhibition_count": _museum_exhibition_count(snapshot, artist_id, major_only=True),
                "major_museum_acquisition_count": _museum_acquisition_count(snapshot, artist_id, major_only=True),
                "gallery_prestige_score": _gallery_prestige_score(snapshot, artist_id),
                "gallery_tier": _gallery_tier(snapshot, artist_id),
                "collector_degree": _neighbor_degree(snapshot, artist_id, "collector", "collects"),
                "collector_centrality_score": _neighbor_centrality_score(
                    snapshot,
                    artist_id,
                    "collector",
                    "collects",
                    centrality,
                ),
                "curator_degree": _neighbor_degree(snapshot, artist_id, "curator", "curated_artist"),
                "curator_centrality_score": _neighbor_centrality_score(
                    snapshot,
                    artist_id,
                    "curator",
                    "curated_artist",
                    centrality,
                ),
                "art_fair_count": _art_fair_count(snapshot, artist_id),
                "auction_lot_count": _auction_lot_count(snapshot, artist_id, None, prediction_timestamp),
                "auction_median_price": _auction_median_price(snapshot, artist_id, None, prediction_timestamp),
                "auction_price_growth_1y": _auction_price_growth(
                    snapshot,
                    artist_id,
                    two_year_prior,
                    one_year_prior,
                    prediction_timestamp,
                ),
                "press_mention_count_1y": _press_mention_count(
                    snapshot,
                    artist_id,
                    one_year_prior,
                    prediction_timestamp,
                ),
                "press_mention_growth_1y": _press_mention_growth(
                    snapshot,
                    artist_id,
                    two_year_prior,
                    one_year_prior,
                    prediction_timestamp,
                ),
                "graph_distance_to_major_institution": _graph_distance(
                    snapshot,
                    artist_id,
                    _major_institution_nodes(snapshot),
                ),
                "graph_distance_to_top_gallery": _graph_distance(
                    snapshot,
                    artist_id,
                    _top_gallery_nodes(snapshot),
                ),
                "career_age_years": _career_age_years(snapshot, artist_id, artist_data, prediction_timestamp),
            }
        )

    return pd.DataFrame(rows, columns=ARTIST_GRAPH_FEATURE_COLUMNS).fillna(0)


def _snapshot_graph(graph: nx.MultiDiGraph, prediction_date: str) -> nx.MultiDiGraph:
    """Copy nodes and edges whose temporal edge interval is active as of a date."""
    cutoff = pd.Timestamp(prediction_date)
    snapshot = nx.MultiDiGraph()
    snapshot.add_nodes_from(graph.nodes(data=True))

    for source, target, key, edge_data in graph.edges(keys=True, data=True):
        start_date = edge_data.get("start_date")
        if start_date is not None and pd.Timestamp(start_date) > cutoff:
            continue
        end_date = edge_data.get("end_date")
        is_current_relationship = edge_data.get("relationship_type") in {"represents"}
        if is_current_relationship and end_date not in {None, ""} and pd.Timestamp(end_date) < cutoff:
            continue
        snapshot.add_edge(source, target, key=key, **edge_data)

    connected_nodes = set()
    for source, target in snapshot.edges():
        connected_nodes.update((source, target))
    artist_nodes = {node for node, data in snapshot.nodes(data=True) if data.get("node_type") == "artist"}
    snapshot.remove_nodes_from(set(snapshot.nodes) - connected_nodes - artist_nodes)
    return snapshot


def _artist_nodes(graph: nx.MultiDiGraph) -> list[tuple[str, dict]]:
    """Return artist nodes from a graph."""
    return [
        (node_id, data)
        for node_id, data in graph.nodes(data=True)
        if data.get("node_type") == "artist"
    ]


def _museum_exhibition_count(
    graph: nx.MultiDiGraph,
    artist_id: str,
    major_only: bool = False,
) -> int:
    """Count museum exhibitions connected to an artist."""
    count = 0
    for _, event_id, edge_data in graph.out_edges(artist_id, data=True):
        if edge_data.get("relationship_type") != "included_in":
            continue
        host_ids = _event_host_institutions(graph, event_id)
        museum_hosts = [
            host_id
            for host_id in host_ids
            if graph.nodes[host_id].get("node_type") == "museum"
        ]
        if major_only:
            museum_hosts = [host_id for host_id in museum_hosts if _is_major_institution(graph.nodes[host_id])]
        if museum_hosts:
            count += 1
    return count


def _museum_acquisition_count(
    graph: nx.MultiDiGraph,
    artist_id: str,
    major_only: bool = False,
) -> int:
    """Count museum acquisitions connected to an artist."""
    count = 0
    for museum_id, _, edge_data in graph.in_edges(artist_id, data=True):
        if edge_data.get("relationship_type") != "acquired_artist":
            continue
        museum_data = graph.nodes[museum_id]
        if museum_data.get("node_type") != "museum":
            continue
        if major_only and not _is_major_institution(museum_data):
            continue
        count += 1
    return count


def _gallery_prestige_score(graph: nx.MultiDiGraph, artist_id: str) -> float:
    """Return the highest current gallery prestige score for an artist."""
    scores = []
    for gallery_id in _represented_by_gallery_ids(graph, artist_id):
        scores.append(float(graph.nodes[gallery_id].get("prestige_score", 0) or 0))
    return max(scores) if scores else 0.0


def _gallery_tier(graph: nx.MultiDiGraph, artist_id: str) -> int:
    """Return the highest current gallery tier rank for an artist."""
    ranks = []
    for gallery_id in _represented_by_gallery_ids(graph, artist_id):
        gallery_data = graph.nodes[gallery_id]
        tier = str(gallery_data.get("tier", "")).lower()
        if tier:
            ranks.append(GALLERY_TIER_RANK.get(tier, 0))
        elif "prestige_score" in gallery_data:
            ranks.append(_prestige_to_tier_rank(float(gallery_data.get("prestige_score") or 0)))
    return max(ranks) if ranks else 0


def _neighbor_degree(
    graph: nx.MultiDiGraph,
    artist_id: str,
    node_type: str,
    relationship_type: str,
) -> int:
    """Count unique incoming neighbors of a type via a relationship."""
    return len(_incoming_neighbors(graph, artist_id, node_type, relationship_type))


def _neighbor_centrality_score(
    graph: nx.MultiDiGraph,
    artist_id: str,
    node_type: str,
    relationship_type: str,
    centrality: dict[str, float],
) -> float:
    """Average centrality score of matching incoming neighbors."""
    neighbors = _incoming_neighbors(graph, artist_id, node_type, relationship_type)
    if not neighbors:
        return 0.0
    return sum(centrality.get(neighbor, 0.0) for neighbor in neighbors) / len(neighbors)


def _art_fair_count(graph: nx.MultiDiGraph, artist_id: str) -> int:
    """Count art fair or biennial event inclusions."""
    count = 0
    for _, event_id, edge_data in graph.out_edges(artist_id, data=True):
        if edge_data.get("relationship_type") != "included_in":
            continue
        event_data = graph.nodes[event_id]
        event_type = str(
            event_data.get("event_type")
            or event_data.get("exhibition_type")
            or event_data.get("institution_type")
            or ""
        ).lower()
        if event_type in ART_FAIR_EVENT_TYPES:
            count += 1
    return count


def _auction_lot_count(
    graph: nx.MultiDiGraph,
    artist_id: str,
    start_date: pd.Timestamp | None,
    end_date: pd.Timestamp,
) -> int:
    """Count auction lots in a date interval."""
    return len(_auction_prices(graph, artist_id, start_date, end_date))


def _auction_median_price(
    graph: nx.MultiDiGraph,
    artist_id: str,
    start_date: pd.Timestamp | None,
    end_date: pd.Timestamp,
) -> float:
    """Return median auction price in a date interval."""
    prices = _auction_prices(graph, artist_id, start_date, end_date)
    if not prices:
        return 0.0
    return float(pd.Series(prices).median())


def _auction_price_growth(
    graph: nx.MultiDiGraph,
    artist_id: str,
    previous_start: pd.Timestamp,
    current_start: pd.Timestamp,
    end_date: pd.Timestamp,
) -> float:
    """Return current-year median auction growth versus the prior year."""
    previous = _auction_median_price(graph, artist_id, previous_start, current_start)
    current = _auction_median_price(graph, artist_id, current_start, end_date)
    if previous <= 0 or current <= 0:
        return 0.0
    return (current - previous) / previous


def _press_mention_count(
    graph: nx.MultiDiGraph,
    artist_id: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> int:
    """Count press mentions in a date interval."""
    total = 0
    for _, press_id, edge_data in graph.out_edges(artist_id, data=True):
        if edge_data.get("relationship_type") != "mentioned_in_press":
            continue
        edge_date = pd.Timestamp(edge_data["start_date"])
        if start_date < edge_date <= end_date:
            total += int(graph.nodes[press_id].get("mentions", graph.nodes[press_id].get("mention_count", 1)) or 0)
    return total


def _press_mention_growth(
    graph: nx.MultiDiGraph,
    artist_id: str,
    previous_start: pd.Timestamp,
    current_start: pd.Timestamp,
    end_date: pd.Timestamp,
) -> float:
    """Return current-year press mention growth versus the prior year."""
    previous = _press_mention_count(graph, artist_id, previous_start, current_start)
    current = _press_mention_count(graph, artist_id, current_start, end_date)
    if previous <= 0:
        return float(current) if current > 0 else 0.0
    return (current - previous) / previous


def _graph_distance(
    graph: nx.MultiDiGraph,
    artist_id: str,
    target_ids: set[str],
) -> int:
    """Return shortest undirected graph distance to any target node."""
    if not target_ids:
        return 0
    undirected = nx.Graph(graph)
    distances = []
    for target_id in target_ids:
        try:
            distances.append(nx.shortest_path_length(undirected, artist_id, target_id))
        except nx.NetworkXNoPath:
            continue
    return min(distances) if distances else 0


def _career_age_years(
    graph: nx.MultiDiGraph,
    artist_id: str,
    artist_data: dict,
    prediction_date: pd.Timestamp,
) -> float:
    """Estimate artist career age from first graph event or birth year fallback."""
    dates = [
        pd.Timestamp(edge_data["start_date"])
        for _, _, edge_data in graph.in_edges(artist_id, data=True)
        if edge_data.get("start_date")
    ]
    dates.extend(
        pd.Timestamp(edge_data["start_date"])
        for _, _, edge_data in graph.out_edges(artist_id, data=True)
        if edge_data.get("start_date")
    )
    if dates:
        return round((prediction_date - min(dates)).days / 365.25, 2)
    birth_year = artist_data.get("birth_year")
    if birth_year and not math.isnan(float(birth_year)):
        return float(prediction_date.year - int(birth_year))
    return 0.0


def _event_host_institutions(graph: nx.MultiDiGraph, event_id: str) -> list[str]:
    """Return institutions hosting an event node."""
    return [
        target_id
        for _, target_id, edge_data in graph.out_edges(event_id, data=True)
        if edge_data.get("relationship_type") == "hosted_by"
    ]


def _represented_by_gallery_ids(graph: nx.MultiDiGraph, artist_id: str) -> list[str]:
    """Return current gallery representation node IDs for an artist."""
    return [
        gallery_id
        for gallery_id, _, edge_data in graph.in_edges(artist_id, data=True)
        if edge_data.get("relationship_type") == "represents"
        and graph.nodes[gallery_id].get("node_type") == "gallery"
    ]


def _incoming_neighbors(
    graph: nx.MultiDiGraph,
    artist_id: str,
    node_type: str,
    relationship_type: str,
) -> set[str]:
    """Return incoming neighbor IDs matching node and relationship types."""
    return {
        source_id
        for source_id, _, edge_data in graph.in_edges(artist_id, data=True)
        if edge_data.get("relationship_type") == relationship_type
        and graph.nodes[source_id].get("node_type") == node_type
    }


def _auction_prices(
    graph: nx.MultiDiGraph,
    artist_id: str,
    start_date: pd.Timestamp | None,
    end_date: pd.Timestamp,
) -> list[float]:
    """Return auction prices in a date interval."""
    prices = []
    for _, auction_id, edge_data in graph.out_edges(artist_id, data=True):
        if edge_data.get("relationship_type") != "has_auction_result":
            continue
        edge_date = pd.Timestamp(edge_data["start_date"])
        if start_date is not None and edge_date <= start_date:
            continue
        if edge_date > end_date:
            continue
        price = graph.nodes[auction_id].get("price_usd")
        if price is not None:
            prices.append(float(price))
    return prices


def _major_institution_nodes(graph: nx.MultiDiGraph) -> set[str]:
    """Return major museum or institution node IDs."""
    return {
        node_id
        for node_id, data in graph.nodes(data=True)
        if data.get("node_type") in {"museum", "institution"}
        and _is_major_institution(data)
    }


def _top_gallery_nodes(graph: nx.MultiDiGraph) -> set[str]:
    """Return top-gallery node IDs."""
    return {
        node_id
        for node_id, data in graph.nodes(data=True)
        if data.get("node_type") == "gallery"
        and (
            str(data.get("tier", "")).lower() in TOP_GALLERY_TIERS
            or float(data.get("prestige_score", 0) or 0) >= 0.85
        )
    }


def _is_major_institution(node_data: dict) -> bool:
    """Return whether a museum or institution should count as major."""
    return bool(node_data.get("is_major")) or str(node_data.get("tier", "")).lower() in MAJOR_INSTITUTION_TIERS


def _prestige_to_tier_rank(prestige_score: float) -> int:
    """Map continuous gallery prestige to a coarse tier rank."""
    if prestige_score >= 0.9:
        return 5
    if prestige_score >= 0.75:
        return 4
    if prestige_score >= 0.55:
        return 3
    if prestige_score > 0:
        return 2
    return 0
