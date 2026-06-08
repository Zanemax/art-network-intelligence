"""Tests for point-in-time artist features from temporal graphs."""

import networkx as nx
import pytest

from src.graph.features import ARTIST_GRAPH_FEATURE_COLUMNS, build_artist_graph_features


def test_artist_graph_features_use_only_edges_before_prediction_date() -> None:
    """Verify future relationships and events are excluded from feature values."""
    graph = _temporal_feature_graph()

    features = build_artist_graph_features(graph, prediction_date="2024-01-01")
    row = features.set_index("artist_id").loc["artist_a"]

    assert list(features.columns) == ARTIST_GRAPH_FEATURE_COLUMNS
    assert row["museum_exhibition_count"] == 1
    assert row["major_museum_exhibition_count"] == 1
    assert row["museum_acquisition_count"] == 1
    assert row["major_museum_acquisition_count"] == 1
    assert row["gallery_prestige_score"] == 0.5
    assert row["gallery_tier"] == 2
    assert row["collector_degree"] == 1
    assert row["curator_degree"] == 1
    assert row["art_fair_count"] == 1
    assert row["auction_lot_count"] == 2
    assert row["auction_median_price"] == 125.0
    assert row["auction_price_growth_1y"] == pytest.approx(0.5)
    assert row["press_mention_count_1y"] == 10
    assert row["press_mention_growth_1y"] == pytest.approx(1.5)
    assert row["graph_distance_to_major_institution"] == 1
    assert row["graph_distance_to_top_gallery"] == 0
    assert row["career_age_years"] > 3


def test_artist_graph_features_include_future_values_after_prediction_date_moves() -> None:
    """Verify later prediction dates can include relationships once observable."""
    graph = _temporal_feature_graph()

    features = build_artist_graph_features(graph, prediction_date="2025-12-31")
    row = features.set_index("artist_id").loc["artist_a"]

    assert row["museum_exhibition_count"] == 2
    assert row["museum_acquisition_count"] == 2
    assert row["gallery_prestige_score"] == 0.95
    assert row["gallery_tier"] == 4
    assert row["collector_degree"] == 2
    assert row["curator_degree"] == 2
    assert row["auction_lot_count"] == 3


def _temporal_feature_graph() -> nx.MultiDiGraph:
    """Build a compact temporal graph with past and future artist signals."""
    graph = nx.MultiDiGraph()
    graph.add_node("artist_a", node_type="artist", birth_year=1990)
    graph.add_node("museum_major", node_type="museum", tier="top", prestige_score=1.0)
    graph.add_node("museum_mid", node_type="museum", tier="mid", prestige_score=0.7)
    graph.add_node("gallery_mid", node_type="gallery", tier="mid", prestige_score=0.5)
    graph.add_node("gallery_top", node_type="gallery", tier="top", prestige_score=0.95)
    graph.add_node("collector_past", node_type="collector")
    graph.add_node("collector_future", node_type="collector")
    graph.add_node("curator_past", node_type="curator")
    graph.add_node("curator_future", node_type="curator")
    graph.add_node("exhibition_past", node_type="exhibition", event_type="solo_show")
    graph.add_node("exhibition_future", node_type="exhibition", event_type="solo_show")
    graph.add_node("fair_past", node_type="exhibition", event_type="art_fair")
    graph.add_node("acquisition_past", node_type="acquisition")
    graph.add_node("acquisition_future", node_type="acquisition")
    graph.add_node("auction_previous", node_type="auction_result", price_usd=100)
    graph.add_node("auction_current", node_type="auction_result", price_usd=150)
    graph.add_node("auction_future", node_type="auction_result", price_usd=1000)
    graph.add_node("press_previous", node_type="press_mentions", mentions=4)
    graph.add_node("press_current", node_type="press_mentions", mentions=10)
    graph.add_node("press_future", node_type="press_mentions", mentions=20)

    _add_edge(graph, "gallery_mid", "artist_a", "represents", "2020-01-01")
    _add_edge(graph, "gallery_top", "artist_a", "represents", "2025-01-01")
    _add_edge(graph, "artist_a", "exhibition_past", "included_in", "2023-02-01")
    _add_edge(graph, "exhibition_past", "museum_major", "hosted_by", "2023-02-01")
    _add_edge(graph, "artist_a", "exhibition_future", "included_in", "2024-02-01")
    _add_edge(graph, "exhibition_future", "museum_mid", "hosted_by", "2024-02-01")
    _add_edge(graph, "artist_a", "fair_past", "included_in", "2023-03-01")
    _add_edge(graph, "fair_past", "gallery_mid", "hosted_by", "2023-03-01")
    _add_edge(graph, "museum_major", "acquisition_past", "made_acquisition", "2023-04-01")
    _add_edge(graph, "acquisition_past", "artist_a", "acquired_work_by", "2023-04-01")
    _add_edge(graph, "museum_major", "artist_a", "acquired_artist", "2023-04-01")
    _add_edge(graph, "museum_mid", "acquisition_future", "made_acquisition", "2024-04-01")
    _add_edge(graph, "acquisition_future", "artist_a", "acquired_work_by", "2024-04-01")
    _add_edge(graph, "museum_mid", "artist_a", "acquired_artist", "2024-04-01")
    _add_edge(graph, "collector_past", "artist_a", "collects", "2023-05-01")
    _add_edge(graph, "collector_future", "artist_a", "collects", "2024-05-01")
    _add_edge(graph, "curator_past", "artist_a", "curated_artist", "2023-06-01")
    _add_edge(graph, "curator_future", "artist_a", "curated_artist", "2024-06-01")
    _add_edge(graph, "artist_a", "auction_previous", "has_auction_result", "2022-06-01")
    _add_edge(graph, "artist_a", "auction_current", "has_auction_result", "2023-06-01")
    _add_edge(graph, "artist_a", "auction_future", "has_auction_result", "2024-06-01")
    _add_edge(graph, "artist_a", "press_previous", "mentioned_in_press", "2022-06-01")
    _add_edge(graph, "artist_a", "press_current", "mentioned_in_press", "2023-06-01")
    _add_edge(graph, "artist_a", "press_future", "mentioned_in_press", "2024-06-01")
    return graph


def _add_edge(
    graph: nx.MultiDiGraph,
    source_id: str,
    target_id: str,
    relationship_type: str,
    start_date: str,
) -> None:
    """Add a temporal test edge."""
    graph.add_edge(
        source_id,
        target_id,
        source_id=source_id,
        target_id=target_id,
        relationship_type=relationship_type,
        start_date=start_date,
        end_date="",
        source_url=f"https://example.com/{relationship_type}/{source_id}/{target_id}",
        confidence_score=1.0,
        notes="test fixture",
    )
