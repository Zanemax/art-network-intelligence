"""Tests for graph evidence filtering and compact ego-network helpers."""

import pandas as pd

from src.app.components.graph_view import (
    filter_graph_by_confidence,
    filter_graph_by_date,
    get_artist_ego_network,
    get_top_signal_edges,
)
from src.data.synthetic import load_synthetic_dataset
from src.graph.build_graph import build_investment_graph


def test_filter_graph_by_date_excludes_future_edges() -> None:
    """Verify graph evidence respects the selected cutoff date."""
    graph = build_investment_graph(load_synthetic_dataset())
    filtered = filter_graph_by_date(graph, "2023-12-31")

    assert all(
        pd.Timestamp(data["start_date"]) <= pd.Timestamp("2023-12-31")
        for _, _, data in filtered.edges(data=True)
    )
    assert "acq_002" not in filtered


def test_filter_graph_by_confidence_excludes_low_confidence_edges() -> None:
    """Verify confidence filtering removes weak relationship evidence."""
    graph = build_investment_graph(load_synthetic_dataset())
    filtered = filter_graph_by_confidence(graph, 0.80)

    assert filtered.number_of_edges() < graph.number_of_edges()
    assert all(float(data["confidence_score"]) >= 0.80 for _, _, data in filtered.edges(data=True))


def test_get_artist_ego_network_limits_nodes_and_keeps_artist() -> None:
    """Verify profile graph defaults to a compact ego-network."""
    graph = build_investment_graph(load_synthetic_dataset())
    ego_graph = get_artist_ego_network(graph, "artist_ada_rios", max_nodes=12)

    assert "artist_ada_rios" in ego_graph
    assert ego_graph.number_of_nodes() <= 12
    assert ego_graph.number_of_edges() > 0


def test_get_top_signal_edges_prioritizes_score_relevant_relationships() -> None:
    """Verify strongest graph evidence uses relationship signal weights."""
    graph = build_investment_graph(load_synthetic_dataset())
    edges = get_top_signal_edges(graph, "artist_ada_rios", limit=5)
    relationship_types = [data["relationship_type"] for _, _, data in edges]

    assert len(edges) == 5
    assert any(relationship in relationship_types for relationship in {"acquired_artist", "included_in", "represents"})
