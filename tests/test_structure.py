"""Smoke tests for the Art Taste Graph investment model MVP."""

from src.app.streamlit_app import _graph_to_dot
from src.data.synthetic import load_synthetic_dataset
import pandas as pd

from src.graph.build_graph import TEMPORAL_EDGE_COLUMNS, build_graph_as_of, build_investment_graph
from src.models.features import FEATURE_COLUMNS, build_artist_features
from src.models.investment_model import train_investment_model


def test_synthetic_dataset_contains_required_tables() -> None:
    """Verify the fixture includes the requested entity and event tables."""
    dataset = load_synthetic_dataset()

    assert set(dataset) >= {
        "artists",
        "galleries",
        "museums",
        "collectors",
        "curators",
        "exhibitions",
        "acquisitions",
        "auction_results",
    }
    assert len(dataset["artists"]) >= 6


def test_investment_graph_has_typed_nodes_and_dated_edges() -> None:
    """Verify synthetic market data can produce a temporal relationship graph."""
    graph = build_investment_graph(load_synthetic_dataset())

    assert graph.number_of_nodes() > 0
    assert graph.number_of_edges() > 0
    assert {data["node_type"] for _, data in graph.nodes(data=True)} >= {
        "artist",
        "gallery",
        "museum",
        "collector",
        "curator",
        "exhibition",
        "acquisition",
        "auction_result",
    }
    assert all(
        set(TEMPORAL_EDGE_COLUMNS).issubset(data)
        for _, _, data in graph.edges(data=True)
    )
    assert all(
        data["source_id"] == source and data["target_id"] == target
        for source, target, data in graph.edges(data=True)
    )


def test_build_graph_as_of_excludes_future_relationships() -> None:
    """Verify snapshot graphs exclude relationships after the cutoff date."""
    cutoff_date = "2023-12-31"
    full_graph = build_investment_graph(load_synthetic_dataset())
    snapshot_graph = build_graph_as_of(cutoff_date)

    assert any(
        pd.Timestamp(data["start_date"]) > pd.Timestamp(cutoff_date)
        for _, _, data in full_graph.edges(data=True)
    )
    assert all(
        pd.Timestamp(data["start_date"]) <= pd.Timestamp(cutoff_date)
        for _, _, data in snapshot_graph.edges(data=True)
    )
    assert "acq_002" in full_graph
    assert "acq_002" not in snapshot_graph
    assert "auction_002_artist_ada_rios" in full_graph
    assert "auction_002_artist_ada_rios" not in snapshot_graph


def test_artist_features_and_model_predictions_are_available() -> None:
    """Verify feature extraction and model scoring produce artist probabilities."""
    dataset = load_synthetic_dataset()
    graph = build_investment_graph(dataset)
    features = build_artist_features(dataset, graph)
    result = train_investment_model(features)

    assert set(FEATURE_COLUMNS).issubset(features.columns)
    assert features["doubled_in_3_years"].nunique() == 2
    assert result.predictions["prediction_probability"].between(0, 1).all()
    assert result.feature_importance["importance"].sum() > 0


def test_dashboard_graph_dot_includes_selected_artist() -> None:
    """Verify the dashboard graph renderer exposes a selected artist ego graph."""
    dataset = load_synthetic_dataset()
    graph = build_investment_graph(dataset)
    dot = _graph_to_dot(graph, "artist_ada_rios")

    assert "Ada Rios" in dot
    assert "represents" in dot
