"""Tests for loading normalized imported data into the MVP graph schema."""

from src.data.loaders import imported_data_available, load_imported_dataset
from src.graph.build_graph import build_investment_graph


def test_imported_data_can_be_adapted_to_graph_schema() -> None:
    """Verify normalized imported CSVs can drive the existing graph path."""
    assert imported_data_available()

    dataset = load_imported_dataset()
    graph = build_investment_graph(dataset)

    assert {"artists", "galleries", "representation", "auction_results"}.issubset(dataset)
    assert not dataset["artists"].empty
    assert {"artist_id", "name"}.issubset(dataset["artists"].columns)
    assert graph.number_of_nodes() >= len(dataset["artists"])
    assert graph.number_of_edges() > 0
