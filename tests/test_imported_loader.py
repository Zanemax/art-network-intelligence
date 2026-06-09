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
    assert "" not in graph
    assert (
        "event_artist_kaye_donachie_i_kept_the_memory_for_myself_22_02_2024_13",
        "gallery_maureen_paley",
    ) in graph.edges()
    auction_nodes = [
        data
        for _, data in graph.nodes(data=True)
        if data.get("node_type") == "auction_result" and data.get("artist_id") == "artist_kaye_donachie"
    ]
    assert auction_nodes
    assert any(
        "Christ" in str(node.get("auction_house")) and float(node.get("price_usd", 0)) > 0 and "Christ" not in str(node.get("title"))
        for node in auction_nodes
    )
