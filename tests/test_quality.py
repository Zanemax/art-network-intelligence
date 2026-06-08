"""Tests for artist and relationship data quality scoring."""

import networkx as nx
import pandas as pd

from src.data.quality import LOW_QUALITY_WARNING, calculate_data_quality, save_data_quality_report
from src.models.explain import explain_artist_prediction


def test_calculate_data_quality_returns_artist_and_relationship_rows() -> None:
    """Verify quality output includes requested fields for artists and relationships."""
    dataset, graph = _quality_fixture()

    quality = calculate_data_quality(dataset, graph, as_of_date="2024-01-01")

    assert {
        "missing_required_fields",
        "stale_data_flag",
        "low_confidence_flag",
        "conflicting_identity_flag",
        "insufficient_history_flag",
        "source_count",
        "average_confidence_score",
    }.issubset(quality.columns)
    assert {"artist", "relationship"}.issubset(set(quality["entity_type"]))


def test_artist_quality_flags_sparse_low_confidence_and_conflicting_identity() -> None:
    """Verify artist quality flags identify sparse or low-confidence data."""
    dataset, graph = _quality_fixture()

    quality = calculate_data_quality(dataset, graph, as_of_date="2024-01-01")
    artist_quality = quality[(quality["entity_type"] == "artist") & (quality["entity_id"] == "artist_a")].iloc[0]

    assert bool(artist_quality["low_confidence_flag"])
    assert bool(artist_quality["conflicting_identity_flag"])
    assert bool(artist_quality["insufficient_history_flag"])
    assert artist_quality["source_count"] == 1
    assert artist_quality["average_confidence_score"] == 0.4


def test_save_data_quality_report_writes_csv(tmp_path) -> None:
    """Verify quality reports can be saved to reports/data_quality.csv."""
    dataset, graph = _quality_fixture()
    output_path = tmp_path / "reports" / "data_quality.csv"

    quality = save_data_quality_report(dataset, graph, output_path=output_path)

    assert output_path.exists()
    saved = pd.read_csv(output_path)
    assert len(saved) == len(quality)


def test_explanation_uses_standard_quality_warning() -> None:
    """Verify poor data quality triggers the required dashboard warning."""
    quality_row = pd.Series(
        {
            "missing_required_fields": False,
            "stale_data_flag": False,
            "low_confidence_flag": True,
            "conflicting_identity_flag": False,
            "insufficient_history_flag": True,
        }
    )
    feature_row = pd.Series({"auction_lot_count": 2, "press_mention_count_1y": 5, "collector_degree": 1})

    explanation = explain_artist_prediction(
        artist_id="artist_a",
        score=0.7,
        feature_row=feature_row,
        quality_row=quality_row,
    )

    assert explanation.data_quality_warning == LOW_QUALITY_WARNING
    assert LOW_QUALITY_WARNING in explanation.explanation_text


def _quality_fixture() -> tuple[dict[str, pd.DataFrame], nx.MultiDiGraph]:
    """Build a small graph with duplicate artist identity and low-confidence edge."""
    dataset = {
        "artists": pd.DataFrame(
            [
                {"artist_id": "artist_a", "name": "Same Name"},
                {"artist_id": "artist_b", "name": "Same Name"},
            ]
        )
    }
    graph = nx.MultiDiGraph()
    graph.add_node("artist_a", node_type="artist", name="Same Name")
    graph.add_node("artist_b", node_type="artist", name="Same Name")
    graph.add_node("gallery_a", node_type="gallery", name="Gallery A")
    graph.add_edge(
        "gallery_a",
        "artist_a",
        source_id="gallery_a",
        target_id="artist_a",
        relationship_type="represents",
        start_date="2023-01-01",
        end_date="",
        source_url="https://example.com/source",
        confidence_score=0.4,
        notes="low confidence",
    )
    return dataset, graph
