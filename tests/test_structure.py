"""Smoke tests for the Universal Graph investment model MVP."""

from src.app.streamlit_app import (
    _artist_exhibition_cards,
    _artist_press_cards,
    _artist_timeline,
    _default_prediction_date,
    _graph_to_dot,
)
from src.data.loaders import load_imported_dataset
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


def test_imported_data_default_prediction_date_uses_latest_evidence() -> None:
    """Verify imported research defaults to the latest available evidence date."""
    dataset = load_imported_dataset()
    expected = max(
        pd.to_datetime(dataset["exhibitions"]["date"], errors="coerce").max(),
        pd.to_datetime(dataset["auction_results"]["sale_date"], errors="coerce").max(),
        pd.Timestamp(f"{int(pd.to_numeric(dataset['press_mentions']['year'], errors='coerce').max())}-12-31"),
    ).strftime("%Y-%m-%d")

    assert _default_prediction_date(dataset, "Imported real data") == expected
    assert _default_prediction_date(dataset, "Synthetic demo data") == "2021-12-31"


def test_imported_exhibition_cards_show_uploaded_exhibitions() -> None:
    """Verify uploaded exhibitions are exposed as first-class evidence cards."""
    dataset = load_imported_dataset()
    graph = build_investment_graph(dataset)

    cards = _artist_exhibition_cards(dataset, graph, "artist_kaye_donachie", "2026-01-17")

    assert "I kept the memory for myself" in set(cards["counterparty"])
    assert "Into The Thousand Mirrors" in set(cards["counterparty"])
    assert "Maureen Paley" in set(cards["counterparty_type"])
    assert not cards["detail"].str.contains("Venue:").any()
    assert cards["detail"].str.contains("Dates: 2024-02-22 to 2024-03-30").any()
    assert cards["detail"].str.contains("Source:").any()
    assert cards["detail"].str.contains("Confidence:").any()


def test_imported_press_cards_show_rich_mention_data() -> None:
    """Verify uploaded press mentions expose article metadata as evidence cards."""
    dataset = load_imported_dataset()

    cards = _artist_press_cards(dataset, "artist_kaye_donachie", "2026-01-17")

    assert "Kaye Donachie, Maureen Paley" in set(cards["counterparty"])
    assert "Artforum" in set(cards["counterparty_type"])
    assert not cards["detail"].str.contains("Outlet:").any()
    assert cards["detail"].str.contains("Author: Andrew Hunt").any()
    assert cards["detail"].str.contains("Attention: 1").any()
    assert cards["detail"].str.contains("Sentiment: 0.7").any()
    assert cards["detail"].str.contains("Source:").any()
    assert cards["detail"].str.contains("Confidence:").any()


def test_imported_press_timeline_uses_article_titles() -> None:
    """Verify career evidence does not fall back to generic artist-year press labels."""
    dataset = load_imported_dataset()
    graph = build_investment_graph(dataset)

    timeline = _artist_timeline(graph, "artist_pavlo_kerestey", "2026-12-31")
    press_rows = timeline[timeline["relationship"] == "Press mention"]

    assert "Did a group of artist eerily predict the Ukrainian crisis?" in set(press_rows["counterparty"])
    assert "CNN" in set(press_rows["counterparty_type"])
    assert "Press mention - Pavlo Kerestey 2014" not in set(press_rows["counterparty"])
    assert not press_rows["detail"].str.contains("Outlet:").any()
    assert press_rows["detail"].str.contains("Author: Jake Wallis Simons").any()


def test_timeline_cards_do_not_repeat_context_in_titles() -> None:
    """Verify card titles do not duplicate outlet or auction metadata."""
    dataset = load_imported_dataset()
    graph = build_investment_graph(dataset)

    press_timeline = _artist_timeline(graph, "artist_wolfgang_tillmans", "2026-12-31")
    ft_rows = press_timeline[press_timeline["counterparty"].str.contains("Pompidou", regex=False)]

    assert not ft_rows.empty
    assert "Financial Times" in set(ft_rows["counterparty_type"])
    assert not ft_rows["counterparty"].str.contains("Financial Times", regex=False).any()

    auction_timeline = _artist_timeline(graph, "artist_kaye_donachie", "2026-12-31")
    auction_rows = auction_timeline[auction_timeline["relationship"] == "Auction result"]

    assert "Song For The Last Act" in set(auction_rows["counterparty"])
    assert not auction_rows["counterparty"].str.contains("Christie", regex=False).any()
    assert auction_rows["detail"].str.contains("Price:").any()
