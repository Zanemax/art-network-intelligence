"""Premium Streamlit MVP product demo for Art Network Intelligence."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import networkx as nx
import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.app.components.ui import (  # noqa: E402
    driver_list,
    empty_state,
    evidence_grid,
    load_design_system,
    metric_card,
    note_box,
    pill,
    relationship_cards,
    render_header,
    render_pill,
    render_sidebar_brand,
    section_heading,
    score_card,
    table_label,
)
from src.app.components.graph_view import (  # noqa: E402
    SIGNAL_TYPES,
    filter_graph_by_confidence,
    filter_graph_by_date,
    get_artist_ego_network,
    graph_signal_summary,
    highlighted_evidence_nodes,
    render_graph_view,
)
from src.data.quality import LOW_QUALITY_WARNING, calculate_data_quality  # noqa: E402
from src.data.loaders import imported_data_available, load_imported_dataset  # noqa: E402
from src.data.synthetic import load_synthetic_dataset  # noqa: E402
from src.data.validate_schemas import validate_template_directory  # noqa: E402
from src.graph.build_graph import build_investment_graph  # noqa: E402
from src.graph.features import build_artist_graph_features  # noqa: E402
from src.models.explain import explain_artist_prediction  # noqa: E402
from src.models.features import FEATURE_COLUMNS, build_artist_features  # noqa: E402
from src.models.investment_model import train_investment_model  # noqa: E402
from src.models.similarity import find_similar_artists  # noqa: E402


PAGES = [
    "Artist Profile",
    "Evidence",
    "Comparables",
    "Taste Graph Explorer",
]

DEFAULT_GRAPH_NODE_TYPES = {"artist", "gallery", "museum", "collector", "curator", "exhibition", "acquisition"}


def main() -> None:
    """Render the commercial-style product demo."""
    st.set_page_config(page_title="Art Network Intelligence", layout="wide", initial_sidebar_state="expanded")
    load_design_system()

    render_sidebar_brand()
    data_source_options = ["Synthetic demo data"]
    if imported_data_available():
        data_source_options.append("Imported real data")
    data_source = st.sidebar.radio("Data source", data_source_options)
    st.sidebar.markdown("#### Search Artist")
    state = _load_demo_state(data_source)
    dataset = state["dataset"]
    graph = state["graph"]
    predictions = state["predictions"]
    quality = state["quality"]
    scoring_mode = state["scoring_mode"]

    artists = _artist_options(predictions)
    selected_label = st.sidebar.selectbox("Artist universe", artists, label_visibility="collapsed")
    selected = predictions.loc[predictions["artist_id"] == artists[selected_label]].iloc[0]
    as_of_date = st.sidebar.text_input("Prediction date", value="2021-12-31")
    page = st.sidebar.radio("Navigation", PAGES, label_visibility="collapsed")

    render_header(
        page,
        f"A collector-facing research brief focused on thesis, evidence, and comparable outcomes. Source: {data_source}.",
        _last_refresh_timestamp(),
    )

    if scoring_mode != "trained_model":
        note_box("Imported data is sparse, so scores are graph-derived heuristics until enough labeled history exists for model training.", warning=True)

    if page == "Artist Profile":
        _render_artist_brief(dataset, graph, quality, selected, as_of_date)
    elif page == "Evidence":
        _render_evidence(dataset, graph, quality, selected, as_of_date)
    elif page == "Comparables":
        _render_comparables(dataset, graph, selected, as_of_date)
    else:
        _render_taste_graph_explorer(graph, selected, as_of_date)


@st.cache_data(show_spinner=False)
def _load_demo_state(data_source: str = "Synthetic demo data") -> dict[str, object]:
    """Load selected data source, graph, and lightweight demo scores."""
    if data_source == "Imported real data" and imported_data_available():
        dataset = load_imported_dataset()
    else:
        dataset = load_synthetic_dataset()
    graph = build_investment_graph(dataset)
    features = build_artist_features(dataset, graph)
    if features["doubled_in_3_years"].nunique() >= 2:
        result = train_investment_model(features)
        predictions = result.predictions
        scoring_mode = "trained_model"
    else:
        predictions = _heuristic_predictions(features)
        scoring_mode = "graph_heuristic"
    quality = calculate_data_quality(dataset, graph)
    return {
        "dataset": dataset,
        "graph": graph,
        "predictions": predictions,
        "quality": quality,
        "scoring_mode": scoring_mode,
    }


def _heuristic_predictions(features: pd.DataFrame) -> pd.DataFrame:
    """Create probability-like scores when imported data has no training labels."""
    rows = features[["artist_id", "name"] + FEATURE_COLUMNS + ["doubled_in_3_years"]].copy()
    numeric = rows[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    scaled = numeric.copy()
    for column in FEATURE_COLUMNS:
        max_value = float(numeric[column].max())
        scaled[column] = numeric[column] / max_value if max_value > 0 else 0.0
    weights = pd.Series(
        {
            "museum_exhibitions": 0.18,
            "museum_acquisitions": 0.20,
            "gallery_prestige_score": 0.16,
            "collector_centrality_score": 0.12,
            "curator_centrality_score": 0.10,
            "auction_price_growth": 0.12,
            "press_mention_velocity": 0.06,
            "distance_to_top_tier_institution": -0.06,
        }
    )
    score = (scaled * weights).sum(axis=1).clip(lower=0.01, upper=0.99)
    rows["prediction_probability"] = score
    return rows.sort_values("prediction_probability", ascending=False).reset_index(drop=True)


def _render_artist_brief(
    dataset: dict[str, pd.DataFrame],
    graph: nx.MultiDiGraph,
    quality: pd.DataFrame,
    selected: pd.Series,
    as_of_date: str,
) -> None:
    """Render a collector-focused decision brief for one artist."""
    context = _artist_context(dataset, graph, quality, selected, as_of_date)
    artist_id = str(selected["artist_id"])
    artist_name = str(selected["name"])
    feature_row = context["feature_row"]
    probabilities = context["probabilities"]
    explanation = context["explanation"]
    similar = context["similar"]
    artist_quality = context["artist_quality"]

    st.markdown(f"## {artist_name}")
    st.markdown(
        f"{pill('Research brief', 'neutral')} {pill(_institutional_status(feature_row), 'success')} "
        f"{pill(_quality_confidence_label(artist_quality) + ' data confidence', _quality_tone(artist_quality))}",
        unsafe_allow_html=True,
    )

    top_cols = st.columns([0.24, 0.24, 0.26, 0.26])
    with top_cols[0]:
        score_card("Breakout score", f"{probabilities['breakout_probability']:.0%}", "Experimental model output", "Model", "success")
    with top_cols[1]:
        score_card("Confidence", explanation.confidence_level.title(), "Use as research priority, not advice", "Quality", _confidence_tone(explanation.confidence_level))
    with top_cols[2]:
        metric_card("Current gallery", _current_gallery(graph, artist_id, as_of_date), "Representation signal", "neutral")
    with top_cols[3]:
        metric_card("Institutional status", _institutional_status(feature_row), "Museum exposure and acquisitions", "neutral")

    section_heading("1. Why Is This Artist Interesting?", "Decision thesis")
    note_box(str(explanation.explanation_text))
    thesis_cols = st.columns(3)
    with thesis_cols[0]:
        metric_card(
            "Institutional signal",
            f"{int(feature_row['major_museum_exhibition_count']) + int(feature_row['major_museum_acquisition_count'])}",
            "Major museum shows and acquisitions",
            "success",
        )
    with thesis_cols[1]:
        metric_card(
            "Market signal",
            f"{probabilities['market_success_probability']:.0%}",
            "Auction growth, lots, and gallery strength",
            "warning",
        )
    with thesis_cols[2]:
        metric_card(
            "Network signal",
            f"{int(feature_row['collector_degree']) + int(feature_row['curator_degree'])}",
            "Collector and curator connections",
            "neutral",
        )

    section_heading("2. What Evidence Supports The Score?", "Only point-in-time evidence")
    evidence_cols = st.columns([0.45, 0.55])
    with evidence_cols[0]:
        evidence_grid(_collector_evidence_summary(feature_row, artist_quality), limit=6)
    with evidence_cols[1]:
        relationship_cards(_artist_timeline(graph, artist_id, as_of_date), limit=6)

    if explanation.data_quality_warning:
        note_box(explanation.data_quality_warning, warning=True)

    section_heading("Network Evidence", "Relationships creating the strongest signal")
    note_box("This network shows the relationships contributing most to the artist's score.")
    profile_graph_cols = st.columns([0.68, 0.32])
    comparable_ids = similar["artist_id"].head(5).tolist()
    ego_graph = get_artist_ego_network(graph, artist_id, max_nodes=18, comparable_artist_ids=comparable_ids)
    highlights = highlighted_evidence_nodes(ego_graph, artist_id, comparable_ids)
    with profile_graph_cols[0]:
        render_graph_view(ego_graph, selected_node=artist_id, highlighted_nodes=highlights, max_edges=28, height=520)
    with profile_graph_cols[1]:
        st.dataframe(graph_signal_summary(ego_graph, artist_id), width="stretch", hide_index=True)
        table_label("Highlight key")
        st.markdown(
            f"{pill('current gallery', 'neutral')} {pill('museums', 'success')} "
            f"{pill('collectors', 'neutral')} {pill('curators', 'warning')} {pill('comparables', 'danger')}",
            unsafe_allow_html=True,
        )

    section_heading("3. Which Comparable Artists Succeeded Or Failed?", "Later outcomes where available")
    comparables = _comparables_table(similar).head(5)
    if comparables.empty:
        empty_state("No comparable artists available", "Add richer artist histories to generate comparable trajectories.")
    else:
        st.dataframe(comparables, width="stretch", hide_index=True)


def _render_evidence(
    dataset: dict[str, pd.DataFrame],
    graph: nx.MultiDiGraph,
    quality: pd.DataFrame,
    selected: pd.Series,
    as_of_date: str,
) -> None:
    """Render the supporting evidence behind one artist score."""
    context = _artist_context(dataset, graph, quality, selected, as_of_date)
    artist_id = str(selected["artist_id"])
    artist_name = str(selected["name"])
    feature_row = context["feature_row"]
    probabilities = context["probabilities"]
    explanation = context["explanation"]
    artist_quality = context["artist_quality"]

    st.markdown(f"## Evidence for {artist_name}")
    score_cols = st.columns(3)
    with score_cols[0]:
        score_card("Institutional probability", f"{probabilities['institutional_success_probability']:.0%}", "3-year label window", "Forecast", "success")
    with score_cols[1]:
        score_card("Market probability", f"{probabilities['market_success_probability']:.0%}", "Auction and demand signal", "Forecast", "warning")
    with score_cols[2]:
        score_card("Gallery advancement", f"{context['gallery_probability']:.0%}", "Representation upgrade signal", "Forecast", "neutral")

    left, right = st.columns([0.48, 0.52])
    with left:
        section_heading("Positive Evidence", "Drivers that raise conviction")
        driver_list(explanation.top_positive_drivers)
        section_heading("Cautions", "Reasons to keep researching")
        driver_list(explanation.top_negative_drivers, empty_text="No major negative drivers in the current feature view.")
        section_heading("Data Quality", "Can this score be trusted?")
        st.dataframe(_quality_summary(artist_quality), width="stretch", hide_index=True)
        if explanation.data_quality_warning:
            note_box(explanation.data_quality_warning, warning=True)
    with right:
        section_heading("Evidence Summary", "Signals used by the model")
        evidence_grid(_collector_evidence_summary(feature_row, artist_quality), limit=7)
        section_heading("Career Evidence", f"Visible on or before {as_of_date}")
        relationship_cards(_artist_timeline(graph, artist_id, as_of_date), limit=8)

    section_heading("Source Tables", "Detailed records, collapsed by default")
    _render_profile_expanders(dataset, graph, artist_id, as_of_date)


def _render_comparables(
    dataset: dict[str, pd.DataFrame],
    graph: nx.MultiDiGraph,
    selected: pd.Series,
    as_of_date: str,
) -> None:
    """Render comparable artist outcomes for collector diligence."""
    artist_id = str(selected["artist_id"])
    artist_name = str(selected["name"])
    similar = find_similar_artists(graph, dataset, artist_id, as_of_date, top_n=5)

    st.markdown(f"## Comparable Artists for {artist_name}")
    note_box("Comparables are matched on normalized career-trajectory and graph-derived features as of the selected prediction date.")

    cols = st.columns(3)
    for column, (_, row) in zip(cols, similar.head(3).iterrows()):
        with column:
            metric_card(
                str(row["artist_name"]),
                f"{float(row['similarity_score']):.0%}",
                _outcome_summary(row),
                "success" if _has_positive_outcome(row) else "warning",
            )

    section_heading("Succeeded Or Failed?", "Later outcomes attached to each comparable")
    st.dataframe(_comparables_table(similar), width="stretch", hide_index=True)

    section_heading("Why They Are Comparable", "Shared signals, not visual similarity")
    st.dataframe(
        similar[["artist_name", "shared_signals"]].rename(
            columns={"artist_name": "artist", "shared_signals": "shared_signals"}
        ),
        width="stretch",
        hide_index=True,
    )

    section_heading("Side-By-Side Signals", f"Point-in-time features as of {as_of_date}")
    feature_frame = build_artist_graph_features(graph, as_of_date)
    selected_ids = [artist_id] + similar["artist_id"].head(5).tolist()
    side_by_side = feature_frame[feature_frame["artist_id"].isin(selected_ids)].merge(
        dataset["artists"][["artist_id", "name"]],
        on="artist_id",
        how="left",
    )
    display_columns = [
        "name",
        "museum_exhibition_count",
        "major_museum_acquisition_count",
        "gallery_prestige_score",
        "collector_degree",
        "curator_degree",
        "auction_price_growth_1y",
        "press_mention_growth_1y",
        "graph_distance_to_major_institution",
    ]
    st.dataframe(side_by_side[display_columns], width="stretch", hide_index=True)


def _render_taste_graph_explorer(
    graph: nx.MultiDiGraph,
    selected: pd.Series,
    as_of_date: str,
) -> None:
    """Render the full graph explorer as a focused evidence workflow."""
    st.markdown("## Taste Graph Explorer")
    note_box("Default view is an ego-network so the graph stays legible. Use full-network mode only after narrowing filters.")

    node_types = sorted({str(data.get("node_type", "unknown")) for _, data in graph.nodes(data=True)})
    relationship_types = sorted({str(data.get("relationship_type", "unknown")) for _, _, data in graph.edges(data=True)})
    entity_options = _entity_options(graph)
    selected_artist_id = str(selected["artist_id"])
    default_entity_index = list(entity_options.values()).index(selected_artist_id) if selected_artist_id in entity_options.values() else 0

    filter_cols = st.columns([0.23, 0.25, 0.15, 0.14, 0.13, 0.10])
    selected_entity_label = filter_cols[0].selectbox("Search artist/entity", entity_options, index=default_entity_index)
    selected_node_id = entity_options[selected_entity_label]
    default_node_types = [node_type for node_type in node_types if node_type in DEFAULT_GRAPH_NODE_TYPES]
    default_relationships = [relationship for relationship in relationship_types if relationship in SIGNAL_TYPES]
    selected_node_types = filter_cols[1].multiselect("Node type", node_types, default=default_node_types)
    selected_relationships = filter_cols[2].multiselect("Relationship", relationship_types, default=default_relationships)
    min_confidence = filter_cols[3].slider("Confidence", 0.0, 1.0, 0.0, 0.05)
    cutoff = filter_cols[4].text_input("Date cutoff", value=as_of_date)
    max_nodes = filter_cols[5].number_input("Max nodes", min_value=10, max_value=80, value=25, step=5)

    mode = st.radio("Graph mode", ["Ego-network", "Full network"], horizontal=True)
    filtered = filter_graph_by_confidence(filter_graph_by_date(graph, cutoff), min_confidence)
    filtered = _filter_graph_by_types(filtered, set(selected_node_types), set(selected_relationships))

    if mode == "Ego-network":
        visible_graph = get_artist_ego_network(filtered, selected_node_id, max_nodes=int(max_nodes))
    else:
        visible_graph = _limit_graph_nodes(filtered, int(max_nodes), selected_node_id)

    metric_cols = st.columns(4)
    with metric_cols[0]:
        metric_card("Visible nodes", str(visible_graph.number_of_nodes()), "Filtered graph entities", "neutral")
    with metric_cols[1]:
        metric_card("Visible edges", str(visible_graph.number_of_edges()), "Relationships in view", "neutral")
    with metric_cols[2]:
        metric_card("Mode", mode, "Default keeps graph legible", "success" if mode == "Ego-network" else "warning")
    with metric_cols[3]:
        metric_card("Min confidence", f"{min_confidence:.2f}", "Relationship evidence threshold", "neutral")

    graph_cols = st.columns([0.70, 0.30])
    with graph_cols[0]:
        render_graph_view(
            visible_graph,
            selected_node=selected_node_id if selected_node_id in visible_graph else None,
            highlighted_nodes=highlighted_evidence_nodes(visible_graph, selected_node_id),
            max_edges=100,
            height=610,
            show_edge_labels=mode == "Ego-network",
        )
    with graph_cols[1]:
        section_heading("Selected Node", "Entity detail panel")
        st.dataframe(_selected_node_detail(graph, selected_node_id), width="stretch", hide_index=True)
        section_heading("Relationships", "Filtered evidence touching selected node")
        relationship_cards(_selected_node_relationships(visible_graph, selected_node_id), limit=8)


def _render_dashboard(
    dataset: dict[str, pd.DataFrame],
    graph: nx.MultiDiGraph,
    predictions: pd.DataFrame,
    quality: pd.DataFrame,
    as_of_date: str,
) -> None:
    """Render the command-center dashboard page."""
    high_conviction = int((predictions["prediction_probability"] >= 0.65).sum())
    metric_cols = st.columns(5)
    with metric_cols[0]:
        metric_card("Artists tracked", str(len(dataset["artists"])), "Synthetic MVP universe", "neutral")
    with metric_cols[1]:
        metric_card("Galleries tracked", str(len(dataset["galleries"])), "Representation signals", "neutral")
    with metric_cols[2]:
        metric_card("Museums tracked", str(len(dataset["museums"])), "Institutional endpoints", "success")
    with metric_cols[3]:
        metric_card("Active relationships", str(graph.number_of_edges()), "Timestamped graph edges", "neutral")
    with metric_cols[4]:
        metric_card("High-conviction opportunities", str(high_conviction), "Score above 65%", "warning")

    left, right = st.columns([0.58, 0.42])
    with left:
        section_heading("Recent Artist Momentum", "Ranked by current model score")
        momentum = _top_artist_table(predictions, quality)
        st.dataframe(momentum, width="stretch", hide_index=True)

        section_heading("Top Emerging Artists", "Institutional and market indicators")
        emerging = _emerging_artist_table(dataset, graph, predictions, as_of_date)
        st.dataframe(emerging, width="stretch", hide_index=True)

    with right:
        section_heading("Prediction Distribution", "Probability bands")
        st.bar_chart(_prediction_distribution(predictions), x="score_band", y="artist_count", height=250)

        section_heading("Institutional Activity Feed", "Latest visible events")
        st.dataframe(_institutional_feed(graph, limit=8), width="stretch", hide_index=True)


def _render_artist_profile(
    dataset: dict[str, pd.DataFrame],
    graph: nx.MultiDiGraph,
    quality: pd.DataFrame,
    selected: pd.Series,
    as_of_date: str,
) -> None:
    """Render the flagship artist profile."""
    artist_id = str(selected["artist_id"])
    artist_name = str(selected["name"])
    graph_features = build_artist_graph_features(graph, as_of_date)
    feature_row = graph_features.loc[graph_features["artist_id"] == artist_id].iloc[0]
    similar = find_similar_artists(graph, dataset, artist_id, as_of_date, top_n=5)
    artist_quality = _artist_quality_row(quality, artist_id)
    probabilities = _demo_probabilities(float(selected["prediction_probability"]), feature_row)
    gallery_probability = _gallery_advancement_probability(feature_row)
    explanation = explain_artist_prediction(
        artist_id=artist_id,
        score=probabilities["breakout_probability"],
        feature_row=feature_row,
        feature_frame=graph_features.drop(columns=["artist_id"]),
        comparable_artists=similar["artist_name"].head(3).tolist(),
        quality_row=artist_quality,
    )

    st.markdown(f"## {artist_name}")
    profile_cols = st.columns([0.22, 0.22, 0.22, 0.34])
    with profile_cols[0]:
        metric_card("Current score", f"{probabilities['breakout_probability']:.0%}", "Composite breakout signal", "success")
    with profile_cols[1]:
        metric_card("Confidence level", explanation.confidence_level.title(), "Driven by source depth", _confidence_tone(explanation.confidence_level))
    with profile_cols[2]:
        metric_card("Current gallery", _current_gallery(graph, artist_id, as_of_date), "Representation as of date", "neutral")
    with profile_cols[3]:
        metric_card("Institutional status", _institutional_status(feature_row), "Museum exposure and acquisitions", "neutral")

    score_cols = st.columns(3)
    with score_cols[0]:
        metric_card("Institutional Success Probability", f"{probabilities['institutional_success_probability']:.0%}", "3-year forward window", "success")
    with score_cols[1]:
        metric_card("Market Success Probability", f"{probabilities['market_success_probability']:.0%}", "Auction and demand signals", "warning")
    with score_cols[2]:
        metric_card("Gallery Advancement Probability", f"{gallery_probability:.0%}", "Representation upgrade signal", "neutral")

    if explanation.data_quality_warning:
        note_box(explanation.data_quality_warning, warning=True)

    left, center, right = st.columns([0.36, 0.34, 0.30])
    with left:
        section_heading("Key Drivers", "Positive contributors")
        driver_list(explanation.top_positive_drivers)
        section_heading("Risk Factors", "Signals lowering conviction")
        driver_list(explanation.top_negative_drivers, empty_text="No major negative drivers in the current feature view.")
    with center:
        section_heading("Career Timeline", f"As of {as_of_date}")
        st.dataframe(_artist_timeline(graph, artist_id, as_of_date).head(12), width="stretch", hide_index=True)
    with right:
        section_heading("Network Summary", "Collector, curator, and institution proximity")
        st.dataframe(_network_summary(feature_row), width="stretch")
        section_heading("Data Quality Indicators", "Coverage and confidence")
        st.dataframe(_quality_summary(artist_quality), width="stretch", hide_index=True)

    section_heading("Similar Artists", "Career trajectory and graph-derived feature match")
    st.dataframe(
        similar[["artist_name", "similarity_score", "later_outcome_summary", "shared_signals"]],
        width="stretch",
        hide_index=True,
    )

    section_heading("Relationship Network", "Highlighted ego-network")
    ego_graph = graph.subgraph(_ego_nodes(graph, artist_id)).copy()
    render_interactive_graph(ego_graph, selected_node=artist_id, max_edges=80, height=620)

    _render_profile_expanders(dataset, graph, artist_id, as_of_date)


def _render_similar_artists(
    dataset: dict[str, pd.DataFrame],
    graph: nx.MultiDiGraph,
    selected: pd.Series,
    as_of_date: str,
) -> None:
    """Render similar artist search as a due-diligence workflow."""
    st.markdown(f"## Similar Artists to {selected['name']}")
    top_n = st.slider("Comparison set", 3, 10, 5)
    similar = find_similar_artists(graph, dataset, str(selected["artist_id"]), as_of_date, top_n=top_n)

    cols = st.columns(min(3, max(1, len(similar))))
    for column, (_, row) in zip(cols, similar.head(3).iterrows()):
        with column:
            metric_card(
                str(row["artist_name"]),
                f"{float(row['similarity_score']):.0%}",
                str(row["later_outcome_summary"]),
                "success" if float(row["similarity_score"]) >= 0.75 else "neutral",
            )

    section_heading("Outcome Comparison", "Later outcomes are shown only where available")
    comparison = similar[
        [
            "artist_name",
            "similarity_score",
            "later_outcome_summary",
            "institutional_success_3y",
            "market_success_3y",
            "gallery_success_3y",
        ]
    ]
    st.dataframe(comparison, width="stretch", hide_index=True)

    section_heading("Shared Characteristics", "Closest normalized feature values")
    st.dataframe(
        similar[["artist_name", "shared_signals"]].rename(
            columns={"artist_name": "artist", "shared_signals": "shared_characteristics"}
        ),
        width="stretch",
        hide_index=True,
    )

    section_heading("Side-by-Side Feature Comparison", f"As of {as_of_date}")
    feature_frame = build_artist_graph_features(graph, as_of_date)
    selected_ids = [str(selected["artist_id"])] + similar["artist_id"].head(top_n).tolist()
    side_by_side = feature_frame[feature_frame["artist_id"].isin(selected_ids)].merge(
        dataset["artists"][["artist_id", "name"]],
        on="artist_id",
        how="left",
    )
    st.dataframe(side_by_side, width="stretch", hide_index=True)


def _render_graph_explorer(graph: nx.MultiDiGraph, selected_artist_id: str) -> None:
    """Render the interactive graph explorer."""
    st.markdown("## Taste Graph Explorer")
    node_types = sorted({str(data.get("node_type", "unknown")) for _, data in graph.nodes(data=True)})
    relationship_types = sorted({str(data.get("relationship_type", "unknown")) for _, _, data in graph.edges(data=True)})

    filters = st.columns([0.24, 0.28, 0.18, 0.18, 0.12])
    selected_node_types = filters[0].multiselect("Node type", node_types, default=node_types)
    selected_relationships = filters[1].multiselect("Relationship type", relationship_types, default=relationship_types)
    cutoff = filters[2].text_input("Date cutoff", value="2024-12-31")
    min_confidence = filters[3].slider("Min confidence", 0.0, 1.0, 0.0, 0.05)
    highlight_ego = filters[4].checkbox("Ego", value=True, help="Highlight selected artist ego-network")

    filtered = _filter_graph(graph, set(selected_node_types), set(selected_relationships), cutoff, min_confidence)
    selected_node = selected_artist_id if selected_artist_id in filtered.nodes else None
    if highlight_ego and selected_node:
        filtered = filtered.subgraph(_ego_nodes(filtered, selected_node)).copy()

    metric_cols = st.columns(4)
    metric_cols[0].metric("Visible Nodes", filtered.number_of_nodes())
    metric_cols[1].metric("Visible Edges", filtered.number_of_edges())
    metric_cols[2].metric("Node Types", len(selected_node_types))
    metric_cols[3].metric("Relationship Types", len(selected_relationships))

    left, right = st.columns([0.72, 0.28])
    with left:
        render_interactive_graph(filtered, selected_node=selected_node, max_edges=140, height=690)
    with right:
        section_heading("Selection Panel", "Filtered network context")
        selected_data = graph.nodes[selected_artist_id] if selected_artist_id in graph.nodes else {}
        render_pill(str(selected_data.get("node_type", "artist")).title(), "neutral")
        st.markdown(f"### {selected_data.get('name', selected_artist_id)}")
        st.dataframe(_selected_node_relationships(filtered, selected_artist_id), width="stretch", hide_index=True)


def _render_model_performance() -> None:
    """Render saved model, baseline, and backtest performance."""
    st.markdown("## Model Performance")
    metrics_path = REPO_ROOT / "reports" / "metrics.json"
    predictions_path = REPO_ROOT / "reports" / "predictions.csv"
    backtest_path = REPO_ROOT / "reports" / "backtest_results.csv"

    metrics = _load_metrics(metrics_path)
    if metrics.empty:
        note_box("Run temporal training to generate reports/metrics.json.")
    else:
        best = metrics.sort_values("roc_auc", ascending=False).iloc[0]
        cols = st.columns(4)
        cols[0].metric("ROC AUC", f"{float(best.get('roc_auc', 0)):.2f}", str(best["model"]))
        cols[1].metric("Precision at K", f"{float(best.get('precision_at_k', 0)):.2f}")
        cols[2].metric("Recall at K", f"{float(best.get('recall_at_k', 0)):.2f}")
        cols[3].metric("Average Precision", f"{float(best.get('average_precision', 0)):.2f}")
        section_heading("Model Comparison", "Temporal holdout metrics and baselines")
        st.dataframe(metrics, width="stretch", hide_index=True)
        chart_metrics = metrics[["model", "roc_auc", "average_precision"]].set_index("model")
        st.bar_chart(chart_metrics, height=280)

    if backtest_path.exists():
        backtests = pd.read_csv(backtest_path)
        section_heading("Backtest Performance", "Annual point-in-time tests")
        st.dataframe(backtests, width="stretch", hide_index=True)

    if predictions_path.exists():
        section_heading("Saved Predictions", "Most recent training output")
        st.dataframe(pd.read_csv(predictions_path).head(100), width="stretch", hide_index=True)


def _render_data_quality(dataset: dict[str, pd.DataFrame], graph: nx.MultiDiGraph, quality: pd.DataFrame) -> None:
    """Render data quality, coverage, source confidence, and validation warnings."""
    st.markdown("## Data Quality")
    template_results = validate_template_directory(REPO_ROOT / "data" / "raw" / "templates")
    issue_count = sum(len(errors) for errors in template_results.values())
    edge_confidence = pd.Series([float(data.get("confidence_score", 0)) for _, _, data in graph.edges(data=True)])
    source_urls = [str(data.get("source_url", "")) for _, _, data in graph.edges(data=True) if data.get("source_url")]

    cols = st.columns(5)
    cols[0].metric("Raw Templates", len(template_results))
    cols[1].metric("Validation Warnings", issue_count)
    cols[2].metric("Tables Covered", len(dataset))
    cols[3].metric("Avg Confidence", f"{edge_confidence.mean():.2f}")
    cols[4].metric("Unique Sources", len(set(source_urls)))

    if issue_count:
        warning_rows = [(name, "; ".join(errors)) for name, errors in template_results.items() if errors]
        note_box("Validation warnings require review before using real-data predictions.", warning=True)
        st.dataframe(pd.DataFrame(warning_rows, columns=["file", "issues"]), width="stretch", hide_index=True)
    else:
        note_box("Raw CSV templates match the registered schemas.")

    left, right = st.columns(2)
    with left:
        section_heading("Coverage Metrics", "Rows by table")
        st.dataframe(
            pd.DataFrame({"table": list(dataset), "rows": [len(frame) for frame in dataset.values()]}),
            width="stretch",
            hide_index=True,
        )
        section_heading("Missing Data Indicators", "Artist and relationship checks")
        st.dataframe(quality, width="stretch", hide_index=True)
    with right:
        section_heading("Confidence Score Distribution", "Relationship source confidence")
        confidence_bins = pd.cut(edge_confidence, bins=[0, 0.5, 0.7, 0.85, 1.0], include_lowest=True).value_counts().sort_index()
        st.bar_chart(confidence_bins.rename_axis("confidence_band").reset_index(name="relationship_count"), x="confidence_band", y="relationship_count", height=260)
        section_heading("Data Source Statistics", "Evidence depth")
        st.dataframe(_source_statistics(graph), width="stretch", hide_index=True)

    poor_artists = quality[
        (quality["entity_type"] == "artist")
        & (
            quality[
                [
                    "missing_required_fields",
                    "stale_data_flag",
                    "low_confidence_flag",
                    "conflicting_identity_flag",
                    "insufficient_history_flag",
                ]
            ].any(axis=1)
        )
    ]
    if not poor_artists.empty:
        note_box(LOW_QUALITY_WARNING, warning=True)


def _render_profile_expanders(
    dataset: dict[str, pd.DataFrame],
    graph: nx.MultiDiGraph,
    artist_id: str,
    as_of_date: str,
) -> None:
    """Render detailed artist evidence tables in expandable sections."""
    with st.expander("Exhibitions", expanded=False):
        table_label("Museum and exhibition evidence")
        st.dataframe(_artist_events(dataset.get("exhibitions", pd.DataFrame()), artist_id, "date", as_of_date), width="stretch", hide_index=True)
    with st.expander("Acquisitions", expanded=False):
        table_label("Museum acquisition evidence")
        st.dataframe(_artist_events(dataset.get("acquisitions", pd.DataFrame()), artist_id, "date", as_of_date), width="stretch", hide_index=True)
    with st.expander("Auction history", expanded=False):
        table_label("Auction price history")
        st.dataframe(_artist_events(dataset.get("auction_results", pd.DataFrame()), artist_id, "sale_date", as_of_date), width="stretch", hide_index=True)
    with st.expander("Press coverage", expanded=False):
        table_label("Press mention history")
        press = dataset.get("press_mentions", pd.DataFrame())
        rows = press[press["artist_id"] == artist_id].sort_values("year", ascending=False) if not press.empty else press
        st.dataframe(rows, width="stretch", hide_index=True)
    with st.expander("Relationship network", expanded=False):
        table_label("Timestamped graph relationships")
        st.dataframe(_artist_timeline(graph, artist_id, as_of_date), width="stretch", hide_index=True)


def _artist_context(
    dataset: dict[str, pd.DataFrame],
    graph: nx.MultiDiGraph,
    quality: pd.DataFrame,
    selected: pd.Series,
    as_of_date: str,
) -> dict[str, object]:
    """Build the artist-specific context shared across the decision pages."""
    artist_id = str(selected["artist_id"])
    graph_features = build_artist_graph_features(graph, as_of_date)
    feature_row = graph_features.loc[graph_features["artist_id"] == artist_id].iloc[0]
    similar = find_similar_artists(graph, dataset, artist_id, as_of_date, top_n=5)
    artist_quality = _artist_quality_row(quality, artist_id)
    probabilities = _demo_probabilities(float(selected["prediction_probability"]), feature_row)
    explanation = explain_artist_prediction(
        artist_id=artist_id,
        score=probabilities["breakout_probability"],
        feature_row=feature_row,
        feature_frame=graph_features.drop(columns=["artist_id"]),
        comparable_artists=similar["artist_name"].head(3).tolist(),
        quality_row=artist_quality,
    )
    return {
        "feature_row": feature_row,
        "similar": similar,
        "artist_quality": artist_quality,
        "probabilities": probabilities,
        "gallery_probability": _gallery_advancement_probability(feature_row),
        "explanation": explanation,
    }


def _collector_evidence_summary(feature_row: pd.Series, quality_row: pd.Series | None) -> pd.DataFrame:
    """Summarize the evidence in terms a collector can scan quickly."""
    rows = [
        {
            "evidence": "Museum exposure",
            "value": str(int(feature_row.get("museum_exhibition_count", 0))),
            "why_it_matters": "Shows institutional validation before the prediction date.",
        },
        {
            "evidence": "Major museum acquisitions",
            "value": str(int(feature_row.get("major_museum_acquisition_count", 0))),
            "why_it_matters": "Signals durable institutional demand.",
        },
        {
            "evidence": "Gallery strength",
            "value": f"{float(feature_row.get('gallery_prestige_score', 0)):.2f}",
            "why_it_matters": "Representation affects access, pricing, and placement.",
        },
        {
            "evidence": "Collector / curator links",
            "value": str(int(feature_row.get("collector_degree", 0)) + int(feature_row.get("curator_degree", 0))),
            "why_it_matters": "Network proximity can precede institutional or market movement.",
        },
        {
            "evidence": "Auction growth",
            "value": f"{float(feature_row.get('auction_price_growth_1y', 0)):.2f}x",
            "why_it_matters": "Measures recent secondary-market price acceleration.",
        },
        {
            "evidence": "Press momentum",
            "value": f"{float(feature_row.get('press_mention_growth_1y', 0)):.2f}x",
            "why_it_matters": "Captures rising attention that may support further diligence.",
        },
        {
            "evidence": "Data confidence",
            "value": _quality_confidence_label(quality_row),
            "why_it_matters": "Low-confidence evidence should reduce reliance on the score.",
        },
    ]
    return pd.DataFrame(rows)


def _comparables_table(similar: pd.DataFrame) -> pd.DataFrame:
    """Return comparables with readable outcome labels."""
    if similar.empty:
        return pd.DataFrame(columns=["artist", "similarity", "outcome", "shared_signals"])
    rows = similar.copy()
    rows["artist"] = rows["artist_name"]
    rows["similarity"] = rows["similarity_score"].map(lambda value: f"{float(value):.0%}")
    rows["outcome"] = rows.apply(_outcome_summary, axis=1)
    return rows[["artist", "similarity", "outcome", "shared_signals"]]


def _outcome_summary(row: pd.Series) -> str:
    """Summarize whether a comparable later succeeded or failed."""
    outcomes = []
    if int(row.get("institutional_success_3y", 0)) == 1:
        outcomes.append("institutional success")
    if int(row.get("market_success_3y", 0)) == 1:
        outcomes.append("market success")
    if int(row.get("gallery_success_3y", 0)) == 1:
        outcomes.append("gallery success")
    if outcomes:
        return "Succeeded: " + ", ".join(outcomes)
    return "No 3-year success outcome recorded"


def _has_positive_outcome(row: pd.Series) -> bool:
    """Return whether a comparable has at least one positive future outcome."""
    return any(
        int(row.get(column, 0)) == 1
        for column in ["institutional_success_3y", "market_success_3y", "gallery_success_3y"]
    )


def _quality_tone(quality_row: pd.Series | None) -> str:
    """Map data quality into visual tone."""
    label = _quality_confidence_label(quality_row)
    if label == "High":
        return "success"
    if label == "Medium":
        return "warning"
    return "danger"


def _demo_probabilities(breakout_probability: float, feature_row: pd.Series) -> dict[str, float]:
    """Create product-demo probability views from point-in-time signals."""
    institutional_raw = (
        0.18 * float(feature_row["museum_exhibition_count"])
        + 0.24 * float(feature_row["major_museum_exhibition_count"])
        + 0.22 * float(feature_row["major_museum_acquisition_count"])
        + 0.12 * max(0, 3 - float(feature_row["graph_distance_to_major_institution"]))
    )
    market_raw = (
        0.22 * float(feature_row["auction_lot_count"])
        + 0.28 * float(feature_row["auction_price_growth_1y"])
        + 0.18 * float(feature_row["gallery_prestige_score"])
        + 0.08 * float(feature_row["press_mention_growth_1y"])
    )
    return {
        "breakout_probability": _clamp_probability(breakout_probability),
        "institutional_success_probability": _clamp_probability(institutional_raw),
        "market_success_probability": _clamp_probability(market_raw),
    }


def _gallery_advancement_probability(feature_row: pd.Series) -> float:
    """Estimate gallery advancement probability for the product demo."""
    raw = (
        0.32 * float(feature_row.get("gallery_prestige_score", 0))
        + 0.16 * float(feature_row.get("press_mention_growth_1y", 0))
        + 0.10 * max(0, 3 - float(feature_row.get("graph_distance_to_top_gallery", 3)))
    )
    return _clamp_probability(raw)


def _clamp_probability(value: float) -> float:
    """Clamp a demo score into probability range."""
    return max(0.01, min(0.99, float(value)))


def _current_gallery(graph: nx.MultiDiGraph, artist_id: str, as_of_date: str) -> str:
    """Return current gallery representation as of a date."""
    cutoff = pd.Timestamp(as_of_date)
    galleries = []
    for gallery_id, _, edge_data in graph.in_edges(artist_id, data=True):
        if edge_data.get("relationship_type") != "represents":
            continue
        if pd.Timestamp(edge_data["start_date"]) > cutoff:
            continue
        end_date = edge_data.get("end_date")
        if end_date and pd.Timestamp(end_date) < cutoff:
            continue
        data = graph.nodes[gallery_id]
        galleries.append((pd.Timestamp(edge_data["start_date"]), data.get("name", gallery_id)))
    if not galleries:
        return "Unrepresented"
    return str(sorted(galleries)[-1][1])


def _artist_timeline(graph: nx.MultiDiGraph, artist_id: str, as_of_date: str) -> pd.DataFrame:
    """Build an artist career timeline from temporal edges."""
    cutoff = pd.Timestamp(as_of_date)
    rows = []
    for source, target, edge_data in graph.edges(data=True):
        if artist_id not in {source, target}:
            continue
        event_date = pd.Timestamp(edge_data["start_date"])
        if event_date > cutoff:
            continue
        other_id = target if source == artist_id else source
        other_data = graph.nodes[other_id]
        rows.append(
            {
                "date": event_date.strftime("%Y-%m-%d"),
                "relationship": edge_data.get("relationship_type"),
                "counterparty": other_data.get("name") or other_data.get("title") or other_id,
                "counterparty_type": other_data.get("node_type"),
                "confidence": float(edge_data.get("confidence_score", 0)),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["date", "relationship", "counterparty", "counterparty_type", "confidence"])
    return pd.DataFrame(rows).sort_values("date", ascending=False)


def _filter_graph(
    graph: nx.MultiDiGraph,
    node_types: set[str],
    relationship_types: set[str],
    cutoff: str,
    min_confidence: float,
) -> nx.MultiDiGraph:
    """Filter graph by node type, relationship type, cutoff date, and confidence."""
    cutoff_ts = pd.Timestamp(cutoff)
    filtered = nx.MultiDiGraph()
    for node_id, data in graph.nodes(data=True):
        if data.get("node_type") in node_types:
            filtered.add_node(node_id, **data)
    for source, target, data in graph.edges(data=True):
        if source not in filtered or target not in filtered:
            continue
        if data.get("relationship_type") not in relationship_types:
            continue
        if pd.Timestamp(data.get("start_date")) > cutoff_ts:
            continue
        if float(data.get("confidence_score", 0)) < min_confidence:
            continue
        filtered.add_edge(source, target, **data)
    return filtered


def _artist_quality_row(quality: pd.DataFrame, artist_id: str) -> pd.Series | None:
    """Return one artist quality row if available."""
    rows = quality[(quality["entity_type"] == "artist") & (quality["entity_id"] == artist_id)]
    if rows.empty:
        return None
    return rows.iloc[0]


def _artist_options(predictions: pd.DataFrame) -> dict[str, str]:
    """Return display labels mapped to stable artist IDs."""
    return {
        f"{row['name']}  ·  {row['artist_id']}": str(row["artist_id"])
        for _, row in predictions.sort_values("name").iterrows()
    }


def _top_artist_table(predictions: pd.DataFrame, quality: pd.DataFrame) -> pd.DataFrame:
    """Build the dashboard artist ranking table."""
    rows = predictions.sort_values("prediction_probability", ascending=False).copy()
    rows["score"] = rows["prediction_probability"].map(lambda value: f"{float(value):.0%}")
    rows["confidence"] = rows["artist_id"].map(lambda artist_id: _quality_confidence_label(_artist_quality_row(quality, artist_id)))
    return rows[["artist_id", "name", "score", "confidence", "doubled_in_3_years"]].head(10)


def _emerging_artist_table(
    dataset: dict[str, pd.DataFrame],
    graph: nx.MultiDiGraph,
    predictions: pd.DataFrame,
    as_of_date: str,
) -> pd.DataFrame:
    """Build a compact emerging artist table from graph features."""
    features = build_artist_graph_features(graph, as_of_date)
    display_features = features[
        [
            "artist_id",
            "gallery_prestige_score",
            "major_museum_exhibition_count",
            "press_mention_growth_1y",
        ]
    ]
    rows = predictions.drop(
        columns=["gallery_prestige_score"],
        errors="ignore",
    ).merge(display_features, on="artist_id", how="left")
    rows = rows.merge(dataset["artists"][["artist_id", "region"]], on="artist_id", how="left")
    rows["breakout_probability"] = rows["prediction_probability"].map(lambda value: f"{float(value):.0%}")
    return rows[
        [
            "name",
            "region",
            "breakout_probability",
            "gallery_prestige_score",
            "major_museum_exhibition_count",
            "press_mention_growth_1y",
        ]
    ].sort_values("breakout_probability", ascending=False).head(8)


def _prediction_distribution(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return prediction counts by score band."""
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
    bands = pd.cut(predictions["prediction_probability"], bins=bins, labels=labels, include_lowest=True)
    return bands.value_counts().sort_index().rename_axis("score_band").reset_index(name="artist_count")


def _institutional_feed(graph: nx.MultiDiGraph, limit: int = 8) -> pd.DataFrame:
    """Build a recent institutional activity feed from graph edges."""
    rows = []
    institutional_relationships = {"included_in", "acquired_artist", "organized_by"}
    for source, target, data in graph.edges(data=True):
        if data.get("relationship_type") not in institutional_relationships:
            continue
        source_data = graph.nodes[source]
        target_data = graph.nodes[target]
        rows.append(
            {
                "date": data.get("start_date"),
                "relationship": data.get("relationship_type"),
                "source": source_data.get("name") or source_data.get("title") or source,
                "target": target_data.get("name") or target_data.get("title") or target,
                "confidence": float(data.get("confidence_score", 0)),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["date", "relationship", "source", "target", "confidence"])
    return pd.DataFrame(rows).sort_values("date", ascending=False).head(limit)


def _network_summary(feature_row: pd.Series) -> pd.DataFrame:
    """Return profile network summary metrics."""
    metrics = [
        "collector_degree",
        "collector_centrality_score",
        "curator_degree",
        "curator_centrality_score",
        "graph_distance_to_major_institution",
        "graph_distance_to_top_gallery",
    ]
    return feature_row[metrics].to_frame("value")


def _quality_summary(quality_row: pd.Series | None) -> pd.DataFrame:
    """Return artist-level quality indicators."""
    if quality_row is None:
        return pd.DataFrame([{"indicator": "quality_record", "value": "Not available"}])
    fields = [
        "missing_required_fields",
        "stale_data_flag",
        "low_confidence_flag",
        "conflicting_identity_flag",
        "insufficient_history_flag",
        "source_count",
        "average_confidence_score",
    ]
    return pd.DataFrame({"indicator": fields, "value": [quality_row.get(field) for field in fields]})


def _institutional_status(feature_row: pd.Series) -> str:
    """Summarize institutional standing."""
    major_exhibitions = int(feature_row.get("major_museum_exhibition_count", 0))
    major_acquisitions = int(feature_row.get("major_museum_acquisition_count", 0))
    if major_acquisitions:
        return "Major museum acquisition"
    if major_exhibitions:
        return "Major museum exposure"
    if int(feature_row.get("museum_exhibition_count", 0)):
        return "Museum exhibited"
    return "Early institutional record"


def _confidence_tone(confidence: str) -> str:
    """Map confidence text to a visual tone."""
    value = str(confidence).lower()
    if value == "high":
        return "success"
    if value == "medium":
        return "warning"
    return "danger"


def _quality_confidence_label(quality_row: pd.Series | None) -> str:
    """Create a human-readable data confidence label."""
    if quality_row is None:
        return "Unknown"
    if bool(quality_row.get("low_confidence_flag")) or bool(quality_row.get("insufficient_history_flag")):
        return "Low"
    if float(quality_row.get("average_confidence_score", 0)) >= 0.85:
        return "High"
    return "Medium"


def _artist_events(frame: pd.DataFrame, artist_id: str, date_column: str, as_of_date: str) -> pd.DataFrame:
    """Filter artist event rows as of a date."""
    if frame.empty or "artist_id" not in frame.columns:
        return frame
    rows = frame[frame["artist_id"] == artist_id].copy()
    if date_column in rows.columns:
        rows = rows[pd.to_datetime(rows[date_column]) <= pd.Timestamp(as_of_date)]
        rows = rows.sort_values(date_column, ascending=False)
    return rows


def _selected_node_relationships(graph: nx.MultiDiGraph, node_id: str) -> pd.DataFrame:
    """Return relationships touching the selected node in the filtered graph."""
    rows = []
    if node_id not in graph:
        return pd.DataFrame(columns=["direction", "relationship", "counterparty", "date", "confidence"])
    for source, target, data in graph.in_edges(node_id, data=True):
        rows.append(_relationship_row(graph, "inbound", source, data))
    for source, target, data in graph.out_edges(node_id, data=True):
        rows.append(_relationship_row(graph, "outbound", target, data))
    return pd.DataFrame(rows).head(16)


def _entity_options(graph: nx.MultiDiGraph) -> dict[str, str]:
    """Return searchable graph entity labels mapped to node IDs."""
    labels = {}
    for node_id, data in sorted(graph.nodes(data=True), key=lambda item: str(item[1].get("name") or item[1].get("title") or item[0])):
        node_type = str(data.get("node_type", "unknown"))
        label = data.get("name") or data.get("title") or node_id
        labels[f"{label}  ·  {node_type}"] = node_id
    return labels


def _filter_graph_by_types(
    graph: nx.MultiDiGraph,
    node_types: set[str],
    relationship_types: set[str],
) -> nx.MultiDiGraph:
    """Filter graph by selected node and relationship types."""
    filtered = nx.MultiDiGraph()
    for node_id, data in graph.nodes(data=True):
        if data.get("node_type") in node_types:
            filtered.add_node(node_id, **data)
    for source, target, data in graph.edges(data=True):
        if source not in filtered or target not in filtered:
            continue
        if data.get("relationship_type") in relationship_types:
            filtered.add_edge(source, target, **data)
    return filtered


def _limit_graph_nodes(graph: nx.MultiDiGraph, max_nodes: int, selected_node_id: str) -> nx.MultiDiGraph:
    """Limit full-network mode to a readable number of relevant nodes."""
    if graph.number_of_nodes() <= max_nodes:
        return graph.copy()
    undirected = nx.Graph(graph)
    centrality = nx.degree_centrality(undirected) if undirected.number_of_nodes() > 1 else {}
    ranked = sorted(graph.nodes(), key=lambda node_id: centrality.get(node_id, 0), reverse=True)
    selected_nodes = [selected_node_id] if selected_node_id in graph else []
    for node_id in ranked:
        if node_id not in selected_nodes:
            selected_nodes.append(node_id)
        if len(selected_nodes) >= max_nodes:
            break
    return graph.subgraph(selected_nodes).copy()


def _selected_node_detail(graph: nx.MultiDiGraph, node_id: str) -> pd.DataFrame:
    """Return concise selected-node metadata for the explorer panel."""
    if node_id not in graph:
        return pd.DataFrame([{"field": "node", "value": "Not found"}])
    data = graph.nodes[node_id]
    fields = ["node_type", "name", "title", "region", "tier", "prestige_score", "birth_year"]
    rows = [{"field": "node_id", "value": node_id}]
    rows.extend(
        {"field": field, "value": data.get(field)}
        for field in fields
        if field in data and pd.notna(data.get(field))
    )
    return pd.DataFrame(rows)


def _relationship_row(graph: nx.MultiDiGraph, direction: str, counterparty_id: str, data: dict[str, object]) -> dict[str, object]:
    """Create a table row for one graph relationship."""
    counterparty = graph.nodes[counterparty_id]
    return {
        "direction": direction,
        "relationship": data.get("relationship_type"),
        "counterparty": counterparty.get("name") or counterparty.get("title") or counterparty_id,
        "date": data.get("start_date"),
        "confidence": float(data.get("confidence_score", 0)),
    }


def _source_statistics(graph: nx.MultiDiGraph) -> pd.DataFrame:
    """Summarize relationship source evidence."""
    rows = []
    for _, _, data in graph.edges(data=True):
        rows.append(
            {
                "relationship_type": data.get("relationship_type", "unknown"),
                "source_url": data.get("source_url", ""),
                "confidence": float(data.get("confidence_score", 0)),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["relationship_type", "relationship_count", "avg_confidence", "source_count"])
    return (
        frame.groupby("relationship_type")
        .agg(
            relationship_count=("relationship_type", "size"),
            avg_confidence=("confidence", "mean"),
            source_count=("source_url", pd.Series.nunique),
        )
        .reset_index()
        .sort_values("relationship_count", ascending=False)
    )


def _load_metrics(metrics_path: Path) -> pd.DataFrame:
    """Load saved model metrics into a display table."""
    if not metrics_path.exists():
        return pd.DataFrame()
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if "models" not in metrics:
        return pd.DataFrame()
    frame = pd.DataFrame(metrics["models"]).T.reset_index(names="model")
    return frame


def _last_refresh_timestamp() -> str:
    """Return a stable last-refresh timestamp from local data and report files."""
    paths = (
        list((REPO_ROOT / "data" / "synthetic").glob("*.csv"))
        + list((REPO_ROOT / "data" / "raw" / "imported").glob("*.csv"))
        + list((REPO_ROOT / "reports").glob("*"))
    )
    existing = [path for path in paths if path.exists()]
    if not existing:
        return "Not available"
    latest = max(path.stat().st_mtime for path in existing)
    return pd.Timestamp(latest, unit="s").strftime("%Y-%m-%d %H:%M")


def _full_graph_to_dot(graph: nx.MultiDiGraph, max_edges: int = 80) -> str:
    """Convert a filtered graph to a compact Graphviz DOT view."""
    node_styles = _node_styles()
    lines = ["digraph G {", "rankdir=LR;", "node [style=filled, fontname=Helvetica];"]
    visible_nodes = set()
    for index, (source, target, edge_data) in enumerate(graph.edges(data=True)):
        if index >= max_edges:
            break
        visible_nodes.update((source, target))
        label = edge_data.get("relationship_type", "")
        lines.append(f'"{source}" -> "{target}" [label="{label}"];')
    for node_id in visible_nodes:
        node_data = graph.nodes[node_id]
        label = node_data.get("name") or node_data.get("title") or node_id
        color, shape = node_styles.get(node_data.get("node_type"), ("#cccccc", "ellipse"))
        lines.append(f'"{node_id}" [label="{label}", fillcolor="{color}", shape="{shape}"];')
    lines.append("}")
    return "\n".join(lines)


def _graph_to_dot(graph: nx.MultiDiGraph, selected_artist_id: str) -> str:
    """Convert a compact artist ego graph to Graphviz DOT for tests and fallback views."""
    node_styles = _node_styles()
    selected_nodes = _ego_nodes(graph, selected_artist_id)
    lines = ["digraph G {", "rankdir=LR;", "node [style=filled, fontname=Helvetica];"]

    for node_id in selected_nodes:
        node_data = graph.nodes[node_id]
        label = node_data.get("name") or node_data.get("title") or node_id
        color, shape = node_styles.get(node_data.get("node_type"), ("#cccccc", "ellipse"))
        pen_width = "3" if node_id == selected_artist_id else "1"
        lines.append(
            f'"{node_id}" [label="{label}", fillcolor="{color}", shape="{shape}", penwidth="{pen_width}"];'
        )

    for source, target, edge_data in graph.edges(data=True):
        if source not in selected_nodes or target not in selected_nodes:
            continue
        label = edge_data.get("relationship_type", "")
        lines.append(f'"{source}" -> "{target}" [label="{label}"];')

    lines.append("}")
    return "\n".join(lines)


def _ego_nodes(graph: nx.MultiDiGraph, selected_artist_id: str) -> set[str]:
    """Return the selected artist and directly connected graph nodes."""
    nodes = {selected_artist_id}
    if selected_artist_id in graph:
        nodes.update(graph.predecessors(selected_artist_id))
        nodes.update(graph.successors(selected_artist_id))
    return nodes


def _node_styles() -> dict[str, tuple[str, str]]:
    """Return visual styles for graph node types."""
    return {
        "artist": ("#2f4a46", "ellipse"),
        "gallery": ("#7a5b2f", "box"),
        "museum": ("#3f654f", "box"),
        "collector": ("#6d5a78", "diamond"),
        "curator": ("#8a6d32", "diamond"),
        "exhibition": ("#68717a", "oval"),
        "acquisition": ("#536f7a", "oval"),
        "auction_result": ("#6a665e", "oval"),
        "press_mentions": ("#8a6d32", "oval"),
    }


if __name__ == "__main__":
    main()
