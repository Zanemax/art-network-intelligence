"""Build NetworkX graphs from art market entities and events.

The investment MVP graph connects artists to galleries, museums, collectors,
curators, exhibitions, acquisitions, and auction events. Nodes carry a
``node_type`` attribute, while every edge carries temporal relationship
metadata for snapshot reasoning.
"""

from __future__ import annotations

from collections.abc import Mapping

import networkx as nx
import pandas as pd

from src.data.synthetic import load_synthetic_dataset


TEMPORAL_EDGE_COLUMNS = (
    "source_id",
    "target_id",
    "relationship_type",
    "start_date",
    "end_date",
    "source_url",
    "confidence_score",
    "notes",
)


def build_investment_graph(
    dataset: dict[str, pd.DataFrame] | None = None,
    cutoff_date: str | None = None,
) -> nx.MultiDiGraph:
    """Create a typed art investment graph from source tables.

    When ``cutoff_date`` is provided, only relationship and event edges whose
    start date is on or before the cutoff are included.
    """
    dataset = dataset or load_synthetic_dataset()
    graph = nx.MultiDiGraph()

    _add_entity_nodes(graph, dataset["artists"], "artist_id", "artist")
    _add_entity_nodes(graph, dataset["galleries"], "gallery_id", "gallery")
    _add_entity_nodes(graph, dataset["museums"], "museum_id", "museum")
    _add_entity_nodes(graph, dataset["collectors"], "collector_id", "collector")
    _add_entity_nodes(graph, dataset["curators"], "curator_id", "curator")

    for record in dataset["representation"].to_dict(orient="records"):
        _add_temporal_edge(
            graph,
            source_id=record["gallery_id"],
            target_id=record["artist_id"],
            relationship_type="represents",
            start_date=record["start_date"],
            end_date=record.get("end_date", ""),
            record=record,
            cutoff_date=cutoff_date,
        )

    for record in dataset["exhibitions"].to_dict(orient="records"):
        if not _included_as_of(record["date"], cutoff_date):
            continue
        exhibition_id = record["exhibition_id"]
        graph.add_node(exhibition_id, node_type="exhibition", **record)
        _add_temporal_edge(
            graph,
            source_id=record["artist_id"],
            target_id=exhibition_id,
            relationship_type="included_in",
            start_date=record["date"],
            end_date=record.get("end_date", record["date"]),
            record=record,
            cutoff_date=cutoff_date,
        )
        _add_temporal_edge(
            graph,
            source_id=exhibition_id,
            target_id=record["institution_id"],
            relationship_type="hosted_by",
            start_date=record["date"],
            end_date=record.get("end_date", record["date"]),
            record=record,
            cutoff_date=cutoff_date,
        )
        if record.get("curator_id"):
            _add_temporal_edge(
                graph,
                source_id=record["curator_id"],
                target_id=exhibition_id,
                relationship_type="curated",
                start_date=record["date"],
                end_date=record.get("end_date", record["date"]),
                record=record,
                cutoff_date=cutoff_date,
            )
            _add_temporal_edge(
                graph,
                source_id=record["curator_id"],
                target_id=record["artist_id"],
                relationship_type="curated_artist",
                start_date=record["date"],
                end_date=record.get("end_date", record["date"]),
                record=record,
                cutoff_date=cutoff_date,
            )

    for record in dataset["acquisitions"].to_dict(orient="records"):
        if not _included_as_of(record["date"], cutoff_date):
            continue
        acquisition_id = record["acquisition_id"]
        graph.add_node(acquisition_id, node_type="acquisition", **record)
        _add_temporal_edge(
            graph,
            source_id=record["museum_id"],
            target_id=acquisition_id,
            relationship_type="made_acquisition",
            start_date=record["date"],
            end_date=record.get("end_date", record["date"]),
            record=record,
            cutoff_date=cutoff_date,
        )
        _add_temporal_edge(
            graph,
            source_id=acquisition_id,
            target_id=record["artist_id"],
            relationship_type="acquired_work_by",
            start_date=record["date"],
            end_date=record.get("end_date", record["date"]),
            record=record,
            cutoff_date=cutoff_date,
        )
        _add_temporal_edge(
            graph,
            source_id=record["museum_id"],
            target_id=record["artist_id"],
            relationship_type="acquired_artist",
            start_date=record["date"],
            end_date=record.get("end_date", record["date"]),
            record=record,
            cutoff_date=cutoff_date,
        )

    for record in dataset["collector_holdings"].to_dict(orient="records"):
        _add_temporal_edge(
            graph,
            source_id=record["collector_id"],
            target_id=record["artist_id"],
            relationship_type="collects",
            start_date=record["date"],
            end_date=record.get("end_date", ""),
            record=record,
            cutoff_date=cutoff_date,
        )

    for index, record in enumerate(dataset["auction_results"].to_dict(orient="records"), start=1):
        if not _included_as_of(record["sale_date"], cutoff_date):
            continue
        auction_id = record.get("auction_result_id") or f"auction_{index:03d}_{record['artist_id']}"
        graph.add_node(auction_id, node_type="auction_result", title=_auction_result_title(record), **record)
        _add_temporal_edge(
            graph,
            source_id=record["artist_id"],
            target_id=auction_id,
            relationship_type="has_auction_result",
            start_date=record["sale_date"],
            end_date=record.get("end_date", record["sale_date"]),
            record=record,
            cutoff_date=cutoff_date,
        )

    for record in dataset["press_mentions"].to_dict(orient="records"):
        press_date = _press_mention_date(record)
        if not _included_as_of(press_date, cutoff_date):
            continue
        press_id = _press_mention_id(record)
        graph.add_node(press_id, node_type="press_mentions", title=_press_mention_title(record), **record)
        _add_temporal_edge(
            graph,
            source_id=record["artist_id"],
            target_id=press_id,
            relationship_type="mentioned_in_press",
            start_date=press_date,
            end_date=record.get("end_date", press_date),
            record=record,
            cutoff_date=cutoff_date,
        )

    return graph


def _auction_result_title(record: Mapping[str, object]) -> str:
    """Return a readable label for an auction result node."""
    work_title = str(record.get("work_title") or "").strip()
    sale_name = str(record.get("sale_name") or "").strip()
    return work_title or sale_name or "Auction result"


def _press_mention_date(record: Mapping[str, object]) -> str:
    """Return the best available date for a press mention."""
    publication_date = str(record.get("publication_date") or "").strip()
    if publication_date:
        return publication_date
    date = str(record.get("date") or "").strip()
    if date:
        return date
    return f"{record['year']}-12-31"


def _press_mention_id(record: Mapping[str, object]) -> str:
    """Return a stable press node ID without collapsing distinct articles."""
    press_mention_id = str(record.get("press_mention_id") or "").strip()
    if press_mention_id:
        return press_mention_id
    return f"press_{record['artist_id']}_{record['year']}"


def _press_mention_title(record: Mapping[str, object]) -> str:
    """Return a readable label for article-level or aggregate press data."""
    article_title = str(record.get("article_title") or "").strip()
    outlet = str(record.get("outlet_name") or "").strip()
    year = str(record.get("year") or "").strip()
    mentions = record.get("mentions", record.get("mention_count", ""))

    if article_title:
        return article_title
    if outlet and year:
        return f"{outlet} press mention ({year})"
    try:
        mention_total = int(float(mentions))
    except (TypeError, ValueError):
        mention_total = 0
    if mention_total > 1 and year:
        return f"{mention_total} press mentions in {year}"
    if year:
        return f"Press mention in {year}"
    return "Press mention"


def build_graph_as_of(cutoff_date: str) -> nx.MultiDiGraph:
    """Build a cumulative synthetic investment graph as of ``cutoff_date``."""
    return build_investment_graph(cutoff_date=cutoff_date)


def build_artwork_graph(artworks: pd.DataFrame) -> nx.Graph:
    """Create a simple artwork graph from a DataFrame of artwork metadata."""
    graph = nx.Graph()

    for record in artworks.to_dict(orient="records"):
        artwork_id = record["artwork_id"]
        graph.add_node(artwork_id, node_type="artwork", **record)

        artist = record.get("artist")
        if artist:
            graph.add_node(artist, node_type="artist")
            graph.add_edge(artwork_id, artist, relationship="created_by")

        genre = record.get("genre")
        if genre:
            graph.add_node(genre, node_type="genre")
            graph.add_edge(artwork_id, genre, relationship="belongs_to")

    return graph


def _add_entity_nodes(
    graph: nx.MultiDiGraph,
    frame: pd.DataFrame,
    id_column: str,
    node_type: str,
) -> None:
    """Add typed entity nodes from a DataFrame."""
    for record in frame.to_dict(orient="records"):
        node_id = record[id_column]
        graph.add_node(node_id, node_type=node_type, **record)


def _add_temporal_edge(
    graph: nx.MultiDiGraph,
    source_id: str,
    target_id: str,
    relationship_type: str,
    start_date: str,
    end_date: str,
    record: Mapping[str, object],
    cutoff_date: str | None,
) -> None:
    """Add an edge with the canonical temporal relationship attributes."""
    if not source_id or not target_id:
        return
    if not _included_as_of(start_date, cutoff_date):
        return

    source_url = str(record.get("source_url") or _default_source_url(relationship_type, source_id, target_id))
    confidence_score = float(record.get("confidence_score") or 0.75)
    notes = str(record.get("notes") or "")

    graph.add_edge(
        source_id,
        target_id,
        source_id=source_id,
        target_id=target_id,
        relationship_type=relationship_type,
        relationship=relationship_type,
        start_date=start_date,
        end_date=end_date or "",
        source_url=source_url,
        confidence_score=confidence_score,
        notes=notes,
    )


def _included_as_of(start_date: str, cutoff_date: str | None) -> bool:
    """Return whether a relationship/event started on or before the cutoff."""
    if cutoff_date is None:
        return True
    return pd.Timestamp(start_date) <= pd.Timestamp(cutoff_date)


def _default_source_url(relationship_type: str, source_id: str, target_id: str) -> str:
    """Create a deterministic placeholder source URL for synthetic edges."""
    return f"https://example.com/synthetic/{relationship_type}/{source_id}/{target_id}"
