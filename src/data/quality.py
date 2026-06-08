"""Data quality scoring for artist and relationship records."""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import pandas as pd


QUALITY_COLUMNS = [
    "entity_type",
    "entity_id",
    "missing_required_fields",
    "stale_data_flag",
    "low_confidence_flag",
    "conflicting_identity_flag",
    "insufficient_history_flag",
    "source_count",
    "average_confidence_score",
]

LOW_QUALITY_WARNING = "Prediction confidence is low because this artist has sparse or low-confidence source data."


def calculate_data_quality(
    dataset: dict[str, pd.DataFrame],
    graph: nx.MultiDiGraph,
    as_of_date: str = "2024-12-31",
    stale_after_years: int = 3,
    low_confidence_threshold: float = 0.6,
    min_artist_history_edges: int = 3,
) -> pd.DataFrame:
    """Calculate quality flags for each artist and relationship edge."""
    rows = []
    artist_names = _artist_names(dataset)
    duplicate_names = {
        name
        for name, count in pd.Series(list(artist_names.values())).value_counts().items()
        if count > 1
    }

    for artist_id, node_data in graph.nodes(data=True):
        if node_data.get("node_type") != "artist":
            continue
        rows.append(
            _artist_quality_row(
                graph,
                artist_id,
                artist_names,
                duplicate_names,
                as_of_date,
                stale_after_years,
                low_confidence_threshold,
                min_artist_history_edges,
            )
        )

    rows.extend(
        _relationship_quality_row(
            source,
            target,
            key,
            data,
            as_of_date,
            stale_after_years,
            low_confidence_threshold,
        )
        for source, target, key, data in graph.edges(keys=True, data=True)
    )

    return pd.DataFrame(rows, columns=QUALITY_COLUMNS)


def save_data_quality_report(
    dataset: dict[str, pd.DataFrame],
    graph: nx.MultiDiGraph,
    output_path: str | Path = "reports/data_quality.csv",
    as_of_date: str = "2024-12-31",
) -> pd.DataFrame:
    """Calculate and save data quality results to CSV."""
    quality = calculate_data_quality(dataset, graph, as_of_date=as_of_date)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    quality.to_csv(path, index=False)
    return quality


def artist_quality_warning(quality_row: pd.Series | None) -> str:
    """Return the standard warning when an artist quality row is poor."""
    if quality_row is None or quality_row.empty:
        return ""
    is_poor = any(
        bool(quality_row.get(flag, False))
        for flag in (
            "missing_required_fields",
            "stale_data_flag",
            "low_confidence_flag",
            "conflicting_identity_flag",
            "insufficient_history_flag",
        )
    )
    return LOW_QUALITY_WARNING if is_poor else ""


def _artist_quality_row(
    graph: nx.MultiDiGraph,
    artist_id: str,
    artist_names: dict[str, str],
    duplicate_names: set[str],
    as_of_date: str,
    stale_after_years: int,
    low_confidence_threshold: float,
    min_artist_history_edges: int,
) -> dict[str, object]:
    """Build one artist quality row."""
    edge_data = _artist_edges(graph, artist_id)
    confidences = [float(data.get("confidence_score", 0)) for data in edge_data]
    source_urls = {data.get("source_url") for data in edge_data if data.get("source_url")}
    latest_date = _latest_edge_date(edge_data)

    return {
        "entity_type": "artist",
        "entity_id": artist_id,
        "missing_required_fields": not bool(artist_names.get(artist_id)),
        "stale_data_flag": _is_stale(latest_date, as_of_date, stale_after_years),
        "low_confidence_flag": _average(confidences) < low_confidence_threshold,
        "conflicting_identity_flag": artist_names.get(artist_id) in duplicate_names,
        "insufficient_history_flag": len(edge_data) < min_artist_history_edges,
        "source_count": len(source_urls),
        "average_confidence_score": round(_average(confidences), 4),
    }


def _relationship_quality_row(
    source: str,
    target: str,
    key: int,
    data: dict,
    as_of_date: str,
    stale_after_years: int,
    low_confidence_threshold: float,
) -> dict[str, object]:
    """Build one relationship quality row."""
    required_fields = [
        "source_id",
        "target_id",
        "relationship_type",
        "start_date",
        "source_url",
        "confidence_score",
    ]
    missing_required = any(not data.get(field) and data.get(field) != 0 for field in required_fields)
    confidence = float(data.get("confidence_score", 0) or 0)
    relationship_id = f"{source}|{target}|{data.get('relationship_type')}|{key}"
    return {
        "entity_type": "relationship",
        "entity_id": relationship_id,
        "missing_required_fields": missing_required,
        "stale_data_flag": _is_stale(pd.Timestamp(data.get("start_date")), as_of_date, stale_after_years),
        "low_confidence_flag": confidence < low_confidence_threshold,
        "conflicting_identity_flag": data.get("source_id") not in {None, source} or data.get("target_id") not in {None, target},
        "insufficient_history_flag": False,
        "source_count": 1 if data.get("source_url") else 0,
        "average_confidence_score": round(confidence, 4),
    }


def _artist_edges(graph: nx.MultiDiGraph, artist_id: str) -> list[dict]:
    """Return edge data connected to an artist."""
    edges = []
    edges.extend(data for _, _, data in graph.in_edges(artist_id, data=True))
    edges.extend(data for _, _, data in graph.out_edges(artist_id, data=True))
    return edges


def _artist_names(dataset: dict[str, pd.DataFrame]) -> dict[str, str]:
    """Return artist names keyed by ID."""
    artists = dataset.get("artists", pd.DataFrame())
    if artists.empty or "artist_id" not in artists.columns:
        return {}
    name_column = "name" if "name" in artists.columns else "canonical_name"
    if name_column not in artists.columns:
        return {}
    return artists.set_index("artist_id")[name_column].fillna("").astype(str).to_dict()


def _latest_edge_date(edge_data: list[dict]) -> pd.Timestamp | None:
    """Return the latest available edge start date."""
    dates = [pd.Timestamp(data["start_date"]) for data in edge_data if data.get("start_date")]
    return max(dates) if dates else None


def _is_stale(
    latest_date: pd.Timestamp | None,
    as_of_date: str,
    stale_after_years: int,
) -> bool:
    """Return whether a record is stale as of the selected date."""
    if latest_date is None:
        return True
    stale_cutoff = pd.Timestamp(as_of_date) - pd.DateOffset(years=stale_after_years)
    return latest_date < stale_cutoff


def _average(values: list[float]) -> float:
    """Return an average with zero for no observations."""
    if not values:
        return 0.0
    return sum(values) / len(values)
