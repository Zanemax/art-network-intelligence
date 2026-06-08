"""Load raw, processed, imported, or synthetic art taste data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


IMPORTED_REQUIRED_FILES = (
    "artists.csv",
    "galleries.csv",
    "museums.csv",
    "collectors.csv",
    "curators.csv",
    "events.csv",
    "relationships.csv",
    "auction_results.csv",
    "press_mentions.csv",
)


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame."""
    return pd.read_csv(path)


def imported_data_available(directory: str | Path = "data/raw/imported") -> bool:
    """Return whether imported normalized CSVs are available."""
    path = Path(directory)
    return path.exists() and all((path / filename).exists() for filename in IMPORTED_REQUIRED_FILES)


def load_imported_dataset(directory: str | Path = "data/raw/imported") -> dict[str, pd.DataFrame]:
    """Load normalized imported CSVs and adapt them to the MVP graph schema.

    The manual importer writes production-shaped raw CSVs. The current MVP graph
    and model still expect the older synthetic fixture shape, so this adapter
    provides a compatibility layer until the graph builder is fully normalized.
    """
    raw = _read_imported_tables(Path(directory))
    museums = _adapt_museums(raw["museums.csv"])
    museum_id_by_institution = _id_map(raw["museums.csv"], "institution_id", "museum_id")
    return {
        "artists": _adapt_artists(raw["artists.csv"]),
        "galleries": _adapt_galleries(raw["galleries.csv"]),
        "museums": museums,
        "collectors": _adapt_collectors(raw["collectors.csv"]),
        "curators": _adapt_curators(raw["curators.csv"]),
        "representation": _adapt_representation(raw["relationships.csv"]),
        "exhibitions": _adapt_exhibitions(raw["events.csv"], museum_id_by_institution),
        "acquisitions": _adapt_acquisitions(raw["relationships.csv"]),
        "collector_holdings": _adapt_collector_holdings(raw["relationships.csv"]),
        "auction_results": _adapt_auction_results(raw["auction_results.csv"]),
        "press_mentions": _adapt_press_mentions(raw["press_mentions.csv"]),
    }


def _read_imported_tables(directory: Path) -> dict[str, pd.DataFrame]:
    """Read all imported raw CSVs with string-safe defaults."""
    tables = {}
    for filename in IMPORTED_REQUIRED_FILES:
        path = directory / filename
        if not path.exists():
            tables[filename] = pd.DataFrame()
            continue
        tables[filename] = pd.read_csv(path, dtype=str, keep_default_na=False).fillna("")
    return tables


def _adapt_artists(frame: pd.DataFrame) -> pd.DataFrame:
    """Adapt imported artists to synthetic artist columns."""
    return pd.DataFrame(
        {
            "artist_id": frame.get("artist_id", pd.Series(dtype=str)),
            "name": frame.get("canonical_name", pd.Series(dtype=str)),
            "birth_year": _numeric(frame.get("birth_year", pd.Series(dtype=str))),
            "region": frame.get("nationality", pd.Series(dtype=str)),
        }
    )


def _adapt_galleries(frame: pd.DataFrame) -> pd.DataFrame:
    """Adapt imported galleries to synthetic gallery columns."""
    return pd.DataFrame(
        {
            "gallery_id": frame.get("gallery_id", pd.Series(dtype=str)),
            "name": frame.get("canonical_name", pd.Series(dtype=str)),
            "prestige_score": _numeric(frame.get("prestige_score", pd.Series(dtype=str)), default=0.5),
        }
    )


def _adapt_museums(frame: pd.DataFrame) -> pd.DataFrame:
    """Adapt imported museums to synthetic museum columns."""
    return pd.DataFrame(
        {
            "museum_id": frame.get("museum_id", pd.Series(dtype=str)),
            "name": frame.get("canonical_name", pd.Series(dtype=str)),
            "tier": frame.get("tier", pd.Series(dtype=str)).replace("", "mid"),
            "prestige_score": _numeric(frame.get("prestige_score", pd.Series(dtype=str)), default=0.6),
        }
    )


def _adapt_collectors(frame: pd.DataFrame) -> pd.DataFrame:
    """Adapt imported collectors to synthetic collector columns."""
    return pd.DataFrame(
        {
            "collector_id": frame.get("collector_id", pd.Series(dtype=str)),
            "name": frame.get("display_name", pd.Series(dtype=str)),
            "prestige_score": _numeric(frame.get("confidence_score", pd.Series(dtype=str)), default=0.5),
        }
    )


def _adapt_curators(frame: pd.DataFrame) -> pd.DataFrame:
    """Adapt imported curators to synthetic curator columns."""
    return pd.DataFrame(
        {
            "curator_id": frame.get("curator_id", pd.Series(dtype=str)),
            "name": frame.get("display_name", pd.Series(dtype=str)),
            "prestige_score": _numeric(frame.get("confidence_score", pd.Series(dtype=str)), default=0.5),
        }
    )


def _adapt_representation(relationships: pd.DataFrame) -> pd.DataFrame:
    """Adapt represents relationships into gallery representation rows."""
    rows = relationships[relationships.get("relationship_type", "") == "represents"].copy()
    if rows.empty:
        return pd.DataFrame(columns=["artist_id", "gallery_id", "start_date", "end_date", "source_url", "confidence_score"])
    return pd.DataFrame(
        {
            "artist_id": rows["target_node_id"],
            "gallery_id": rows["source_node_id"],
            "start_date": rows["start_date"].replace("", "1900-01-01"),
            "end_date": rows.get("end_date", ""),
            "source_url": rows.get("source_url", ""),
            "confidence_score": rows.get("confidence_score", "0.7"),
        }
    )


def _adapt_exhibitions(events: pd.DataFrame, museum_id_by_institution: dict[str, str]) -> pd.DataFrame:
    """Adapt imported museum events into exhibition rows."""
    if events.empty:
        return pd.DataFrame(columns=["exhibition_id", "artist_id", "institution_id", "curator_id", "date", "institution_type"])
    rows = events[events["event_type"].ne("art_fair")].copy()
    if rows.empty:
        return pd.DataFrame(columns=["exhibition_id", "artist_id", "institution_id", "curator_id", "date", "institution_type"])
    return pd.DataFrame(
        {
            "exhibition_id": rows["event_id"],
            "artist_id": rows["artist_id"],
            "institution_id": rows["institution_id"].map(museum_id_by_institution).fillna(rows["institution_id"]),
            "curator_id": rows.get("curator_id", ""),
            "date": rows["event_date"].replace("", rows["start_date"]).replace("", "1900-01-01"),
            "institution_type": "museum",
            "source_url": rows.get("source_url", ""),
            "confidence_score": rows.get("confidence_score", "0.7"),
        }
    )


def _adapt_acquisitions(relationships: pd.DataFrame) -> pd.DataFrame:
    """Adapt acquired_artist relationships into acquisition rows."""
    rows = relationships[relationships.get("relationship_type", "") == "acquired_artist"].copy()
    if rows.empty:
        return pd.DataFrame(columns=["acquisition_id", "artist_id", "museum_id", "date", "value_usd"])
    return pd.DataFrame(
        {
            "acquisition_id": rows["relationship_id"],
            "artist_id": rows["target_node_id"],
            "museum_id": rows["source_node_id"],
            "date": rows["relationship_date"].replace("", rows["start_date"]).replace("", "1900-01-01"),
            "value_usd": 0,
            "source_url": rows.get("source_url", ""),
            "confidence_score": rows.get("confidence_score", "0.7"),
        }
    )


def _adapt_collector_holdings(relationships: pd.DataFrame) -> pd.DataFrame:
    """Adapt collects relationships into collector holding rows."""
    rows = relationships[relationships.get("relationship_type", "") == "collects"].copy()
    if rows.empty:
        return pd.DataFrame(columns=["collector_id", "artist_id", "date"])
    return pd.DataFrame(
        {
            "collector_id": rows["source_node_id"],
            "artist_id": rows["target_node_id"],
            "date": rows["relationship_date"].replace("", rows["start_date"]).replace("", "1900-01-01"),
            "source_url": rows.get("source_url", ""),
            "confidence_score": rows.get("confidence_score", "0.7"),
        }
    )


def _adapt_auction_results(frame: pd.DataFrame) -> pd.DataFrame:
    """Adapt imported auction rows to synthetic auction columns."""
    if frame.empty:
        return pd.DataFrame(columns=["artist_id", "sale_date", "price_usd"])
    return pd.DataFrame(
        {
            "artist_id": frame["artist_id"],
            "sale_date": frame["sale_date"].replace("", "1900-01-01"),
            "price_usd": _numeric(frame["price_usd"], default=0),
            "source_url": frame.get("source_url", ""),
            "confidence_score": frame.get("confidence_score", "0.7"),
        }
    )


def _adapt_press_mentions(frame: pd.DataFrame) -> pd.DataFrame:
    """Adapt imported press rows to synthetic press columns."""
    if frame.empty:
        return pd.DataFrame(columns=["artist_id", "year", "mentions"])
    dates = pd.to_datetime(frame["publication_date"].replace("", "1900-01-01"), errors="coerce")
    return pd.DataFrame(
        {
            "artist_id": frame["artist_id"],
            "year": dates.dt.year.fillna(1900).astype(int),
            "mentions": _numeric(frame["mention_count"], default=1),
            "source_url": frame.get("source_url", ""),
            "confidence_score": frame.get("confidence_score", "0.7"),
        }
    )


def _id_map(frame: pd.DataFrame, key_column: str, value_column: str) -> dict[str, str]:
    """Build a string dictionary from two columns if present."""
    if frame.empty or key_column not in frame.columns or value_column not in frame.columns:
        return {}
    return frame.set_index(key_column)[value_column].to_dict()


def _numeric(values: pd.Series, default: float = 0.0) -> pd.Series:
    """Convert a string series to numeric values with a default fill."""
    return pd.to_numeric(values, errors="coerce").fillna(default)
