"""Validate raw CSV files against the production art taste graph schemas.

The schemas in this module define required columns and lightweight parse rules
for replacing synthetic MVP data with real production data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class CsvSchema:
    """Schema metadata for one raw CSV file."""

    required_columns: tuple[str, ...]
    primary_key: str
    string_columns: tuple[str, ...] = ()
    integer_columns: tuple[str, ...] = ()
    float_columns: tuple[str, ...] = ()
    date_columns: tuple[str, ...] = ()
    url_columns: tuple[str, ...] = ("source_url",)
    confidence_column: str = "confidence_score"
    required_value_columns: tuple[str, ...] = ()


SCHEMAS: dict[str, CsvSchema] = {
    "artists.csv": CsvSchema(
        required_columns=(
            "artist_id",
            "canonical_name",
            "birth_year",
            "death_year",
            "nationality",
            "gender",
            "primary_medium",
            "artist_website_url",
            "source_url",
            "confidence_score",
            "notes",
        ),
        primary_key="artist_id",
        string_columns=(
            "artist_id",
            "canonical_name",
            "nationality",
            "gender",
            "primary_medium",
            "notes",
        ),
        integer_columns=("birth_year", "death_year"),
        float_columns=("confidence_score",),
        url_columns=("artist_website_url", "source_url"),
        required_value_columns=("artist_id", "canonical_name", "source_url", "confidence_score"),
    ),
    "galleries.csv": CsvSchema(
        required_columns=(
            "gallery_id",
            "institution_id",
            "canonical_name",
            "city",
            "country",
            "founded_year",
            "prestige_score",
            "source_url",
            "confidence_score",
            "notes",
        ),
        primary_key="gallery_id",
        string_columns=("gallery_id", "institution_id", "canonical_name", "city", "country", "notes"),
        integer_columns=("founded_year",),
        float_columns=("prestige_score", "confidence_score"),
        required_value_columns=("gallery_id", "canonical_name", "source_url", "confidence_score"),
    ),
    "museums.csv": CsvSchema(
        required_columns=(
            "museum_id",
            "institution_id",
            "canonical_name",
            "city",
            "country",
            "founded_year",
            "tier",
            "prestige_score",
            "source_url",
            "confidence_score",
            "notes",
        ),
        primary_key="museum_id",
        string_columns=("museum_id", "institution_id", "canonical_name", "city", "country", "tier", "notes"),
        integer_columns=("founded_year",),
        float_columns=("prestige_score", "confidence_score"),
        required_value_columns=("museum_id", "canonical_name", "source_url", "confidence_score"),
    ),
    "collectors.csv": CsvSchema(
        required_columns=(
            "collector_id",
            "display_name",
            "collector_type",
            "city",
            "country",
            "visibility_level",
            "source_url",
            "confidence_score",
            "notes",
        ),
        primary_key="collector_id",
        string_columns=(
            "collector_id",
            "display_name",
            "collector_type",
            "city",
            "country",
            "visibility_level",
            "notes",
        ),
        float_columns=("confidence_score",),
        required_value_columns=("collector_id", "display_name", "source_url", "confidence_score"),
    ),
    "curators.csv": CsvSchema(
        required_columns=(
            "curator_id",
            "display_name",
            "affiliated_institution_id",
            "city",
            "country",
            "source_url",
            "confidence_score",
            "notes",
        ),
        primary_key="curator_id",
        string_columns=(
            "curator_id",
            "display_name",
            "affiliated_institution_id",
            "city",
            "country",
            "notes",
        ),
        float_columns=("confidence_score",),
        required_value_columns=("curator_id", "display_name", "source_url", "confidence_score"),
    ),
    "events.csv": CsvSchema(
        required_columns=(
            "event_id",
            "event_type",
            "event_name",
            "institution_id",
            "artist_id",
            "curator_id",
            "start_date",
            "end_date",
            "event_date",
            "city",
            "country",
            "source_url",
            "confidence_score",
            "notes",
        ),
        primary_key="event_id",
        string_columns=(
            "event_id",
            "event_type",
            "event_name",
            "institution_id",
            "artist_id",
            "curator_id",
            "city",
            "country",
            "notes",
        ),
        float_columns=("confidence_score",),
        date_columns=("start_date", "end_date", "event_date"),
        required_value_columns=("event_id", "event_type", "event_date", "source_url", "confidence_score"),
    ),
    "relationships.csv": CsvSchema(
        required_columns=(
            "relationship_id",
            "source_node_id",
            "source_node_type",
            "target_node_id",
            "target_node_type",
            "relationship_type",
            "relationship_date",
            "start_date",
            "end_date",
            "source_url",
            "confidence_score",
            "notes",
        ),
        primary_key="relationship_id",
        string_columns=(
            "relationship_id",
            "source_node_id",
            "source_node_type",
            "target_node_id",
            "target_node_type",
            "relationship_type",
            "notes",
        ),
        float_columns=("confidence_score",),
        date_columns=("relationship_date", "start_date", "end_date"),
        required_value_columns=(
            "relationship_id",
            "source_node_id",
            "target_node_id",
            "relationship_type",
            "relationship_date",
            "source_url",
            "confidence_score",
        ),
    ),
    "auction_results.csv": CsvSchema(
        required_columns=(
            "auction_result_id",
            "artist_id",
            "auction_house",
            "sale_name",
            "lot_number",
            "sale_date",
            "work_title",
            "medium",
            "creation_year",
            "estimate_low_usd",
            "estimate_high_usd",
            "price_usd",
            "currency",
            "source_url",
            "confidence_score",
            "notes",
        ),
        primary_key="auction_result_id",
        string_columns=(
            "auction_result_id",
            "artist_id",
            "auction_house",
            "sale_name",
            "lot_number",
            "work_title",
            "medium",
            "currency",
            "notes",
        ),
        integer_columns=("creation_year",),
        float_columns=("estimate_low_usd", "estimate_high_usd", "price_usd", "confidence_score"),
        date_columns=("sale_date",),
        required_value_columns=("auction_result_id", "artist_id", "sale_date", "source_url", "confidence_score"),
    ),
    "press_mentions.csv": CsvSchema(
        required_columns=(
            "press_mention_id",
            "artist_id",
            "outlet_name",
            "article_title",
            "author",
            "publication_date",
            "url",
            "mention_count",
            "sentiment_score",
            "source_url",
            "confidence_score",
            "notes",
        ),
        primary_key="press_mention_id",
        string_columns=(
            "press_mention_id",
            "artist_id",
            "outlet_name",
            "article_title",
            "author",
            "notes",
        ),
        integer_columns=("mention_count",),
        float_columns=("sentiment_score", "confidence_score"),
        date_columns=("publication_date",),
        url_columns=("url", "source_url"),
        required_value_columns=("press_mention_id", "artist_id", "publication_date", "source_url", "confidence_score"),
    ),
    "institutions.csv": CsvSchema(
        required_columns=(
            "institution_id",
            "institution_type",
            "canonical_name",
            "city",
            "country",
            "founded_year",
            "tier",
            "prestige_score",
            "source_url",
            "confidence_score",
            "notes",
        ),
        primary_key="institution_id",
        string_columns=(
            "institution_id",
            "institution_type",
            "canonical_name",
            "city",
            "country",
            "tier",
            "notes",
        ),
        integer_columns=("founded_year",),
        float_columns=("prestige_score", "confidence_score"),
        required_value_columns=("institution_id", "institution_type", "canonical_name", "source_url", "confidence_score"),
    ),
}


def validate_csv(path: str | Path) -> list[str]:
    """Validate one CSV file and return a list of human-readable errors."""
    csv_path = Path(path)
    schema = SCHEMAS.get(csv_path.name)
    if schema is None:
        return [f"No schema registered for {csv_path.name}."]

    frame = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    return validate_dataframe(frame, schema)


def validate_dataframe(frame: pd.DataFrame, schema: CsvSchema) -> list[str]:
    """Validate a DataFrame against a schema and return all found errors."""
    errors = _validate_required_columns(frame, schema)
    if errors or frame.empty:
        return errors

    errors.extend(_validate_required_values(frame, schema.required_value_columns))
    errors.extend(_validate_primary_key(frame, schema.primary_key))
    errors.extend(_validate_string_columns(frame, schema.string_columns))
    errors.extend(_validate_integer_columns(frame, schema.integer_columns))
    errors.extend(_validate_float_columns(frame, schema.float_columns))
    errors.extend(_validate_date_columns(frame, schema.date_columns))
    errors.extend(_validate_url_columns(frame, schema.url_columns))
    errors.extend(_validate_confidence_score(frame, schema.confidence_column))
    return errors


def validate_template_directory(directory: str | Path) -> dict[str, list[str]]:
    """Validate every registered CSV template in a directory."""
    template_dir = Path(directory)
    return {
        filename: validate_csv(template_dir / filename)
        for filename in sorted(SCHEMAS)
    }


def _validate_required_columns(frame: pd.DataFrame, schema: CsvSchema) -> list[str]:
    missing = [column for column in schema.required_columns if column not in frame.columns]
    extra = [column for column in frame.columns if column not in schema.required_columns]
    errors = []
    if missing:
        errors.append(f"Missing required columns: {', '.join(missing)}.")
    if extra:
        errors.append(f"Unexpected columns: {', '.join(extra)}.")
    return errors


def _validate_primary_key(frame: pd.DataFrame, column: str) -> list[str]:
    errors = []
    if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
        errors.append(f"{column} must be populated for every row.")
    if frame[column].duplicated().any():
        errors.append(f"{column} must be unique.")
    return errors


def _validate_required_values(frame: pd.DataFrame, columns: tuple[str, ...]) -> list[str]:
    errors = []
    for column in columns:
        values = frame[column]
        if values.isna().any() or values.astype(str).str.strip().eq("").any():
            errors.append(f"{column} must be populated for every row.")
    return errors


def _validate_string_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> list[str]:
    errors = []
    for column in columns:
        if frame[column].dropna().map(lambda value: isinstance(value, str)).all():
            continue
        errors.append(f"{column} must contain string values.")
    return errors


def _validate_integer_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> list[str]:
    errors = []
    for column in columns:
        values = frame[column].dropna()
        values = values[values.astype(str).str.strip() != ""]
        if values.empty:
            continue
        numeric = pd.to_numeric(values, errors="coerce")
        if numeric.isna().any() or (numeric % 1 != 0).any():
            errors.append(f"{column} must contain integer values.")
    return errors


def _validate_float_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> list[str]:
    errors = []
    for column in columns:
        values = frame[column].dropna()
        values = values[values.astype(str).str.strip() != ""]
        if values.empty:
            continue
        if pd.to_numeric(values, errors="coerce").isna().any():
            errors.append(f"{column} must contain numeric values.")
    return errors


def _validate_date_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> list[str]:
    errors = []
    for column in columns:
        values = frame[column].dropna()
        values = values[values.astype(str).str.strip() != ""]
        if values.empty:
            continue
        if pd.to_datetime(values, format="%Y-%m-%d", errors="coerce").isna().any():
            errors.append(f"{column} must use YYYY-MM-DD dates.")
    return errors


def _validate_url_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> list[str]:
    errors = []
    for column in columns:
        values = frame[column].dropna().astype(str).str.strip()
        values = values[values != ""]
        if values.empty:
            continue
        if not values.str.startswith(("http://", "https://")).all():
            errors.append(f"{column} must contain http or https URLs.")
    return errors


def _validate_confidence_score(frame: pd.DataFrame, column: str) -> list[str]:
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.isna().any():
        return [f"{column} must contain numeric values."]
    if (~values.between(0, 1)).any():
        return [f"{column} must be between 0.0 and 1.0."]
    return []
