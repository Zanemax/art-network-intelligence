"""Tests for production raw CSV templates and schema validation."""

from pathlib import Path
import re

import pandas as pd

from src.data.validate_schemas import SCHEMAS, validate_csv, validate_dataframe


TEMPLATE_DIR = Path("data/raw/templates")
SNAKE_CASE = re.compile(r"^[a-z][a-z0-9_]*$")


def test_all_registered_templates_exist_and_match_required_columns() -> None:
    """Verify template headers are the schema source of truth."""
    for filename, schema in SCHEMAS.items():
        path = TEMPLATE_DIR / filename
        frame = pd.read_csv(path)

        assert path.exists()
        assert tuple(frame.columns) == schema.required_columns
        assert all(SNAKE_CASE.match(column) for column in frame.columns)
        assert validate_csv(path) == []


def test_template_set_matches_expected_raw_files() -> None:
    """Verify the raw template directory contains the requested schema files."""
    assert {path.name for path in TEMPLATE_DIR.glob("*.csv")} == set(SCHEMAS) | {
        "artist_research_template.csv"
    }


def test_valid_rows_pass_type_validation() -> None:
    """Verify representative production rows satisfy required columns and types."""
    for filename, schema in SCHEMAS.items():
        row = _valid_row(schema.required_columns, schema.primary_key)
        frame = pd.DataFrame([row])

        assert validate_dataframe(frame, schema) == [], filename


def test_invalid_required_values_and_types_are_reported() -> None:
    """Verify validators catch missing IDs, malformed dates, and bad numbers."""
    schema = SCHEMAS["relationships.csv"]
    row = _valid_row(schema.required_columns, schema.primary_key)
    row["relationship_id"] = ""
    row["relationship_date"] = "06/08/2026"
    row["confidence_score"] = "1.7"
    frame = pd.DataFrame([row])

    errors = validate_dataframe(frame, schema)

    assert "relationship_id must be populated for every row." in errors
    assert "relationship_date must use YYYY-MM-DD dates." in errors
    assert "confidence_score must be between 0.0 and 1.0." in errors


def test_invalid_numeric_columns_are_reported() -> None:
    """Verify numeric parse errors are caught for market data."""
    schema = SCHEMAS["auction_results.csv"]
    row = _valid_row(schema.required_columns, schema.primary_key)
    row["price_usd"] = "not-a-number"
    row["creation_year"] = "2020.5"
    frame = pd.DataFrame([row])

    errors = validate_dataframe(frame, schema)

    assert "price_usd must contain numeric values." in errors
    assert "creation_year must contain integer values." in errors


def _valid_row(columns: tuple[str, ...], primary_key: str) -> dict[str, object]:
    """Build one valid row for a registered schema."""
    row: dict[str, object] = {}
    for column in columns:
        if column == primary_key or column.endswith("_id"):
            row[column] = f"{column}_001"
        elif column.endswith("_url") or column == "url":
            row[column] = f"https://example.com/{column}"
        elif column.endswith("_date") or column in {"start_date", "end_date"}:
            row[column] = "2026-06-08"
        elif column in {"confidence_score", "prestige_score", "sentiment_score"}:
            row[column] = 0.8
        elif column in {
            "birth_year",
            "death_year",
            "founded_year",
            "creation_year",
            "mention_count",
        }:
            row[column] = 2020 if column != "mention_count" else 3
        elif column.endswith("_usd"):
            row[column] = 100000.0
        else:
            row[column] = f"{column}_value"
    return row
