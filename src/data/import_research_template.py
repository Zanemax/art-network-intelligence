"""Import manual artist research rows into normalized raw CSV files."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

from src.data.validate_schemas import SCHEMAS, validate_csv


RESEARCH_REQUIRED_COLUMNS = tuple(
    pd.read_csv(Path("data/raw/templates/artist_research_template.csv"), nrows=0).columns
)
DEFAULT_OUTPUT_DIR = Path("data/raw/imported")


def import_research_template(
    template_path: str | Path,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, pd.DataFrame]:
    """Validate and convert manual research rows into normalized CSV tables."""
    path = Path(template_path)
    rows = pd.read_csv(path, dtype=str).fillna("")
    _validate_research_columns(rows)

    output = {
        "artists.csv": _build_artists(rows),
        "institutions.csv": _build_institutions(rows),
        "galleries.csv": _build_galleries(rows),
        "museums.csv": _build_museums(rows),
        "events.csv": _build_events(rows),
        "relationships.csv": _build_relationships(rows),
        "auction_results.csv": _build_auction_results(rows),
        "press_mentions.csv": _build_press_mentions(rows),
        "collectors.csv": _build_collectors(rows),
        "curators.csv": _build_curators(rows),
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    for filename, frame in output.items():
        frame = _dedupe_by_primary_key(frame, SCHEMAS[filename].primary_key)
        frame.to_csv(output_path / filename, index=False)
        errors = validate_csv(output_path / filename)
        if errors:
            raise ValueError(f"{filename} failed validation: {'; '.join(errors)}")
        output[filename] = frame

    return output


def _validate_research_columns(rows: pd.DataFrame) -> None:
    """Ensure manual research rows match the template header."""
    missing = [column for column in RESEARCH_REQUIRED_COLUMNS if column not in rows.columns]
    if missing:
        raise ValueError(f"Missing research template columns: {', '.join(missing)}")
    if rows.empty:
        return
    if rows["artist_id"].str.strip().eq("").any():
        raise ValueError("artist_id is required for every research row.")
    if rows["artist_name"].str.strip().eq("").any():
        raise ValueError("artist_name is required for every research row.")


def _build_artists(rows: pd.DataFrame) -> pd.DataFrame:
    """Build normalized artist entity rows."""
    frame = pd.DataFrame(
        {
            "artist_id": rows["artist_id"],
            "canonical_name": rows["artist_name"],
            "birth_year": rows["birth_year"],
            "death_year": rows["death_year"],
            "nationality": rows["nationality"],
            "gender": rows["gender"],
            "primary_medium": rows["primary_medium"],
            "artist_website_url": rows["artist_website_url"],
            "source_url": rows["bio_source_url"],
            "confidence_score": rows["bio_confidence_score"].replace("", "0.7"),
            "notes": rows["notes"],
        }
    )
    return _align(frame, "artists.csv")


def _build_institutions(rows: pd.DataFrame) -> pd.DataFrame:
    """Build shared institution registry rows."""
    institutions = []
    for _, row in rows.iterrows():
        if row["gallery_name"]:
            institutions.append(
                {
                    "institution_id": _id("institution", row["gallery_name"]),
                    "institution_type": "gallery",
                    "canonical_name": row["gallery_name"],
                    "city": row["gallery_city"],
                    "country": row["gallery_country"],
                    "founded_year": "",
                    "tier": row["gallery_tier"],
                    "prestige_score": row["gallery_prestige_score"] or "0.5",
                    "source_url": _first(row["gallery_source_url"], row["event_source_url"], row["art_fair_source_url"]),
                    "confidence_score": row["gallery_confidence_score"] or "0.7",
                    "notes": row["notes"],
                }
            )
        if row["museum_name"]:
            institutions.append(
                {
                    "institution_id": _id("institution", row["museum_name"]),
                    "institution_type": "museum",
                    "canonical_name": row["museum_name"],
                    "city": row["museum_city"],
                    "country": row["museum_country"],
                    "founded_year": "",
                    "tier": row["museum_tier"],
                    "prestige_score": "0.8" if row["museum_tier"] in {"major", "top"} else "0.6",
                    "source_url": _first(row["event_source_url"], row["acquisition_source_url"]),
                    "confidence_score": _first(row["event_confidence_score"], row["acquisition_confidence_score"], "0.7"),
                    "notes": row["notes"],
                }
            )
        if row["art_fair_name"]:
            institutions.append(
                {
                    "institution_id": _id("institution", row["art_fair_name"]),
                    "institution_type": "art_fair",
                    "canonical_name": row["art_fair_name"],
                    "city": row["art_fair_city"],
                    "country": row["art_fair_country"],
                    "founded_year": "",
                    "tier": "",
                    "prestige_score": "0.5",
                    "source_url": row["art_fair_source_url"],
                    "confidence_score": row["art_fair_confidence_score"] or "0.7",
                    "notes": row["notes"],
                }
            )
    return _align(pd.DataFrame(institutions), "institutions.csv")


def _build_galleries(rows: pd.DataFrame) -> pd.DataFrame:
    """Build gallery entity rows."""
    frame = pd.DataFrame(
        [
            {
                "gallery_id": _id("gallery", row["gallery_name"]),
                "institution_id": _id("institution", row["gallery_name"]),
                "canonical_name": row["gallery_name"],
                "city": row["gallery_city"],
                "country": row["gallery_country"],
                "founded_year": "",
                "prestige_score": row["gallery_prestige_score"] or "0.5",
                "source_url": _first(row["gallery_source_url"], row["event_source_url"], row["art_fair_source_url"]),
                "confidence_score": row["gallery_confidence_score"] or "0.7",
                "notes": row["notes"],
            }
            for _, row in rows.iterrows()
            if row["gallery_name"]
        ]
    )
    return _align(frame, "galleries.csv")


def _build_museums(rows: pd.DataFrame) -> pd.DataFrame:
    """Build museum entity rows."""
    frame = pd.DataFrame(
        [
            {
                "museum_id": _id("museum", row["museum_name"]),
                "institution_id": _id("institution", row["museum_name"]),
                "canonical_name": row["museum_name"],
                "city": row["museum_city"],
                "country": row["museum_country"],
                "founded_year": "",
                "tier": row["museum_tier"],
                "prestige_score": "0.8" if row["museum_tier"] in {"major", "top"} else "0.6",
                "source_url": _first(row["event_source_url"], row["acquisition_source_url"]),
                "confidence_score": _first(row["event_confidence_score"], row["acquisition_confidence_score"], "0.7"),
                "notes": row["notes"],
            }
            for _, row in rows.iterrows()
            if row["museum_name"]
        ]
    )
    return _align(frame, "museums.csv")


def _build_events(rows: pd.DataFrame) -> pd.DataFrame:
    """Build exhibition and art fair event rows."""
    events = []
    gallery_lookup = _gallery_domain_lookup(rows)
    for index, row in rows.iterrows():
        if row["event_name"]:
            inferred_gallery = _infer_gallery_from_event_url(row, gallery_lookup)
            gallery_name = row["gallery_name"] or inferred_gallery.get("gallery_name", "")
            gallery_city = row["gallery_city"] or inferred_gallery.get("gallery_city", "")
            gallery_country = row["gallery_country"] or inferred_gallery.get("gallery_country", "")
            host_name = row["museum_name"] or gallery_name
            host_city = row["museum_city"] or gallery_city
            host_country = row["museum_country"] or gallery_country
            event_type = row["museum_event_type"] or ("gallery_exhibition" if gallery_name and not row["museum_name"] else "museum_exhibition")
            start_date = _date(row["event_start_date"] or row["event_end_date"])
            end_date = _date(row["event_end_date"])
            events.append(
                {
                    "event_id": _event_id("event", row["artist_id"], row["event_name"], row["event_start_date"], index),
                    "event_type": event_type,
                    "event_name": row["event_name"],
                    "institution_id": _id("institution", host_name),
                    "artist_id": row["artist_id"],
                    "curator_id": _id("curator", row["curator_name"]) if row["curator_name"] else "",
                    "start_date": start_date,
                    "end_date": end_date,
                    "event_date": start_date,
                    "city": host_city,
                    "country": host_country,
                    "source_url": row["event_source_url"],
                    "confidence_score": row["event_confidence_score"] or "0.7",
                    "notes": row["notes"],
                }
            )
        if row["art_fair_name"]:
            events.append(
                {
                    "event_id": _event_id("fair", row["artist_id"], row["art_fair_name"], row["art_fair_date"], index),
                    "event_type": "art_fair",
                    "event_name": row["art_fair_name"],
                    "institution_id": _id("institution", row["art_fair_name"]),
                    "artist_id": row["artist_id"],
                    "curator_id": "",
                    "start_date": _date(row["art_fair_date"]),
                    "end_date": _date(row["art_fair_date"]),
                    "event_date": _date(row["art_fair_date"]),
                    "city": row["art_fair_city"],
                    "country": row["art_fair_country"],
                    "source_url": row["art_fair_source_url"],
                    "confidence_score": row["art_fair_confidence_score"] or "0.7",
                    "notes": row["notes"],
                }
            )
    return _align(pd.DataFrame(events), "events.csv")


def _build_relationships(rows: pd.DataFrame) -> pd.DataFrame:
    """Build normalized relationship edge rows."""
    relationships = []
    gallery_lookup = _gallery_domain_lookup(rows)
    for index, row in rows.iterrows():
        if _has_gallery_representation(row):
            relationships.append(_relationship(row, index, _id("gallery", row["gallery_name"]), "gallery", row["artist_id"], "artist", "represents", _date(row["gallery_start_date"]), row["gallery_source_url"], row["gallery_confidence_score"]))
        if row["museum_name"] and row["event_name"]:
            relationships.append(_relationship(row, index, row["artist_id"], "artist", _id("museum", row["museum_name"]), "museum", "museum_exhibition", _date(row["event_start_date"]), row["event_source_url"], row["event_confidence_score"]))
        if row["gallery_name"] and row["event_name"] and not row["museum_name"]:
            relationships.append(_relationship(row, index, row["artist_id"], "artist", _id("gallery", row["gallery_name"]), "gallery", "gallery_exhibition", _date(row["event_start_date"]), row["event_source_url"], row["event_confidence_score"]))
        inferred_gallery = _infer_gallery_from_event_url(row, gallery_lookup)
        if inferred_gallery and row["event_name"] and not row["museum_name"] and not row["gallery_name"]:
            relationships.append(_relationship(row, index, row["artist_id"], "artist", _id("gallery", inferred_gallery["gallery_name"]), "gallery", "gallery_exhibition", _date(row["event_start_date"]), row["event_source_url"], row["event_confidence_score"]))
        if row["museum_name"] and row["acquisition_date"]:
            relationships.append(_relationship(row, index, _id("museum", row["museum_name"]), "museum", row["artist_id"], "artist", "acquired_artist", _date(row["acquisition_date"]), row["acquisition_source_url"], row["acquisition_confidence_score"]))
        if row["collector_name"]:
            relationships.append(_relationship(row, index, _id("collector", row["collector_name"]), "collector", row["artist_id"], "artist", "collects", _date(_first(row["gallery_start_date"], row["event_start_date"], row["sale_date"], row["publication_date"])), row["collector_source_url"], row["collector_confidence_score"]))
        if row["curator_name"]:
            relationships.append(_relationship(row, index, _id("curator", row["curator_name"]), "curator", row["artist_id"], "artist", "curated_artist", _date(row["event_start_date"]), row["curator_source_url"], row["curator_confidence_score"]))
    return _align(pd.DataFrame(relationships), "relationships.csv")


def _build_auction_results(rows: pd.DataFrame) -> pd.DataFrame:
    """Build auction result rows."""
    frame = pd.DataFrame(
        [
            {
                "auction_result_id": _event_id("auction", row["artist_id"], row["lot_number"], row["sale_date"], index),
                "artist_id": row["artist_id"],
                "auction_house": row["auction_house"],
                "sale_name": row["sale_name"],
                "lot_number": row["lot_number"],
                "sale_date": _date(row["sale_date"]),
                "work_title": row["work_title"],
                "medium": row["work_medium"],
                "creation_year": row["creation_year"],
                "estimate_low_usd": row["estimate_low_usd"],
                "estimate_high_usd": row["estimate_high_usd"],
                "price_usd": row["price_usd"],
                "currency": row["currency"] or "USD",
                "source_url": row["auction_source_url"],
                "confidence_score": row["auction_confidence_score"] or "0.7",
                "notes": row["notes"],
            }
            for index, row in rows.iterrows()
            if row["sale_date"] or row["auction_house"] or row["price_usd"]
        ]
    )
    return _align(frame, "auction_results.csv")


def _build_press_mentions(rows: pd.DataFrame) -> pd.DataFrame:
    """Build press mention rows."""
    frame = pd.DataFrame(
        [
            {
                "press_mention_id": _event_id("press", row["artist_id"], row["article_title"], row["publication_date"], index),
                "artist_id": row["artist_id"],
                "outlet_name": row["press_outlet"],
                "article_title": row["article_title"],
                "author": row["article_author"],
                "publication_date": _date(row["publication_date"]),
                "url": row["article_url"],
                "mention_count": row["mention_count"] or "1",
                "sentiment_score": row["sentiment_score"] or "0",
                "source_url": row["article_url"],
                "confidence_score": row["press_confidence_score"] or "0.7",
                "notes": row["notes"],
            }
            for index, row in rows.iterrows()
            if row["article_title"] or row["article_url"]
        ]
    )
    return _align(frame, "press_mentions.csv")


def _build_collectors(rows: pd.DataFrame) -> pd.DataFrame:
    """Build collector entity rows."""
    frame = pd.DataFrame(
        [
            {
                "collector_id": _id("collector", row["collector_name"]),
                "display_name": row["collector_name"],
                "collector_type": row["collector_type"] or "individual",
                "city": "",
                "country": "",
                "visibility_level": "public",
                "source_url": row["collector_source_url"],
                "confidence_score": row["collector_confidence_score"] or "0.7",
                "notes": row["notes"],
            }
            for _, row in rows.iterrows()
            if row["collector_name"]
        ]
    )
    return _align(frame, "collectors.csv")


def _build_curators(rows: pd.DataFrame) -> pd.DataFrame:
    """Build curator entity rows."""
    frame = pd.DataFrame(
        [
            {
                "curator_id": _id("curator", row["curator_name"]),
                "display_name": row["curator_name"],
                "affiliated_institution_id": _id("institution", row["museum_name"]) if row["museum_name"] else "",
                "city": "",
                "country": "",
                "source_url": row["curator_source_url"],
                "confidence_score": row["curator_confidence_score"] or "0.7",
                "notes": row["notes"],
            }
            for _, row in rows.iterrows()
            if row["curator_name"]
        ]
    )
    return _align(frame, "curators.csv")


def _relationship(
    row: pd.Series,
    index: int,
    source_id: str,
    source_type: str,
    target_id: str,
    target_type: str,
    relationship_type: str,
    date: str,
    source_url: str,
    confidence_score: str,
) -> dict[str, str]:
    """Build one normalized relationship row."""
    relationship_date = date or "1900-01-01"
    return {
        "relationship_id": _event_id("rel", source_id, target_id, relationship_type, index),
        "source_node_id": source_id,
        "source_node_type": source_type,
        "target_node_id": target_id,
        "target_node_type": target_type,
        "relationship_type": relationship_type,
        "relationship_date": relationship_date,
        "start_date": relationship_date,
        "end_date": "",
        "source_url": source_url,
        "confidence_score": confidence_score or "0.7",
        "notes": row["notes"],
    }


def _align(frame: pd.DataFrame, filename: str) -> pd.DataFrame:
    """Align a generated DataFrame to a registered raw schema."""
    columns = list(SCHEMAS[filename].required_columns)
    if frame.empty:
        return pd.DataFrame(columns=columns)
    for column in columns:
        if column not in frame.columns:
            frame[column] = ""
    return frame[columns].fillna("")


def _dedupe_by_primary_key(frame: pd.DataFrame, primary_key: str) -> pd.DataFrame:
    """Drop duplicate normalized entity rows."""
    if frame.empty:
        return frame
    return frame.drop_duplicates(subset=[primary_key], keep="first").reset_index(drop=True)


def _id(prefix: str, value: str) -> str:
    """Create a stable slug ID."""
    slug = re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")
    return f"{prefix}_{slug}" if slug else ""


def _event_id(prefix: str, *parts: object) -> str:
    """Create a stable event ID from row parts."""
    slug = re.sub(r"[^a-z0-9]+", "_", "_".join(str(part).lower() for part in parts if str(part))).strip("_")
    return f"{prefix}_{slug}"


def _date(value: object) -> str:
    """Normalize template dates to ISO format while allowing blank values."""
    text = str(value or "").strip()
    if not text:
        return ""
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return text
    parsed = pd.to_datetime(text, dayfirst=True, errors="coerce")
    return parsed.strftime("%Y-%m-%d") if pd.notna(parsed) else text


def _gallery_domain_lookup(rows: pd.DataFrame) -> dict[tuple[str, str], dict[str, str]]:
    """Map artist/gallery source domains to gallery metadata."""
    lookup = {}
    for _, row in rows.iterrows():
        domain = _url_domain(row["gallery_source_url"])
        if not row["artist_id"] or not row["gallery_name"] or not domain:
            continue
        lookup[(row["artist_id"], domain)] = {
            "gallery_name": row["gallery_name"],
            "gallery_city": row["gallery_city"],
            "gallery_country": row["gallery_country"],
        }
    return lookup


def _infer_gallery_from_event_url(row: pd.Series, gallery_lookup: dict[tuple[str, str], dict[str, str]]) -> dict[str, str]:
    """Infer a gallery host when an event URL matches a known gallery domain."""
    event_domain = _url_domain(row["event_source_url"])
    if not event_domain:
        return {}
    for (artist_id, gallery_domain), gallery in gallery_lookup.items():
        if artist_id != row["artist_id"]:
            continue
        if event_domain == gallery_domain or event_domain.endswith(f".{gallery_domain}") or gallery_domain.endswith(f".{event_domain}"):
            return gallery
    return {}


def _url_domain(value: object) -> str:
    """Return a normalized URL domain for source matching."""
    text = str(value or "").strip()
    if not text:
        return ""
    parsed = urlparse(text if "://" in text else f"https://{text}")
    return parsed.netloc.lower().removeprefix("www.")


def _has_gallery_representation(row: pd.Series) -> bool:
    """Return whether a gallery row describes representation, not just an event venue."""
    if not row["gallery_name"]:
        return False
    if row["gallery_start_date"] or row["gallery_tier"] or row["gallery_prestige_score"]:
        return True
    if row["event_name"] or row["art_fair_name"]:
        return False
    return bool(row["gallery_source_url"])


def _first(*values: str) -> str:
    """Return the first non-empty value."""
    return next((value for value in values if value), "")


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Import manual artist research template rows.")
    parser.add_argument("template_path", help="Path to artist_research_template.csv")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for normalized CSV output.")
    return parser.parse_args()


def main() -> None:
    """Run the manual research importer."""
    args = _parse_args()
    output = import_research_template(args.template_path, args.output_dir)
    for filename, frame in output.items():
        print(f"Wrote {len(frame)} rows to {Path(args.output_dir) / filename}")


if __name__ == "__main__":
    main()
