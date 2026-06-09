"""Tests for importing manual artist research templates."""

import pandas as pd

from src.data.import_research_template import RESEARCH_REQUIRED_COLUMNS, import_research_template


def test_import_research_template_converts_rows_to_normalized_csvs(tmp_path) -> None:
    """Verify a filled research row creates normalized entities and relationships."""
    template_path = tmp_path / "artist_research_template.csv"
    output_dir = tmp_path / "imported"
    row = _valid_research_row()
    pd.DataFrame([row], columns=RESEARCH_REQUIRED_COLUMNS).to_csv(template_path, index=False)

    output = import_research_template(template_path, output_dir)

    assert (output_dir / "artists.csv").exists()
    assert (output_dir / "relationships.csv").exists()
    assert output["artists.csv"].loc[0, "artist_id"] == "artist_jane_doe"
    assert output["galleries.csv"].loc[0, "canonical_name"] == "Example Gallery"
    assert set(output["relationships.csv"]["relationship_type"]) >= {
        "represents",
        "museum_exhibition",
        "acquired_artist",
        "collects",
        "curated_artist",
    }
    assert output["auction_results.csv"].loc[0, "price_usd"] == "250000"
    assert output["auction_results.csv"].loc[0, "sale_date"] == "2023-06-10"
    assert output["press_mentions.csv"].loc[0, "outlet_name"] == "Art Journal"
    assert output["events.csv"].loc[0, "institution_id"] == "institution_example_museum"


def test_import_research_template_infers_gallery_exhibition_from_source_domain(tmp_path) -> None:
    """Verify gallery exhibition rows can be linked through a known gallery domain."""
    template_path = tmp_path / "artist_research_template.csv"
    output_dir = tmp_path / "imported"
    gallery_row = {column: "" for column in RESEARCH_REQUIRED_COLUMNS}
    gallery_row.update(
        {
            "artist_id": "artist_jane_doe",
            "artist_name": "Jane Doe",
            "bio_source_url": "https://example.com/jane/bio",
            "bio_confidence_score": "0.9",
            "gallery_name": "Example Gallery",
            "gallery_city": "New York",
            "gallery_country": "US",
            "gallery_source_url": "https://example-gallery.test/about",
            "gallery_confidence_score": "0.9",
        }
    )
    event_row = {column: "" for column in RESEARCH_REQUIRED_COLUMNS}
    event_row.update(
        {
            "artist_id": "artist_jane_doe",
            "artist_name": "Jane Doe",
            "bio_source_url": "https://example.com/jane/bio",
            "bio_confidence_score": "0.9",
            "event_name": "Jane Doe: Gallery Exhibition",
            "event_start_date": "22/02/2024",
            "event_end_date": "30/03/2024",
            "event_source_url": "https://www.example-gallery.test/exhibitions/jane-doe",
            "event_confidence_score": "1.0",
        }
    )
    pd.DataFrame([gallery_row, event_row], columns=RESEARCH_REQUIRED_COLUMNS).to_csv(template_path, index=False)

    output = import_research_template(template_path, output_dir)

    event = output["events.csv"][output["events.csv"]["event_name"] == "Jane Doe: Gallery Exhibition"].iloc[0]
    assert event["event_type"] == "gallery_exhibition"
    assert event["institution_id"] == "institution_example_gallery"
    assert event["start_date"] == "2024-02-22"
    assert (
        output["relationships.csv"][output["relationships.csv"]["relationship_type"] == "gallery_exhibition"]
        .iloc[0]["target_node_id"]
        == "gallery_example_gallery"
    )


def test_gallery_exhibition_does_not_create_representation(tmp_path) -> None:
    """Verify a gallery-hosted exhibition is not treated as gallery representation."""
    template_path = tmp_path / "artist_research_template.csv"
    output_dir = tmp_path / "imported"
    row = {column: "" for column in RESEARCH_REQUIRED_COLUMNS}
    row.update(
        {
            "artist_id": "artist_jane_doe",
            "artist_name": "Jane Doe",
            "bio_source_url": "https://example.com/jane/bio",
            "bio_confidence_score": "0.9",
            "gallery_name": "Venue Gallery",
            "gallery_city": "London",
            "gallery_country": "UK",
            "gallery_confidence_score": "1.0",
            "museum_event_type": "gallery_exhibition",
            "event_name": "Jane Doe at Venue Gallery",
            "event_start_date": "2024-02-22",
            "event_source_url": "https://venue-gallery.test/exhibitions/jane",
            "event_confidence_score": "1.0",
        }
    )
    pd.DataFrame([row], columns=RESEARCH_REQUIRED_COLUMNS).to_csv(template_path, index=False)

    output = import_research_template(template_path, output_dir)

    relationship_types = set(output["relationships.csv"]["relationship_type"])
    assert output["galleries.csv"].loc[0, "source_url"] == "https://venue-gallery.test/exhibitions/jane"
    assert "gallery_exhibition" in relationship_types
    assert "represents" not in relationship_types


def test_import_research_template_allows_header_only_template(tmp_path) -> None:
    """Verify the distributed blank template can be imported before rows are added."""
    template_path = tmp_path / "artist_research_template.csv"
    output_dir = tmp_path / "imported"
    pd.DataFrame(columns=RESEARCH_REQUIRED_COLUMNS).to_csv(template_path, index=False)

    output = import_research_template(template_path, output_dir)

    assert all(frame.empty for frame in output.values())
    assert (output_dir / "artists.csv").exists()


def _valid_research_row() -> dict[str, str]:
    """Return a complete manual research row."""
    row = {column: "" for column in RESEARCH_REQUIRED_COLUMNS}
    row.update(
        {
            "artist_id": "artist_jane_doe",
            "artist_name": "Jane Doe",
            "birth_year": "1988",
            "nationality": "US",
            "gender": "female",
            "primary_medium": "painting",
            "artist_website_url": "https://example.com/jane-doe",
            "bio_source_url": "https://example.com/jane-doe/bio",
            "bio_confidence_score": "0.9",
            "gallery_name": "Example Gallery",
            "gallery_city": "New York",
            "gallery_country": "US",
            "gallery_tier": "major",
            "gallery_prestige_score": "0.85",
            "gallery_start_date": "2020-01-01",
            "gallery_source_url": "https://example.com/gallery/jane-doe",
            "gallery_confidence_score": "0.9",
            "museum_name": "Example Museum",
            "museum_city": "Los Angeles",
            "museum_country": "US",
            "museum_tier": "top",
            "museum_event_type": "major_solo_show",
            "event_name": "Jane Doe: New Work",
            "event_start_date": "2022-04-01",
            "event_end_date": "2022-08-01",
            "event_source_url": "https://example.com/museum/exhibition",
            "event_confidence_score": "1.0",
            "acquisition_date": "2023-01-15",
            "acquisition_value_usd": "100000",
            "acquisition_source_url": "https://example.com/museum/acquisition",
            "acquisition_confidence_score": "1.0",
            "art_fair_name": "Example Fair",
            "art_fair_date": "2021-05-01",
            "art_fair_city": "Basel",
            "art_fair_country": "CH",
            "art_fair_source_url": "https://example.com/fair",
            "art_fair_confidence_score": "0.8",
            "auction_house": "Example Auctions",
            "sale_name": "Contemporary Evening",
            "lot_number": "12",
            "sale_date": "2023-06-10",
            "work_title": "Untitled",
            "work_medium": "oil on canvas",
            "creation_year": "2020",
            "estimate_low_usd": "150000",
            "estimate_high_usd": "220000",
            "price_usd": "250000",
            "currency": "USD",
            "auction_source_url": "https://example.com/auction/lot-12",
            "auction_confidence_score": "1.0",
            "press_outlet": "Art Journal",
            "article_title": "Jane Doe Breaks Through",
            "article_author": "Critic Name",
            "publication_date": "2023-07-01",
            "article_url": "https://example.com/press",
            "mention_count": "3",
            "sentiment_score": "0.7",
            "press_confidence_score": "0.8",
            "collector_name": "Example Collector",
            "collector_type": "individual",
            "collector_source_url": "https://example.com/collector",
            "collector_confidence_score": "0.7",
            "curator_name": "Example Curator",
            "curator_source_url": "https://example.com/curator",
            "curator_confidence_score": "0.8",
            "notes": "test row",
        }
    )
    return row
