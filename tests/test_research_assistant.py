"""Tests for the semi-automated research assistant workflow."""

from __future__ import annotations

import pandas as pd

from src.data.research import generate_candidates as generator
from src.data.research.promote_accepted_candidates import promote_accepted_candidates


def test_generate_candidates_preserves_existing_review_fields(tmp_path, monkeypatch) -> None:
    """Existing candidate review decisions should not be overwritten."""
    seed_path = tmp_path / "artist_seed_list.csv"
    output_path = tmp_path / "candidate_observations.csv"
    pd.DataFrame(
        [
            {
                "artist_id": "artist_001",
                "canonical_name": "Ada Rios",
                "primary_medium": "painting",
                "notes": "",
            }
        ]
    ).to_csv(seed_path, index=False)
    existing = {
        "candidate_id": "candidate_existing",
        "artist_id": "artist_001",
        "canonical_name": "Ada Rios",
        "observation_type": "artist_profile",
        "observed_entity_name": "Ada Rios",
        "event_name": "",
        "event_date": "",
        "relationship_type": "identity_match",
        "source_url": "https://www.wikidata.org/wiki/Q1",
        "source_name": "Wikidata",
        "raw_text_excerpt": "Ada Rios - artist",
        "suggested_confidence_score": "0.75",
        "accepted": "yes",
        "needs_human_review_reason": "already reviewed",
        "why_matched": "prior match",
        "review_notes": "confirmed by researcher",
    }
    pd.DataFrame([existing]).to_csv(output_path, index=False)

    def fake_wikidata_candidates(seed, max_results, timeout_seconds):
        row = dict(existing)
        row["accepted"] = "no"
        row["review_notes"] = ""
        return [row]

    monkeypatch.setattr(generator, "_wikidata_candidates", fake_wikidata_candidates)

    output = generator.generate_candidates(seed_path, output_path)

    assert len(output) == 1
    assert output.loc[0, "accepted"] == "yes"
    assert output.loc[0, "needs_human_review_reason"] == "already reviewed"
    assert output.loc[0, "review_notes"] == "confirmed by researcher"


def test_wikidata_classification_keeps_true_artist_profile() -> None:
    """Visual artist pages should become artist profile candidates."""
    seed = pd.Series({"artist_id": "artist_001", "canonical_name": "Ada Rios", "primary_medium": "painting"})

    candidate = generator._classify_wikidata_result(seed, "Ada Rios", "Mexican painter and visual artist")

    assert candidate["observation_type"] == "artist_profile"
    assert candidate["relationship_type"] == "identity_match"
    assert float(candidate["suggested_confidence_score"]) >= 0.65
    assert "visual-artist" in candidate["why_matched"]


def test_wikidata_classification_flags_researcher_false_match() -> None:
    """Researcher-like pages should not be classified as artist profiles."""
    seed = pd.Series({"artist_id": "artist_001", "canonical_name": "Ada Rios", "primary_medium": "painting"})

    candidate = generator._classify_wikidata_result(seed, "Ada Rios", "computer science researcher and professor")

    assert candidate["observation_type"] == "rejected_match"
    assert candidate["relationship_type"] == "not_applicable"
    assert float(candidate["suggested_confidence_score"]) <= 0.25
    assert "Potential false match" in candidate["needs_human_review_reason"]


def test_wikidata_classification_handles_accents_and_generic_artist_pages() -> None:
    """Accent differences should match, while generic artist pages stay lower confidence."""
    accented_seed = pd.Series({"artist_id": "artist_001", "canonical_name": "Jade Fadojutimi", "primary_medium": "painting"})
    generic_seed = pd.Series({"artist_id": "artist_002", "canonical_name": "Tunji Adeniyi-Jones", "primary_medium": ""})

    accented = generator._classify_wikidata_result(accented_seed, "Jadé Fadojutimi", "British visual artist")
    generic = generator._classify_wikidata_result(generic_seed, "Tunji Adeniyi-Jones", "artist born 1992")

    assert accented["observation_type"] == "artist_profile"
    assert float(accented["suggested_confidence_score"]) >= 0.65
    assert generic["observation_type"] == "artist_profile"
    assert generic["suggested_confidence_score"] == "0.45"
    assert "not clearly visual artist" in generic["needs_human_review_reason"]


def test_wikidata_exhibition_result_becomes_museum_exhibition() -> None:
    """Exhibition-like Wikidata hits should become museum exhibition candidates."""
    seed = pd.Series({"artist_id": "artist_001", "canonical_name": "Ada Rios", "primary_medium": "painting"})

    candidate = generator._classify_wikidata_result(seed, "Ada Rios Retrospective", "2021 museum exhibition")

    assert candidate["observation_type"] == "museum_exhibition"
    assert candidate["relationship_type"] == "exhibited_at"
    assert "exhibition" in candidate["needs_human_review_reason"].lower()


def test_wikidata_market_language_creates_market_research_leads() -> None:
    """Market-language structured hits should become reviewable lead types."""
    seed = pd.Series({"artist_id": "artist_001", "canonical_name": "Ada Rios", "primary_medium": "painting"})

    gallery = generator._classify_wikidata_result(seed, "Ada Rios", "gallery profile represented by North Axis")
    auction = generator._classify_wikidata_result(seed, "Ada Rios", "auction result for a painting lot")
    press = generator._classify_wikidata_result(seed, "Ada Rios", "newspaper article and interview")

    assert gallery["observation_type"] == "gallery_representation"
    assert auction["observation_type"] == "auction_result"
    assert press["observation_type"] == "press_mention"
    assert gallery["suggested_confidence_score"] == "0.40"
    assert "verify" in gallery["needs_human_review_reason"].lower()


def test_wikidata_dedupes_artist_profile_candidates() -> None:
    """Only the best Wikidata identity profile should survive per artist."""
    rows = [
        {
            "candidate_id": "candidate_1",
            "artist_id": "artist_001",
            "canonical_name": "Ada Rios",
            "observation_type": "artist_profile",
            "observed_entity_name": "Ada Rios",
            "source_url": "https://www.wikidata.org/wiki/Q1",
            "suggested_confidence_score": "0.80",
        },
        {
            "candidate_id": "candidate_2",
            "artist_id": "artist_001",
            "canonical_name": "Ada Rios",
            "observation_type": "artist_profile",
            "observed_entity_name": "Ada Rios",
            "source_url": "https://www.wikidata.org/wiki/Q2",
            "suggested_confidence_score": "0.55",
        },
        {
            "candidate_id": "candidate_3",
            "artist_id": "artist_001",
            "canonical_name": "Ada Rios",
            "observation_type": "museum_exhibition",
            "observed_entity_name": "Ada Rios Retrospective",
            "source_url": "https://www.wikidata.org/wiki/Q3",
            "suggested_confidence_score": "0.45",
        },
    ]

    deduped = generator._dedupe_wikidata_candidates(rows)

    assert [row["candidate_id"] for row in deduped] == ["candidate_1", "candidate_3"]


def test_reclassifies_only_unreviewed_wikidata_rows() -> None:
    """Old unreviewed Wikidata rows should migrate, reviewed rows should remain stable."""
    existing = pd.DataFrame(
        [
            {
                "candidate_id": "old_unreviewed",
                "artist_id": "artist_001",
                "canonical_name": "Ada Rios",
                "observation_type": "artist_profile",
                "observed_entity_name": "Ada Rios Retrospective",
                "event_name": "",
                "event_date": "",
                "relationship_type": "identity_match",
                "source_url": "https://www.wikidata.org/wiki/Q1",
                "source_name": "Wikidata",
                "raw_text_excerpt": "Ada Rios Retrospective — museum exhibition",
                "suggested_confidence_score": "0.75",
                "accepted": "no",
                "needs_human_review_reason": "",
                "why_matched": "",
                "review_notes": "",
            },
            {
                "candidate_id": "reviewed",
                "artist_id": "artist_001",
                "canonical_name": "Ada Rios",
                "observation_type": "artist_profile",
                "observed_entity_name": "Ada Rios",
                "event_name": "",
                "event_date": "",
                "relationship_type": "identity_match",
                "source_url": "https://www.wikidata.org/wiki/Q2",
                "source_name": "Wikidata",
                "raw_text_excerpt": "Ada Rios — computer science researcher",
                "suggested_confidence_score": "0.75",
                "accepted": "no",
                "needs_human_review_reason": "",
                "why_matched": "",
                "review_notes": "researcher decided this is actually the right artist",
            },
        ]
    )

    output = generator._reclassify_unreviewed_wikidata_rows(existing)

    migrated = output[output["source_url"] == "https://www.wikidata.org/wiki/Q1"].iloc[0]
    reviewed = output[output["source_url"] == "https://www.wikidata.org/wiki/Q2"].iloc[0]
    assert migrated["observation_type"] == "museum_exhibition"
    assert migrated["relationship_type"] == "exhibited_at"
    assert reviewed["candidate_id"] == "reviewed"
    assert reviewed["observation_type"] == "artist_profile"


def test_promote_accepted_candidates_writes_manual_research_rows(tmp_path) -> None:
    """Only accepted candidates should become manual research template rows."""
    candidate_path = tmp_path / "candidate_observations.csv"
    promoted_path = tmp_path / "accepted_artist_research_template.csv"
    rows = [
        {
            "candidate_id": "candidate_gallery",
            "artist_id": "artist_001",
            "canonical_name": "Ada Rios",
            "observation_type": "gallery_representation",
            "observed_entity_name": "North Axis",
            "event_name": "",
            "event_date": "2021-01-01",
            "relationship_type": "represents",
            "source_url": "https://example.com/gallery",
            "source_name": "Gallery website",
            "raw_text_excerpt": "Ada Rios is represented by North Axis.",
            "suggested_confidence_score": "0.9",
            "accepted": "yes",
            "needs_human_review_reason": "Confirm representation date.",
            "why_matched": "Official gallery text matched the artist name.",
            "review_notes": "official page",
        },
        {
            "candidate_id": "candidate_rejected",
            "artist_id": "artist_002",
            "canonical_name": "Bea Sol",
            "observation_type": "artist_profile",
            "observed_entity_name": "Bea Sol",
            "event_name": "",
            "event_date": "",
            "relationship_type": "identity_match",
            "source_url": "https://example.com/rejected",
            "source_name": "Example",
            "raw_text_excerpt": "",
            "suggested_confidence_score": "0.5",
            "accepted": "no",
            "needs_human_review_reason": "",
            "why_matched": "",
            "review_notes": "",
        },
    ]
    pd.DataFrame(rows).to_csv(candidate_path, index=False)

    output = promote_accepted_candidates(candidate_path, promoted_path)
    rerun = promote_accepted_candidates(candidate_path, promoted_path)

    assert len(output) == 1
    assert len(rerun) == 1
    assert output.loc[0, "artist_id"] == "artist_001"
    assert output.loc[0, "gallery_name"] == "North Axis"
    assert output.loc[0, "gallery_source_url"] == "https://example.com/gallery"
    assert "candidate_id=candidate_gallery" in output.loc[0, "notes"]
    assert "why_matched=Official gallery text matched the artist name." in output.loc[0, "notes"]
