"""Generate sourced candidate observations for human research review.

This module intentionally does not ingest data into the graph. It reads a seed
list of artists, queries conservative structured sources where available, and
writes candidate observations that a human must review before promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


SEED_COLUMNS = (
    "artist_id",
    "canonical_name",
    "primary_medium",
    "notes",
)
CANDIDATE_COLUMNS = (
    "candidate_id",
    "artist_id",
    "canonical_name",
    "observation_type",
    "observed_entity_name",
    "event_name",
    "event_date",
    "relationship_type",
    "source_url",
    "source_name",
    "raw_text_excerpt",
    "suggested_confidence_score",
    "accepted",
    "needs_human_review_reason",
    "why_matched",
    "review_notes",
)
DEFAULT_SEED_LIST = Path("data/raw/manual/artist_seed_list.csv")
DEFAULT_OUTPUT = Path("data/raw/candidates/candidate_observations.csv")
WIKIDATA_API_URL = "https://www.wikidata.org/w/api.php"

STRUCTURED_OBSERVATION_TYPES = {
    "artist_profile",
    "gallery_representation",
    "museum_exhibition",
    "museum_acquisition",
    "auction_result",
    "press_mention",
}
VISUAL_ARTIST_TERMS = {
    "visual artist",
    "painter",
    "sculptor",
    "photographer",
    "printmaker",
    "installation artist",
    "performance artist",
    "conceptual artist",
    "video artist",
    "mixed-media artist",
    "ceramicist",
    "textile artist",
    "contemporary artist",
}
GENERIC_ARTIST_TERMS = {"artist", "draughtsman", "draftsman"}
VISUAL_MEDIUM_TERMS = {
    "painting",
    "sculpture",
    "photography",
    "drawing",
    "installation",
    "video",
    "performance",
    "printmaking",
    "ceramic",
    "textile",
    "mixed media",
}
EXHIBITION_TERMS = {"exhibition", "biennial", "triennial", "retrospective", "solo show", "group show"}
GALLERY_REPRESENTATION_TERMS = {"represented by", "representation", "gallery profile", "gallery"}
ACQUISITION_TERMS = {"acquisition", "acquired", "collection", "accession"}
AUCTION_TERMS = {"auction", "auction result", "sale result", "lot"}
PRESS_TERMS = {"press", "article", "interview", "review", "magazine", "newspaper"}
FALSE_MATCH_TERMS = {
    "researcher",
    "scientist",
    "professor",
    "politician",
    "athlete",
    "footballer",
    "singer",
    "musician",
    "album",
    "song",
    "film",
    "novel",
    "artwork",
    "painting by",
    "museum",
    "gallery",
}


def generate_candidates(
    seed_list_path: str | Path = DEFAULT_SEED_LIST,
    output_path: str | Path = DEFAULT_OUTPUT,
    max_results_per_artist: int = 3,
    timeout_seconds: float = 8.0,
) -> pd.DataFrame:
    """Generate candidate observations and merge them with existing review rows."""
    seeds = _read_seed_list(Path(seed_list_path))
    generated = []
    for _, seed in seeds.iterrows():
        if not str(seed["canonical_name"]).strip():
            continue
        generated.extend(_wikidata_candidates(seed, max_results_per_artist, timeout_seconds))

    generated_frame = _align_candidates(pd.DataFrame(generated))
    existing = _reclassify_unreviewed_wikidata_rows(_read_candidates(Path(output_path)))
    output = _merge_preserving_review_fields(existing, generated_frame)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    return output


def _read_seed_list(path: Path) -> pd.DataFrame:
    """Read seed artists, creating an empty template when needed."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=SEED_COLUMNS).to_csv(path, index=False)
    seeds = pd.read_csv(path, dtype=str).fillna("")
    missing = [column for column in SEED_COLUMNS if column not in seeds.columns]
    if missing:
        raise ValueError(f"Missing seed list columns: {', '.join(missing)}")
    return seeds[list(SEED_COLUMNS)]


def _read_candidates(path: Path) -> pd.DataFrame:
    """Read candidate observations, creating an empty template when needed."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=CANDIDATE_COLUMNS).to_csv(path, index=False)
    rows = pd.read_csv(path, dtype=str).fillna("")
    for column in CANDIDATE_COLUMNS:
        if column not in rows.columns:
            rows[column] = ""
    return rows[list(CANDIDATE_COLUMNS)]


def _wikidata_candidates(seed: pd.Series, max_results: int, timeout_seconds: float) -> list[dict[str, str]]:
    """Return classified candidate observations from Wikidata entity search."""
    params = {
        "action": "wbsearchentities",
        "format": "json",
        "language": "en",
        "uselang": "en",
        "type": "item",
        "limit": str(max_results),
        "search": str(seed["canonical_name"]),
    }
    url = f"{WIKIDATA_API_URL}?{urlencode(params)}"
    request = Request(url, headers={"User-Agent": "art-network-intelligence-mvp/0.1"})
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return []

    raw_candidates = []
    for result in payload.get("search", []):
        source_url = str(result.get("concepturi") or "")
        if not source_url:
            continue
        observed_name = str(result.get("label") or "")
        description = str(result.get("description") or "")
        excerpt = _excerpt(observed_name, description)
        classification = _classify_wikidata_result(seed, observed_name, description)
        candidate = {
            "artist_id": str(seed["artist_id"]),
            "canonical_name": str(seed["canonical_name"]),
            "observation_type": classification["observation_type"],
            "observed_entity_name": observed_name,
            "event_name": observed_name if classification["observation_type"] == "museum_exhibition" else "",
            "event_date": "",
            "relationship_type": classification["relationship_type"],
            "source_url": source_url,
            "source_name": "Wikidata",
            "raw_text_excerpt": excerpt,
            "suggested_confidence_score": classification["suggested_confidence_score"],
            "accepted": "no",
            "needs_human_review_reason": classification["needs_human_review_reason"],
            "why_matched": classification["why_matched"],
            "review_notes": "",
        }
        candidate["candidate_id"] = _candidate_id(candidate)
        raw_candidates.append(candidate)
    return _dedupe_wikidata_candidates(raw_candidates)


def _classify_wikidata_result(seed: pd.Series, observed_name: str, description: str) -> dict[str, str]:
    """Classify a Wikidata search hit into a review-safe observation type."""
    seed_name = str(seed["canonical_name"])
    primary_medium = str(seed.get("primary_medium", ""))
    text = f"{observed_name} {description}".casefold()
    name_match = _name_match_strength(seed_name, observed_name)
    has_visual_artist_term = any(term in text for term in VISUAL_ARTIST_TERMS)
    has_generic_artist_term = any(term in text for term in GENERIC_ARTIST_TERMS)
    seed_has_visual_medium = any(term in primary_medium.casefold() for term in VISUAL_MEDIUM_TERMS)
    looks_like_exhibition = any(term in text for term in EXHIBITION_TERMS)
    looks_false = any(term in text for term in FALSE_MATCH_TERMS)

    if looks_like_exhibition:
        confidence = "0.45" if name_match else "0.25"
        return {
            "observation_type": "museum_exhibition",
            "relationship_type": "exhibited_at",
            "suggested_confidence_score": confidence,
            "needs_human_review_reason": "Wikidata result looks like an exhibition; confirm the artist participated and identify the host institution.",
            "why_matched": _why_matched(seed_name, observed_name, description, "exhibition-like Wikidata result"),
        }

    if any(term in text for term in GALLERY_REPRESENTATION_TERMS) and name_match:
        return _market_lead_classification(
            seed_name,
            observed_name,
            description,
            "gallery_representation",
            "represents",
            "Structured source contains gallery or representation language; verify the gallery and dates.",
        )

    if any(term in text for term in ACQUISITION_TERMS) and name_match:
        return _market_lead_classification(
            seed_name,
            observed_name,
            description,
            "museum_acquisition",
            "acquired_by",
            "Structured source contains acquisition or collection language; verify museum acquisition details.",
        )

    if any(term in text for term in AUCTION_TERMS) and name_match:
        return _market_lead_classification(
            seed_name,
            observed_name,
            description,
            "auction_result",
            "has_auction_result",
            "Structured source contains auction language; verify sale result, date, and price.",
        )

    if any(term in text for term in PRESS_TERMS) and name_match:
        return _market_lead_classification(
            seed_name,
            observed_name,
            description,
            "press_mention",
            "mentioned_in_press",
            "Structured source contains press/article language; verify outlet and publication date.",
        )

    if has_visual_artist_term and name_match:
        return {
            "observation_type": "artist_profile",
            "relationship_type": "identity_match",
            "suggested_confidence_score": "0.80" if name_match == "exact" else "0.65",
            "needs_human_review_reason": "Structured source appears to describe a visual artist; verify identity before accepting.",
            "why_matched": _why_matched(seed_name, observed_name, description, "name matched a visual-artist description"),
        }

    if has_generic_artist_term and name_match and not looks_false:
        confidence = "0.55" if seed_has_visual_medium else "0.45"
        return {
            "observation_type": "artist_profile",
            "relationship_type": "identity_match",
            "suggested_confidence_score": confidence,
            "needs_human_review_reason": "The source says artist but not clearly visual artist or painter; confirm this is the correct art-market artist.",
            "why_matched": _why_matched(seed_name, observed_name, description, "name matched a generic artist description"),
        }

    reason = "Result does not clearly describe the queried artist as a visual artist."
    if looks_false:
        reason = "Potential false match: result appears to be a non-artist, artwork, institution, exhibition, or unrelated entity."
    return {
        "observation_type": "rejected_match",
        "relationship_type": "not_applicable",
        "suggested_confidence_score": "0.10" if looks_false else "0.25",
        "needs_human_review_reason": reason,
        "why_matched": _why_matched(seed_name, observed_name, description, "Wikidata search returned a lexical match, but type evidence is weak"),
    }


def _market_lead_classification(
    seed_name: str,
    observed_name: str,
    description: str,
    observation_type: str,
    relationship_type: str,
    review_reason: str,
) -> dict[str, str]:
    """Return a conservative market-research lead from structured text."""
    return {
        "observation_type": observation_type,
        "relationship_type": relationship_type,
        "suggested_confidence_score": "0.40",
        "needs_human_review_reason": review_reason,
        "why_matched": _why_matched(seed_name, observed_name, description, f"{observation_type} language appeared in structured-source text"),
    }


def _merge_preserving_review_fields(existing: pd.DataFrame, generated: pd.DataFrame) -> pd.DataFrame:
    """Append new candidates without overwriting human accepted/review notes."""
    if existing.empty:
        return _align_candidates(generated)
    if generated.empty:
        return _align_candidates(existing)

    existing_ids = set(existing["candidate_id"].astype(str))
    new_rows = generated[~generated["candidate_id"].astype(str).isin(existing_ids)]
    merged = pd.concat([existing, new_rows], ignore_index=True)
    return _align_candidates(merged).drop_duplicates(subset=["candidate_id"], keep="first")


def _reclassify_unreviewed_wikidata_rows(existing: pd.DataFrame) -> pd.DataFrame:
    """Reclassify old unreviewed Wikidata candidates without touching reviewed rows."""
    if existing.empty:
        return existing
    rows = []
    for _, row in existing.iterrows():
        if not _can_reclassify_existing_candidate(row):
            rows.append(row.to_dict())
            continue
        seed = pd.Series(
            {
                "artist_id": row.get("artist_id", ""),
                "canonical_name": row.get("canonical_name", ""),
                "primary_medium": "",
            }
        )
        classification = _classify_wikidata_result(
            seed,
            str(row.get("observed_entity_name", "")),
            str(row.get("raw_text_excerpt", "")),
        )
        updated = row.to_dict()
        updated.update(
            {
                "observation_type": classification["observation_type"],
                "event_name": str(row.get("observed_entity_name", "")) if classification["observation_type"] == "museum_exhibition" else "",
                "relationship_type": classification["relationship_type"],
                "suggested_confidence_score": classification["suggested_confidence_score"],
                "accepted": "no",
                "needs_human_review_reason": classification["needs_human_review_reason"],
                "why_matched": classification["why_matched"],
            }
        )
        updated["candidate_id"] = _candidate_id(updated)
        rows.append(updated)
    return _align_candidates(pd.DataFrame(rows)).drop_duplicates(subset=["candidate_id"], keep="first")


def _can_reclassify_existing_candidate(row: pd.Series) -> bool:
    """Return whether an existing row is safe to update automatically."""
    if str(row.get("source_name", "")).strip().casefold() != "wikidata":
        return False
    if str(row.get("accepted", "")).strip().casefold() not in {"", "no"}:
        return False
    if str(row.get("review_notes", "")).strip():
        return False
    return True


def _align_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    """Align a candidate frame to the review schema."""
    if frame.empty:
        return pd.DataFrame(columns=CANDIDATE_COLUMNS)
    for column in CANDIDATE_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    frame = frame[list(CANDIDATE_COLUMNS)].fillna("")
    frame["accepted"] = frame["accepted"].replace("", "no")
    return frame


def _dedupe_wikidata_candidates(candidates: list[dict[str, str]]) -> list[dict[str, str]]:
    """Deduplicate Wikidata results, keeping only the best artist identity hit."""
    deduped_by_url = {candidate["source_url"]: candidate for candidate in candidates}
    ordered = sorted(deduped_by_url.values(), key=_candidate_sort_key, reverse=True)
    output = []
    artist_profile_seen = False
    seen_keys = set()
    for candidate in ordered:
        key = (
            candidate["artist_id"],
            candidate["observation_type"],
            candidate["observed_entity_name"].casefold(),
        )
        if key in seen_keys:
            continue
        if candidate["observation_type"] == "artist_profile":
            if artist_profile_seen:
                continue
            artist_profile_seen = True
        seen_keys.add(key)
        output.append(candidate)
    return output


def _candidate_sort_key(candidate: dict[str, str]) -> tuple[int, float]:
    """Sort candidates by product usefulness and confidence."""
    type_rank = {
        "artist_profile": 5,
        "museum_exhibition": 4,
        "gallery_representation": 3,
        "museum_acquisition": 3,
        "auction_result": 3,
        "press_mention": 2,
        "rejected_match": 1,
    }
    return (type_rank.get(candidate["observation_type"], 0), float(candidate["suggested_confidence_score"] or 0))


def _name_match_strength(seed_name: str, observed_name: str) -> str:
    """Return exact, partial, or blank name-match strength."""
    seed_normalized = _normalize_name(seed_name)
    observed_normalized = _normalize_name(observed_name)
    if seed_normalized and seed_normalized == observed_normalized:
        return "exact"
    if seed_normalized and (seed_normalized in observed_normalized or observed_normalized in seed_normalized):
        return "partial"
    return ""


def _normalize_name(value: str) -> str:
    """Normalize names for conservative entity matching."""
    decomposed = unicodedata.normalize("NFKD", str(value))
    ascii_text = "".join(character for character in decomposed if not unicodedata.combining(character))
    return " ".join(ascii_text.casefold().replace("-", " ").split())


def _why_matched(seed_name: str, observed_name: str, description: str, reason: str) -> str:
    """Explain why a candidate was generated."""
    name_strength = _name_match_strength(seed_name, observed_name) or "lexical"
    description_text = description or "no description"
    return f"{reason}; {name_strength} name match to '{seed_name}'; Wikidata description: {description_text}"


def _excerpt(label: str, description: str) -> str:
    """Build a short raw excerpt from structured-source text."""
    parts = [part for part in [label, description] if part]
    return " — ".join(parts)


def _candidate_id(candidate: dict[str, str]) -> str:
    """Create a stable candidate ID."""
    raw = "|".join(
        str(candidate.get(column, ""))
        for column in ["artist_id", "observation_type", "observed_entity_name", "event_name", "event_date", "source_url"]
    )
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"candidate_{digest}"


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Generate sourced candidate observations for review.")
    parser.add_argument("--seed-list", default=str(DEFAULT_SEED_LIST), help="Path to artist seed list CSV.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Path to candidate observations CSV.")
    parser.add_argument("--max-results-per-artist", type=int, default=3, help="Maximum structured-source matches per artist.")
    return parser.parse_args()


def main() -> None:
    """Run candidate generation."""
    args = _parse_args()
    output = generate_candidates(args.seed_list, args.output, args.max_results_per_artist)
    print(f"Wrote {len(output)} candidate rows to {args.output}")
    print("Review candidates and set accepted=yes before promotion.")


if __name__ == "__main__":
    main()
