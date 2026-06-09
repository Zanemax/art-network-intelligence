"""Promote human-approved candidate observations into manual research format.

Only rows with ``accepted=yes`` are promoted. The output is a manual research
CSV compatible with ``python -m src.data.import_research_template``; this keeps
human review in the loop before anything reaches the graph.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.import_research_template import RESEARCH_REQUIRED_COLUMNS
from src.data.research.generate_candidates import CANDIDATE_COLUMNS, DEFAULT_OUTPUT as DEFAULT_CANDIDATES


DEFAULT_PROMOTED_TEMPLATE = Path("data/raw/manual/accepted_artist_research_template.csv")
ACCEPTED_VALUES = {"yes", "y", "true", "1"}
REVIEW_OVERRIDE_PREFIX = "template."
REVIEW_OVERRIDE_PROTECTED_COLUMNS = {"artist_id", "artist_name", "notes"}


def promote_accepted_candidates(
    candidate_path: str | Path = DEFAULT_CANDIDATES,
    output_path: str | Path = DEFAULT_PROMOTED_TEMPLATE,
) -> pd.DataFrame:
    """Append accepted candidate observations to a manual research template CSV."""
    candidates = _read_candidate_observations(Path(candidate_path))
    accepted = candidates[candidates["accepted"].map(_is_accepted)].copy()
    _validate_accepted_candidates(accepted)

    promoted = pd.DataFrame([_candidate_to_research_row(row) for _, row in accepted.iterrows()])
    promoted = _align_research_rows(promoted)

    existing = _read_existing_output(Path(output_path))
    output = _append_new_promotions(existing, promoted)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    return output


def _read_candidate_observations(path: Path) -> pd.DataFrame:
    """Read candidate observations and validate the review schema."""
    if not path.exists():
        raise FileNotFoundError(f"Candidate observations file does not exist: {path}")
    rows = pd.read_csv(path, dtype=str).fillna("")
    for column in CANDIDATE_COLUMNS:
        if column not in rows.columns:
            rows[column] = ""
    return rows[list(CANDIDATE_COLUMNS)]


def _read_existing_output(path: Path) -> pd.DataFrame:
    """Read an existing promoted template, or return an empty template."""
    if not path.exists():
        return pd.DataFrame(columns=RESEARCH_REQUIRED_COLUMNS)
    rows = pd.read_csv(path, dtype=str).fillna("")
    missing = [column for column in RESEARCH_REQUIRED_COLUMNS if column not in rows.columns]
    if missing:
        raise ValueError(f"Existing promoted template is missing columns: {', '.join(missing)}")
    return rows[list(RESEARCH_REQUIRED_COLUMNS)]


def _validate_accepted_candidates(rows: pd.DataFrame) -> None:
    """Ensure accepted candidates have enough provenance for promotion."""
    if rows.empty:
        return
    missing_source = rows[rows["source_url"].str.strip().eq("")]
    if not missing_source.empty:
        ids = ", ".join(missing_source["candidate_id"].astype(str).tolist())
        raise ValueError(f"Accepted candidates must include source_url: {ids}")
    missing_artist = rows[rows["artist_id"].str.strip().eq("") | rows["canonical_name"].str.strip().eq("")]
    if not missing_artist.empty:
        ids = ", ".join(missing_artist["candidate_id"].astype(str).tolist())
        raise ValueError(f"Accepted candidates must include artist_id and canonical_name: {ids}")


def _candidate_to_research_row(candidate: pd.Series) -> dict[str, str]:
    """Convert one accepted candidate into the broad manual research row shape."""
    row = {column: "" for column in RESEARCH_REQUIRED_COLUMNS}
    row["artist_id"] = str(candidate["artist_id"])
    row["artist_name"] = str(candidate["canonical_name"])
    row["notes"] = _notes(candidate)

    observation_type = str(candidate["observation_type"]).strip().lower()
    relationship_type = str(candidate["relationship_type"]).strip().lower()
    observed_entity = str(candidate["observed_entity_name"]).strip()
    event_name = str(candidate["event_name"]).strip()
    event_date = str(candidate["event_date"]).strip()
    source_url = str(candidate["source_url"]).strip()
    confidence = str(candidate["suggested_confidence_score"]).strip() or "0.6"

    if observation_type in {"artist_profile", "identity", "bio"}:
        row["bio_source_url"] = source_url
        row["bio_confidence_score"] = confidence
    elif observation_type in {"gallery_exhibition"} or relationship_type in {"gallery_exhibition"}:
        row["gallery_name"] = observed_entity
        row["museum_event_type"] = "gallery_exhibition"
        row["event_name"] = event_name or observed_entity
        row["event_start_date"] = event_date
        row["event_source_url"] = source_url
        row["event_confidence_score"] = confidence
    elif observation_type in {"gallery_representation", "gallery"} or relationship_type in {"represents", "represented_by"}:
        row["gallery_name"] = observed_entity
        row["gallery_start_date"] = event_date
        row["gallery_source_url"] = source_url
        row["gallery_confidence_score"] = confidence
    elif observation_type in {"museum_exhibition", "exhibition", "biennial_inclusion"} or relationship_type in {"exhibited_at", "included_in", "museum_exhibition"}:
        row["museum_name"] = observed_entity
        row["museum_event_type"] = observation_type or "museum_exhibition"
        row["event_name"] = event_name or observed_entity
        row["event_start_date"] = event_date
        row["event_source_url"] = source_url
        row["event_confidence_score"] = confidence
    elif observation_type in {"museum_acquisition", "acquisition"} or relationship_type in {"acquired_by", "acquired_artist"}:
        row["museum_name"] = observed_entity
        row["acquisition_date"] = event_date
        row["acquisition_source_url"] = source_url
        row["acquisition_confidence_score"] = confidence
    elif observation_type in {"art_fair", "fair"}:
        row["art_fair_name"] = observed_entity or event_name
        row["art_fair_date"] = event_date
        row["art_fair_source_url"] = source_url
        row["art_fair_confidence_score"] = confidence
    elif observation_type in {"auction_result", "auction"}:
        row["auction_house"] = observed_entity
        row["sale_name"] = event_name
        row["sale_date"] = event_date
        row["auction_source_url"] = source_url
        row["auction_confidence_score"] = confidence
    elif observation_type in {"press_mention", "press"} or relationship_type in {"mentioned_in_press", "press_mention"}:
        row["press_outlet"] = observed_entity
        row["article_title"] = event_name
        row["publication_date"] = event_date
        row["article_url"] = source_url
        row["press_confidence_score"] = confidence
    elif observation_type in {"collector_signal", "collector"} or relationship_type in {"collects", "collected_by"}:
        row["collector_name"] = observed_entity
        row["collector_source_url"] = source_url
        row["collector_confidence_score"] = confidence
    elif observation_type in {"curator_signal", "curator"} or relationship_type in {"curated_by", "curated_artist"}:
        row["curator_name"] = observed_entity
        row["curator_source_url"] = source_url
        row["curator_confidence_score"] = confidence
        row["event_start_date"] = event_date
    else:
        row["bio_source_url"] = source_url
        row["bio_confidence_score"] = confidence
    row.update(_review_field_overrides(candidate))
    return row


def _append_new_promotions(existing: pd.DataFrame, promoted: pd.DataFrame) -> pd.DataFrame:
    """Append promotions that are not already present by candidate marker."""
    if promoted.empty:
        return _align_research_rows(existing)
    existing_notes = "\n".join(existing.get("notes", pd.Series(dtype=str)).astype(str).tolist())
    new_rows = promoted[
        ~promoted["notes"].map(lambda note: _candidate_marker_from_notes(note) in existing_notes)
    ]
    output = pd.concat([existing, new_rows], ignore_index=True)
    return _align_research_rows(output)


def _align_research_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Align rows to the manual research template schema."""
    if frame.empty:
        return pd.DataFrame(columns=RESEARCH_REQUIRED_COLUMNS)
    for column in RESEARCH_REQUIRED_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    return frame[list(RESEARCH_REQUIRED_COLUMNS)].fillna("")


def _is_accepted(value: str) -> bool:
    """Return whether a review value means accepted."""
    return str(value).strip().casefold() in ACCEPTED_VALUES


def _review_field_overrides(candidate: pd.Series) -> dict[str, str]:
    """Parse template-column overrides from review notes."""
    notes = str(candidate.get("review_notes", "")).strip()
    if not notes:
        return {}
    overrides = {}
    for part in _review_note_parts(notes):
        if "=" not in part:
            continue
        raw_key, raw_value = part.split("=", 1)
        key = raw_key.strip()
        value = raw_value.strip()
        if key.startswith(REVIEW_OVERRIDE_PREFIX):
            key = key.removeprefix(REVIEW_OVERRIDE_PREFIX).strip()
            _validate_review_override_key(key, candidate)
        elif key not in RESEARCH_REQUIRED_COLUMNS:
            continue
        elif key in REVIEW_OVERRIDE_PROTECTED_COLUMNS:
            _validate_review_override_key(key, candidate)
        overrides[key] = value
    return overrides


def _review_note_parts(notes: str) -> list[str]:
    """Split review notes into possible override fragments."""
    return [part.strip() for chunk in str(notes).split("|") for part in chunk.split(";") if part.strip()]


def _validate_review_override_key(key: str, candidate: pd.Series) -> None:
    """Validate a requested template override key."""
    if key in REVIEW_OVERRIDE_PROTECTED_COLUMNS:
        raise ValueError(f"Review override cannot set protected column {key}: {candidate['candidate_id']}")
    if key not in RESEARCH_REQUIRED_COLUMNS:
        raise ValueError(f"Unknown review override column {key}: {candidate['candidate_id']}")


def _notes(candidate: pd.Series) -> str:
    """Build provenance notes for an accepted candidate."""
    parts = [
        _candidate_marker(str(candidate["candidate_id"])),
        f"source_name={candidate['source_name']}",
    ]
    if str(candidate["raw_text_excerpt"]).strip():
        parts.append(f"excerpt={candidate['raw_text_excerpt']}")
    if str(candidate.get("why_matched", "")).strip():
        parts.append(f"why_matched={candidate['why_matched']}")
    if str(candidate.get("needs_human_review_reason", "")).strip():
        parts.append(f"review_reason={candidate['needs_human_review_reason']}")
    if str(candidate["review_notes"]).strip():
        parts.append(f"review_notes={candidate['review_notes']}")
    return " | ".join(parts)


def _candidate_marker(candidate_id: str) -> str:
    """Return a stable marker used to avoid duplicate promotion."""
    return f"candidate_id={candidate_id}"


def _candidate_marker_from_notes(notes: str) -> str:
    """Extract the candidate marker from promoted notes."""
    return str(notes).split(" | ", 1)[0]


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Promote accepted candidate observations into manual research format.")
    parser.add_argument("candidate_path", nargs="?", default=str(DEFAULT_CANDIDATES), help="Path to candidate_observations.csv")
    parser.add_argument("--output", default=str(DEFAULT_PROMOTED_TEMPLATE), help="Manual research CSV output path.")
    return parser.parse_args()


def main() -> None:
    """Run candidate promotion."""
    args = _parse_args()
    output = promote_accepted_candidates(args.candidate_path, args.output)
    print(f"Wrote {len(output)} promoted manual research rows to {args.output}")
    print(f"Next: python -m src.data.import_research_template {args.output}")


if __name__ == "__main__":
    main()
