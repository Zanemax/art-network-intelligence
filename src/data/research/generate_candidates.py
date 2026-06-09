"""Generate sourced candidate observations for human research review.

This module intentionally does not ingest data into the graph. It reads a seed
list of artists, queries conservative structured sources where available, and
writes candidate observations that a human must review before promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import json
import unicodedata
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlencode, urljoin, urlparse
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
DEFAULT_SOURCE_PAGES = Path("data/raw/manual/source_pages.csv")
WIKIDATA_API_URL = "https://www.wikidata.org/w/api.php"
SOURCE_PAGE_COLUMNS = (
    "source_id",
    "artist_id",
    "canonical_name",
    "source_type",
    "source_name",
    "url",
    "notes",
)

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
    source_pages_path: str | Path = DEFAULT_SOURCE_PAGES,
    max_source_links_per_page: int = 12,
) -> pd.DataFrame:
    """Generate candidate observations and merge them with existing review rows."""
    seeds = _read_seed_list(Path(seed_list_path))
    source_pages = _read_source_pages(Path(source_pages_path))
    generated = []
    for _, seed in seeds.iterrows():
        if not str(seed["canonical_name"]).strip():
            continue
        generated.extend(_wikidata_candidates(seed, max_results_per_artist, timeout_seconds))
        generated.extend(_source_page_candidates(seed, source_pages, timeout_seconds, max_source_links_per_page))

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


def _read_source_pages(path: Path) -> pd.DataFrame:
    """Read curated source pages for source-specific candidate collection."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=SOURCE_PAGE_COLUMNS).to_csv(path, index=False)
    rows = pd.read_csv(path, dtype=str).fillna("")
    for column in SOURCE_PAGE_COLUMNS:
        if column not in rows.columns:
            rows[column] = ""
    return rows[list(SOURCE_PAGE_COLUMNS)]


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


def _source_page_candidates(
    seed: pd.Series,
    source_pages: pd.DataFrame,
    timeout_seconds: float,
    max_links_per_page: int,
) -> list[dict[str, str]]:
    """Return candidates from curated gallery, museum, auction, and press pages."""
    rows = []
    for _, source in source_pages.iterrows():
        if not _source_applies_to_seed(source, seed):
            continue
        source_url = str(source["url"]).strip()
        if not source_url:
            continue
        page = _fetch_source_page(source_url, timeout_seconds)
        if not page:
            continue
        rows.extend(_candidates_from_source_page(seed, source, source_url, page))
        for link in _artist_links_from_page(page, source_url, str(seed["canonical_name"]), max_links_per_page):
            if link["url"] == source_url:
                continue
            linked_page = _fetch_source_page(link["url"], timeout_seconds)
            if not linked_page:
                linked_page = {"title": link["text"], "text": link["text"], "links": []}
            rows.extend(_candidates_from_source_page(seed, source, link["url"], linked_page, link_text=link["text"]))
    return _dedupe_source_candidates(rows)


def _source_applies_to_seed(source: pd.Series, seed: pd.Series) -> bool:
    """Return whether a configured source should be searched for a seed artist."""
    source_artist_id = str(source.get("artist_id", "")).strip()
    source_name = str(source.get("canonical_name", "")).strip()
    if source_artist_id and source_artist_id != str(seed["artist_id"]):
        return False
    if source_name and not _name_match_strength(str(seed["canonical_name"]), source_name):
        return False
    return True


def _fetch_source_page(url: str, timeout_seconds: float) -> dict[str, object]:
    """Fetch and parse a source page into text and links."""
    request = Request(url, headers={"User-Agent": "art-network-intelligence-research/0.2"})
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            content_type = response.headers.get("content-type", "")
            if "html" not in content_type and "text" not in content_type:
                return {}
            html = response.read().decode("utf-8", errors="replace")
    except Exception:
        return {}
    parser = _SourceHTMLParser()
    parser.feed(html)
    return {
        "title": parser.title.strip(),
        "text": " ".join(parser.text_parts),
        "links": parser.links,
    }


def _candidates_from_source_page(
    seed: pd.Series,
    source: pd.Series,
    url: str,
    page: dict[str, object],
    link_text: str = "",
) -> list[dict[str, str]]:
    """Create review candidates when a source page appears to mention a seed artist."""
    seed_name = str(seed["canonical_name"])
    page_text = str(page.get("text", ""))
    title = str(page.get("title", ""))
    excerpt = _source_excerpt(seed_name, " ".join(part for part in [link_text, title, page_text] if part))
    if not excerpt and not _name_match_strength(seed_name, title) and not _name_match_strength(seed_name, link_text):
        return []

    source_type = str(source["source_type"]).strip().casefold()
    classification = _classify_source_page(source_type, seed_name, title, link_text, page_text)
    observed_entity = classification["observed_entity_name"] or str(source["source_name"]) or title or seed_name
    if classification["observation_type"] == "gallery_exhibition" and str(source["source_name"]).strip():
        observed_entity = str(source["source_name"]).strip()
    candidate = {
        "artist_id": str(seed["artist_id"]),
        "canonical_name": seed_name,
        "observation_type": classification["observation_type"],
        "observed_entity_name": observed_entity,
        "event_name": classification["event_name"],
        "event_date": classification["event_date"],
        "relationship_type": classification["relationship_type"],
        "source_url": url,
        "source_name": str(source["source_name"]) or _source_name_from_url(url),
        "raw_text_excerpt": excerpt or _excerpt(title or link_text, page_text[:180]),
        "suggested_confidence_score": classification["suggested_confidence_score"],
        "accepted": "no",
        "needs_human_review_reason": classification["needs_human_review_reason"],
        "why_matched": classification["why_matched"],
        "review_notes": classification["review_notes"],
    }
    candidate["candidate_id"] = _candidate_id(candidate)
    return [candidate]


def _classify_source_page(
    source_type: str,
    seed_name: str,
    title: str,
    link_text: str,
    page_text: str,
) -> dict[str, str]:
    """Classify a curated source-page hit into a reviewable observation."""
    text = " ".join(part for part in [source_type, title, link_text, page_text] if part)
    text_casefold = text.casefold()
    event_date = _extract_date(text)
    event_name = _clean_event_name(title or link_text)
    observed = event_name or title or link_text or seed_name
    why = f"Curated {source_type or 'source'} page mentioned '{seed_name}'."
    review_notes = ""

    if source_type in {"gallery", "gallery_exhibition", "exhibition"} or any(term in text_casefold for term in EXHIBITION_TERMS):
        is_representation = _looks_like_representation(text_casefold)
        is_gallery_exhibition = (source_type in {"gallery", "gallery_exhibition"} or "gallery" in text_casefold) and not is_representation
        return {
            "observation_type": "gallery_representation" if is_representation else "gallery_exhibition" if is_gallery_exhibition else "museum_exhibition",
            "observed_entity_name": observed,
            "event_name": event_name or observed,
            "event_date": event_date,
            "relationship_type": "represents" if is_representation else "gallery_exhibition" if is_gallery_exhibition else "exhibited_at",
            "suggested_confidence_score": "0.85" if source_type in {"gallery", "museum"} else "0.70",
            "needs_human_review_reason": "Source page appears to describe an exhibition or gallery relationship; verify host, dates, and whether this is representation or only a venue.",
            "why_matched": why,
            "review_notes": review_notes,
        }

    if source_type in {"museum", "institution"}:
        return {
            "observation_type": "museum_exhibition",
            "observed_entity_name": observed,
            "event_name": event_name or observed,
            "event_date": event_date,
            "relationship_type": "exhibited_at",
            "suggested_confidence_score": "0.85",
            "needs_human_review_reason": "Museum or institution page mentioned the artist; verify event/acquisition details.",
            "why_matched": why,
            "review_notes": "",
        }

    if source_type in {"auction", "auction_house"} or any(term in text_casefold for term in AUCTION_TERMS):
        return {
            "observation_type": "auction_result",
            "observed_entity_name": observed,
            "event_name": "",
            "event_date": event_date,
            "relationship_type": "has_auction_result",
            "suggested_confidence_score": "0.85" if source_type in {"auction", "auction_house"} else "0.65",
            "needs_human_review_reason": "Auction-like source page mentioned the artist; verify lot, sale date, and price.",
            "why_matched": why,
            "review_notes": "",
        }

    if source_type in {"press", "publication"} or any(term in text_casefold for term in PRESS_TERMS):
        return {
            "observation_type": "press_mention",
            "observed_entity_name": observed,
            "event_name": "",
            "event_date": event_date,
            "relationship_type": "mentioned_in_press",
            "suggested_confidence_score": "0.75",
            "needs_human_review_reason": "Press-like source page mentioned the artist; verify outlet, date, and article relevance.",
            "why_matched": why,
            "review_notes": "",
        }

    return {
        "observation_type": "artist_profile",
        "observed_entity_name": observed,
        "event_name": "",
        "event_date": event_date,
        "relationship_type": "identity_match",
        "suggested_confidence_score": "0.65",
        "needs_human_review_reason": "Curated source page mentioned the artist; classify the observation before accepting.",
        "why_matched": why,
        "review_notes": "",
    }


def _artist_links_from_page(
    page: dict[str, object],
    base_url: str,
    artist_name: str,
    max_links: int,
) -> list[dict[str, str]]:
    """Find same-domain links likely to contain artist-specific evidence."""
    base_domain = _domain(base_url)
    artist_tokens = set(_normalize_name(artist_name).split())
    candidates = []
    seen = set()
    for link in page.get("links", []):
        href = str(link.get("href", "")).strip()
        text = str(link.get("text", "")).strip()
        if not href:
            continue
        url = urljoin(base_url, href)
        if _domain(url) != base_domain or url in seen:
            continue
        link_text = f"{text} {urlparse(url).path}".casefold()
        if not artist_tokens.intersection(set(_normalize_name(link_text).split())):
            continue
        seen.add(url)
        candidates.append({"url": url, "text": text})
        if len(candidates) >= max_links:
            break
    return candidates


def _dedupe_source_candidates(candidates: list[dict[str, str]]) -> list[dict[str, str]]:
    """Deduplicate source candidates by stable ID."""
    return list({candidate["candidate_id"]: candidate for candidate in candidates}.values())


def _source_excerpt(artist_name: str, text: str, window: int = 260) -> str:
    """Return a compact excerpt around the artist name."""
    clean = " ".join(str(text or "").split())
    if not clean:
        return ""
    index = clean.casefold().find(artist_name.casefold())
    if index < 0:
        return clean[:window]
    start = max(0, index - window // 2)
    end = min(len(clean), index + len(artist_name) + window // 2)
    return clean[start:end]


def _clean_event_name(value: str) -> str:
    """Clean a page title or link text into a candidate event name."""
    text = re.sub(r"\s+", " ", str(value or "")).strip(" |–—-")
    return text[:180]


def _extract_date(text: str) -> str:
    """Extract a conservative date-like value from source text."""
    raw = str(text or "")
    iso = re.search(r"\b(20\d{2}|19\d{2})[-/.](\d{1,2})[-/.](\d{1,2})\b", raw)
    if iso:
        year, month, day = iso.groups()
        return f"{year}-{int(month):02d}-{int(day):02d}"
    day_month_year = re.search(
        r"\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(20\d{2}|19\d{2})\b",
        raw,
        flags=re.IGNORECASE,
    )
    if day_month_year:
        day, month_name, year = day_month_year.groups()
        month = pd.Timestamp(f"1 {month_name} {year}").month
        return f"{year}-{month:02d}-{int(day):02d}"
    year = re.search(r"\b(20\d{2}|19\d{2})\b", raw)
    return year.group(1) if year else ""


def _looks_like_representation(text_casefold: str) -> bool:
    """Return whether text looks like gallery representation rather than a venue listing."""
    return any(phrase in text_casefold for phrase in {"represented by", "represents ", "is pleased to represent", "gallery artists"})


def _source_name_from_url(url: str) -> str:
    """Return a readable source name from a URL host."""
    domain = _domain(url)
    return domain or "Source page"


def _domain(url: str) -> str:
    """Return normalized URL domain."""
    return urlparse(str(url)).netloc.lower().removeprefix("www.")


class _SourceHTMLParser(HTMLParser):
    """Small HTML text/link parser for curated source pages."""

    def __init__(self) -> None:
        super().__init__()
        self.text_parts: list[str] = []
        self.links: list[dict[str, str]] = []
        self.title = ""
        self._in_title = False
        self._active_href = ""
        self._active_link_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "title":
            self._in_title = True
        if tag == "a":
            attr_map = dict(attrs)
            self._active_href = str(attr_map.get("href") or "")
            self._active_link_text = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        if tag == "a" and self._active_href:
            self.links.append({"href": self._active_href, "text": " ".join(self._active_link_text).strip()})
            self._active_href = ""
            self._active_link_text = []

    def handle_data(self, data: str) -> None:
        text = " ".join(str(data or "").split())
        if not text:
            return
        if self._in_title:
            self.title = f"{self.title} {text}".strip()
        if self._active_href:
            self._active_link_text.append(text)
        self.text_parts.append(text)


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
        "gallery_exhibition": 4,
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
    parser.add_argument("--source-pages", default=str(DEFAULT_SOURCE_PAGES), help="Path to curated source pages CSV.")
    parser.add_argument("--max-results-per-artist", type=int, default=3, help="Maximum structured-source matches per artist.")
    parser.add_argument("--max-source-links-per-page", type=int, default=12, help="Maximum artist-matching links to follow per source page.")
    return parser.parse_args()


def main() -> None:
    """Run candidate generation."""
    args = _parse_args()
    output = generate_candidates(
        args.seed_list,
        args.output,
        args.max_results_per_artist,
        source_pages_path=args.source_pages,
        max_source_links_per_page=args.max_source_links_per_page,
    )
    print(f"Wrote {len(output)} candidate rows to {args.output}")
    print("Review candidates and set accepted=yes before promotion.")


if __name__ == "__main__":
    main()
