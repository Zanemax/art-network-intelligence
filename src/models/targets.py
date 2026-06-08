"""Target-label generation for artist success outcomes.

Labels intentionally use the future window after ``prediction_date``. Feature
generation should be run separately on data filtered as of ``prediction_date``
so model inputs do not leak future information.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class TargetConfig:
    """Configurable definitions for artist success labels."""

    prediction_window_years: int = 3
    institutional_success_events: tuple[str, ...] = (
        "major_solo_show",
        "solo_show",
        "biennial",
        "biennial_inclusion",
    )
    major_museum_tiers: tuple[str, ...] = ("top", "major")
    market_growth_multiple: float = 2.0
    gallery_tier_rank: dict[str, int] = field(
        default_factory=lambda: {
            "emerging": 1,
            "local": 1,
            "mid": 2,
            "regional": 2,
            "major": 3,
            "top": 4,
            "mega": 5,
        }
    )
    require_existing_gallery_for_upgrade: bool = True


TARGET_COLUMNS = (
    "institutional_success_3y",
    "market_success_3y",
    "gallery_success_3y",
)


def generate_artist_targets(
    dataset: dict[str, pd.DataFrame],
    prediction_date: str,
    config: TargetConfig | None = None,
) -> pd.DataFrame:
    """Generate configurable artist success labels after ``prediction_date``."""
    config = config or TargetConfig()
    artists = dataset["artists"][["artist_id"]].copy()
    window = _future_window(prediction_date, config.prediction_window_years)

    targets = artists.copy()
    targets["institutional_success_3y"] = targets["artist_id"].map(
        _institutional_success(dataset, window, config)
    ).fillna(0).astype(int)
    targets["market_success_3y"] = targets["artist_id"].map(
        _market_success(dataset, window, config)
    ).fillna(0).astype(int)
    targets["gallery_success_3y"] = targets["artist_id"].map(
        _gallery_success(dataset, window, config)
    ).fillna(0).astype(int)

    return targets


def _future_window(prediction_date: str, years: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return the inclusive future labeling window."""
    start = pd.Timestamp(prediction_date)
    end = start + pd.DateOffset(years=years)
    return start, end


def _institutional_success(
    dataset: dict[str, pd.DataFrame],
    window: tuple[pd.Timestamp, pd.Timestamp],
    config: TargetConfig,
) -> dict[str, int]:
    """Label artists with major institutional acquisition or event success."""
    successes: set[str] = set()
    acquisitions = dataset.get("acquisitions", pd.DataFrame())
    museums = dataset.get("museums", pd.DataFrame())

    if not acquisitions.empty and not museums.empty:
        museum_id_column = "museum_id" if "museum_id" in museums.columns else "institution_id"
        joined = acquisitions.merge(
            museums,
            left_on="museum_id",
            right_on=museum_id_column,
            how="left",
            suffixes=("", "_museum"),
        )
        date_column = _first_existing_column(joined, ("date", "acquisition_date", "event_date"))
        if date_column:
            future = _rows_in_window(joined, date_column, window)
            if "is_major" in future.columns:
                major = future[future["is_major"].astype(bool)]
            elif "tier" in future.columns:
                major = future[future["tier"].isin(config.major_museum_tiers)]
            else:
                major = future.iloc[0:0]
            successes.update(major["artist_id"].dropna().astype(str))

    event_successes = _institutional_event_successes(dataset, window, config)
    successes.update(event_successes)
    return {artist_id: 1 for artist_id in successes}


def _institutional_event_successes(
    dataset: dict[str, pd.DataFrame],
    window: tuple[pd.Timestamp, pd.Timestamp],
    config: TargetConfig,
) -> set[str]:
    """Find artists with major solo shows or biennial inclusions."""
    event_tables = [
        dataset.get("events", pd.DataFrame()),
        dataset.get("exhibitions", pd.DataFrame()),
    ]
    successes: set[str] = set()

    for events in event_tables:
        if events.empty or "artist_id" not in events.columns:
            continue
        date_column = _first_existing_column(events, ("event_date", "start_date", "date"))
        type_column = _first_existing_column(events, ("event_type", "exhibition_type", "type"))
        if not date_column:
            continue
        future = _rows_in_window(events, date_column, window)
        if type_column:
            future = future[future[type_column].isin(config.institutional_success_events)]
        elif "is_major" in future.columns:
            future = future[future["is_major"].astype(bool)]
        else:
            future = future.iloc[0:0]
        successes.update(future["artist_id"].dropna().astype(str))

    return successes


def _market_success(
    dataset: dict[str, pd.DataFrame],
    window: tuple[pd.Timestamp, pd.Timestamp],
    config: TargetConfig,
) -> dict[str, int]:
    """Label artists whose future median auction price reaches the threshold."""
    auction_results = dataset.get("auction_results", pd.DataFrame())
    if auction_results.empty:
        return {}

    sales = auction_results.copy()
    sales["sale_date"] = pd.to_datetime(sales["sale_date"], errors="coerce")
    prediction_date, window_end = window
    before = sales[sales["sale_date"] <= prediction_date]
    future = sales[(sales["sale_date"] > prediction_date) & (sales["sale_date"] <= window_end)]

    baseline = before.groupby("artist_id")["price_usd"].median()
    future_median = future.groupby("artist_id")["price_usd"].median()
    growth = future_median / baseline
    successes = growth[growth >= config.market_growth_multiple].index
    return {str(artist_id): 1 for artist_id in successes}


def _gallery_success(
    dataset: dict[str, pd.DataFrame],
    window: tuple[pd.Timestamp, pd.Timestamp],
    config: TargetConfig,
) -> dict[str, int]:
    """Label artists who move to a higher-tier gallery in the future window."""
    representation = dataset.get("representation", pd.DataFrame())
    galleries = dataset.get("galleries", pd.DataFrame())
    if representation.empty or galleries.empty:
        return {}

    gallery_scores = _gallery_scores(galleries, config)
    reps = representation.copy()
    reps["start_date"] = pd.to_datetime(reps["start_date"], errors="coerce")
    reps["gallery_score"] = reps["gallery_id"].map(gallery_scores).fillna(0)
    prediction_date, window_end = window

    current = reps[reps["start_date"] <= prediction_date].groupby("artist_id")["gallery_score"].max()
    future = reps[
        (reps["start_date"] > prediction_date) & (reps["start_date"] <= window_end)
    ].groupby("artist_id")["gallery_score"].max()

    successes: dict[str, int] = {}
    for artist_id, future_score in future.items():
        current_score = current.get(artist_id)
        if pd.isna(current_score):
            if not config.require_existing_gallery_for_upgrade and future_score > 0:
                successes[str(artist_id)] = 1
            continue
        if future_score > current_score:
            successes[str(artist_id)] = 1

    return successes


def _gallery_scores(galleries: pd.DataFrame, config: TargetConfig) -> pd.Series:
    """Return comparable gallery scores from tier labels or prestige scores."""
    if "tier" in galleries.columns:
        return galleries.set_index("gallery_id")["tier"].map(config.gallery_tier_rank).fillna(0)
    if "prestige_score" in galleries.columns:
        return galleries.set_index("gallery_id")["prestige_score"]
    return pd.Series(0, index=galleries["gallery_id"])


def _rows_in_window(
    frame: pd.DataFrame,
    date_column: str,
    window: tuple[pd.Timestamp, pd.Timestamp],
) -> pd.DataFrame:
    """Return rows after prediction date and on or before window end."""
    prediction_date, window_end = window
    dated = frame.copy()
    dated[date_column] = pd.to_datetime(dated[date_column], errors="coerce")
    return dated[(dated[date_column] > prediction_date) & (dated[date_column] <= window_end)]


def _first_existing_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    """Return the first candidate column present in a DataFrame."""
    return next((column for column in candidates if column in frame.columns), None)
