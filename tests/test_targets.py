"""Tests for artist success target-label generation."""

import pandas as pd

from src.models.targets import TargetConfig, TARGET_COLUMNS, generate_artist_targets


def test_generate_artist_targets_uses_future_window_only() -> None:
    """Verify each configured target is labeled from the post-prediction window."""
    dataset = _target_fixture()

    targets = generate_artist_targets(dataset, prediction_date="2021-01-01")
    target_by_artist = targets.set_index("artist_id")

    assert set(TARGET_COLUMNS).issubset(targets.columns)
    assert target_by_artist.loc["artist_a", "institutional_success_3y"] == 1
    assert target_by_artist.loc["artist_b", "market_success_3y"] == 1
    assert target_by_artist.loc["artist_c", "gallery_success_3y"] == 1
    assert target_by_artist.loc["artist_d", list(TARGET_COLUMNS)].sum() == 0


def test_market_success_requires_baseline_before_prediction_date() -> None:
    """Verify future auction results alone are not treated as price growth."""
    dataset = _target_fixture()
    dataset["auction_results"] = pd.concat(
        [
            dataset["auction_results"],
            pd.DataFrame(
                [
                    {
                        "artist_id": "artist_d",
                        "sale_date": "2022-05-01",
                        "price_usd": 500000,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    targets = generate_artist_targets(dataset, prediction_date="2021-01-01")

    assert targets.set_index("artist_id").loc["artist_d", "market_success_3y"] == 0


def test_target_definitions_are_configurable() -> None:
    """Verify configurable thresholds and event definitions affect labels."""
    dataset = _target_fixture()
    config = TargetConfig(
        market_growth_multiple=3.0,
        institutional_success_events=("biennial_inclusion",),
        major_museum_tiers=("top",),
        gallery_tier_rank={"emerging": 1, "mid": 2, "top": 3},
    )

    targets = generate_artist_targets(dataset, prediction_date="2021-01-01", config=config)
    target_by_artist = targets.set_index("artist_id")

    assert target_by_artist.loc["artist_a", "institutional_success_3y"] == 1
    assert target_by_artist.loc["artist_b", "market_success_3y"] == 0


def test_success_after_three_year_window_is_excluded() -> None:
    """Verify labels ignore outcomes occurring after the configured horizon."""
    dataset = _target_fixture()
    targets = generate_artist_targets(dataset, prediction_date="2021-01-01")

    assert targets.set_index("artist_id").loc["artist_d", "institutional_success_3y"] == 0


def _target_fixture() -> dict[str, pd.DataFrame]:
    """Create a compact fixture for target-label tests."""
    return {
        "artists": pd.DataFrame(
            [
                {"artist_id": "artist_a", "name": "Artist A"},
                {"artist_id": "artist_b", "name": "Artist B"},
                {"artist_id": "artist_c", "name": "Artist C"},
                {"artist_id": "artist_d", "name": "Artist D"},
            ]
        ),
        "museums": pd.DataFrame(
            [
                {"museum_id": "museum_top", "name": "Top Museum", "tier": "top"},
                {"museum_id": "museum_mid", "name": "Mid Museum", "tier": "mid"},
            ]
        ),
        "acquisitions": pd.DataFrame(
            [
                {
                    "acquisition_id": "acq_a",
                    "artist_id": "artist_a",
                    "museum_id": "museum_top",
                    "date": "2022-06-01",
                },
                {
                    "acquisition_id": "acq_d",
                    "artist_id": "artist_d",
                    "museum_id": "museum_top",
                    "date": "2025-01-02",
                },
            ]
        ),
        "events": pd.DataFrame(
            [
                {
                    "event_id": "event_a",
                    "artist_id": "artist_a",
                    "event_type": "biennial_inclusion",
                    "event_date": "2023-03-01",
                },
                {
                    "event_id": "event_d",
                    "artist_id": "artist_d",
                    "event_type": "major_solo_show",
                    "event_date": "2025-02-01",
                },
            ]
        ),
        "auction_results": pd.DataFrame(
            [
                {"artist_id": "artist_b", "sale_date": "2020-06-01", "price_usd": 100000},
                {"artist_id": "artist_b", "sale_date": "2020-09-01", "price_usd": 120000},
                {"artist_id": "artist_b", "sale_date": "2022-06-01", "price_usd": 250000},
                {"artist_id": "artist_b", "sale_date": "2022-09-01", "price_usd": 260000},
            ]
        ),
        "galleries": pd.DataFrame(
            [
                {"gallery_id": "gallery_emerging", "name": "Emerging Gallery", "tier": "emerging"},
                {"gallery_id": "gallery_top", "name": "Top Gallery", "tier": "top"},
            ]
        ),
        "representation": pd.DataFrame(
            [
                {
                    "artist_id": "artist_c",
                    "gallery_id": "gallery_emerging",
                    "start_date": "2020-01-01",
                },
                {
                    "artist_id": "artist_c",
                    "gallery_id": "gallery_top",
                    "start_date": "2022-01-01",
                },
            ]
        ),
    }
