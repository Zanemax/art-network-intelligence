"""Generate synthetic art market datasets for the investment model MVP.

The data in this module is deliberately small and explainable. It gives the
graph, feature engineering, model, dashboard, and tests a shared fixture that
resembles the entities an art investment workflow would eventually ingest from
real market, institutional, and press sources.
"""

import pandas as pd


def load_synthetic_dataset() -> dict[str, pd.DataFrame]:
    """Return synthetic entities and events for the art taste graph MVP."""
    artists = pd.DataFrame(
        [
            {"artist_id": "artist_ada_rios", "name": "Ada Rios", "birth_year": 1985, "region": "Mexico City"},
            {"artist_id": "artist_mina_park", "name": "Mina Park", "birth_year": 1991, "region": "Seoul"},
            {"artist_id": "artist_luca_voss", "name": "Luca Voss", "birth_year": 1978, "region": "Berlin"},
            {"artist_id": "artist_noor_haddad", "name": "Noor Haddad", "birth_year": 1988, "region": "Beirut"},
            {"artist_id": "artist_elias_stone", "name": "Elias Stone", "birth_year": 1972, "region": "New York"},
            {"artist_id": "artist_sana_okoye", "name": "Sana Okoye", "birth_year": 1994, "region": "Lagos"},
            {"artist_id": "artist_jules_marin", "name": "Jules Marin", "birth_year": 1982, "region": "Paris"},
            {"artist_id": "artist_iris_klein", "name": "Iris Klein", "birth_year": 1969, "region": "Vienna"},
        ]
    )

    galleries = pd.DataFrame(
        [
            {"gallery_id": "gallery_north_axis", "name": "North Axis", "prestige_score": 0.92},
            {"gallery_id": "gallery_terra", "name": "Terra Contemporary", "prestige_score": 0.74},
            {"gallery_id": "gallery_lumen", "name": "Lumen Works", "prestige_score": 0.65},
            {"gallery_id": "gallery_open_field", "name": "Open Field", "prestige_score": 0.48},
        ]
    )

    museums = pd.DataFrame(
        [
            {"museum_id": "museum_moma", "name": "MoMA", "tier": "top", "prestige_score": 1.00},
            {"museum_id": "museum_tate", "name": "Tate Modern", "tier": "top", "prestige_score": 0.97},
            {"museum_id": "museum_hammer", "name": "Hammer Museum", "tier": "mid", "prestige_score": 0.78},
            {"museum_id": "museum_walker", "name": "Walker Art Center", "tier": "mid", "prestige_score": 0.76},
        ]
    )

    collectors = pd.DataFrame(
        [
            {"collector_id": "collector_chen", "name": "Lena Chen", "prestige_score": 0.88},
            {"collector_id": "collector_singh", "name": "Arjun Singh", "prestige_score": 0.79},
            {"collector_id": "collector_morgan", "name": "Eve Morgan", "prestige_score": 0.93},
            {"collector_id": "collector_sato", "name": "Ken Sato", "prestige_score": 0.63},
        ]
    )

    curators = pd.DataFrame(
        [
            {"curator_id": "curator_ortiz", "name": "Marisol Ortiz", "prestige_score": 0.91},
            {"curator_id": "curator_blake", "name": "Theo Blake", "prestige_score": 0.82},
            {"curator_id": "curator_nguyen", "name": "Lan Nguyen", "prestige_score": 0.73},
        ]
    )

    representation = pd.DataFrame(
        [
            {"artist_id": "artist_ada_rios", "gallery_id": "gallery_north_axis", "start_date": "2020-01-15"},
            {"artist_id": "artist_mina_park", "gallery_id": "gallery_terra", "start_date": "2021-03-10"},
            {"artist_id": "artist_luca_voss", "gallery_id": "gallery_lumen", "start_date": "2019-09-01"},
            {"artist_id": "artist_noor_haddad", "gallery_id": "gallery_north_axis", "start_date": "2022-02-20"},
            {"artist_id": "artist_elias_stone", "gallery_id": "gallery_terra", "start_date": "2018-06-12"},
            {"artist_id": "artist_sana_okoye", "gallery_id": "gallery_open_field", "start_date": "2022-11-05"},
            {"artist_id": "artist_jules_marin", "gallery_id": "gallery_lumen", "start_date": "2020-07-19"},
            {"artist_id": "artist_iris_klein", "gallery_id": "gallery_terra", "start_date": "2017-04-03"},
        ]
    )

    exhibitions = pd.DataFrame(
        [
            {"exhibition_id": "exh_001", "artist_id": "artist_ada_rios", "institution_id": "museum_moma", "curator_id": "curator_ortiz", "date": "2022-04-01", "institution_type": "museum"},
            {"exhibition_id": "exh_002", "artist_id": "artist_ada_rios", "institution_id": "museum_hammer", "curator_id": "curator_blake", "date": "2023-06-15", "institution_type": "museum"},
            {"exhibition_id": "exh_003", "artist_id": "artist_mina_park", "institution_id": "museum_walker", "curator_id": "curator_nguyen", "date": "2022-09-10", "institution_type": "museum"},
            {"exhibition_id": "exh_004", "artist_id": "artist_luca_voss", "institution_id": "museum_hammer", "curator_id": "curator_blake", "date": "2021-05-08", "institution_type": "museum"},
            {"exhibition_id": "exh_005", "artist_id": "artist_noor_haddad", "institution_id": "museum_tate", "curator_id": "curator_ortiz", "date": "2023-03-21", "institution_type": "museum"},
            {"exhibition_id": "exh_006", "artist_id": "artist_elias_stone", "institution_id": "museum_walker", "curator_id": "curator_nguyen", "date": "2020-11-11", "institution_type": "museum"},
            {"exhibition_id": "exh_007", "artist_id": "artist_sana_okoye", "institution_id": "museum_hammer", "curator_id": "curator_blake", "date": "2024-01-30", "institution_type": "museum"},
            {"exhibition_id": "exh_008", "artist_id": "artist_jules_marin", "institution_id": "museum_tate", "curator_id": "curator_ortiz", "date": "2021-10-18", "institution_type": "museum"},
            {"exhibition_id": "exh_009", "artist_id": "artist_iris_klein", "institution_id": "museum_walker", "curator_id": "curator_nguyen", "date": "2021-02-14", "institution_type": "museum"},
        ]
    )

    acquisitions = pd.DataFrame(
        [
            {"acquisition_id": "acq_001", "artist_id": "artist_ada_rios", "museum_id": "museum_moma", "date": "2023-01-20", "value_usd": 180000},
            {"acquisition_id": "acq_002", "artist_id": "artist_ada_rios", "museum_id": "museum_hammer", "date": "2024-02-12", "value_usd": 95000},
            {"acquisition_id": "acq_003", "artist_id": "artist_mina_park", "museum_id": "museum_walker", "date": "2023-07-05", "value_usd": 70000},
            {"acquisition_id": "acq_004", "artist_id": "artist_noor_haddad", "museum_id": "museum_tate", "date": "2024-03-03", "value_usd": 160000},
            {"acquisition_id": "acq_005", "artist_id": "artist_sana_okoye", "museum_id": "museum_hammer", "date": "2024-04-22", "value_usd": 52000},
            {"acquisition_id": "acq_006", "artist_id": "artist_jules_marin", "museum_id": "museum_tate", "date": "2022-06-30", "value_usd": 115000},
            {"acquisition_id": "acq_007", "artist_id": "artist_iris_klein", "museum_id": "museum_walker", "date": "2021-12-01", "value_usd": 85000},
        ]
    )

    collector_holdings = pd.DataFrame(
        [
            {"collector_id": "collector_chen", "artist_id": "artist_ada_rios", "date": "2021-08-12"},
            {"collector_id": "collector_morgan", "artist_id": "artist_ada_rios", "date": "2022-12-02"},
            {"collector_id": "collector_singh", "artist_id": "artist_mina_park", "date": "2022-05-17"},
            {"collector_id": "collector_sato", "artist_id": "artist_luca_voss", "date": "2020-10-10"},
            {"collector_id": "collector_morgan", "artist_id": "artist_noor_haddad", "date": "2023-08-01"},
            {"collector_id": "collector_chen", "artist_id": "artist_sana_okoye", "date": "2024-01-12"},
            {"collector_id": "collector_singh", "artist_id": "artist_jules_marin", "date": "2021-09-08"},
            {"collector_id": "collector_sato", "artist_id": "artist_iris_klein", "date": "2020-03-03"},
        ]
    )

    auction_results = pd.DataFrame(
        [
            {"artist_id": "artist_ada_rios", "sale_date": "2021-01-15", "price_usd": 65000},
            {"artist_id": "artist_ada_rios", "sale_date": "2024-01-15", "price_usd": 155000},
            {"artist_id": "artist_mina_park", "sale_date": "2021-03-12", "price_usd": 42000},
            {"artist_id": "artist_mina_park", "sale_date": "2024-03-12", "price_usd": 91000},
            {"artist_id": "artist_luca_voss", "sale_date": "2020-11-20", "price_usd": 70000},
            {"artist_id": "artist_luca_voss", "sale_date": "2023-11-20", "price_usd": 98000},
            {"artist_id": "artist_noor_haddad", "sale_date": "2021-06-01", "price_usd": 53000},
            {"artist_id": "artist_noor_haddad", "sale_date": "2024-06-01", "price_usd": 128000},
            {"artist_id": "artist_elias_stone", "sale_date": "2020-04-30", "price_usd": 120000},
            {"artist_id": "artist_elias_stone", "sale_date": "2023-04-30", "price_usd": 162000},
            {"artist_id": "artist_sana_okoye", "sale_date": "2021-10-09", "price_usd": 28000},
            {"artist_id": "artist_sana_okoye", "sale_date": "2024-10-09", "price_usd": 72000},
            {"artist_id": "artist_jules_marin", "sale_date": "2020-07-07", "price_usd": 80000},
            {"artist_id": "artist_jules_marin", "sale_date": "2023-07-07", "price_usd": 136000},
            {"artist_id": "artist_iris_klein", "sale_date": "2020-02-14", "price_usd": 140000},
            {"artist_id": "artist_iris_klein", "sale_date": "2023-02-14", "price_usd": 190000},
        ]
    )

    press_mentions = pd.DataFrame(
        [
            {"artist_id": "artist_ada_rios", "year": 2021, "mentions": 8},
            {"artist_id": "artist_ada_rios", "year": 2024, "mentions": 31},
            {"artist_id": "artist_mina_park", "year": 2021, "mentions": 5},
            {"artist_id": "artist_mina_park", "year": 2024, "mentions": 15},
            {"artist_id": "artist_luca_voss", "year": 2020, "mentions": 11},
            {"artist_id": "artist_luca_voss", "year": 2023, "mentions": 14},
            {"artist_id": "artist_noor_haddad", "year": 2021, "mentions": 7},
            {"artist_id": "artist_noor_haddad", "year": 2024, "mentions": 28},
            {"artist_id": "artist_elias_stone", "year": 2020, "mentions": 18},
            {"artist_id": "artist_elias_stone", "year": 2023, "mentions": 20},
            {"artist_id": "artist_sana_okoye", "year": 2021, "mentions": 3},
            {"artist_id": "artist_sana_okoye", "year": 2024, "mentions": 17},
            {"artist_id": "artist_jules_marin", "year": 2020, "mentions": 10},
            {"artist_id": "artist_jules_marin", "year": 2023, "mentions": 19},
            {"artist_id": "artist_iris_klein", "year": 2020, "mentions": 16},
            {"artist_id": "artist_iris_klein", "year": 2023, "mentions": 18},
        ]
    )

    return {
        "artists": artists,
        "galleries": galleries,
        "museums": museums,
        "collectors": collectors,
        "curators": curators,
        "representation": representation,
        "exhibitions": exhibitions,
        "acquisitions": acquisitions,
        "collector_holdings": collector_holdings,
        "auction_results": auction_results,
        "press_mentions": press_mentions,
    }


def sample_artworks() -> pd.DataFrame:
    """Return a compatibility sample derived from the synthetic artists table."""
    artists = load_synthetic_dataset()["artists"]
    return artists.rename(columns={"artist_id": "artwork_id", "region": "genre"})[
        ["artwork_id", "name", "genre"]
    ].rename(columns={"name": "title"})
