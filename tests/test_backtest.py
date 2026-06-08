"""Tests for historical backtesting."""

import pandas as pd

from src.data.synthetic import load_synthetic_dataset
from src.models.backtest import (
    annual_prediction_dates,
    build_backtest_panel,
    run_backtest,
    walk_forward_backtest,
)


def test_annual_prediction_dates_are_inclusive() -> None:
    """Verify annual prediction dates cover the requested range."""
    assert annual_prediction_dates(2017, 2019) == (
        "2017-01-01",
        "2018-01-01",
        "2019-01-01",
    )


def test_build_backtest_panel_creates_artist_date_rows() -> None:
    """Verify each prediction date gets one feature-label row per artist."""
    dataset = load_synthetic_dataset()
    panel = build_backtest_panel(dataset, ("2019-01-01", "2020-01-01"))

    assert len(panel) == len(dataset["artists"]) * 2
    assert {"artist_id", "prediction_date", "institutional_success_3y"}.issubset(panel.columns)
    assert panel["prediction_date"].min() == pd.Timestamp("2019-01-01")


def test_walk_forward_backtest_trains_only_on_past_dates() -> None:
    """Verify fold train/test counts imply strictly historical training."""
    panel = build_backtest_panel(
        load_synthetic_dataset(),
        ("2020-01-01", "2021-01-01", "2022-01-01"),
    )

    results = walk_forward_backtest(panel, "institutional_success_3y")
    first_fold = results[results["prediction_date"] == "2021-01-01"]
    second_fold = results[results["prediction_date"] == "2022-01-01"]

    assert not first_fold.empty
    assert first_fold["train_rows"].max() == 8
    assert second_fold["train_rows"].max() == 16
    assert first_fold["test_rows"].max() == 8


def test_run_backtest_writes_reports(tmp_path) -> None:
    """Verify the backtest CLI core writes CSV and markdown outputs."""
    results = run_backtest(
        target="institutional_success_3y",
        start_year=2020,
        end_year=2022,
        reports_dir=tmp_path,
    )

    assert not results.empty
    assert (tmp_path / "backtest_results.csv").exists()
    assert (tmp_path / "backtest_summary.md").exists()
    assert "Backtest Summary" in (tmp_path / "backtest_summary.md").read_text(encoding="utf-8")
