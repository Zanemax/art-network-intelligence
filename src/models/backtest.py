"""Historical walk-forward backtests for graph-derived artist success models."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.synthetic import load_synthetic_dataset
from src.graph.build_graph import build_investment_graph
from src.graph.features import build_artist_graph_features
from src.models.baselines import get_baselines
from src.models.evaluate import evaluate_predictions
from src.models.targets import TARGET_COLUMNS, generate_artist_targets
from src.models.train import MODEL_FEATURE_COLUMNS, get_model_registry


def annual_prediction_dates(start_year: int, end_year: int) -> tuple[str, ...]:
    """Return annual January 1 prediction dates for an inclusive year range."""
    if start_year > end_year:
        raise ValueError("start_year must be less than or equal to end_year.")
    return tuple(f"{year}-01-01" for year in range(start_year, end_year + 1))


def build_backtest_panel(
    dataset: dict[str, pd.DataFrame],
    prediction_dates: tuple[str, ...],
) -> pd.DataFrame:
    """Build annual graph snapshots, point-in-time features, and 3-year labels."""
    frames = []

    for prediction_date in prediction_dates:
        snapshot = build_investment_graph(dataset, cutoff_date=prediction_date)
        features = build_artist_graph_features(snapshot, prediction_date)
        features["prediction_date"] = prediction_date
        labels = generate_artist_targets(dataset, prediction_date)
        labels["prediction_date"] = prediction_date
        frames.append(features.merge(labels, on=["artist_id", "prediction_date"], how="inner"))

    panel = pd.concat(frames, ignore_index=True)
    panel["prediction_date"] = pd.to_datetime(panel["prediction_date"])
    return panel


def run_backtest(
    target: str,
    start_year: int = 2017,
    end_year: int = 2023,
    reports_dir: str | Path = "reports",
    k: int = 5,
) -> pd.DataFrame:
    """Run a historical walk-forward backtest and save reports."""
    if target not in TARGET_COLUMNS:
        valid_targets = ", ".join(TARGET_COLUMNS)
        raise ValueError(f"Unknown target '{target}'. Valid targets: {valid_targets}.")

    dataset = load_synthetic_dataset()
    prediction_dates = annual_prediction_dates(start_year, end_year)
    panel = build_backtest_panel(dataset, prediction_dates)
    results = walk_forward_backtest(panel, target, k=k)

    reports_path = Path(reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)
    results.to_csv(reports_path / "backtest_results.csv", index=False)
    (reports_path / "backtest_summary.md").write_text(
        _summary_markdown(results, target, start_year, end_year),
        encoding="utf-8",
    )
    return results


def walk_forward_backtest(panel: pd.DataFrame, target: str, k: int = 5) -> pd.DataFrame:
    """Evaluate each annual test date using only strictly earlier training dates."""
    rows = []
    dated_panel = panel.copy()
    dated_panel["prediction_date"] = pd.to_datetime(dated_panel["prediction_date"])
    prediction_dates = sorted(dated_panel["prediction_date"].unique())

    for test_date in prediction_dates[1:]:
        train = dated_panel[dated_panel["prediction_date"] < test_date].copy()
        test = dated_panel[dated_panel["prediction_date"] == test_date].copy()
        if train.empty or test.empty:
            continue

        rows.extend(_baseline_rows(train, test, target, test_date, k))
        if train[target].nunique() < 2:
            continue
        rows.extend(_graph_model_rows(train, test, target, test_date, k))

    return pd.DataFrame(rows).sort_values(["prediction_date", "model"]).reset_index(drop=True)


def _graph_model_rows(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    test_date: pd.Timestamp,
    k: int,
) -> list[dict[str, object]]:
    """Train and evaluate graph-feature ML models for one fold."""
    rows = []
    x_train = train[MODEL_FEATURE_COLUMNS]
    y_train = train[target].astype(int)
    x_test = test[MODEL_FEATURE_COLUMNS]

    for model_name, model in get_model_registry().items():
        model.fit(x_train, y_train)
        scores = model.predict_proba(x_test)[:, 1]
        predictions = _prediction_frame(test, model_name, scores, target)
        metrics = evaluate_predictions(predictions, test, target, k=k)
        rows.append(_result_row(test_date, model_name, "graph_model", len(train), len(test), metrics))

    return rows


def _baseline_rows(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    test_date: pd.Timestamp,
    k: int,
) -> list[dict[str, object]]:
    """Evaluate all baselines for one fold."""
    rows = []
    date_string = pd.Timestamp(test_date).strftime("%Y-%m-%d")
    for baseline_name, baseline_fn in get_baselines().items():
        predictions = baseline_fn(test.reset_index(drop=True), date_string)
        metrics = evaluate_predictions(predictions, test, target, k=k)
        rows.append(_result_row(test_date, baseline_name, "baseline", len(train), len(test), metrics))
    return rows


def _prediction_frame(
    test: pd.DataFrame,
    model_name: str,
    scores,
    target: str,
) -> pd.DataFrame:
    """Format model scores for metric evaluation."""
    predictions = test[["artist_id", "prediction_date", target]].copy()
    predictions["prediction_date"] = pd.to_datetime(predictions["prediction_date"]).dt.strftime("%Y-%m-%d")
    predictions["model"] = model_name
    predictions["score"] = scores
    predictions["probability_like_score"] = scores
    return predictions


def _result_row(
    test_date: pd.Timestamp,
    model_name: str,
    model_type: str,
    train_rows: int,
    test_rows: int,
    metrics: dict[str, float | None],
) -> dict[str, object]:
    """Build one backtest result row."""
    return {
        "prediction_date": pd.Timestamp(test_date).strftime("%Y-%m-%d"),
        "model": model_name,
        "model_type": model_type,
        "train_rows": train_rows,
        "test_rows": test_rows,
        **metrics,
    }


def _summary_markdown(
    results: pd.DataFrame,
    target: str,
    start_year: int,
    end_year: int,
) -> str:
    """Create a concise markdown summary for the backtest."""
    if results.empty:
        return f"# Backtest Summary\n\nNo folds were evaluated for `{target}`.\n"

    metric_columns = ["roc_auc", "precision_at_k", "recall_at_k", "average_precision"]
    aggregate = (
        results.groupby(["model_type", "model"])[metric_columns]
        .mean(numeric_only=True)
        .reset_index()
        .sort_values("average_precision", ascending=False, na_position="last")
    )

    lines = [
        "# Backtest Summary",
        "",
        f"- Target: `{target}`",
        f"- Prediction years: {start_year}-{end_year}",
        f"- Folds evaluated: {results['prediction_date'].nunique()}",
        "",
        "## Average Metrics",
        "",
        _markdown_table(aggregate),
        "",
        "## Notes",
        "",
        "Each fold trains only on prediction dates before the tested year. Labels use the 3-year future window after each prediction date.",
    ]
    return "\n".join(lines)


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a small DataFrame as a markdown table without optional deps."""
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.4f}" if pd.notna(value) else "")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run historical graph-model backtests.")
    parser.add_argument("--target", required=True, help="Target column to evaluate.")
    parser.add_argument("--start-year", type=int, default=2017, help="First annual prediction year.")
    parser.add_argument("--end-year", type=int, default=2023, help="Last annual prediction year.")
    parser.add_argument("--k", type=int, default=5, help="k for precision@k and recall@k.")
    return parser.parse_args()


def main() -> None:
    """Run the historical backtest CLI."""
    args = _parse_args()
    results = run_backtest(args.target, args.start_year, args.end_year, k=args.k)
    print(f"Saved {len(results)} rows to reports/backtest_results.csv")
    print("Saved summary to reports/backtest_summary.md")


if __name__ == "__main__":
    main()
