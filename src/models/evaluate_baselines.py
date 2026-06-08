"""Command-line evaluation for baseline artist success rankers."""

from __future__ import annotations

import argparse

import pandas as pd

from src.data.synthetic import load_synthetic_dataset
from src.graph.build_graph import build_investment_graph
from src.graph.features import build_artist_graph_features
from src.models.baselines import evaluate_rankings, get_baselines
from src.models.targets import TARGET_COLUMNS, generate_artist_targets


def main() -> None:
    """Run baseline evaluation from the command line."""
    args = _parse_args()
    if args.target not in TARGET_COLUMNS:
        valid_targets = ", ".join(TARGET_COLUMNS)
        raise SystemExit(f"Unknown target '{args.target}'. Valid targets: {valid_targets}.")

    dataset = load_synthetic_dataset()
    graph = build_investment_graph(dataset)
    features = build_artist_graph_features(graph, args.prediction_date)
    labels = generate_artist_targets(dataset, args.prediction_date)

    rows = []
    for baseline_name, baseline_fn in get_baselines().items():
        predictions = baseline_fn(features, args.prediction_date)
        metrics = evaluate_rankings(predictions, labels, args.target, k=args.k)
        rows.append({"baseline": baseline_name, **metrics})

    results = pd.DataFrame(rows).sort_values("average_precision", ascending=False, na_position="last")
    print(results.to_string(index=False))


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Evaluate baseline artist success rankers.")
    parser.add_argument("--target", required=True, help="Target column to evaluate.")
    parser.add_argument("--prediction-date", default="2021-01-01", help="Point-in-time feature cutoff date.")
    parser.add_argument("--k", type=int, default=5, help="k for precision@k and recall@k.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
