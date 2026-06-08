"""Evaluation utilities for temporal artist success prediction."""

from __future__ import annotations

import math

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def evaluate_predictions(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    target_column: str,
    k: int = 5,
    score_column: str = "probability_like_score",
) -> dict[str, float | None]:
    """Evaluate artist-date predictions against binary labels."""
    predictions = predictions.copy()
    labels = labels.copy()
    join_columns = ["artist_id"]
    if "prediction_date" in predictions.columns and "prediction_date" in labels.columns:
        predictions["prediction_date"] = pd.to_datetime(predictions["prediction_date"]).dt.strftime("%Y-%m-%d")
        labels["prediction_date"] = pd.to_datetime(labels["prediction_date"]).dt.strftime("%Y-%m-%d")
        join_columns.append("prediction_date")

    joined = predictions.merge(labels[join_columns + [target_column]], on=join_columns, how="inner")
    if f"{target_column}_y" in joined.columns:
        joined[target_column] = joined[f"{target_column}_y"]
    y_true = joined[target_column].astype(int)
    y_score = joined[score_column].astype(float)

    return {
        "roc_auc": _json_safe(_safe_roc_auc(y_true, y_score)),
        "precision_at_k": precision_at_k(joined, target_column, k, score_column),
        "recall_at_k": recall_at_k(joined, target_column, k, score_column),
        "average_precision": _json_safe(_safe_average_precision(y_true, y_score)),
    }


def precision_at_k(
    scored_labels: pd.DataFrame,
    target_column: str,
    k: int,
    score_column: str = "probability_like_score",
) -> float:
    """Return precision among the top-k rows by ranking score."""
    if k <= 0 or scored_labels.empty:
        return 0.0
    top_k = scored_labels.sort_values(score_column, ascending=False).head(k)
    return float(top_k[target_column].astype(int).sum() / len(top_k))


def recall_at_k(
    scored_labels: pd.DataFrame,
    target_column: str,
    k: int,
    score_column: str = "probability_like_score",
) -> float:
    """Return recall among the top-k rows by ranking score."""
    positives = int(scored_labels[target_column].astype(int).sum())
    if k <= 0 or positives == 0:
        return 0.0
    top_k = scored_labels.sort_values(score_column, ascending=False).head(k)
    return float(top_k[target_column].astype(int).sum() / positives)


def _safe_roc_auc(y_true: pd.Series, y_score: pd.Series) -> float:
    """Return ROC AUC, or NaN when labels contain a single class."""
    if y_true.nunique() < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _safe_average_precision(y_true: pd.Series, y_score: pd.Series) -> float:
    """Return average precision, or NaN when no positives are present."""
    if y_true.sum() == 0:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def _json_safe(value: float) -> float | None:
    """Convert NaN metric values to None for JSON output."""
    if isinstance(value, float) and math.isnan(value):
        return None
    return value
