"""Train and score a simple art investment classifier.

The MVP predicts whether an artist's auction price doubles within three years.
It uses a small RandomForestClassifier so the dashboard can expose intuitive
feature importance values without additional model explainability libraries.
"""

from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.features import FEATURE_COLUMNS


@dataclass(frozen=True)
class ModelResult:
    """Container for fitted model output and dashboard-ready scores."""

    model: Pipeline
    predictions: pd.DataFrame
    feature_importance: pd.DataFrame


def train_investment_model(features: pd.DataFrame) -> ModelResult:
    """Train a classifier and return artist probabilities plus feature weights."""
    x = features[FEATURE_COLUMNS]
    y = features["doubled_in_3_years"]

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=3,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    model.fit(x, y)

    probabilities = model.predict_proba(x)[:, 1]
    predictions = features[["artist_id", "name"] + FEATURE_COLUMNS + ["doubled_in_3_years"]].copy()
    predictions["prediction_probability"] = probabilities
    predictions = predictions.sort_values("prediction_probability", ascending=False).reset_index(drop=True)

    classifier = model.named_steps["classifier"]
    feature_importance = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": classifier.feature_importances_,
        }
    ).sort_values("importance", ascending=False, ignore_index=True)

    return ModelResult(
        model=model,
        predictions=predictions,
        feature_importance=feature_importance,
    )
