"""Placeholder recommendation logic for the Art Taste Graph MVP.

This module will eventually combine graph neighborhoods, content features, and
scikit-learn similarity models to suggest artworks or artists.
"""

from collections.abc import Iterable


def recommend_from_neighbors(seed_ids: Iterable[str], limit: int = 10) -> list[str]:
    """Return placeholder recommendations for the provided seed node IDs."""
    return list(seed_ids)[:limit]
