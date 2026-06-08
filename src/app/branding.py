"""Central branding configuration for the Streamlit application."""

from __future__ import annotations

from pathlib import Path


PRODUCT_NAME = "Free Exchange Intelligence"
SHORT_NAME = "FXI"
TAGLINE = "Mapping the networks behind cultural value."

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = REPO_ROOT / "assets"
LOGO_PATH = ASSETS_DIR / "logo.png"
FAVICON_PATH = ASSETS_DIR / "favicon.png"


def existing_asset(path: Path) -> str | None:
    """Return a string path for Streamlit when the asset exists."""
    return str(path) if path.exists() else None
