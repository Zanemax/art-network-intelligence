"""Root Streamlit entrypoint for Streamlit Community Cloud.

The product code lives in ``src/app/streamlit_app.py``. Keeping this small
wrapper at the repo root lets Streamlit Cloud use its default app discovery
while preserving the organized source layout.
"""

from src.app.streamlit_app import main


if __name__ == "__main__":
    main()
