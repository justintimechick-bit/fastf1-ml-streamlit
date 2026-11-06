"""Streamlit Community Cloud entrypoint for the real-observation model lab."""

import sys
from pathlib import Path

source = Path(__file__).resolve().parent / "src"
if str(source) not in sys.path:
    sys.path.insert(0, str(source))

from f1ml_app.lab_app import main  # noqa: E402

main()
