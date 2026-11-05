"""Streamlit Community Cloud entrypoint; the real-data review is the landing page."""

import os
import runpy
from pathlib import Path

os.environ.setdefault("F1ML_APP_HOME", "real-experiment")
runpy.run_path(str(Path(__file__).parent / "src/f1ml_app/streamlit_app.py"), run_name="__main__")
