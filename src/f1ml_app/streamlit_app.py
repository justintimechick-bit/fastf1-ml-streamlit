"""Streamlit entry point. From the project root: ``streamlit run src/f1ml_app/streamlit_app.py``."""

import os
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:  # also works from a plain checkout without an editable install
    sys.path.insert(0, str(_SRC))

import streamlit as st  # noqa: E402

from f1ml_app import theme, ui  # noqa: E402

st.set_page_config(
    page_title="Pit Wall · F1 ML forecasts",
    page_icon="🏁",
    layout="wide",
    initial_sidebar_state="expanded",
)
theme.inject()
st.logo(str(Path(__file__).parent / "assets" / "logo.svg"), size="large")
real_home = os.environ.get("F1ML_APP_HOME") == "real-experiment"
views = Path(__file__).with_name("views")

navigation = st.navigation(
    {
        "Race weekend": [
            st.Page(
                views / "overview.py",
                title="Event overview",
                icon=":material/flag:",
                default=not real_home,
            ),
            st.Page(
                views / "qualifying.py", title="Qualifying predictions", icon=":material/timer:"
            ),
            st.Page(views / "race.py", title="Race predictions", icon=":material/emoji_events:"),
        ],
        "Evaluation": [
            st.Page(
                views / "real_experiment.py",
                title="Real-data experiment",
                icon=":material/fact_check:",
                url_path="real-experiment",
                default=real_home,
            ),
            st.Page(
                views / "comparison.py", title="Model comparison", icon=":material/leaderboard:"
            ),
            st.Page(
                views / "methodology.py",
                title="Data quality & methodology",
                icon=":material/rule:",
            ),
        ],
    }
)
ui.sidebar(fixed_source="real" if navigation.title == "Real-data experiment" else None)
navigation.run()
