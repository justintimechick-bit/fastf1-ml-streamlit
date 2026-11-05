"""The cloud entrypoint opens real source evidence without loading demo models."""

from pathlib import Path

import pandas as pd
import pytest

from f1ml_app.real_dashboard import coverage_summary

ROOT = Path(__file__).resolve().parents[1]


def test_coverage_denominators_exclude_future_and_sprint_sessions():
    sessions = pd.DataFrame(
        [
            {
                "year": 2026,
                "kind": "Qualifying",
                "completion_state": "results_available",
                "results_rows": 22,
                "laps_rows": 0,
                "weather_rows": 0,
            },
            {
                "year": 2026,
                "kind": "Race",
                "completion_state": "not_started",
                "results_rows": 0,
                "laps_rows": 0,
                "weather_rows": 0,
            },
            {
                "year": 2026,
                "kind": "Sprint",
                "completion_state": "unknown",
                "results_rows": 0,
                "laps_rows": 0,
                "weather_rows": 0,
            },
        ]
    )
    row = coverage_summary(sessions).iloc[0]
    assert row["Qualifying results"] == "1 / 1"
    assert row["Race results"] == "0 / 0"
    assert row["Weather"] == "0 / 1"


@pytest.mark.skipif(
    not (ROOT / "data/real-20261105-core-v2/receipt.json").exists(),
    reason="local real artifacts absent",
)
def test_cloud_home_and_filters_never_call_demo_models_or_training(monkeypatch):
    from sklearn.pipeline import Pipeline
    from streamlit.testing.v1 import AppTest

    from f1ml.ingestion.provider import FastF1Provider
    from f1ml_app import ui

    def forbidden(*args, **kwargs):
        raise AssertionError("Cloud review attempted demo inference, training or a download")

    monkeypatch.setenv("F1ML_APP_HOME", "real-experiment")
    monkeypatch.setattr(ui, "experiment", forbidden)
    monkeypatch.setattr(Pipeline, "fit", forbidden)
    monkeypatch.setattr(FastF1Provider, "fetch_results", forbidden)
    monkeypatch.setattr(FastF1Provider, "fetch_timing", forbidden)
    at = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=60).run()
    assert not at.exception and not at.error
    assert at.session_state["source"] == "real"
    assert at.segmented_control(key="source").disabled
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Result sessions collected"] == "112"
    assert metrics["Real models trained"] == "0"
    assert len(at.download_button) == 4
    at.checkbox(key="real_missing_only").check().run()
    at.segmented_control(key="season_real").set_value(2025).run()
    at.selectbox(key="real_evaluation_task").set_value("podium").run()
    assert not at.exception and not at.error
    assert at.dataframe[1].value["Grand Prix"].empty  # all 2025 Q/R results were collected
