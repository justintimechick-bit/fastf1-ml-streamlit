"""Navigation preserves filters and the context follows verified displayed artifacts."""

import json
import shutil

import joblib
import pytest
import requests
from sklearn.pipeline import Pipeline
from streamlit.testing.v1 import AppTest

from f1ml_app.settings import load_paths
from f1ml_real import lab


def _navigation_harness():
    import streamlit as st

    from f1ml_app.lab_app import data_methodology, historical_test, three_models

    route = st.selectbox("Test route", ["Historical", "Models", "Data"], key="test_route")
    {"Historical": historical_test, "Models": three_models, "Data": data_methodology}[route]()


def _review_page():
    from f1ml_app.real_dashboard import render

    render()


def _ribbon(at):
    return next(
        node.proto.body
        for node in at.get("html")
        if 'aria-label="Current telemetry context"' in node.proto.body
    )


@pytest.fixture
def readonly_ui(monkeypatch, dataset_root, replay_path, tmp_path):
    monkeypatch.setenv("F1ML_REAL_DATASET", str(dataset_root))
    monkeypatch.setenv("F1ML_REAL_RUNS", str(replay_path.parent))
    monkeypatch.setenv("F1ML_LIVE_DATA", str(tmp_path / "no-live-data"))

    def forbidden(*args, **kwargs):
        raise AssertionError("Navigation attempted a download, training, or model deserialization")

    monkeypatch.setattr(Pipeline, "fit", forbidden)
    monkeypatch.setattr(joblib, "load", forbidden)
    monkeypatch.setattr(requests.Session, "request", forbidden)
    monkeypatch.setattr(lab, "cached_run", forbidden)
    monkeypatch.setattr(lab, "run_backtest", forbidden)


def test_inspected_artifact_and_filters_survive_modes_pages_and_seasons(readonly_ui, replay_path):
    at = AppTest.from_function(_navigation_harness, default_timeout=30).run()
    assert not at.exception and not at.error
    assert at.radio(key="view_mode").value == "Full Telemetry & Audits"
    at.selectbox(key="lab_replay_artifact").set_value(str(replay_path)).run()
    at.selectbox(key="lab_prediction_family").set_value("gradient_boosting").run()
    at.radio(key="lab_forecast_tab").set_value("After qualifying · race").run()
    at.selectbox(key="lab_target_event").set_value("2026-18")
    at.selectbox(key="lab_history_window").set_value("Previous 6 races that year").run()
    ribbon = _ribbon(at)
    assert replay_path.name in ribbon and "R14" in ribbon
    assert "Next request:" in ribbon and "R18" in ribbon
    assert "Gradient Boosting" in ribbon and "Predictions locked" in ribbon
    assert "Receipt verified" in ribbon

    at.radio(key="view_mode").set_value("Executive Skim").run()
    assert not at.exception and not at.error and not at.dataframe
    assert "Highest independent podium probabilities" in "\n".join(x.value for x in at.markdown)
    at.selectbox(key="test_route").set_value("Models").run()
    assert not at.exception and replay_path.name in _ribbon(at)
    at.selectbox(key="test_route").set_value("Data").run()
    assert not at.exception and "Source coverage" in _ribbon(at)
    at.selectbox(key="lab_season").set_value(2025).run()
    assert not at.exception
    at.selectbox(key="lab_season").set_value(2026).run()
    at.selectbox(key="test_route").set_value("Historical").run()
    assert not at.exception and not at.error
    assert at.selectbox(key="lab_target_event").value == "2026-18"
    assert at.selectbox(key="lab_history_window").value == "Previous 6 races that year"
    assert at.selectbox(key="lab_replay_artifact").value == str(replay_path)
    assert at.selectbox(key="lab_prediction_family").value == "gradient_boosting"
    assert at.radio(key="lab_forecast_tab").value == "After qualifying · race"
    assert at.radio(key="view_mode").value == "Executive Skim"
    at.radio(key="view_mode").set_value("Full Telemetry & Audits").run()
    assert not at.exception and not at.error
    predictions = next(df.value for df in at.dataframe if "Podium chance" in df.value)
    assert len(predictions) == 22


def test_changed_lock_revokes_success_badges_and_hides_predictions(
    readonly_ui, replay_path, tmp_path, monkeypatch
):
    copied = tmp_path / "runs" / replay_path.name
    shutil.copytree(replay_path, copied)
    monkeypatch.setenv("F1ML_REAL_RUNS", str(copied.parent))
    at = AppTest.from_function(_navigation_harness, default_timeout=30).run()
    at.selectbox(key="lab_replay_artifact").set_value(str(copied)).run()
    assert "Predictions locked" in _ribbon(at)
    marker = copied / "prediction_lock.json"
    lock = json.loads(marker.read_text())
    lock["sealed_before_target_scoring"] = False
    marker.write_text(json.dumps(lock))
    at.run()
    assert not at.exception
    ribbon = _ribbon(at)
    assert copied.name in ribbon and "unavailable" in ribbon
    assert "Predictions locked" not in ribbon and "Receipt verified" not in ribbon
    assert "Receipt not verified" in ribbon and not at.dataframe
    assert any("could not be verified" in x.value for x in at.warning)


def test_coverage_first_visit_keeps_a_valid_history_default(readonly_ui):
    at = AppTest.from_function(_navigation_harness, default_timeout=30)
    at.session_state["test_route"] = "Data"
    at.run()
    at.selectbox(key="lab_season").set_value(2025).run()
    at.selectbox(key="lab_season").set_value(2026).run()
    at.selectbox(key="test_route").set_value("Historical").run()
    assert not at.exception and not at.error
    assert at.selectbox(key="lab_history_window").value == "All earlier races that year"


def test_blocked_review_preserves_task_and_missing_filter_across_modes(readonly_ui):
    if not (load_paths().real_experiment / "receipt.json").is_file():
        pytest.skip("Local-only archived experiment is excluded from deployment releases")
    at = AppTest.from_function(_review_page, default_timeout=30).run()
    assert not at.exception and not at.error
    at.checkbox(key="real_missing_only").check().run()
    at.selectbox(key="real_evaluation_task").set_value("podium").run()
    at.radio(key="view_mode").set_value("Executive Skim").run()
    ribbon = _ribbon(at)
    assert "no trained real model" in ribbon and "training blocked" in ribbon
    assert "podium" in ribbon and "Receipt verified" in ribbon
    assert "Predictions locked" not in ribbon and not at.dataframe
    at.radio(key="view_mode").set_value("Full Telemetry & Audits").run()
    assert not at.exception and not at.error
    assert at.checkbox(key="real_missing_only").value
    assert at.selectbox(key="real_evaluation_task").value == "podium"
