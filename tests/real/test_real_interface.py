"""Real UI smoke checks; ordinary reruns must remain read-only."""

from pathlib import Path

import joblib
import requests
from sklearn.pipeline import Pipeline
from streamlit.testing.v1 import AppTest

from f1ml_real import lab

ROOT = Path(__file__).resolve().parents[2]


def _page_models():
    from f1ml_app.lab_app import three_models

    three_models()


def _page_data():
    from f1ml_app.lab_app import data_methodology

    data_methodology()


def _page_race_results(path):
    from pathlib import Path

    from f1ml_app.lab_app import _task_results
    from f1ml_real.lab import load_run

    _task_results(load_run(Path(path)), "race", "linear")


def _no_side_effects(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("A display rerun attempted network access or model fitting")

    monkeypatch.setattr(Pipeline, "fit", forbidden)
    monkeypatch.setattr(joblib, "load", forbidden)
    monkeypatch.setattr(requests.Session, "request", forbidden)
    monkeypatch.setattr(lab, "cached_run", forbidden)
    monkeypatch.setattr(lab, "run_backtest", forbidden)


def _text(at):
    return "\n".join(
        str(element.value)
        for kind in ("markdown", "caption", "info", "warning")
        for element in at.get(kind)
    )


def test_real_only_landing_page_and_ordinary_rerun_do_not_download_or_fit(
    monkeypatch, dataset_root, tmp_path
):
    monkeypatch.setenv("F1ML_REAL_DATASET", str(dataset_root))
    monkeypatch.setenv("F1ML_REAL_RUNS", str(tmp_path / "no-saved-runs"))
    monkeypatch.setenv("F1ML_LIVE_DATA", str(tmp_path / "live"))
    _no_side_effects(monkeypatch)
    at = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=30).run()
    assert not at.exception and not at.error
    assert at.selectbox(key="lab_target_event").value == "2026-19"
    assert at.selectbox(key="lab_history_window").value == "All earlier races that year"
    assert at.button(key="lab_run").label == "Run historical test"
    assert at.sidebar.button(key="lab_refresh").label == "Refresh FastF1 data"
    assert not any(button.label == "Refresh FastF1 data" for button in at.main.button)
    assert "Synthetic demo" not in _text(at)
    assert "Linear models" in _text(at)
    assert "Random forest" in _text(at)
    assert "Gradient boosting" in _text(at)
    at.selectbox(key="lab_target_event").set_value("2026-18").run()
    assert not at.exception and not at.error
    assert not (tmp_path / "no-saved-runs").exists()


def test_explicit_form_submission_calls_real_engine_with_selected_history(
    monkeypatch, dataset_root, replay_path
):
    monkeypatch.setenv("F1ML_REAL_DATASET", str(dataset_root))
    monkeypatch.setenv("F1ML_REAL_RUNS", str(replay_path.parent))
    monkeypatch.setenv("F1ML_LIVE_DATA", str(replay_path.parent / "live"))
    _no_side_effects(monkeypatch)
    calls = []

    def selected_run(dataset, event, output, *, last_n):
        calls.append((dataset, event, output, last_n))
        return replay_path

    monkeypatch.setattr(lab, "cached_run", selected_run)
    at = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=30).run()
    assert not calls
    at.selectbox(key="lab_target_event").set_value("2026-14")
    at.selectbox(key="lab_history_window").set_value("Previous 12 races that year")
    at.button(key="lab_run").click().run()
    assert not at.exception and not at.error
    assert calls == [(dataset_root, "2026-14", replay_path.parent, 12)]
    assert at.metric[0].value == "12"
    assert any("Historical test saved" in element.value for element in at.success)
    assert len(at.dataframe) >= 2  # actual baseline/model scores and predictions
    at.run()
    assert len(calls) == 1


def test_model_and_data_pages_use_real_coverage_without_training(
    monkeypatch, dataset_root, tmp_path
):
    monkeypatch.setenv("F1ML_REAL_DATASET", str(dataset_root))
    monkeypatch.setenv("F1ML_LIVE_DATA", str(tmp_path / "live"))
    _no_side_effects(monkeypatch)
    models = AppTest.from_function(_page_models, default_timeout=30).run()
    assert not models.exception and not models.error
    text = _text(models)
    assert "Ridge regression + logistic regression" in text
    assert "Course connection" in text
    data = AppTest.from_function(_page_data, default_timeout=30).run()
    assert not data.exception and not data.error
    assert any(element.value == "19" for element in data.metric)
    assert "none are presented as an untouched final holdout" in _text(data)
    coverage = next(frame.value for frame in data.dataframe if "Known numeric ranks" in frame.value)
    assert len(coverage) == 38
    assert coverage["Rostered drivers"].eq(22).all()


def test_race_view_displays_independent_classifier_probabilities(monkeypatch, replay_path):
    _no_side_effects(monkeypatch)
    at = AppTest.from_function(
        _page_race_results, args=(str(replay_path),), default_timeout=30
    ).run()
    assert not at.exception and not at.error
    predictions = next(frame.value for frame in at.dataframe if "Podium chance" in frame.value)
    assert len(predictions) == 22
    assert predictions["Podium chance"].between(0, 1).all()
    assert predictions["Podium chance"].nunique() > 1
