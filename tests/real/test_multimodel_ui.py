"""Multi-model comparisons preserve saved outputs and global UI state on read-only reruns."""

import shutil
from itertools import combinations
from pathlib import Path

import joblib
import pandas as pd
import pytest
import requests
from pandas.testing import assert_frame_equal
from sklearn.pipeline import Pipeline
from streamlit.testing.v1 import AppTest

from f1ml_app import charts
from f1ml_app import forecasts as fc
from f1ml_app.settings import MODEL_LABELS
from f1ml_real import lab

ROOT = Path(__file__).resolve().parents[2]
SELECTIONS = [list(group) for size in (1, 2, 3) for group in combinations(MODEL_LABELS, size)]


@pytest.mark.parametrize("models", SELECTIONS)
@pytest.mark.parametrize("task", ["qualifying", "race"])
def test_comparisons_align_saved_ranks_and_probabilities_by_driver(replay, models, task):
    saved = replay["predictions"]
    before = saved.copy(deep=True)
    field = fc.comparison_frame(saved, task=task, models=models, primary=models[0])
    assert len(field) == 22 and field.driver_id.is_unique
    assert field.attrs["models"] == models
    if len(models) > 1:
        ranks = field[[f"rank__{model}" for model in models]]
        assert field["Consensus / Mean Rank"].equals(ranks.mean(axis=1))
        assert field["Model Spread (Max - Min Rank)"].equals(ranks.max(axis=1) - ranks.min(axis=1))
        for model in models:
            original = fc.presentation_frame(saved, task=task, family=model).set_index("driver_id")
            assert field[f"rank__{model}"].equals(field.driver_id.map(original.predicted_rank))
            assert field[f"podium__{model}"].equals(
                field.driver_id.map(original.podium_probability)
            )
        chart = charts.model_field_comparison(field).to_dict()
        points = next(iter(chart["datasets"].values()))
        assert {row["family"] for row in points} == set(models)
    assert_frame_equal(saved, before)


def test_missing_models_and_driver_outputs_stay_unknown(replay):
    saved = replay["predictions"]
    subset = saved.loc[~saved.family.eq("random_forest")].copy()
    # A missing primary output must not drop that driver or borrow another model's rank.
    driver = subset.loc[subset.task.eq("race") & subset.family.eq("linear"), "driver_id"].iloc[0]
    subset = subset.loc[~(subset.driver_id.eq(driver) & subset.family.eq("linear"))]
    field = fc.comparison_frame(subset, task="race", models=list(MODEL_LABELS), primary="linear")
    assert len(field) == 22
    assert field["rank__random_forest"].isna().all()
    row = field.set_index("driver_id").loc[driver]
    assert pd.isna(row.predicted_rank) and pd.isna(row["rank__linear"])
    assert row["Models with rank"] == 1
    assert row["Consensus / Mean Rank"] == row["rank__gradient_boosting"]
    charts.model_field_comparison(field, "rank_error").to_dict()
    empty = fc.comparison_frame(pd.DataFrame(), task="race", models=["linear"], primary="linear")
    assert empty.empty


@pytest.fixture
def app(monkeypatch, dataset_root, replay_path, tmp_path):
    monkeypatch.setenv("F1ML_REAL_DATASET", str(dataset_root))
    monkeypatch.setenv("F1ML_REAL_RUNS", str(replay_path.parent))
    monkeypatch.setenv("F1ML_LIVE_DATA", str(tmp_path / "live"))

    def forbidden(*args, **kwargs):
        raise AssertionError("A filter or navigation change attempted training or network access")

    monkeypatch.setattr(Pipeline, "fit", forbidden)
    monkeypatch.setattr(joblib, "load", forbidden)
    monkeypatch.setattr(requests.Session, "request", forbidden)
    monkeypatch.setattr(lab, "cached_run", forbidden)
    monkeypatch.setattr(lab, "run_backtest", forbidden)
    at = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=30).run()
    at.selectbox(key="lab_replay_artifact").set_value(str(replay_path)).run()
    assert not at.exception and not at.error
    return at


def _field(at):
    return next(node.value for node in at.dataframe if "Predicted rank" in node.value)


def _ribbon(at):
    return next(
        node.proto.body
        for node in at.get("html")
        if 'aria-label="Current telemetry context"' in node.proto.body
    )


def test_all_model_combinations_render_both_tasks_and_calibration(app):
    at = app
    for task in ("Before qualifying", "After qualifying · race"):
        at.radio(key="lab_forecast_tab").set_value(task).run()
        for models in SELECTIONS:
            at.multiselect(key="lab_selected_models").set_value(models).run()
            assert not at.exception and not at.error
            field = _field(at)
            assert len(field) == 22
            assert at.selectbox(key="lab_prediction_family").value in models
            for model, label in MODEL_LABELS.items():
                assert (f"{label} · Predicted Rank" in field) == (
                    len(models) > 1 and model in models
                )
            ribbon = _ribbon(at)
            assert f"{len(models)}/3 active" in ribbon
    at.multiselect(key="lab_selected_models").set_value([]).run()
    assert len(at.multiselect(key="lab_selected_models").value) == 1
    assert not at.exception and not at.error
    at.button(key="select_all_models").click().run()
    assert at.multiselect(key="lab_selected_models").value == list(MODEL_LABELS)
    assert not at.exception and not at.error


def test_top_navigation_preserves_global_filters_and_reset_restores_defaults(app, replay_path):
    at = app
    assert at.main.radio(key="lab_page").options == [
        "Historical test",
        "Refresh & predict",
        "Three models",
        "Data & methodology",
    ]
    assert not any(node.key == "lab_page" for node in at.sidebar.radio)
    assert {node.key for node in at.sidebar.button} == {"select_all_models", "reset_filters"}
    at.multiselect(key="lab_selected_models").set_value(["linear", "random_forest"]).run()
    at.selectbox(key="lab_prediction_family").set_value("random_forest").run()
    at.radio(key="lab_forecast_tab").set_value("After qualifying · race").run()
    at.multiselect(key="grid_teams").set_value(["ferrari"]).run()
    assert _field(at).Team.eq("Ferrari").all() and len(_field(at)) == 2
    assert "Ferrari" in _ribbon(at)
    at.text_input(key="grid_driver_search").set_value("no matching driver").run()
    assert not at.exception and not at.error
    assert not any("Predicted rank" in node.value for node in at.dataframe)
    at.text_input(key="grid_driver_search").set_value("").run()
    at.radio(key="view_mode").set_value("Executive Skim").run()
    for page in ("Three models", "Data & methodology", "Refresh & predict", "Historical test"):
        at.radio(key="lab_page").set_value(page).run()
        assert not at.exception and not at.error
        assert at.multiselect(key="lab_selected_models").value == ["linear", "random_forest"]
        assert at.multiselect(key="grid_teams").value == ["ferrari"]
        assert at.radio(key="view_mode").value == "Executive Skim"
        assert at.session_state["lab_replay_artifact"] == str(replay_path)
    assert at.selectbox(key="lab_prediction_family").value == "random_forest"
    assert at.selectbox(key="lab_target_event").value == "2026-19"
    at.button(key="reset_filters").click().run()
    assert not at.exception and not at.error
    assert at.multiselect(key="lab_selected_models").value == list(MODEL_LABELS)
    assert at.multiselect(key="grid_teams").value == []
    assert at.text_input(key="grid_driver_search").value == ""
    assert at.radio(key="view_mode").value == "Full Telemetry & Audits"
    assert at.radio(key="lab_page").value == "Historical test"


def test_switching_replay_artifacts_keeps_model_and_grid_selection(
    app, replay_path, tmp_path, monkeypatch
):
    copies = [tmp_path / "runs" / f"replay-2026-14-12-{suffix}" for suffix in ("first", "second")]
    for path in copies:
        shutil.copytree(replay_path, path)
    monkeypatch.setenv("F1ML_REAL_RUNS", str(copies[0].parent))
    app.multiselect(key="lab_selected_models").set_value(["linear", "gradient_boosting"]).run()
    app.multiselect(key="grid_teams").set_value(["ferrari"]).run()
    for path in copies:
        app.selectbox(key="lab_replay_artifact").set_value(str(path)).run()
        assert not app.exception and not app.error
        assert path.name in _ribbon(app)
        assert app.multiselect(key="lab_selected_models").value == ["linear", "gradient_boosting"]
        assert len(_field(app)) == 2
