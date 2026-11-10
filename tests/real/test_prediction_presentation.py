"""Saved rank/probability presentation preserves model outputs and missing outcomes."""

import json

import pandas as pd
from pandas.testing import assert_frame_equal
from streamlit.testing.v1 import AppTest

from f1ml_app import charts
from f1ml_app import forecasts as fc


def _race_page(path):
    from pathlib import Path

    from f1ml_app.lab_app import _task_results
    from f1ml_real.lab import load_run

    _task_results(load_run(Path(path)), "race", "linear")


def _calibration_page(path):
    from f1ml_app import ui
    from f1ml_app.lab_app import _read_saved

    run = _read_saved(path)
    ui.calibration_audit(run["calibration_predictions"], family="linear", key="test_calibration")


def test_display_uses_discrete_ranks_and_independent_classifier_probabilities(replay):
    saved = replay["predictions"]
    before = saved.copy(deep=True)
    field = fc.presentation_frame(saved, task="race", family="linear")
    assert len(field) == 22
    assert field.predicted_rank.tolist() == list(range(1, 23))
    podium = saved.loc[saved.task.eq("podium") & saved.family.eq("linear")].set_index("driver_id")
    for row in field.itertuples():
        assert row.podium_probability == podium.loc[row.driver_id, "podium_probability"]
    assert fc.spotlight_candidates(field).driver_id.tolist() == field.head(4).driver_id.tolist()
    assert_frame_equal(saved, before)


def test_missing_actuals_stay_unknown_and_replay_gate_removes_results(replay):
    field = fc.presentation_frame(replay["predictions"], task="race", family="linear")
    field.loc[0, ["actual_rank", "actual_podium"]] = float("nan")
    field.loc[0, "outcome"] = "non_classified"
    shown = fc.presentation_frame(field)
    unknown = shown.loc[shown.driver_id.eq(field.driver_id.iloc[0])].iloc[0]
    assert pd.isna(unknown.rank_error) and unknown.rank_delta_label == "—"
    hidden = fc.presentation_frame(field, outcome=fc.Outcome("race", False, "Not yet revealed"))
    assert hidden.actual_rank.isna().all() and hidden.actual_podium.isna().all()
    assert hidden.rank_error.isna().all()
    assert "outcome" not in hidden and "label_source" not in hidden
    assert fc.rank_delta_label(3) == "▲ +3"
    assert fc.rank_delta_label(-2) == "▼ -2"
    assert fc.rank_delta_label(0) == "— 0"
    # Vega-Lite must retain all driver rows even without actual endpoints.
    spec = charts.rank_deltas(shown, color_by="error").to_dict()
    assert len(spec["layer"][1]["encoding"]["y"]["scale"]["domain"]) == 22
    assert any(param.get("select", {}).get("fields") == ["driver_id"] for param in spec["params"])


def test_spotlight_filters_highlights_and_probability_column_config(replay_path):
    at = AppTest.from_function(_race_page, args=(str(replay_path),), default_timeout=30).run()
    assert not at.exception and not at.error
    html = "\n".join(node.proto.body for node in at.get("html"))
    assert html.count('class="f1-spotlight"') == 3
    assert "P4 Challenger" in html and 'role="meter"' in html
    assert "Podium hit" in html or "Podium miss" in html
    table = next(node for node in at.dataframe if "Podium chance" in node.value)
    config = json.loads(table.proto.columns)
    assert config["Podium chance"]["type_config"]["type"] == "progress"
    assert config["Podium chance"]["type_config"]["min_value"] == 0
    assert config["Podium chance"]["type_config"]["max_value"] == 1
    assert config["driver_id"]["hidden"]
    at.checkbox(key="field_lab_predictions_race_linear_surprises").check().run()
    assert not at.exception and not at.error
    data = next(node for node in at.dataframe if "Podium chance" in node.value).value
    assert 1 <= data.is_surprise.sum() <= 3
    assert data.loc[data.is_surprise, "Rank error"].gt(0).all()
    at.selectbox(key="field_lab_predictions_race_linear_team").set_value("ferrari").run()
    assert not at.exception and not at.error
    filtered = next(node for node in at.dataframe if "Podium chance" in node.value).value
    assert filtered.Team.eq("Ferrari").all() and len(filtered) == 2


def test_calibration_probabilities_are_separate_fit_diagnostics(replay_path):
    at = AppTest.from_function(
        _calibration_page, args=(str(replay_path),), default_timeout=30
    ).run()
    assert not at.exception and not at.error
    table = at.dataframe[0]
    data = table.value
    assert len(data) == 22
    assert data["Before calibration"].between(0, 1).all()
    assert data["Saved calibrated probability"].between(0, 1).all()
    assert any("fit diagnostics" in node.value for node in at.caption)
