"""Offline replay of real Brazil source captures, with no invented event/result rows."""

from contextlib import nullcontext
from pathlib import Path
from shutil import copytree
from types import SimpleNamespace

import pandas as pd
import pytest
import requests
from streamlit.testing.v1 import AppTest

from f1ml.ingestion.types import SourceBatch
from f1ml.storage import read_json, seal_directory, verify_directory
from f1ml_real import lab
from f1ml_real import refresh as live

FIXTURE = Path(__file__).parent / "fixtures/brazil-refresh"


@pytest.fixture
def captured_sources(monkeypatch):
    sources = read_json(FIXTURE / "sources.json")
    requests_made = []

    class Provider:
        def __init__(self, cache):
            self.fastf1 = SimpleNamespace(Cache=SimpleNamespace(disabled=nullcontext))

        def schedule(self, year):
            assert year == 2026
            return read_json(FIXTURE / "schedule.json")

        def fetch_results(self, event, spec):
            requests_made.append((event["event_id"], spec["kind"]))
            assert (event["event_id"], spec["kind"]) == ("2026-20", "Qualifying")
            return SourceBatch(
                {"results": pd.read_parquet(FIXTURE / "results.parquet")},
                read_json(FIXTURE / "results-metadata.json"),
            )

    class Cache:
        def __init__(self, root):
            pass

        def get(self, url):
            if url not in sources:
                raise requests.HTTPError("No current-season documents in captured response")
            return FIXTURE / sources[url]["fixture"]

    def forbid_network(*args, **kwargs):
        raise AssertionError("Offline real-source replay attempted network access")

    monkeypatch.setattr(live, "FastF1Provider", Provider)
    monkeypatch.setattr(live, "PublicCache", Cache)
    monkeypatch.setattr(live, "pdf_text", lambda path, **kwargs: path.read_text())
    monkeypatch.setattr(live, "_now", lambda: pd.Timestamp("2026-11-08T14:50:00.123456Z"))
    monkeypatch.setattr(requests.Session, "request", forbid_network)
    return requests_made


@pytest.fixture
def refreshed(dataset_root, tmp_path, captured_sources):
    state = tmp_path / "live"
    path = live.refresh(dataset_root, state)
    return state, path, lab.RealDataset(path)


def test_real_refresh_adds_brazil_and_preserves_reviewed_history(
    refreshed, dataset_root, real_dataset, captured_sources
):
    state, path, dataset = refreshed
    assert live.current_dataset(dataset_root, state) == path.resolve()
    assert verify_directory(dataset_root) == real_dataset.receipt
    assert len(dataset.events) == 23
    assert captured_sources == [("2026-20", "Qualifying")]
    assert live.forecast_tasks(dataset, "2026-20") == ("race", "podium")
    assert not live.forecast_tasks(dataset, "2026-21")
    for name in ("rosters", "labels", "documents"):
        original = getattr(real_dataset, name)
        actual = getattr(dataset, name)
        actual = actual.loc[actual.event_id.isin(real_dataset.events.event_id)]
        pd.testing.assert_frame_equal(actual.reset_index(drop=True), original)
    roster = dataset.rosters.query("event_id == '2026-20' and task == 'race'")
    assert len(roster) == 22
    assert "rafael_camara" in set(roster.driver_id)
    assert "ocon" not in set(roster.driver_id)
    assert dataset.labels.query("event_id == '2026-20' and task == 'race'").empty
    assert read_json(path / "identity_mapping.json")["CAM"] == "rafael_camara"


def test_refresh_failure_keeps_active_snapshot_and_partial_sources_keep_previous_event(
    refreshed, dataset_root, monkeypatch
):
    state, path, dataset = refreshed
    pointer = (state / "CURRENT.json").read_bytes()
    provider = live.FastF1Provider

    class Unavailable(provider):
        def schedule(self, year):
            raise ValueError("Calendar service unavailable")

    monkeypatch.setattr(live, "FastF1Provider", Unavailable)
    with pytest.raises(ValueError, match="Calendar service unavailable"):
        live.refresh(dataset_root, state)
    assert (state / "CURRENT.json").read_bytes() == pointer
    assert live.current_dataset(dataset_root, state) == path.resolve()
    monkeypatch.setattr(live, "FastF1Provider", provider)
    original_cache = live.PublicCache

    class MissingEvent(original_cache):
        def get(self, url):
            if "S%C3%A3o" in url:
                raise requests.HTTPError("429: try later")
            return super().get(url)

    monkeypatch.setattr(live, "PublicCache", MissingEvent)
    next_path = live.refresh(dataset_root, state)
    new = lab.RealDataset(next_path)
    pd.testing.assert_frame_equal(
        new.rosters.query("event_id == '2026-20'").reset_index(drop=True),
        dataset.rosters.query("event_id == '2026-20'").reset_index(drop=True),
    )
    assert read_json(next_path / "refresh_status.json")[0]["retained_previous"]


def test_unstarted_tasks_enforce_clock_roster_and_qualifying_availability(refreshed):
    _, _, dataset = refreshed
    assert live.forecast_tasks(dataset, "2026-20", "2026-11-07T17:50:00Z") == ("qualifying",)
    assert live.forecast_tasks(dataset, "2026-20", "2026-11-07T19:00:00Z") == ()
    assert live.forecast_tasks(dataset, "2026-20", "2026-11-08T17:00:00Z") == ()
    dataset.rosters = dataset.rosters.loc[
        ~(dataset.rosters.event_id.eq("2026-20") & dataset.rosters.task.eq("race"))
    ]
    assert live.forecast_tasks(dataset, "2026-20") == ()


def test_live_forecast_never_scores_target_and_reuses_verified_artifacts(
    refreshed, tmp_path, monkeypatch
):
    _, path, dataset = refreshed
    attach = lab._attach_actuals

    def earlier_labels_only(source, forecasts):
        assert "2026-20" not in set(forecasts.event_id)
        return attach(source, forecasts)

    monkeypatch.setattr(lab, "_attach_actuals", earlier_labels_only)
    output = lab.cached_forecast(path, "2026-20", tmp_path / "runs", last_n=6)
    run = lab.load_run(output)
    assert run["summary"]["mode"] == "live_forecast"
    assert run["summary"]["training_event_count"] == 6
    assert run["metrics"].empty
    assert set(run["forecasts"].task) == {"race", "podium"}
    assert not {"actual_rank", "actual_podium", "label_source"} & set(run["predictions"])
    for (_, _), group in run["forecasts"].groupby(["task", "family"]):
        assert len(group) == 22 and group.driver_id.nunique() == 22
        assert set(group.predicted_rank) == set(range(1, 23))
    assert set(run["config"]["training_event_ids"]) <= set(
        dataset.history_events("2026-20").event_id
    )

    def forbid_fit(*args, **kwargs):
        raise AssertionError("Reopening a saved forecast must not fit")

    monkeypatch.setattr(lab, "_fit", forbid_fit)
    assert lab.cached_forecast(path, "2026-20", tmp_path / "runs", last_n=6) == output
    monkeypatch.setattr(live, "_now", lambda: pd.Timestamp("2026-11-08T17:00:00Z"))
    with pytest.raises(ValueError, match="no unstarted target"):
        lab.cached_forecast(path, "2026-20", tmp_path / "runs", last_n=6)


def test_live_page_ordinary_rerun_is_read_only_and_displays_brazil(
    refreshed, dataset_root, tmp_path, monkeypatch
):
    state, _, _ = refreshed
    monkeypatch.setenv("F1ML_REAL_DATASET", str(dataset_root))
    monkeypatch.setenv("F1ML_LIVE_DATA", str(state))
    monkeypatch.setenv("F1ML_REAL_RUNS", str(tmp_path / "runs"))

    def forbidden(*args, **kwargs):
        raise AssertionError("Displaying live data must not download or train")

    monkeypatch.setattr(live, "refresh", forbidden)
    monkeypatch.setattr(lab, "cached_forecast", forbidden)
    at = AppTest.from_string(
        "from f1ml_app.lab_app import live_forecast\nlive_forecast()", default_timeout=30
    ).run()
    assert not at.exception and not at.error
    assert at.selectbox(key="lab_live_event").value == "2026-20"
    assert not any(button.label == "Refresh FastF1 data" for button in at.button)
    assert not at.button(key="lab_live_run").disabled
    at.run()
    assert not at.exception and not at.error


def test_historical_refresh_updates_options_and_uses_selected_snapshot(
    refreshed, dataset_root, tmp_path, monkeypatch
):
    state, active, _ = refreshed
    # Replay a bundled dataset from before Mexico using only real captured facts.
    older = tmp_path / "older-base"
    copytree(dataset_root, older)
    metadata = read_json(older / "receipt.json")
    (older / "receipt.json").unlink()
    for name in ("events", "rosters", "labels", "documents"):
        table = older / f"{name}.parquet"
        frame = pd.read_parquet(table)
        frame.loc[~frame.event_id.eq("2026-19")].to_parquet(table, index=False)
    seal_directory(older, {k: v for k, v in metadata.items() if k not in {"files", "id"}})
    monkeypatch.setenv("F1ML_REAL_DATASET", str(older))
    monkeypatch.setenv("F1ML_LIVE_DATA", str(state))
    monkeypatch.setenv("F1ML_REAL_RUNS", str(tmp_path / "runs"))
    selected = [older]
    refresh_calls, run_calls = [], []

    def current(base, live_state):
        assert (base, live_state) == (older, state)
        return selected[0]

    def refresh(base, live_state):
        refresh_calls.append((base, live_state))
        selected[0] = active
        return active

    def run(dataset, target, output, *, last_n):
        run_calls.append((dataset, target, output, last_n))
        raise ValueError("Stopped before fitting in this interface test")

    monkeypatch.setattr(live, "current_dataset", current)
    monkeypatch.setattr(live, "refresh", refresh)
    monkeypatch.setattr(lab, "cached_run", run)
    at = AppTest.from_string("from f1ml_app.lab_app import main\nmain()", default_timeout=30).run()
    assert not at.exception and not at.error
    assert len(at.selectbox(key="lab_target_event").options) == 18
    at.selectbox(key="lab_target_event").set_value("2026-18").run()
    assert not refresh_calls and not run_calls
    at.main.button(key="lab_refresh").click().run()
    assert not at.exception and not at.error
    assert refresh_calls == [(older, state)]
    assert len(at.selectbox(key="lab_target_event").options) == 19
    assert any("Mexico" in option for option in at.selectbox(key="lab_target_event").options)
    assert not any("Paulo" in option for option in at.selectbox(key="lab_target_event").options)
    assert any("São Paulo" in item.value for item in at.info)
    assert at.selectbox(key="lab_target_event").value == "2026-18"
    assert not run_calls
    at.selectbox(key="lab_target_event").set_value("2026-19")
    at.button(key="lab_run").click().run()
    assert not at.exception
    assert run_calls == [(active, "2026-19", tmp_path / "runs", None)]
    at.run()
    assert len(refresh_calls) == len(run_calls) == 1


def test_historical_refresh_failure_keeps_previous_options(
    refreshed, dataset_root, tmp_path, monkeypatch
):
    state, _, _ = refreshed
    monkeypatch.setenv("F1ML_REAL_DATASET", str(dataset_root))
    monkeypatch.setenv("F1ML_LIVE_DATA", str(state))
    monkeypatch.setenv("F1ML_REAL_RUNS", str(tmp_path / "runs"))
    pointer = (state / "CURRENT.json").read_bytes()

    def unavailable(*args):
        raise requests.ConnectionError("Calendar service unavailable")

    monkeypatch.setattr(live, "refresh", unavailable)
    at = AppTest.from_string("from f1ml_app.lab_app import main\nmain()", default_timeout=30).run()
    assert not at.exception and not at.error
    previous = at.selectbox(key="lab_target_event").options
    at.main.button(key="lab_refresh").click().run()
    assert not at.exception
    assert "previous snapshot is still available" in at.error[0].value
    assert at.selectbox(key="lab_target_event").options == previous
    assert (state / "CURRENT.json").read_bytes() == pointer


def test_methodology_refresh_updates_coverage_sources_and_timestamp(
    dataset_root, tmp_path, captured_sources, monkeypatch
):
    state = tmp_path / "live"
    monkeypatch.setenv("F1ML_REAL_DATASET", str(dataset_root))
    monkeypatch.setenv("F1ML_LIVE_DATA", str(state))
    monkeypatch.setenv("F1ML_REAL_RUNS", str(tmp_path / "runs"))

    def forbidden(*args, **kwargs):
        raise AssertionError("Refreshing methodology must not train models")

    monkeypatch.setattr(lab, "_fit", forbidden)
    at = AppTest.from_string(
        "from f1ml_app.lab_app import _sidebar_refresh, data_methodology\n"
        "_sidebar_refresh()\ndata_methodology()",
        default_timeout=30,
    ).run()
    assert not at.exception and not at.error
    before = next(frame.value for frame in at.dataframe if "Known numeric ranks" in frame.value)
    assert len(before) == 38
    assert len(at.sidebar.button) == 2  # Select All Models and Reset Filters
    assert at.main.button(key="lab_refresh").label == "Refresh FastF1 data"
    at.main.button(key="lab_refresh").click().run()
    assert not at.exception and not at.error
    coverage = next(frame.value for frame in at.dataframe if "Known numeric ranks" in frame.value)
    assert len(coverage) == 46
    brazil = coverage.loc[coverage.Event.str.contains("São Paulo")]
    assert len(brazil) == 2
    assert brazil["Rostered drivers"].eq(22).all()
    assert brazil.loc[brazil.Session.eq("Qualifying"), "Known numeric ranks"].iloc[0] > 0
    assert brazil.loc[brazil.Session.eq("Race"), "Known numeric ranks"].iloc[0] == 0
    assert (
        brazil.loc[brazil.Session.eq("Race"), "Status"].iloc[0] == "Awaiting final classification"
    )
    assert next(metric.value for metric in at.metric if metric.label == "Completed events") == "19"
    assert any("Last refreshed: 08 Nov 2026, 14:50 UTC" in caption.value for caption in at.caption)
    docs = next(frame.value for frame in at.dataframe if "published_at" in frame.value)
    assert "2026-20" in set(docs.event_id)
    details = next(frame.value for frame in at.dataframe if "issues" in frame.value)
    assert "2026-20" in set(details.event_id)
    pointer = (state / "CURRENT.json").read_bytes()
    at.run()
    assert not at.exception and not at.error
    assert (state / "CURRENT.json").read_bytes() == pointer
    assert captured_sources == [("2026-20", "Qualifying")]


def test_sidebar_refresh_uses_selected_season_and_live_page_uses_current_season(
    dataset_root, tmp_path, monkeypatch
):
    root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("F1ML_REAL_DATASET", str(dataset_root))
    monkeypatch.setenv("F1ML_HISTORICAL_DATA", str(root / "data/historical"))
    monkeypatch.setenv("F1ML_LIVE_DATA", str(tmp_path / "live"))
    monkeypatch.setenv("F1ML_REAL_RUNS", str(tmp_path / "runs"))
    calls = []

    def refresh(base, state):
        calls.append((base, state))

    monkeypatch.setattr(live, "refresh", refresh)
    at = AppTest.from_string(
        "import streamlit as st\n"
        "from f1ml_app.lab_app import _sidebar_refresh, data_methodology\n"
        "_sidebar_refresh(live_page=st.session_state.get('test_live_page', False))\n"
        "data_methodology()",
        default_timeout=30,
    ).run()
    at.selectbox(key="lab_season").set_value(2024).run()
    assert not at.exception and not at.error and not calls
    assert any(caption.value == "Refresh season: 2024" for caption in at.main.caption)
    at.main.button(key="lab_refresh").click().run()
    assert calls == [(root / "data/historical/2024", tmp_path / "live/2024")]
    at.session_state["test_live_page"] = True
    at.run()
    assert any(caption.value == "Refresh season: 2026" for caption in at.main.caption)
    at.main.button(key="lab_refresh").click().run()
    assert calls[-1] == (dataset_root, tmp_path / "live")


def test_forecast_is_not_sealed_if_session_starts_during_fit(refreshed, tmp_path, monkeypatch):
    _, path, _ = refreshed
    fit = lab._fit

    def crosses_start(*args, **kwargs):
        model = fit(*args, **kwargs)
        monkeypatch.setattr(live, "_now", lambda: pd.Timestamp("2026-11-08T17:00:00Z"))
        return model

    monkeypatch.setattr(lab, "_fit", crosses_start)
    runs = tmp_path / "late-runs"
    with pytest.raises(ValueError, match="started during training"):
        lab.cached_forecast(path, "2026-20", runs, last_n=3)
    assert not list(runs.glob("*/prediction_lock.json"))
    assert not list(runs.glob("*/receipt.json"))


def test_corrupt_refreshed_snapshot_is_never_loaded(refreshed, dataset_root):
    state, path, _ = refreshed
    (path / "refresh_status.json").write_text("[]")
    with pytest.raises(ValueError, match="Snapshot contents changed"):
        live.current_dataset(dataset_root, state)
