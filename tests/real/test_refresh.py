"""Offline replay of real Brazil source captures, with no invented event/result rows."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from pathlib import Path
from shutil import copytree
from threading import Event
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
    assert live.forecast_tasks(dataset, "2026-21") == ("qualifying", "race", "podium")
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
    # A snapshot acquired after the queried time cannot be used as an earlier forecast.
    assert live.forecast_tasks(dataset, "2026-20", "2026-11-07T17:50:00Z") == ()
    assert live.forecast_tasks(dataset, "2026-20", "2026-11-07T19:00:00Z") == ()
    assert live.forecast_tasks(dataset, "2026-20", "2026-11-08T17:00:00Z") == ()
    dataset.rosters = dataset.rosters.loc[
        ~(dataset.rosters.event_id.eq("2026-20") & dataset.rosters.task.eq("race"))
    ]
    # The event's qualifying entry list remains a usable known field.
    assert live.forecast_tasks(dataset, "2026-20") == ("race", "podium")


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


@pytest.mark.parametrize("last_n", [None, 12, 6])
def test_pre_weekend_forecast_chains_each_familys_qualifying(
    refreshed, tmp_path, monkeypatch, last_n
):
    _, path, dataset = refreshed
    original_labels = dataset.labels.copy(deep=True)
    attach = lab._attach_actuals

    def no_target_labels(source, forecasts):
        assert "2026-21" not in set(forecasts.event_id)
        return attach(source, forecasts)

    monkeypatch.setattr(lab, "_attach_actuals", no_target_labels)
    output = lab.cached_forecast(path, "2026-21", tmp_path / "runs", last_n=last_n)
    run = lab.load_run(output)
    expected_ids = dataset.history_events("2026-21").event_id.tolist()
    if last_n:
        expected_ids = expected_ids[-last_n:]
    assert run["summary"]["training_event_ids"] == expected_ids
    assert run["summary"]["qualifying_source"] == "predicted"
    assert run["metrics"].empty
    assert set(run["forecasts"].task) == {"qualifying", "race", "podium"}
    assert run["forecasts"].field_source.eq("latest_known_field_assumption").all()
    assert set(run["forecasts"].field_source_event_id) == {"2026-20"}
    assert not {"actual_rank", "actual_podium"} & set(run["forecasts"])
    audits = read_json(output / "feature_audits.json")
    for family in (*lab.FAMILIES, "baseline"):
        q = run["forecasts"].query('task == "qualifying" and family == @family')
        assert len(q) == 22
        assert set(q.predicted_rank) == set(range(1, 23))
        for task in ("race", "podium"):
            inputs = pd.read_parquet(output / "inputs" / f"{task}-{family}.parquet")
            ranks = inputs.set_index("driver_id").current_qualifying_rank * len(inputs)
            expected = q.set_index("driver_id").predicted_rank
            pd.testing.assert_series_equal(
                ranks.sort_index(), expected.sort_index(), check_names=False
            )
            forecast = run["forecasts"].query("task == @task and family == @family")
            assert forecast.qualifying_source.eq("predicted").all()
            assert set(forecast.predicted_rank) == set(range(1, 23))
            audit = audits[f"2026-21/{task}/{family}"]
            assert audit["qualifying_source"] == "predicted"
            assert not audit["current_qualifying_used"]
    pd.testing.assert_frame_equal(dataset.labels, original_labels)


def test_finished_weekend_leaves_upcoming_list_even_without_final_labels(
    refreshed, dataset_root, tmp_path, monkeypatch
):
    state, _, dataset = refreshed
    monkeypatch.setattr(live, "_now", lambda: pd.Timestamp("2026-11-11T14:00:00Z"))
    assert "2026-20" not in set(live.upcoming_events(dataset).event_id)
    assert set(live.upcoming_events(dataset).event_id) == {"2026-21", "2026-22", "2026-23"}
    monkeypatch.setenv("F1ML_REAL_DATASET", str(dataset_root))
    monkeypatch.setenv("F1ML_LIVE_DATA", str(state))
    monkeypatch.setenv("F1ML_REAL_RUNS", str(tmp_path / "runs"))
    at = AppTest.from_string(
        "from f1ml_app.lab_app import live_forecast\nlive_forecast()", default_timeout=30
    ).run()
    assert not at.exception and not at.error
    assert at.selectbox(key="lab_live_event").value == "2026-21"
    assert not any("Paulo" in option for option in at.selectbox(key="lab_live_event").options)
    assert not at.button(key="lab_live_run").disabled
    assert any("Provisional field" in message.value for message in at.info)
    at.button(key="lab_live_run").click().run(timeout=60)
    assert not at.exception and not at.error
    assert any("predicted qualifying" in message.value for message in at.info)
    # Changing training scope cannot display the previous window's predictions as its result.
    at.selectbox(key="lab_live_window").set_value("Previous 6 races that year").run()
    assert any("selected training window" in message.value for message in at.info)


def test_live_field_prefers_event_entry_list_and_rejects_later_information(refreshed):
    _, _, dataset = refreshed
    cutoff = dataset.event("2026-21").qualifying_cutoff
    field = dataset.forecast_field("2026-21", "qualifying", cutoff)
    assert set(field.field_source_event_id) == {"2026-20"}
    known = dataset.forecast_field("2026-20", "race", dataset.event("2026-20").race_cutoff)
    assert known.field_source.eq("event_entry_list").all()
    # A roster from a later weekend, even if present, cannot cross the prediction cutoff.
    later = field.copy()
    later["event_id"] = "2026-21"
    later["known_at"] = "2026-11-20T00:00:00Z"
    dataset.rosters = pd.concat([dataset.rosters, later], ignore_index=True)
    again = dataset.forecast_field("2026-21", "qualifying", cutoff)
    assert again.field_source.eq("latest_known_field_assumption").all()


def test_interrupted_forecast_can_retry_without_exposing_partial_output(
    refreshed, tmp_path, monkeypatch
):
    _, dataset, _ = refreshed
    runs = tmp_path / "runs"
    build = lab._run_backtest

    def interrupted(dataset, event_id, output, **kwargs):
        output.mkdir()
        (output / "attempt.txt").write_text("Interrupted before the receipt was written")
        raise RuntimeError("Training interrupted")

    monkeypatch.setattr(lab, "_run_backtest", interrupted)
    with pytest.raises(RuntimeError, match="Training interrupted"):
        lab.cached_forecast(dataset, "2026-21", runs, last_n=3)
    assert not list(runs.glob("forecast-*"))
    assert not list((runs / ".pending").iterdir())

    monkeypatch.setattr(lab, "_run_backtest", build)
    output = lab.cached_forecast(dataset, "2026-21", runs, last_n=3)
    assert lab.load_run(output)["summary"]["training_event_count"] == 3


def test_legacy_incomplete_forecast_is_preserved_and_retried_but_sealed_corruption_is_rejected(
    refreshed, tmp_path, monkeypatch
):
    _, dataset, _ = refreshed
    runs = tmp_path / "runs"
    output = lab.cached_forecast(dataset, "2026-21", runs, last_n=3)
    # Simulate an older run interrupted just before the final receipt write.
    (output / "receipt.json").unlink()
    previous = {
        str(p.relative_to(output)): p.read_bytes() for p in output.rglob("*") if p.is_file()
    }
    assert lab.cached_forecast(dataset, "2026-21", runs, last_n=3) == output
    assert lab.load_run(output)["summary"]["training_event_count"] == 3
    (archived,) = (runs / ".incomplete").iterdir()
    assert {
        str(p.relative_to(archived)): p.read_bytes() for p in archived.rglob("*") if p.is_file()
    } == previous

    def forbidden(*args, **kwargs):
        raise AssertionError("Sealed but corrupt runs must never be silently regenerated")

    monkeypatch.setattr(lab, "_run_backtest", forbidden)
    config = output / "config.json"
    config.write_text(config.read_text() + "\n")
    with pytest.raises(ValueError, match="Snapshot contents changed"):
        lab.cached_forecast(dataset, "2026-21", runs, last_n=3)


def test_simultaneous_forecast_requests_publish_once_and_reuse_the_completed_run(
    refreshed, tmp_path, monkeypatch
):
    _, dataset, _ = refreshed
    build = lab._run_backtest
    cache_lock = lab._run_cache_lock
    ready, release, second_request = Event(), Event(), Event()
    builds, requests = [], []

    def pause_before_publication(*args, **kwargs):
        builds.append(args[2])
        result = build(*args, **kwargs)
        ready.set()
        assert release.wait(timeout=30)
        return result

    @contextmanager
    def observed_lock(path):
        requests.append(path)
        if len(requests) == 2:
            second_request.set()
        with cache_lock(path):
            yield

    monkeypatch.setattr(lab, "_run_backtest", pause_before_publication)
    monkeypatch.setattr(lab, "_run_cache_lock", observed_lock)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(lab.cached_forecast, dataset, "2026-21", tmp_path, last_n=3)
        try:
            assert ready.wait(timeout=30)
            assert not requests[0].exists()
            assert not list(tmp_path.glob("*/config.json"))
            second = pool.submit(lab.cached_forecast, dataset, "2026-21", tmp_path, last_n=3)
            assert second_request.wait(timeout=10)
            assert not second.done()
        finally:
            release.set()
        first_output, second_output = first.result(timeout=30), second.result(timeout=30)
    assert first_output == second_output
    assert len(builds) == 1
    assert lab.load_run(first_output)["summary"]["training_event_count"] == 3
    assert not list((tmp_path / ".pending").iterdir())


def test_waiting_for_cache_lock_cannot_bypass_session_start(refreshed, tmp_path, monkeypatch):
    _, dataset, _ = refreshed

    @contextmanager
    def after_session_starts(path):
        monkeypatch.setattr(live, "_now", lambda: pd.Timestamp("2026-11-08T17:00:00Z"))
        yield

    def forbidden(*args, **kwargs):
        raise AssertionError("A request that waited past session start must not train")

    monkeypatch.setattr(lab, "_run_cache_lock", after_session_starts)
    monkeypatch.setattr(lab, "_run_backtest", forbidden)
    with pytest.raises(ValueError, match="target session has started"):
        lab.cached_forecast(dataset, "2026-20", tmp_path, last_n=3)


def test_saved_run_discovery_skips_unsealed_attempts(refreshed, tmp_path, monkeypatch):
    from f1ml.storage import write_json
    from f1ml_app import lab_app

    _, dataset, source = refreshed
    runs = tmp_path / "runs"
    write_json(
        runs / "forecast-unfinished" / "config.json",
        {"target_event_id": "2026-21", "source_snapshot_id": source.receipt["id"], "last_n": None},
    )
    monkeypatch.setattr(lab_app, "paths", lambda: (dataset, runs))
    assert lab_app._default_saved("2026-21", source.receipt["id"]) is None
    assert lab_app._saved_options(source, 2026) == []
    assert not lab_app._has_saved("2026-21", None, source.receipt["id"])
