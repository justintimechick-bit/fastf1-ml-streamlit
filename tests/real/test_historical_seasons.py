"""Historical season selection and training windows on actual FIA observations."""

from contextlib import nullcontext
from copy import copy
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import requests
from streamlit.testing.v1 import AppTest

from f1ml.storage import file_digest, read_json, verify_directory
from f1ml_real import lab
from f1ml_real import refresh as live

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(params=(2024, 2025))
def season_dataset(request):
    return lab.RealDataset(ROOT / "data/historical" / str(request.param))


def test_historical_sources_have_independent_fields_and_cutoff_valid_labels(season_dataset):
    dataset = season_dataset
    assert verify_directory(dataset.root)["data_kind"] == "real"
    year = dataset.receipt["year"]
    assert set(dataset.events.year) == {year}
    assert len(dataset.events) == 24
    assert dataset.rosters.groupby(["event_id", "task"]).size().eq(20).all()
    assert dataset.documents.url.str.contains(str(year)).all()
    assert dataset.labels.query("task == 'race'").groupby("event_id").podium.sum().eq(3).all()
    sauber_drivers = ("bottas", "zhou") if year == 2024 else ("hulkenberg", "bortoleto")
    assert (
        dataset.rosters.loc[dataset.rosters.driver_id.isin(sauber_drivers), "team_id"]
        .eq("sauber")
        .all()
    )
    for document in dataset.documents.loc[dataset.documents.source_file.notna()].itertuples():
        assert file_digest(dataset.root / document.source_file) == document.sha256
    fallback = dataset.labels.source.str.contains("api.jolpi.ca", regex=False)
    assert fallback.any()
    assert dataset.receipt["fallback_semantics"]
    special = fallback & dataset.labels.outcome.ne("ordered")
    assert dataset.labels.loc[special, "official_position"].isna().all()
    for event in dataset.events.itertuples():
        assert pd.Timestamp(event.qualifying_cutoff) < pd.Timestamp(event.qualifying_available_at)
        assert pd.Timestamp(event.qualifying_available_at) < pd.Timestamp(event.race_cutoff)
        assert pd.Timestamp(event.race_cutoff) < pd.Timestamp(event.race_available_at)
    keys = ["event_id", "task", "driver_id"]
    assert set(map(tuple, dataset.rosters[keys].values)) == set(
        map(tuple, dataset.labels[keys].values)
    )


@pytest.mark.parametrize("last_n", (None, 6, 12))
def test_each_training_window_fits_only_earlier_races_that_year(season_dataset, last_n, tmp_path):
    dataset = season_dataset
    target = dataset.events.iloc[-1].event_id
    eligible = dataset.history_events(target)
    expected = eligible if last_n is None else eligible.tail(last_n)
    output = lab.cached_run(dataset.root, target, tmp_path / "runs", last_n=last_n)
    run = lab.load_run(output)
    assert run["config"]["training_event_ids"] == expected.event_id.tolist()
    assert run["summary"]["training_history"] == f"{dataset.receipt['year']}-only"
    assert run["summary"]["training_event_count"] == len(expected)
    assert set(run["training_events"].year) == {dataset.receipt["year"]}
    assert target not in set(run["training_events"].event_id)
    assert set(run["forecasts"].task) == set(lab.TASKS)
    assert run["summary"]["model_count"] == 9
    assert len(run["forecasts"]) == 20 * 3 * 4  # Three tasks, three families and baseline.


def test_season_opener_cannot_train_on_another_year(season_dataset, tmp_path):
    first = season_dataset.events.iloc[0].event_id
    with pytest.raises(ValueError, match="No earlier published real events"):
        lab.cached_run(season_dataset.root, first, tmp_path / "runs")


def test_short_training_window_uses_available_earlier_races(season_dataset, tmp_path):
    target = season_dataset.events.iloc[2].event_id
    output = lab.cached_run(season_dataset.root, target, tmp_path / "runs", last_n=12)
    assert lab.load_run(output)["config"]["training_event_ids"] == (
        season_dataset.history_events(target).event_id.tolist()
    )


def test_refresh_preserves_archived_api_evidence_and_limitations(
    season_dataset, tmp_path, monkeypatch
):
    captures = read_json(Path(__file__).parent / "fixtures/historical-schedules.json")
    schedule = [
        event for event in captures["events"] if event["year"] == season_dataset.receipt["year"]
    ]

    class Provider:
        def __init__(self, cache):
            self.fastf1 = SimpleNamespace(Cache=SimpleNamespace(disabled=nullcontext))

        def schedule(self, year):
            assert year == season_dataset.receipt["year"]
            return schedule

    index = tmp_path / "empty-index.html"
    index.write_text("")  # No new events: existing captured calendar facts are unchanged.
    monkeypatch.setattr(live, "FastF1Provider", Provider)
    monkeypatch.setattr(live.PublicCache, "get", lambda self, url: index)
    path = live.refresh(season_dataset.root, tmp_path / "live")
    revised = lab.RealDataset(path)
    assert revised.receipt["api_fallback_events"] == season_dataset.receipt["api_fallback_events"]
    assert revised.receipt["fallback_semantics"] == season_dataset.receipt["fallback_semantics"]
    pd.testing.assert_frame_equal(revised.labels, season_dataset.labels)
    for document in revised.documents.loc[revised.documents.source_file.notna()].itertuples():
        assert file_digest(path / document.source_file) == document.sha256
    assert (path / "fallbacks.parquet").exists()


def test_combined_facts_never_leak_another_season_into_history_or_features(real_dataset):
    old = lab.RealDataset(ROOT / "data/historical/2024")
    current = lab.RealDataset(ROOT / "data/historical/2025")
    mixed = copy(current)
    mixed._history_cache = {}
    for table in ("events", "rosters", "labels", "documents"):
        setattr(
            mixed,
            table,
            pd.concat(
                [getattr(old, table), getattr(current, table), getattr(real_dataset, table)],
                ignore_index=True,
            ),
        )
    target = current.events.iloc[-1].event_id
    pd.testing.assert_frame_equal(
        mixed.history_events(target), current.history_events(target), check_dtype=False
    )
    for task in lab.TASKS:
        expected, _ = current.feature_frame(target, task)
        actual, audit = mixed.feature_frame(target, task)
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
        assert all(eid.startswith("2025-") for eid in audit["feature_history_event_ids"])


def test_season_switching_and_selected_training_window_are_explicit(
    dataset_root, tmp_path, monkeypatch
):
    monkeypatch.setenv("F1ML_REAL_DATASET", str(dataset_root))
    monkeypatch.setenv("F1ML_HISTORICAL_DATA", str(ROOT / "data/historical"))
    monkeypatch.setenv("F1ML_LIVE_DATA", str(tmp_path / "live"))
    monkeypatch.setenv("F1ML_REAL_RUNS", str(tmp_path / "runs"))
    calls = []

    def forbidden(*args, **kwargs):
        raise AssertionError("An ordinary season change must not download or train")

    def selected_run(dataset, target, output, *, last_n):
        calls.append((dataset, target, last_n))
        raise ValueError("Stopped before fitting in this interface test")

    monkeypatch.setattr(requests.Session, "request", forbidden)
    monkeypatch.setattr(lab, "_fit", forbidden)
    monkeypatch.setattr(lab, "cached_run", selected_run)
    at = AppTest.from_string(
        "from f1ml_app.lab_app import historical_test\nhistorical_test()", default_timeout=30
    ).run()
    assert not at.exception and not at.error
    assert at.selectbox(key="lab_season").options == ["2026", "2025", "2024"]
    for year in (2025, 2024, 2026):
        at.selectbox(key="lab_season").set_value(year).run()
        assert not at.exception and not at.error
        assert at.selectbox(key="lab_target_event").value.startswith(f"{year}-")
        assert at.session_state["lab_active_run"] is None
        assert not calls
    at.selectbox(key="lab_season").set_value(2024).run()
    at.selectbox(key="lab_target_event").set_value("2024-19")
    at.selectbox(key="lab_history_window").set_value("Previous 6 races that year")
    at.button(key="lab_run").click().run()
    assert not at.exception
    assert calls == [(ROOT / "data/historical/2024", "2024-19", 6)]
    at.selectbox(key="lab_season").set_value(2025).run()
    assert not at.exception and not at.error
