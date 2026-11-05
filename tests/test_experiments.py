"""Optional-channel units, cutoff isolation, caching, paired comparisons and core preservation."""

import copy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from f1ml.config import Config, ExperimentConfig
from f1ml.domain import utc
from f1ml.evaluation.splits import load_manifest
from f1ml.experiments.acquisition import TelemetryCatalog
from f1ml.experiments.aggregates import lap_telemetry, pace_summary, weather_summary
from f1ml.experiments.build import build_supplements, event_supplement
from f1ml.experiments.contract import (
    EXTRA,
    GROUPS,
    feature_columns,
    matrix,
    merge_features,
    specification,
)
from f1ml.experiments.evaluate import fit_cell, prediction, retention, run_experiments
from f1ml.features.contract import model_matrix
from f1ml.features.source import as_source
from f1ml.models.qualifying import fit_qualifying, predict_scores, validate_permutation
from f1ml.storage import read_json, verify_directory


def car_samples():
    return pd.DataFrame(
        {
            "SessionTime": pd.to_timedelta([0, 0.1, 1], unit="s"),
            "Speed": [100, 200, 300],
            "Throttle": [100, 0, 0],
            "Brake": [False, True, True],
            "Source": "car",
            "DRS": [0, 12, 14],
        }
    )


def test_variable_intervals_are_time_weighted_and_units_stay_explicit():
    result = lap_telemetry(car_samples(), {"start": 0, "end": 1, "seconds": 1}, ExperimentConfig())
    assert result["speed_mean_kph"] == pytest.approx(240)
    assert result["speed_p95_kph"] == 200
    assert result["full_throttle_fraction"] == pytest.approx(0.1)
    assert result["braking_fraction"] == pytest.approx(0.9)
    assert result["speed_coverage"] == 1
    assert not any("drs" in name.casefold() or "energy" in name for name in result)


def test_missing_channels_sentinels_gaps_and_interpolated_samples_stay_missing():
    raw = car_samples().drop(columns="Brake")
    raw["Throttle"] = 104
    result = lap_telemetry(raw, {"start": 0, "end": 1, "seconds": 1}, ExperimentConfig())
    assert pd.isna(result["braking_fraction"]) and result["brake_coverage"] == 0
    assert pd.isna(result["full_throttle_fraction"]) and result["throttle_coverage"] == 0
    raw["Source"] = "interpolation"
    assert pd.isna(
        lap_telemetry(raw, {"start": 0, "end": 1, "seconds": 1}, ExperimentConfig())[
            "speed_mean_kph"
        ]
    )
    raw = car_samples().iloc[[0, -1]].copy()
    raw["SessionTime"] = pd.to_timedelta([0, 5], unit="s")
    result = lap_telemetry(raw, {"start": 0, "end": 5, "seconds": 5}, ExperimentConfig())
    assert pd.isna(result["speed_mean_kph"]) and result["speed_coverage"] == 0
    with pytest.raises(ValueError, match="Duplicate telemetry"):
        lap_telemetry(
            pd.concat([car_samples(), car_samples().iloc[:1]]),
            {"start": 0, "end": 1, "seconds": 1},
            ExperimentConfig(),
        )


def test_consistency_does_not_pool_qualifying_segments():
    laps = pd.DataFrame(
        {
            "driver_id": ["a", "a", "b", "a"],
            "seconds": [80, 82, 79, 50],
            "segment": ["Q1", "Q1", "Q1", "Q3"],
            "compound": "SOFT",
            "time_bin": 1,
            "rain": False,
            "temperature_bin": 6,
            "sector1": [20, 22, 19, 10],
            "sector2": [30, 30, 29, 20],
            "sector3": [30, 30, 31, 20],
        }
    )
    summary = pace_summary(laps, "a")
    assert summary["lap_cvs"] == [pytest.approx(1 / 81)]
    assert summary["sector_cvs"] == [pytest.approx(1 / 21), 0, 0]
    assert summary["laps"] == 2
    changed = laps.copy()
    changed.loc[changed.segment.eq("Q3"), "seconds"] = 500
    assert pace_summary(changed, "a")["lap_cvs"] == summary["lap_cvs"]


def test_weather_rain_fraction_uses_elapsed_time_not_sample_counts():
    weather = pd.DataFrame(
        {
            "Time": pd.to_timedelta([0, 10, 60], unit="s"),
            "Rainfall": [True, False, False],
            "AirTemp": [20, 22, 24],
        }
    )
    result = weather_summary(weather)
    assert result["observed_rain_fraction"] == pytest.approx(1 / 6)
    assert result["observed_air_temp_c"] == pytest.approx((20 * 10 + 22 * 50) / 60)
    assert pd.isna(weather_summary(weather.iloc[:1])["observed_rain_fraction"])


def test_weather_samples_need_absolute_timestamps_and_exclude_post_session_values():
    from f1ml.experiments.build import bounded_weather

    start = utc("2026-03-01T10:00:00Z")
    spec = {
        "scheduled_start": start.isoformat(),
        "ended_at": (start + pd.Timedelta(hours=1)).isoformat(),
    }
    cutoff = start + pd.Timedelta(hours=2)
    raw = pd.DataFrame(
        {"Time": pd.to_timedelta([0, 60, 7200], unit="s"), "AirTemp": [20.0, 21.0, 9999.0]}
    )
    assert bounded_weather(raw, spec, {}, cutoff).empty
    raw["observed_at"] = start + raw.Time
    eligible = bounded_weather(raw, spec, {}, cutoff)
    assert len(eligible) == 2
    before = weather_summary(eligible)
    raw.loc[2, "AirTemp"] = -9999
    assert weather_summary(bounded_weather(raw, spec, {}, cutoff)) == before


@pytest.fixture(scope="module")
def experiment_run(workflow, tmp_path_factory):
    from f1ml.features.source import load_source

    root = tmp_path_factory.mktemp("experiments")
    manifest = load_manifest(workflow / "manifest.json")
    options = ExperimentConfig(families=("linear",))
    source = load_source(workflow / "dataset")
    build_supplements(
        source, manifest, workflow / "features", root / "capture", root / "optional", options
    )
    config = replace(Config(), experiments=options)
    run_experiments(manifest, workflow / "features", root / "optional", root / "comparison", config)
    return root, config


def test_paired_comparisons_keep_missing_optional_rows_and_all_development_events(
    experiment_run, manifest
):
    root, _ = experiment_run
    receipt = verify_directory(root / "comparison")
    assert receipt["final_test_accessed"] is False and receipt["promoted_groups"] == []
    metrics = pd.read_parquet(root / "comparison/event_metrics.parquet")
    assert len(metrics) == 3 * 3 * 6
    for _, cell in metrics.groupby(["task", "family", "strategy"]):
        sets = [set(g.event_id) for _, g in cell.groupby("group")]
        assert all(ids == {"2026-01", "2026-02", "2026-03"} for ids in sets)
    assert not set(metrics.event_id) & set(manifest["final_test"])
    assert metrics.loc[metrics.group.eq("telemetry"), "observed_fraction"].eq(0).all()
    assert all(
        r["decision"] == "do_not_promote" for r in read_json(root / "comparison/retention.json")
    )
    predictions = pd.read_parquet(root / "comparison/predictions.parquet")
    for (task, _, _, _event), group in predictions.groupby(["task", "family", "group", "event_id"]):
        assert not group.driver_id.duplicated().any()
        if task != "podium":
            validate_permutation(group, group)
    assert (root / "comparison/missingness.parquet").exists()


def test_group_allowlists_keep_history_and_prevent_qualifying_and_race_leakage():
    history = feature_columns("race", "history")
    assert "qualifying_rank_mean_3" in history and "race_rank_mean_3" in history
    assert "qualifying_rank_norm" not in history and "practice_gap_pct" not in history
    assert "qualifying_rank_norm" in feature_columns("race", "sessions")
    assert "qualifying_rank_norm" not in feature_columns("qualifying", "weather")
    assert "practice_lap_cv" in feature_columns("race", "practice")
    assert "telemetry_speed_mean_kph" not in feature_columns("race", "practice")
    assert "observed_rain_fraction" not in feature_columns("race", "telemetry")
    assert not any("DRS" in c or "fuel" in c or "battery" in c for c in EXTRA)


@pytest.mark.parametrize(
    "task,family",
    [
        (t, f)
        for t in ("qualifying", "race", "podium")
        for f in ("linear", "random_forest", "gradient_boosting")
    ],
)
def test_existing_model_families_accept_explicit_experiment_spec(
    workflow, manifest, experiment_run, task, family
):
    root, config = experiment_run
    source_task = "race" if task == "podium" else task
    core = pd.read_parquet(workflow / f"features/{source_task}_features.parquet")
    extra = pd.read_parquet(root / f"optional/{source_task}_optional.parquet")
    features = merge_features(core, extra)
    labels = pd.read_parquet(workflow / f"features/{source_task}_labels.parquet")
    config = replace(
        config,
        qualifying=replace(config.qualifying, forest_trees=5, boosting_iterations=5),
        race=replace(config.race, forest_trees=5, boosting_iterations=5),
    )
    fold = manifest["folds"][-1]
    model = fit_cell(
        features, labels, manifest, fold["train"], task, family, "equal", "weather", config
    )
    field = features.loc[features.event_id.eq(fold["validate"])]
    predicted = prediction(model, field, manifest, task)
    assert set(predicted.driver_id) == set(field.driver_id)
    assert model.metadata["experiment"] == specification("weather")
    with pytest.raises(ValueError):
        matrix(field.assign(target_weather=1), task, specification("weather"))


def test_core_matrix_and_predictions_are_preserved(workflow, manifest, experiment_run):
    root, _ = experiment_run
    core = pd.read_parquet(workflow / "features/qualifying_features.parquet")
    extra = pd.read_parquet(root / "optional/qualifying_optional.parquet")
    joined = merge_features(core, extra)
    actual, _, _ = matrix(joined, "qualifying", specification("core"))
    pd.testing.assert_frame_equal(actual, model_matrix(core, "qualifying"))
    labels = pd.read_parquet(workflow / "features/qualifying_labels.parquet")
    fold = manifest["folds"][-1]
    targets = labels.loc[labels.event_id.isin(fold["train"])]
    first = fit_qualifying(
        core.loc[core.event_id.isin(fold["train"])],
        targets,
        manifest,
        "ridge",
        "equal",
        {"alpha": 10},
        {},
        42,
    )
    second = fit_qualifying(
        joined.loc[joined.event_id.isin(fold["train"])],
        targets,
        manifest,
        "ridge",
        "equal",
        {"alpha": 10},
        {},
        42,
        experiment=specification("core"),
    )
    pd.testing.assert_frame_equal(
        predict_scores(first, core.loc[core.event_id.eq(fold["validate"])]),
        predict_scores(second, joined.loc[joined.event_id.eq(fold["validate"])]),
    )


def test_telemetry_cache_reuse_failure_cooldown_and_forbidden_session_guards(
    dataset, manifest, event_inputs, tmp_path
):
    source = as_source(dataset)
    _, cutoff, _ = event_inputs("2026-03")
    options = ExperimentConfig(max_attempts=1)

    class Failure:
        calls = 0

        def capture(self, event, spec):
            self.calls += 1
            raise OSError("explicit fixture access failure")

    provider = Failure()
    requests = pd.DataFrame(
        [
            {
                "event_id": "2026-03",
                "kind": "Practice 1",
                "task": "qualifying",
                "cutoff": cutoff.isoformat(),
                "mode": "historical_reconstruction",
            }
        ]
    )
    catalog = TelemetryCatalog(tmp_path / "capture")
    first = catalog.run(source, manifest, requests, provider, options)
    assert first[0]["status"] == "failed" and provider.calls == 1
    second = catalog.run(source, manifest, requests, provider, options)
    assert second[0]["action"] == "cooldown" and provider.calls == 1
    catalog.run(source, manifest, requests, provider, options, retry_failed=True)
    assert provider.calls == 2
    for kind in ("Qualifying", "Race", "Sprint"):
        with pytest.raises(ValueError, match="forbidden"):
            catalog.run(source, manifest, requests.assign(kind=kind), provider, options)
    with pytest.raises(ValueError, match="Final test"):
        catalog.run(
            source, manifest, requests.assign(event_id=manifest["final_test"][0]), provider, options
        )


def test_cutoffs_exclude_target_sessions_even_if_optional_observations_exist(
    dataset, manifest, event_inputs
):
    event, cutoff, entrants = event_inputs("2026-03")
    from f1ml.features.build import build_event

    core, _ = build_event(dataset, manifest, event["event_id"], "qualifying", cutoff, entrants)
    reads = []

    class Guard:
        def read(self, source_id, event_id, kind, cutoff, mode, options):
            assert kind not in ("Qualifying", "Race", "Sprint", "Sprint Qualifying")
            reads.append(kind)
            return {}, {"state": "not_requested"}

    extra, audit = event_supplement(as_source(dataset), manifest, core, Guard(), ExperimentConfig())
    assert len(extra) == len(core) and reads
    assert extra.telemetry_laps.eq(0).all()
    assert all(a["kind"] != "Qualifying" for a in audit)


def test_retention_requires_incremental_benefit_coverage_runtime_and_actual_evidence():
    rows = []
    loss = {
        "core": 3.0,
        "history": 5.0,
        "sessions": 4.0,
        "practice": 2.0,
        "telemetry": 1.0,
        "weather": 1.1,
    }
    for group, value in loss.items():
        for event in ("one", "two", "three"):
            rows.append(
                {
                    "task": "race",
                    "family": "linear",
                    "strategy": "equal",
                    "group": group,
                    "event_id": event,
                    "position_mae": value,
                    "runtime_seconds": 1.0,
                    "observed_fraction": 1.0,
                }
            )
    frame = pd.DataFrame(rows)
    paired, decisions = retention(frame, "observed_source_snapshot", ExperimentConfig())
    by_group = {r["group"]: r for r in decisions}
    assert by_group["telemetry"]["decision"] == "candidate_for_protocol_review"
    assert by_group["weather"]["decision"] == "do_not_promote"
    assert by_group["weather"]["mean_improvement_vs_core"] > 0
    assert "insufficient_loss_improvement" in by_group["weather"]["reasons"]
    _, synthetic = retention(frame, "synthetic", ExperimentConfig())
    assert all(r["decision"] == "do_not_promote" for r in synthetic)
    frame.loc[frame.group.eq("telemetry"), "observed_fraction"] = 0
    _, low_coverage = retention(frame, "observed_source_snapshot", ExperimentConfig())
    assert (
        "insufficient_optional_observations"
        in next(r for r in low_coverage if r["group"] == "telemetry")["reasons"]
    )
    with pytest.raises(ValueError, match="identical"):
        retention(frame.iloc[:-1], "observed_source_snapshot", ExperimentConfig())
    assert len(paired) == len(GROUPS) * 3


def test_optional_source_and_implementation_changes_fail_closed(
    workflow, experiment_run, manifest, tmp_path
):
    root, _ = experiment_run
    fake = copy.deepcopy(manifest)
    fake["id"] = "another_manifest"
    with pytest.raises(ValueError, match="different frozen core"):
        run_experiments(fake, workflow / "features", root / "optional", tmp_path / "bad")
    extra = pd.read_parquet(root / "optional/qualifying_optional.parquet")
    core = pd.read_parquet(workflow / "features/qualifying_features.parquet")
    with pytest.raises(ValueError, match="complete core field"):
        merge_features(core, extra.iloc[:-1])
    joined = merge_features(core, extra)
    with pytest.raises(ValueError, match="implementation/version"):
        matrix(joined, "qualifying", {**specification("telemetry"), "implementation_id": "old"})


def test_cached_lap_features_weather_and_teammates_flow_into_supplement(
    dataset, manifest, event_inputs, tmp_path
):
    from f1ml.features.build import build_event

    source = as_source(dataset)
    event, cutoff, entrants = event_inputs("2026-03")
    drivers = entrants.driver_id.iloc[:2].tolist()
    laps, samples = [], []
    for driver_index, driver in enumerate(drivers):
        for lap_number in (1, 2):
            start, duration = 100 * lap_number, 90 + driver_index
            laps.append(
                {
                    "driver_id": driver,
                    "lap_number": lap_number,
                    "LapStartTime": pd.Timedelta(seconds=start),
                    "Time": pd.Timedelta(seconds=start + duration),
                    "LapTime": pd.Timedelta(seconds=duration),
                    "IsAccurate": True,
                    "Deleted": False,
                    "FastF1Generated": False,
                    "PitInTime": pd.NaT,
                    "PitOutTime": pd.NaT,
                    "TrackStatus": "1",
                    "Compound": "SOFT",
                    "Sector1Time": pd.Timedelta(seconds=30),
                    "Sector2Time": pd.Timedelta(seconds=30),
                    "Sector3Time": pd.Timedelta(seconds=30 + driver_index),
                }
            )
            for t in np.arange(start, start + duration + 0.1, 0.5):
                samples.append(
                    {
                        "driver_id": driver,
                        "lap_number": lap_number,
                        "SessionTime": pd.Timedelta(seconds=t),
                        "Speed": 200 + 10 * driver_index,
                        "Throttle": 100,
                        "Brake": False,
                        "Source": "car",
                        "DRS": 12,
                    }
                )
    raw_laps = pd.DataFrame(laps)
    for pit in ("PitInTime", "PitOutTime"):
        raw_laps[pit] = pd.Series(pd.NaT, index=raw_laps.index, dtype="timedelta64[ns]")
    frames = {
        "telemetry": pd.DataFrame(samples),
        "laps": raw_laps,
        "session_laps": raw_laps,
        "weather": pd.DataFrame(
            {
                "Time": pd.to_timedelta([0, 60, 120, 180, 240, 300], unit="s"),
                "TrackTemp": 30,
                "AirTemp": 20,
                "Rainfall": False,
            }
        ),
    }
    frames["weather"]["observed_at"] = (
        utc(event["sessions"][0]["scheduled_start"]) + frames["weather"].Time
    )

    class SyntheticCapture:
        calls = 0

        def capture(self, event, spec):
            self.calls += 1
            return frames, {
                "source": "SYNTHETIC_channel_fixture",
                "completion_state": "completed",
                "completed_at": spec["ended_at"],
                "retrieved_at": "2026-07-01T00:00:00Z",
                "errors": [],
            }

    options, provider = ExperimentConfig(), SyntheticCapture()
    catalog = TelemetryCatalog(tmp_path / "captures")
    request = pd.DataFrame(
        [
            {
                "event_id": event["event_id"],
                "task": "qualifying",
                "kind": "Practice 1",
                "cutoff": cutoff.isoformat(),
                "mode": "historical_reconstruction",
            }
        ]
    )
    first = catalog.run(source, manifest, request, provider, options)
    assert first[0]["status"] == "complete"
    catalog.run(source, manifest, request, provider, options)
    assert provider.calls == 1
    core, _ = build_event(dataset, manifest, event["event_id"], "qualifying", cutoff, entrants)
    extra, audit = event_supplement(source, manifest, core, catalog, options)
    rows = extra.set_index("driver_id")
    assert rows.loc[drivers[0], "telemetry_speed_mean_kph"] == 200, audit
    assert rows.loc[drivers[1], "telemetry_speed_mean_kph"] == 210
    assert rows.loc[drivers[0], "telemetry_teammate_speed_delta_kph"] == -10
    assert rows.loc[drivers[1], "telemetry_teammate_speed_delta_kph"] == 10
    assert rows.loc[drivers[0], "practice_consistency_laps"] == 2
    assert rows.loc[drivers[0], "observed_air_temp_c"] == 20
    assert rows.loc[drivers[0], "telemetry_full_throttle_fraction"] == 1
    assert rows.loc[drivers[0], "telemetry_braking_fraction"] == 0
    assert extra.telemetry_laps.sum() == 4
    loaded, audit = catalog.read(
        source.receipt["id"],
        event["event_id"],
        "Practice 1",
        cutoff,
        "publication_snapshot",
        options,
    )
    assert not loaded and audit["state"] == "unavailable_at_cutoff"
    # A later completed capture cannot become eligible merely by reusing its cache key.
    early = utc(event["sessions"][0]["scheduled_start"]) + pd.Timedelta(minutes=1)
    report = catalog.run(
        source, manifest, request.assign(cutoff=early.isoformat()), provider, options
    )
    assert report[0]["status"] == "ineligible_at_cutoff" and provider.calls == 1

    # Exercise the public loader boundary with native Lap/Laps-shaped test doubles.
    from types import SimpleNamespace

    from f1ml.experiments.acquisition import TelemetryProvider

    class FakeLap(pd.Series):
        def get_car_data(self, **kwargs):
            assert kwargs == {"interpolate_edges": False}
            return frames["telemetry"].loc[
                frames["telemetry"].driver_id.eq(self.driver_id)
                & frames["telemetry"].lap_number.eq(self.LapNumber)
            ]

    class FakeLaps(pd.DataFrame):
        @property
        def _constructor_sliced(self):
            return FakeLap

    original = FakeLaps(raw_laps)
    original["LapNumber"] = original.lap_number
    original["DriverNumber"] = original.driver_id.map({d: str(i) for i, d in enumerate(drivers)})
    session = SimpleNamespace(
        laps=original,
        results=pd.DataFrame({"DriverNumber": ["0", "1"], "DriverId": drivers}),
        weather_data=frames["weather"],
        t0_date=utc(event["sessions"][0]["scheduled_start"]),
        session_status=pd.DataFrame(
            {
                "Status": ["Started", "Finalised", "Ends"],
                "Time": pd.to_timedelta([0, 3500, 3600], unit="s"),
            }
        ),
    )
    calls = []
    session.load = lambda **kwargs: calls.append(kwargs)
    boundary = object.__new__(TelemetryProvider)
    boundary.options = options
    boundary.fastf1 = SimpleNamespace(get_session=lambda *args: session)
    captured, info = boundary.capture(event, event["sessions"][0])
    assert calls == [{"laps": True, "telemetry": True, "weather": True, "messages": True}]
    assert len(captured["laps"]) == 4 and len(captured["telemetry"]) > 0
    assert info["completion_state"] == "completed"
