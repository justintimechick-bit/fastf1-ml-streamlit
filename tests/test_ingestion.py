"""Offline contract checks. Synthetic fixtures never substitute for failed live downloads."""

from contextlib import contextmanager
from dataclasses import replace

import pandas as pd
import pytest
from requests import ConnectionError

from f1ml.config import Config, IngestionConfig
from f1ml.demo import SyntheticProvider
from f1ml.ingestion.catalog import Catalog, current_snapshot
from f1ml.ingestion.coverage import inspect_coverage
from f1ml.ingestion.identity import TeamMapping
from f1ml.ingestion.normalize import normalize_results
from f1ml.ingestion.tables import assert_keys, normalize_batch
from f1ml.ingestion.types import SourceBatch
from f1ml.storage import read_json, verify_directory


class FixtureProvider(SyntheticProvider):
    def __init__(self, fail_timing=False, transient_results=0, duplicate_laps=False):
        self.calls = []
        self.fail_timing = fail_timing
        self.transient_results = transient_results
        self.duplicate_laps = duplicate_laps

    def schedule(self, year):
        schedule = super().schedule(year)
        if year == 2025:
            return [schedule[1]]  # Sprint format.
        if year == 2026:
            return [schedule[0], schedule[-1]]  # Include a future event.
        return [schedule[0]]

    @contextmanager
    def event_cache(self, event_id, generation):
        self.generation = generation
        yield

    def fetch_results(self, event, spec):
        self.calls.append((event["event_id"], spec["kind"], "results", self.generation))
        if self.transient_results:
            self.transient_results -= 1
            raise ConnectionError("synthetic network outage")
        return SourceBatch(
            {
                "results": pd.DataFrame(
                    {
                        "driverId": ["driver_a", "driver_b", "driver_c"],
                        "number": ["1", "2", "3"],
                        "givenName": ["A", "B", "C"],
                        "familyName": ["One", "Two", "Three"],
                        "constructorId": ["rb"] * 3,
                        "constructorName": [
                            "RB F1 Team" if event["year"] == 2024 else "Racing Bulls"
                        ]
                        * 3,
                        "position": ["1", "2", "3"],
                        "positionText": ["1", "2", "3"],
                        "status": ["Finished", "Finished", "Engine"],
                    }
                )
            },
            self.metadata("results"),
        )

    @staticmethod
    def metadata(component):
        return {
            "retrieved_at": "2026-07-01T00:00:00Z",
            "fastf1_version": "fixture",
            "source": "SYNTHETIC_TEST_FIXTURE",
            "errors": [],
            "warnings": [],
            "response_complete": True,
            "absolute_time_available": False,
            "completion_state": "results_available" if component == "results" else "completed",
        }

    def fetch_timing(self, event, spec):
        self.calls.append((event["event_id"], spec["kind"], "timing", self.generation))
        meta = self.metadata("timing")
        if self.fail_timing:
            meta.update(
                completion_state="unknown",
                errors=[{"type": "DataNotLoadedError", "message": "synthetic timing denied"}],
            )
            return SourceBatch(metadata=meta)
        entries = pd.DataFrame(
            {
                "DriverId": ["driver_a", "driver_b", None],
                "DriverNumber": ["1", "2", "3"],
                "TeamId": ["rb"] * 3,
                "TeamName": ["RB F1 Team" if event["year"] == 2024 else "Racing Bulls"] * 3,
            }
        )
        laps = pd.DataFrame(
            {
                "DriverNumber": ["1", "2", "3"],
                "LapNumber": [1.0, 1.0, 1.0],
                "LapTime": pd.to_timedelta([90, 91, None], unit="s"),
                "Time": pd.to_timedelta([180, 181, 182], unit="s"),
                "IsAccurate": [True, True, None],
                "Deleted": [False, False, None],
            }
        )
        if self.duplicate_laps:
            laps = pd.concat([laps, laps.iloc[[0]]], ignore_index=True)
        frames = {
            "entries": entries,
            "weather": pd.DataFrame(
                {"Time": pd.to_timedelta([60], unit="s"), "AirTemp": [22.0], "Rainfall": [False]}
            ),
        }
        if spec["kind"] != "Race":
            frames.update(
                laps=laps,
                status=pd.DataFrame(
                    {
                        "Time": pd.to_timedelta([0, 400], unit="s"),
                        "Status": ["Started", "Finalised"],
                    }
                ),
            )
        return SourceBatch(frames, meta)


@pytest.fixture
def config():
    return Config(ingestion=IngestionConfig(retry_base_seconds=0, retry_max_seconds=0))


def test_resume_reuses_completed_components_and_keeps_all_scheduled_sessions(tmp_path, config):
    provider = FixtureProvider()
    catalog = Catalog(tmp_path / "catalog", config, provider)
    first = catalog.run(create=True, as_of="2026-07-01T00:00:00Z")
    calls = len(provider.calls)
    second = catalog.run()
    assert len(provider.calls) == calls
    assert second["run"]["attempted_jobs"] == 0
    assert second["run"]["reused_jobs"] == calls
    verify_directory(tmp_path / "catalog" / "snapshots" / first["snapshot"].split("/")[-1])
    sessions = inspect_coverage(tmp_path / "catalog", "sessions")
    assert len(sessions) == sum(len(e["sessions"]) for e in catalog.state["events"])
    assert {"Sprint", "Sprint Qualifying", "Qualifying", "Race"}.issubset(set(sessions.kind))
    assert all(kind not in ("Sprint", "Sprint Qualifying") for _, kind, _, _ in provider.calls)
    assert sessions.loc[sessions.event_id.eq("2026-08"), "completion_state"].eq("not_started").all()
    events = inspect_coverage(tmp_path / "catalog", "events").set_index("event_id")
    assert events.loc["2025-02", "practice_expected"] == 1
    assert events.loc["2024-01", "practice_expected"] == 3
    snapshot, _ = current_snapshot(tmp_path / "catalog")
    drivers = pd.read_parquet(snapshot / "tables/drivers.parquet")
    assert "team_id" not in drivers and set(drivers.driver_id) == {
        "driver_a",
        "driver_b",
        "driver_c",
    }
    laps = pd.read_parquet(snapshot / "tables/practice_laps.parquet")
    assert laps.loc[laps.driver_number.eq("3"), "driver_id"].isna().all()
    assert laps.loc[laps.driver_number.eq("3"), "lap_time_missing"].all()


def test_failed_timing_does_not_discard_official_results_and_opens_bounded_circuit(
    tmp_path, config
):
    provider = FixtureProvider(fail_timing=True)
    report = Catalog(tmp_path / "catalog", config, provider).run(
        create=True, as_of="2026-07-01T00:00:00Z"
    )
    assert report["timing_circuit"]["blocked_until"] is not None
    assert report["partial_jobs"] > 0 and report["failure_rows"] > 0
    events = inspect_coverage(tmp_path / "catalog", "events")
    assert events.result_sessions.sum() == 6
    assert events.practice_with_laps.sum() == 0
    assert events.blocked_sessions.sum() > 0
    catalog = Catalog(tmp_path / "catalog", config, provider)
    count = len(provider.calls)
    resumed = catalog.run()
    assert resumed["run"]["attempted_jobs"] == 0 and len(provider.calls) == count


def test_network_retries_are_bounded_and_checkpointed(tmp_path, config):
    provider = FixtureProvider(transient_results=2)
    report = Catalog(tmp_path / "catalog", config, provider).run(
        create=True, selected=["2024-01"], as_of="2026-07-01T00:00:00Z"
    )
    state = read_json(tmp_path / "catalog/state.json")
    assert state["jobs"]["2024-01/Q|results"]["attempts_total"] == 3
    assert report["failure_rows"] == 2  # Recovered attempts remain visible.
    broken = FixtureProvider(transient_results=100)
    Catalog(tmp_path / "broken", config, broken).run(
        create=True, selected=["2024-01"], as_of="2026-07-01T00:00:00Z"
    )
    failed = read_json(tmp_path / "broken/state.json")
    assert failed["jobs"]["2024-01/Q|results"]["attempts_total"] == config.ingestion.max_attempts
    assert failed["jobs"]["2024-01/Q|results"]["status"] == "failed"


def test_refresh_is_scoped_and_previous_source_revisions_remain_verified(tmp_path, config):
    provider = FixtureProvider()
    catalog = Catalog(tmp_path / "catalog", config, provider)
    catalog.run(create=True, as_of="2026-07-01T00:00:00Z")
    before = len(provider.calls)
    old = catalog.state["jobs"]["2024-01/Q|results"]["latest"]
    catalog.run(refresh="2024-01")
    new_calls = provider.calls[before:]
    assert {event for event, _, _, _ in new_calls} == {"2024-01"}
    assert {generation for _, _, _, generation in new_calls} == {1}
    job = catalog.state["jobs"]["2024-01/Q|results"]
    assert job["latest"] != old and old in job["history"]
    verify_directory(tmp_path / "catalog" / old)


def test_parse_failure_retains_raw_laps_and_other_available_components(tmp_path, config):
    provider = FixtureProvider(duplicate_laps=True)
    report = Catalog(tmp_path / "catalog", config, provider).run(
        create=True, selected=["2024-01"], as_of="2026-07-01T00:00:00Z"
    )
    assert report["partial_jobs"] > 0
    failures = inspect_coverage(tmp_path / "catalog", "failures")
    assert failures.message.str.contains("Duplicate practice_laps keys").any()
    state = read_json(tmp_path / "catalog/state.json")
    record = tmp_path / "catalog" / state["jobs"]["2024-01/FP1|timing"]["latest"]
    assert len(pd.read_parquet(record / "source/laps.parquet")) == 4
    assert (record / "normalized/session_weather.parquet").exists()
    assert not (record / "normalized/practice_laps.parquet").exists()


def test_classified_retirement_is_distinct_from_unclassified_retirement():
    raw = pd.DataFrame(
        {
            "driverId": list("abcde"),
            "position": [3, 4, 5, None, None],
            "positionText": ["3", "R", "D", "W", ""],
            "status": ["Engine", "Collision", "Disqualified", "Withdrew", ""],
        }
    )
    result = normalize_results(raw, "Race").set_index("driver_id")
    assert result.loc["a", "is_classified"] and result.loc["a", "is_retired"]
    assert result.loc["a", "outcome"] == "retired_classified" and result.loc["a", "podium"] == 1
    assert not result.loc["b", "is_classified"] and result.loc["b", "is_retired"]
    assert result.loc["b", "official_position"] == 4 and result.loc["b", "podium"] == 0
    assert result.loc["c", "outcome"] == "dsq" and result.loc["d", "outcome"] == "withdrawn"
    assert result.loc["a", "source_status"] == "Engine"
    assert result.loc["b", "source_positionText"] == "R"
    assert pd.isna(result.loc["e", "official_position"]) and pd.isna(result.loc["e", "podium"])


def test_explicit_team_mapping_separates_name_aliases_and_history():
    from pathlib import Path

    mapping = TeamMapping(Path("configs/team_mapping.json"))
    old = mapping.resolve(2024, "rb", "RB F1 Team")
    new = mapping.resolve(2025, "rb", "Racing Bulls")
    assert old["team_id"] == new["team_id"]
    assert old["team_history_id"] != new["team_history_id"]
    assert mapping.resolve(2026, "sauber", "Audi")["team_mapping_state"] == "unmapped"
    assert (
        mapping.resolve(2025, "unknown", "X")["team_id"]
        != mapping.resolve(2026, "unknown", "X")["team_id"]
    )


def test_session_separation_is_validated_without_driver_or_lap_count_assumptions():
    data = pd.DataFrame(
        {
            "session_id": ["2024-01/Q", "2024-01/SQ"],
            "event_id": ["2024-01"] * 2,
            "kind": ["Qualifying", "Sprint Qualifying"],
            "driver_id": ["a"] * 2,
        }
    )
    with pytest.raises(ValueError, match="Wrong session kind"):
        assert_keys(data, "results")
    data.loc[1, ["session_id", "kind"]] = ["2024-01/R", "Race"]
    assert_keys(data, "results")


def test_unfinalised_practice_laps_are_withheld_from_normalized_tables(config):
    from pathlib import Path

    provider = FixtureProvider()
    event = provider.schedule(2024)[0]
    spec = event["sessions"][0]
    provider.generation = 0
    batch = provider.fetch_timing(event, spec)
    batch.metadata["completion_state"] = "in_progress"
    tables, errors = normalize_batch(
        batch, event, spec, TeamMapping(Path(config.ingestion.team_mapping_path)), "timing"
    )
    assert not errors and "practice_laps" not in tables and "session_weather" in tables


def test_configuration_cannot_turn_retries_into_unbounded_downloads(config):
    with pytest.raises(ValueError):
        replace(config, ingestion=IngestionConfig(max_attempts=6))


def test_lapped_is_a_finish_and_repeated_source_positions_are_preserved_but_flagged():
    raw = pd.DataFrame(
        {
            "driverId": ["a", "b"],
            "position": [12, 12],
            "positionText": ["12", "12"],
            "status": ["Lapped", "+1 Lap"],
        }
    )
    result = normalize_results(raw, "Race", allow_ambiguous_positions=True)
    assert result.official_position.tolist() == [12, 12]
    assert result.position_ambiguous.all()
    assert result.is_classified.all() and not result.is_retired.any()
    assert result.source_status.tolist() == ["Lapped", "+1 Lap"]


def test_offline_reparse_applies_mapping_changes_without_network(tmp_path, config):
    import json
    from pathlib import Path

    provider = FixtureProvider()
    catalog = Catalog(tmp_path / "catalog", config, provider)
    original = catalog.run(create=True, selected=["2024-01"], as_of="2026-07-01T00:00:00Z")
    calls = len(provider.calls)
    mapping = json.loads(Path(config.ingestion.team_mapping_path).read_text())
    mapping["rules"][8]["history_id"] = "explicitly_reviewed_new_era"
    revised = tmp_path / "mapping.json"
    revised.write_text(json.dumps(mapping))
    new_config = replace(
        config, ingestion=replace(config.ingestion, team_mapping_path=str(revised))
    )
    with pytest.raises(ValueError, match="run reparse"):
        Catalog(tmp_path / "catalog", new_config, provider).run()
    report = Catalog(tmp_path / "catalog", new_config, None).reparse()
    assert len(provider.calls) == calls and report["run"]["attempted_jobs"] == 0
    snapshot, _ = current_snapshot(tmp_path / "catalog")
    entries = pd.read_parquet(snapshot / "tables/driver_entries.parquet")
    assert entries.team_history_id.eq("explicitly_reviewed_new_era").all()
    verify_directory(Path(original["snapshot"]))


def test_upstream_throttle_stops_requests_and_persists_cooldown(tmp_path, config):
    from fastf1.exceptions import ErgastInvalidRequestError

    class Throttled(FixtureProvider):
        def fetch_results(self, event, spec):
            self.calls.append((event["event_id"], spec["kind"], "results", self.generation))
            raise ErgastInvalidRequestError("Server response: 'Too Many Requests'")

    provider = Throttled()
    catalog = Catalog(tmp_path / "catalog", config, provider)
    report = catalog.run(create=True, as_of="2026-07-01T00:00:00Z")
    assert len(provider.calls) == 1
    assert report["results_circuit"]["blocked_until"] is not None
    assert report["run"]["deferred_jobs"] > 0
    catalog.run(retry_failed=True)
    assert len(provider.calls) == 1  # Explicit retries cannot override source rate-limit cooldown.


def test_interrupted_catalog_resumes_completed_jobs(tmp_path, config):
    class Interrupted(FixtureProvider):
        def fetch_timing(self, event, spec):
            raise KeyboardInterrupt

    catalog = Catalog(tmp_path / "catalog", config, Interrupted())
    with pytest.raises(KeyboardInterrupt):
        catalog.run(create=True, selected=["2024-01"], as_of="2026-07-01T00:00:00Z")
    assert len(read_json(tmp_path / "catalog/state.json")["jobs"]) == 2
    provider = FixtureProvider()
    resumed = Catalog(tmp_path / "catalog", config, provider).run(selected=["2024-01"])
    assert all(component == "timing" for _, _, component, _ in provider.calls)
    assert resumed["run"]["reused_jobs"] == 2


def test_coverage_counts_observed_entries_missing_from_the_official_result_field(tmp_path, config):
    class MissingResult(FixtureProvider):
        def fetch_results(self, event, spec):
            batch = super().fetch_results(event, spec)
            if spec["kind"] == "Qualifying":
                batch.frames["results"] = batch.frames["results"].iloc[:2].copy()
            return batch

    Catalog(tmp_path / "catalog", config, MissingResult()).run(
        create=True, selected=["2024-01"], as_of="2026-07-01T00:00:00Z"
    )
    sessions = inspect_coverage(tmp_path / "catalog", "sessions").set_index("session_id")
    assert sessions.loc["2024-01/Q", "result_driver_count"] == 2
    assert sessions.loc["2024-01/Q", "driver_count"] == 3
    assert sessions.loc["2024-01/Q", "entries_without_results"] == 1
    assert sessions.loc["2024-01/Q", "missing_rank_targets"] == 1
