"""Metamorphic leakage tests and small exact examples, without network access."""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from f1ml.config import Config, FeatureConfig
from f1ml.domain import utc
from f1ml.evaluation.splits import assert_disjoint_events, freeze, gp_session
from f1ml.features.build import build_development, build_event
from f1ml.features.contract import model_matrix
from f1ml.features.dictionary import FEATURE_VERSION, columns
from f1ml.features.history import normalized_results
from f1ml.features.pace import comparable_pace
from f1ml.features.source import CatalogSource, available
from f1ml.models.baseline import fit_baseline, predict
from f1ml.storage import read_json, save_table, seal_directory


class ExampleData:
    def __init__(self):
        self.receipt = {"id": "explicit_synthetic_feature_fixture"}
        self.events, self.frames, self.reads = [], {}, []
        appearances = [(2024, 1), (2024, 2), (2025, 1), (2025, 2)] + [
            (2026, i) for i in range(1, 8)
        ]
        for index, (year, number) in enumerate(appearances):
            # Calendar dates and session names deliberately do not encode Friday/Saturday.
            start = pd.Timestamp(f"{year}-03-01T10:00:00Z") + pd.Timedelta(days=number * 14)
            identifier = f"{year}-{number:02d}"
            specs = [
                ("Practice 1", 0),
                ("Practice 2", 3),
                ("Practice 3", 7),
                ("Qualifying", 10),
                ("Race", 40),
            ]
            if number == 2:
                specs = [
                    ("Practice 1", 0),
                    ("Sprint Qualifying", 3),
                    ("Sprint", 7),
                    ("Qualifying", 10),
                    ("Race", 40),
                ]
            event = {
                "event_id": identifier,
                "year": year,
                "round": number,
                "name": "SYNTHETIC",
                "circuit_id": "verified_fixture_circuit",
                "sessions": [],
            }
            for kind, hours in specs:
                begin = start + pd.Timedelta(hours=hours)
                spec = {
                    "kind": kind,
                    "scheduled_start": begin.isoformat(),
                    "completed": True,
                    "ended_at": (begin + pd.Timedelta(hours=1)).isoformat(),
                    "retrieved_at": "2026-07-01T00:00:00Z",
                    "official_results": kind in ("Qualifying", "Race"),
                }
                if kind.startswith("Sprint"):
                    event["sessions"].append(spec)
                    continue
                spec["table"] = f"{identifier}/{kind}.parquet"
                if kind in ("Qualifying", "Race"):
                    # Driver a's normalized ranks cycle 0, 1/3, 2/3, 1.
                    order = list("abcd")[index % 4 :] + list("abcd")[: index % 4]
                    positions = {driver: order.index(driver) + 1 for driver in "abcd"}
                    self.frames[spec["table"]] = pd.DataFrame(
                        {
                            "driver_id": list("abcd"),
                            "team_id": ["org_red", "org_red", "org_red", "org_blue"],
                            "team_history_id": ["reviewed_red"] * 3 + ["reviewed_blue"],
                            "official_position": [positions[d] for d in "abcd"],
                            "outcome": "ordered",
                            "podium": [float(positions[d] <= 3) for d in "abcd"],
                            "is_retired": [index % 2 == 0, False, False, False],
                        }
                    )
                else:
                    self.frames[spec["table"]] = pd.DataFrame(
                        {
                            "driver_id": list("abcd"),
                            "best_lap_seconds": [90.0, 91.0, 92.0, 93.0],
                            "clean_laps": 2,
                        }
                    )
                if kind != "Race":
                    spec["laps_table"] = f"{identifier}/{kind}_laps.parquet"
                    spec["weather_table"] = f"{identifier}/{kind}_weather.parquet"
                    lap_rows = []
                    segments = ["Q1", "Q2", "Q3"] if kind == "Qualifying" else ["PRACTICE"]
                    for segment in segments:
                        for driver_index, driver in enumerate("abcd"):
                            lap_rows.append(
                                {
                                    "driver_id": driver,
                                    "lap_time_seconds": 90.0 + driver_index,
                                    "end_time_seconds": 690.0 + driver_index,
                                    "lap_start_seconds": 600.0,
                                    "is_accurate": True,
                                    "deleted": False,
                                    "fastf1_generated": False,
                                    "source_PitInTime": None,
                                    "source_PitOutTime": None,
                                    "source_TrackStatus": "1",
                                    "source_Compound": "SOFT",
                                    "qualifying_segment": segment,
                                }
                            )
                    self.frames[spec["laps_table"]] = pd.DataFrame(lap_rows)
                    self.frames[spec["weather_table"]] = pd.DataFrame(
                        {
                            "session_time_seconds": [580.0],
                            "rainfall": [False],
                            "track_temp_c": [35.0],
                        }
                    )
                event["sessions"].append(spec)
            self.events.append(event)

    def event(self, event_id):
        return next(e for e in self.events if e["event_id"] == event_id)

    def table(self, spec):
        self.reads.append(spec["table"])
        return self.frames[spec["table"]].copy()

    def roster(self, event_id, task, *, driver="a"):
        event = self.event(event_id)
        q = gp_session(event, "Qualifying")
        cutoff = (
            utc(q["scheduled_start"]) - pd.Timedelta(seconds=1)
            if task == "qualifying"
            else utc(q["ended_at"]) + pd.Timedelta(hours=1)
        )
        roster = pd.DataFrame(
            {
                "event_id": event_id,
                "task": task,
                "driver_id": list("abcd"),
                "team_id": ["org_red"] * 3 + ["org_blue"],
                "team_history_id": ["reviewed_red"] * 3 + ["reviewed_blue"],
                "known_at": (cutoff - pd.Timedelta(days=1)).isoformat(),
                "source": "explicit_fixture_entry_list",
            }
        )
        return cutoff, roster


@pytest.fixture
def example(tmp_path):
    data = ExampleData()
    manifest = freeze(data, Config(), tmp_path / "manifest.json")
    return data, manifest


def features(example, event="2026-03", task="qualifying"):
    data, manifest = example
    cutoff, roster = data.roster(event, task)
    frame, audit = build_event(data, manifest, event, task, cutoff, roster)
    return frame.set_index("driver_id"), audit


@pytest.mark.parametrize("task", ["qualifying", "race"])
def test_future_results_cannot_change_earlier_features(example, task):
    before, before_audit = features(example, "2026-01", task)
    data, _ = example
    for name, frame in data.frames.items():
        if (
            name.startswith("2026-02/")
            or name.startswith("2026-03/")
            or name.startswith("2026-04/")
        ):
            if "official_position" in frame:
                frame["official_position"] = [4, 3, 2, 1]
                frame["podium"] = 0.0
                frame["is_retired"] = True
    after, after_audit = features(example, "2026-01", task)
    assert_frame_equal(before, after)
    assert before_audit == after_audit


def test_target_qualifying_values_never_enter_qualifying_inputs(example):
    before, _ = features(example)
    data, _ = example
    spec = gp_session(data.event("2026-03"), "Qualifying")
    data.frames[spec["table"]]["official_position"] = [4, 3, 2, 1]
    data.frames[spec["laps_table"]]["lap_time_seconds"] = 1.0
    data.reads.clear()
    after, _ = features(example)
    assert_frame_equal(before, after)
    assert all("2026-03/Qualifying" not in path for path in data.reads)
    assert "qualifying_rank_norm" not in before


def test_target_race_results_never_enter_race_inputs(example):
    before, audit = features(example, task="race")
    data, _ = example
    spec = gp_session(data.event("2026-03"), "Race")
    data.frames[spec["table"]] = data.frames[spec["table"]].assign(
        official_position=99, podium=0, is_retired=True
    )
    data.reads.clear()
    after, _ = features(example, task="race")
    assert_frame_equal(before, after)
    assert spec["table"] not in data.reads
    assert all(s["kind"] not in ("Sprint", "Sprint Qualifying") for s in audit["sources"])


def test_rolling_3_and_5_are_shifted_calendar_weekend_windows(example):
    frame, audit = features(example)
    # Before target index 6, a's ranks at indices 1..5 are 1, 2/3, 1/3, 0, 1.
    a = frame.loc["a"]
    assert a.previous_qualifying_rank == 1.0
    assert a.qualifying_rank_mean_3 == pytest.approx((1 / 3 + 0 + 1) / 3)
    assert a.qualifying_rank_mean_5 == pytest.approx((1 + 2 / 3 + 1 / 3 + 0 + 1) / 5)
    assert a.qualifying_rank_count_3 == 3 and a.qualifying_rank_count_5 == 5
    assert a.retirement_rate_3 == pytest.approx(1 / 3)
    assert "2026-03" not in audit["history_weekends"]


def test_missing_previous_appearance_does_not_expand_the_calendar_window(example):
    data, _ = example
    for kind in ("Qualifying", "Race"):
        spec = gp_session(data.event("2026-02"), kind)
        raw = data.frames[spec["table"]]
        data.frames[spec["table"]] = raw.loc[raw.driver_id.ne("a")].copy()
    frame, _ = features(example)
    assert pd.isna(frame.loc["a", "previous_race_rank"])
    assert frame.loc["a", "race_rank_count_3"] == 2


def test_team_2026_form_prior_and_teammate_comparisons_are_separate(example):
    frame, _ = features(example)
    a = frame.loc["a"]
    assert a.team_qualifying_strength_count_5 == 2  # Only the two prior 2026 weekends.
    assert a.team_prior_qualifying_count == 4
    assert a.teammate_qualifying_delta_count_5 == 2
    assert a.teammate_qualifying_delta_mean_5 == pytest.approx(1 / 6)
    assert a.circuit_prior_qualifying_count == 6
    data, manifest = example
    cutoff, roster = data.roster("2026-03", "qualifying")
    roster["team_history_id"] = "brand_new_2026_era"
    new, _ = build_event(data, manifest, "2026-03", "qualifying", cutoff, roster)
    assert new.team_qualifying_strength_count_5.eq(0).all()
    assert new.team_prior_qualifying_count.eq(0).all()


def test_actual_qualifying_and_pace_are_race_only_and_parts_are_not_mixed(example):
    frame, _ = features(example, task="race")
    assert frame.loc["b", "qualifying_q1_gap_pct"] == pytest.approx(100 * (91 / 90 - 1))
    data, _ = example
    spec = gp_session(data.event("2026-03"), "Qualifying")
    raw = data.frames[spec["laps_table"]]
    raw.loc[raw.qualifying_segment.eq("Q3"), "lap_time_seconds"] *= 0.5
    changed, _ = features(example, task="race")
    assert frame.qualifying_q1_gap_pct.equals(changed.qualifying_q1_gap_pct)
    assert changed.loc["b", "qualifying_q3_gap_pct"] == pytest.approx(
        frame.loc["b", "qualifying_q3_gap_pct"]
    )
    assert "qualifying_q1_gap_pct" not in features(example)[0]


def test_practice_filters_disruptions_pits_deleted_generated_and_unknowns(example):
    baseline, _ = features(example)
    data, _ = example
    for spec in data.event("2026-03")["sessions"]:
        if not spec["kind"].startswith("Practice"):
            continue
        raw = data.frames[spec["laps_table"]]
        bad = pd.concat([raw.iloc[[0]]] * 6, ignore_index=True)
        bad["lap_time_seconds"] = 1.0
        bad.loc[0, "source_PitInTime"] = "0 days 00:10:00"
        bad.loc[1, "source_PitOutTime"] = "0 days 00:10:00"
        bad.loc[2, "deleted"] = True
        bad.loc[3, "is_accurate"] = False
        bad.loc[4, "fastf1_generated"] = True
        bad.loc[5, "source_TrackStatus"] = "145"
        data.frames[spec["laps_table"]] = pd.concat([raw, bad], ignore_index=True)
    changed, _ = features(example)
    assert baseline.practice_gap_pct.equals(changed.practice_gap_pct)
    assert baseline.practice_clean_laps.equals(changed.practice_clean_laps)


def test_unknown_weather_or_segment_leaves_pace_missing_without_dropping_core(example):
    data, manifest = example
    before, _ = features(example, task="race")
    for spec in data.event("2026-03")["sessions"]:
        if "weather_table" in spec:
            data.frames[spec["weather_table"]] = pd.DataFrame()
    after, _ = features(example, task="race")
    assert len(before) == len(after) == 4
    assert after.practice_gap_pct.isna().all() and after.practice_gap_pct_missing.eq(1).all()
    assert after.qualifying_q1_gap_pct.isna().all()
    assert_frame_equal(
        before[["previous_race_rank", "qualifying_rank_norm"]],
        after[["previous_race_rank", "qualifying_rank_norm"]],
    )


def test_weather_cannot_be_matched_forward_into_a_lap(example):
    data, _ = example
    spec = gp_session(data.event("2026-03"), "Practice 1")
    laps = data.frames[spec["laps_table"]]
    weather = data.frames[spec["weather_table"]].copy()
    weather["session_time_seconds"] = 601.0  # All laps started at 600.
    pace = comparable_pace(laps, weather, "Practice 1", FeatureConfig())
    assert pace.gap_pct.isna().all()


def test_unseen_entrant_and_missing_optional_practice_keep_the_field(example):
    data, manifest = example
    cutoff, roster = data.roster("2026-03", "qualifying")
    roster.loc[0, "driver_id"] = "rookie_no_history"
    for spec in data.event("2026-03")["sessions"]:
        if spec["kind"].startswith("Practice"):
            spec["completed"] = False
    frame, _ = build_event(data, manifest, "2026-03", "qualifying", cutoff, roster)
    rookie = frame.loc[frame.driver_id.eq("rookie_no_history")].iloc[0]
    assert len(frame) == 4 and rookie.history_count == 0
    assert pd.isna(rookie.race_rank_mean_5) and rookie.race_rank_mean_5_missing == 1
    assert frame.practice_sessions.eq(0).all()


@pytest.mark.parametrize("roles", [(["a"], ["a"], []), (["a"], [], ["a"]), ([], ["a"], ["a"])])
def test_event_roles_must_be_disjoint(roles):
    with pytest.raises(ValueError, match="must not overlap"):
        assert_disjoint_events(*roles)


def build_all(data, manifest, output):
    cutoffs, rosters = [], []
    for event in manifest["development"]:
        for task in ("qualifying", "race"):
            cutoff, roster = data.roster(event, task)
            cutoffs.append(
                {
                    "event_id": event,
                    "task": task,
                    "cutoff": cutoff.isoformat(),
                    "mode": "historical_reconstruction",
                }
            )
            rosters.append(roster)
    return build_development(
        data, manifest, pd.concat(rosters, ignore_index=True), pd.DataFrame(cutoffs), output
    )


def test_artifacts_are_reproducible_and_dictionary_matches_both_matrices(example, tmp_path):
    data, manifest = example
    a = build_all(data, manifest, tmp_path / "a")
    b = build_all(data, manifest, tmp_path / "b")
    assert a["id"] == b["id"]
    dictionary = read_json(tmp_path / "a/feature_dictionary.json")["features"]
    for task in ("qualifying", "race"):
        frame = pd.read_parquet(tmp_path / f"a/{task}_features.parquet")
        assert not frame.duplicated(["event_id", "driver_id", "task"]).any()
        assert set(model_matrix(frame, task)) == {
            name for name, spec in dictionary.items() if task in spec["tasks"]
        }
        assert frame.feature_version.eq(FEATURE_VERSION).all()
        assert len(frame) == 4 * len(manifest["development"])


def test_fold_preprocessing_ignores_validation_values_and_accepts_unseen_categories(
    example, tmp_path
):
    data, manifest = example
    build_all(data, manifest, tmp_path / "features")
    frame = pd.read_parquet(tmp_path / "features/race_features.parquet")
    targets = pd.read_parquet(tmp_path / "features/race_labels.parquet")
    fold = manifest["folds"][0]
    train = frame.loc[frame.event_id.isin(fold["train"])]
    labels = targets.loc[targets.event_id.isin(fold["train"])]
    validation = frame.loc[frame.event_id.eq(fold["validate"])].copy()
    validation["field_size"] = 1e9
    validation["team_history_id"] = "never_seen_team"
    artifact = fit_baseline(train, labels, "race", manifest, 42)
    imputer = (
        artifact.pipeline.named_steps["transform"]
        .named_transformers_["numeric"]
        .named_steps["impute"]
    )
    assert imputer.statistics_[list(columns("race")).index("field_size")] == 4
    assert len(predict(artifact, validation)) == len(validation)


def test_catalog_adapter_binds_reviews_and_preserves_core_when_timing_is_unknown(tmp_path):
    from f1ml.ingestion.types import session_id

    data = ExampleData()
    event = data.events[0]
    root = tmp_path / "snapshot"
    root.mkdir()
    save_table(
        root / "tables/events.parquet",
        pd.DataFrame([{k: v for k, v in event.items() if k not in ("sessions", "circuit_id")}]),
    )
    sessions, results, laps, weather, reviews = [], [], [], [], []
    for spec in event["sessions"]:
        sid = session_id(event["event_id"], spec["kind"])
        official = spec["kind"] in ("Qualifying", "Race")
        sessions.append(
            {
                "event_id": event["event_id"],
                "session_id": sid,
                "kind": spec["kind"],
                "scheduled_start": spec["scheduled_start"],
                "completion_state": "unknown",
                "completed_at": None,
                "results_rows": 4 if official else 0,
                "laps_rows": 4 if spec["kind"] != "Race" else 0,
                "official_results_complete": official,
                "results_retrieved_at": spec["retrieved_at"],
                "timing_retrieved_at": spec["retrieved_at"],
            }
        )
        reviews.append(
            {
                "session_id": sid,
                "completed_at": spec["ended_at"],
                "available_at": spec["ended_at"],
                "evidence": "explicit_synthetic_review",
            }
        )
        if official:
            results.append(data.frames[spec["table"]].assign(session_id=sid))
        if "laps_table" in spec:
            laps.append(data.frames[spec["laps_table"]].assign(session_id=sid, kind=spec["kind"]))
            weather.append(data.frames[spec["weather_table"]].assign(session_id=sid))
    save_table(root / "tables/sessions.parquet", pd.DataFrame(sessions))
    save_table(root / "tables/results.parquet", pd.concat(results, ignore_index=True))
    lap_frame = pd.concat(laps, ignore_index=True)
    save_table(
        root / "tables/practice_laps.parquet",
        lap_frame.loc[lap_frame.kind.str.startswith("Practice")],
    )
    save_table(
        root / "tables/qualifying_laps.parquet", lap_frame.loc[lap_frame.kind.eq("Qualifying")]
    )
    save_table(root / "tables/session_weather.parquet", pd.concat(weather, ignore_index=True))
    seal_directory(root, {"schema_version": 2})
    source = CatalogSource(root)
    plan = {
        "dataset_id": source.receipt["id"],
        "config": Config().to_dict(),
        "development": [event["event_id"]],
        "final_test": [],
    }
    cutoff, roster = data.roster(event["event_id"], "qualifying")
    core, _ = build_event(source, plan, event["event_id"], "qualifying", cutoff, roster)
    assert len(core) == 4 and core.practice_gap_pct.isna().all()
    reviewed = CatalogSource(
        root,
        pd.DataFrame(reviews),
        pd.DataFrame(
            [
                {
                    "event_id": event["event_id"],
                    "circuit_id": "reviewed_layout",
                    "evidence": "fixture_circuit",
                }
            ]
        ),
    )
    assert reviewed.receipt["id"] != source.receipt["id"]
    with pytest.raises(ValueError, match="differs from frozen dataset"):
        build_event(reviewed, plan, event["event_id"], "qualifying", cutoff, roster)
    plan["dataset_id"] = reviewed.receipt["id"]
    enriched, _ = build_event(reviewed, plan, event["event_id"], "qualifying", cutoff, roster)
    assert len(enriched) == len(core) and enriched.practice_gap_pct.notna().all()
    assert enriched.circuit_id.eq("reviewed_layout").all()


def test_missing_target_label_keeps_core_rows_and_label_mask(example, tmp_path):
    data, manifest = example
    for kind in ("Qualifying", "Race"):
        spec = gp_session(data.event("2026-03"), kind)
        data.frames[spec["table"]] = (
            data.frames[spec["table"]].loc[lambda f: f.driver_id.ne("a")].copy()
        )
    build_all(data, manifest, tmp_path / "features")
    frame = pd.read_parquet(tmp_path / "features/qualifying_features.parquet")
    labels = pd.read_parquet(tmp_path / "features/qualifying_labels.parquet")
    assert len(frame.loc[frame.event_id.eq("2026-03")]) == 4
    assert pd.isna(
        labels.loc[labels.event_id.eq("2026-03") & labels.driver_id.eq("a"), "rank_target"].iloc[0]
    )


def test_session_time_order_not_practice_number_controls_availability(example):
    data, _ = example
    event = data.event("2026-03")
    spec = gp_session(event, "Practice 1")
    q = gp_session(event, "Qualifying")
    spec["scheduled_start"] = (utc(q["scheduled_start"]) + pd.Timedelta(hours=4)).isoformat()
    spec["ended_at"] = (utc(q["scheduled_start"]) + pd.Timedelta(hours=5)).isoformat()
    frame, audit = features(example)
    assert frame.practice_sessions.eq(2).all()
    assert not any(
        s["event_id"] == "2026-03" and s["kind"] == "Practice 1" for s in audit["sources"]
    )


def test_early_prediction_excludes_weekends_whose_race_has_not_finished_at_cutoff(example):
    data, manifest = example
    _, roster = data.roster("2026-03", "qualifying")
    ongoing = gp_session(data.event("2026-02"), "Qualifying")
    cutoff = utc(ongoing["ended_at"]) + pd.Timedelta(hours=1)
    roster["known_at"] = (cutoff - pd.Timedelta(days=1)).isoformat()
    frame, audit = build_event(data, manifest, "2026-03", "qualifying", cutoff, roster)
    assert "2026-02" not in audit["history_weekends"]
    assert frame.loc[frame.driver_id.eq("a"), "previous_qualifying_rank"].iloc[0] == 0


def test_missing_timing_revision_time_cannot_borrow_a_results_publication_timestamp():
    spec = {
        "completed": True,
        "ended_at": "2026-05-01T10:00:00Z",
        "retrieved_at": "2026-05-01T10:05:00Z",
        "timing_retrieved_at": None,
    }
    cutoff = utc("2026-05-01T11:00:00Z")
    assert available(spec, cutoff, "publication_snapshot")
    assert not available(spec, cutoff, "publication_snapshot", timing=True)


def test_rank_normalization_uses_each_actual_field_and_masks_incomplete_fields():
    small = pd.DataFrame({"driver_id": list("abc"), "official_position": [2, 1, 3]})
    large = pd.DataFrame({"driver_id": list("abcde"), "official_position": [2, 1, 3, 4, 5]})
    assert normalized_results(small).iloc[0].rank_norm == 0.5
    assert normalized_results(large).iloc[0].rank_norm == 0.25
    large["official_field_complete"] = False
    assert normalized_results(large).rank_norm.isna().all()
