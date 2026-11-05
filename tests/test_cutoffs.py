import pandas as pd
import pytest

from f1ml.domain import utc
from f1ml.evaluation.splits import gp_session
from f1ml.features.build import build_event, labels_for
from f1ml.features.contract import model_matrix


@pytest.mark.parametrize(
    "task,forbidden", [("qualifying", {"Qualifying", "Race"}), ("race", {"Race"})]
)
def test_target_and_future_observations_are_never_read(
    dataset, manifest, event_inputs, monkeypatch, task, forbidden
):
    event, cutoff, entrants = event_inputs(task=task)
    original = dataset.table
    read = []

    def guarded(spec):
        event_id = spec["table"].split("/")[0]
        assert event_id not in manifest["final_test"]
        assert utc(spec["ended_at"]) < cutoff
        if event_id == event["event_id"]:
            assert spec["kind"] not in forbidden
        read.append((event_id, spec["kind"]))
        return original(spec)

    monkeypatch.setattr(dataset, "table", guarded)
    features, audit = build_event(dataset, manifest, event["event_id"], task, cutoff, entrants)
    assert features.practice_sessions.eq(1).all()  # Sprint weekend only has FP1.
    assert ("qualifying_position" in features) == (task == "race")
    assert read and len(audit["sources"]) == len(read)


def test_uncompleted_or_later_practice_is_not_used(dataset, manifest, event_inputs):
    event, _, entrants = event_inputs("2026-01")
    cutoff = utc(gp_session(event, "Practice 2")["scheduled_start"]) - pd.Timedelta(seconds=1)
    entrants["known_at"] = (cutoff - pd.Timedelta(days=1)).isoformat()
    features, audit = build_event(
        dataset, manifest, event["event_id"], "qualifying", cutoff, entrants
    )
    assert features.practice_sessions.eq(1).all()
    assert all(s["kind"] != "Practice 2" for s in audit["sources"])


@pytest.mark.parametrize("task", ["qualifying", "race"])
def test_cutoff_must_precede_target_start(dataset, manifest, event_inputs, task):
    event, _, entrants = event_inputs(task=task)
    target = gp_session(event, "Qualifying" if task == "qualifying" else "Race")
    with pytest.raises(ValueError, match="strictly before"):
        build_event(dataset, manifest, event["event_id"], task, target["scheduled_start"], entrants)


def test_race_requires_available_qualifying_and_rejects_fake_publication_snapshot(
    dataset, manifest, event_inputs
):
    event, cutoff, entrants = event_inputs(task="race")
    with pytest.raises(ValueError, match="available official GP qualifying"):
        build_event(
            dataset, manifest, event["event_id"], "race", cutoff, entrants, "publication_snapshot"
        )
    too_early = utc(gp_session(event, "Qualifying")["ended_at"])
    with pytest.raises(ValueError, match="available official GP qualifying"):
        build_event(dataset, manifest, event["event_id"], "race", too_early, entrants)


def test_rosters_and_grid_penalties_must_be_known_at_cutoff(dataset, manifest, event_inputs):
    event, cutoff, entrants = event_inputs(task="race")
    entrants["known_at"] = (cutoff + pd.Timedelta(seconds=1)).isoformat()
    with pytest.raises(ValueError, match="Roster was not known"):
        build_event(dataset, manifest, event["event_id"], "race", cutoff, entrants)
    entrants["known_at"] = (cutoff - pd.Timedelta(hours=1)).isoformat()
    entrants["grid_position"] = range(1, len(entrants) + 1)
    entrants["grid_known_at"] = (cutoff + pd.Timedelta(seconds=1)).isoformat()
    entrants["grid_source"] = "synthetic_stewards_notice"
    with pytest.raises(ValueError, match="Grid position was not known"):
        build_event(dataset, manifest, event["event_id"], "race", cutoff, entrants)
    entrants["grid_known_at"] = (cutoff - pd.Timedelta(seconds=1)).isoformat()
    features, _ = build_event(dataset, manifest, event["event_id"], "race", cutoff, entrants)
    assert features.known_grid_position.tolist() == entrants.grid_position.tolist()


def test_rookies_missing_practice_and_no_history_stay_in_field(dataset, manifest, event_inputs):
    event, cutoff, entrants = event_inputs("2024-01")
    entrants.loc[0, "driver_id"] = "new_entrant"
    features, _ = build_event(dataset, manifest, event["event_id"], "qualifying", cutoff, entrants)
    rookie = features.loc[features.driver_id.eq("new_entrant")].iloc[0]
    assert len(features) == len(entrants)
    assert pd.isna(rookie.history_quali_mean) and pd.isna(rookie.practice_gap_pct)
    assert rookie.history_count == rookie.practice_sessions == 0
    with pytest.raises(ValueError, match="roster/result field differs"):
        labels_for(dataset, event["event_id"], "qualifying", features.driver_id)


def test_feature_allowlist_rejects_any_accidental_target_column(dataset, manifest, event_inputs):
    event, cutoff, entrants = event_inputs()
    features, _ = build_event(dataset, manifest, event["event_id"], "qualifying", cutoff, entrants)
    features["official_position"] = 1
    with pytest.raises(ValueError, match="schema mismatch"):
        model_matrix(features, "qualifying")


def test_final_test_event_cannot_build_features_even_for_inference(dataset, manifest, event_inputs):
    event, cutoff, entrants = event_inputs(manifest["final_test"][0])
    with pytest.raises(ValueError, match="Final test block is sealed"):
        build_event(
            dataset, manifest, event["event_id"], "qualifying", cutoff, entrants, inference=True
        )
