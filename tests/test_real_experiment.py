"""Phase 9 review guards and read-only UI; fixtures here are software tests only."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from f1ml.features.contract import COMMON_NUMERIC, IDENTIFIERS
from f1ml.storage import file_digest, read_json, verify_directory
from f1ml_real.audit import failure_category, planned_results, review_event, target_checks
from f1ml_real.preprocessing import ObservedCorePreprocessor

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "data/real-20261105-core-v2"


def labels(kind="Race", drivers=("a", "b", "c", "d")):
    n = len(drivers)
    return pd.DataFrame(
        {
            "event_id": ["2026-01"] * n,
            "driver_id": drivers,
            "team_id": ["team"] * n,
            "kind": [kind] * n,
            "source": ["jolpica_via_fastf1_ergast"] * n,
            "official_position": range(1, n + 1),
            "team_mapping_state": ["mapped"] * n,
            "position_ambiguous": [False] * n,
            "official_field_complete": [True] * n,
            "outcome": ["ordered"] * n,
            "podium": [float(i < 3) for i in range(n)],
        }
    )


def test_target_audit_preserves_administrative_tail_and_rejects_repeated_positions():
    frame = labels()
    frame.loc[3, ["outcome", "podium"]] = ["dsq", 0.0]
    original = frame.copy(deep=True)
    assert target_checks(frame, "Race") == []
    pd.testing.assert_frame_equal(frame, original)
    frame.loc[2, "official_position"] = 2
    assert "ambiguous_official_positions" in target_checks(frame, "Race")
    assert "incomplete_official_order" in target_checks(frame, "Race")


def test_target_audit_rejects_unofficial_labels_unknown_podium_and_unmapped_team():
    frame = labels()
    frame.loc[0, "source"] = "timing_fallback"
    frame.loc[1, "podium"] = np.nan
    frame.loc[2, "team_mapping_state"] = "unmapped"
    reasons = target_checks(frame, "Race")
    assert {
        "unapproved_result_provenance",
        "incomplete_or_contradictory_podium",
        "unresolved_team_mapping",
    } <= set(reasons)


@pytest.mark.parametrize(
    "message,category",
    [
        ("HTTP 401 Unauthorized", "source_authentication"),
        ("Too Many Requests", "source_rate_limit"),
        ("HTTP 404", "source_endpoint_missing"),
        ("Failed to resolve host", "transport_failure"),
    ],
)
def test_access_diagnosis_does_not_call_unavailable_timing_a_parsing_defect(message, category):
    assert failure_category({"message": message, "type": "RequestError"}) == category


def reviewed_inputs():
    event = {
        "event_id": "2026-01",
        "year": 2026,
        "round": 1,
        "name": "Software test",
        "sessions": [
            {
                "kind": "Qualifying",
                "scheduled_start": "2026-03-07T15:00:00Z",
                "completed": True,
                "official_results": True,
                "ended_at": "2026-03-07T16:00:00Z",
            },
            {
                "kind": "Race",
                "scheduled_start": "2026-03-08T15:00:00Z",
                "completed": True,
                "official_results": True,
                "ended_at": "2026-03-08T17:00:00Z",
            },
        ],
    }
    cutoffs = pd.DataFrame(
        [
            {
                "event_id": "2026-01",
                "task": "qualifying",
                "cutoff": "2026-03-07T14:59:00Z",
                "mode": "historical_reconstruction",
            },
            {
                "event_id": "2026-01",
                "task": "race",
                "cutoff": "2026-03-08T14:59:00Z",
                "mode": "historical_reconstruction",
            },
        ]
    )
    rosters = pd.DataFrame(
        [
            {
                "event_id": "2026-01",
                "task": task,
                "driver_id": driver,
                "team_id": "team",
                "known_at": "2026-03-06T00:00:00Z",
                "source": "independent test evidence",
            }
            for task in ("qualifying", "race")
            for driver in ("a", "b", "c", "d")
        ]
    )
    return event, pd.concat([labels("Qualifying"), labels()]), rosters, cutoffs


def test_review_requires_real_completion_and_cutoff_roster_evidence():
    event, results, rosters, cutoffs = reviewed_inputs()
    as_of = pd.Timestamp("2026-11-05T14:00:00Z")
    assert review_event(None, event, results, rosters, cutoffs, as_of)["eligible"]
    no_completion = deepcopy(event)
    no_completion["sessions"][0]["ended_at"] = None
    row = review_event(None, no_completion, results, rosters, cutoffs, as_of)
    assert not row["eligible"] and "completion_time_unverified" in row["reasons"]
    rosters.loc[0, "known_at"] = "2026-03-08T00:00:00Z"
    assert not review_event(None, event, results, rosters, cutoffs, as_of)["eligible"]


def test_independently_reviewed_task_rosters_can_explain_different_q_r_fields():
    event, results, rosters, cutoffs = reviewed_inputs()
    q = results.kind.eq("Qualifying") & results.driver_id.eq("d")
    results.loc[q, "driver_id"] = "substitute"
    rosters.loc[rosters.task.eq("qualifying") & rosters.driver_id.eq("d"), "driver_id"] = (
        "substitute"
    )
    assert review_event(
        None, event, results, rosters, cutoffs, pd.Timestamp("2026-11-05T14:00:00Z")
    )["eligible"]


def test_preprocessing_excludes_unobserved_training_features_without_using_later_rows():
    frame = pd.DataFrame({c: [1.0, 3.0, 5.0] for c in COMMON_NUMERIC})
    frame["practice_gap_pct"] = np.nan
    for c in IDENTIFIERS:
        frame[c] = "test"
    frame["event_id"] = ["2024-01", "2024-02", "2024-03"]
    frame["driver_id"] = ["a", "b", "c"]
    frame["team_id"] = "team"
    frame["task"] = "qualifying"
    fitted = ObservedCorePreprocessor("qualifying").fit(frame)
    assert fitted.excluded_features_ == ["practice_gap_pct"]
    assert not any("practice_gap_pct" in name for name in fitted.get_feature_names_out())
    later = frame.copy()
    later["driver_id"] = ["unseen1", "unseen2", "unseen3"]
    later["history_quali_mean"] = 10000.0
    later["practice_gap_pct"] = 9.0
    assert fitted.transform(later).shape == fitted.transform(frame).shape
    imputer = fitted.transformer_.named_transformers_["numeric"].named_steps["impute"]
    ix = fitted.numeric_features_.index("history_quali_mean")
    assert imputer.statistics_[ix] == 3.0


def test_planned_results_cannot_be_mistaken_for_measured_zero_loss():
    rows = planned_results()
    assert set(rows.phase) == {"development", "final_test"}
    assert set(rows.task) == {"qualifying", "race", "podium"}
    assert rows.events.eq(0).all() and rows.value.isna().all()
    assert rows.status.eq("not_run_ineligible_dataset").all()


@pytest.mark.skipif(not (REAL / "receipt.json").exists(), reason="local Phase 9 artifacts absent")
def test_real_artifacts_fail_closed_and_preserve_the_demo():
    from f1ml.finalization.locking import code_id
    from f1ml_app.catalog import RealExperiment
    from f1ml_real.__main__ import main

    exp = RealExperiment(REAL)
    assert exp.status["state"] == "blocked_before_training"
    assert exp.status["models_trained"] == 0
    assert not exp.status["final_manifest_frozen"]
    assert not exp.status["evaluation_ledger_claimed"]
    assert main(["check-ready", "--experiment", str(REAL)]) == 2
    assert exp.forecast("2026-01", "qualifying") is None
    assert exp.forecast("2026-01", "race") is None
    assert exp.evaluation_records.value.isna().all()
    assert verify_directory(ROOT / "data/phase7-lock")
    assert code_id() == read_json(ROOT / "data/phase7-lock/protocol.json")["code_id"]
    preserved = read_json(ROOT / "data/phase9-preservation.json")
    assert all(file_digest(ROOT / p) == h for p, h in preserved.items())


@pytest.mark.skipif(not (REAL / "receipt.json").exists(), reason="local Phase 9 artifacts absent")
def test_every_real_result_replays_from_retained_raw_frames(tmp_path):
    from f1ml_real.validate import validate_sources

    receipt = validate_sources(REAL, tmp_path / "replay")
    assert receipt["mismatched_sessions"] == 0
    assert receipt["result_sessions"] == 112
    assert receipt["driver_rows"] == 2263
    with pytest.raises(FileExistsError):
        validate_sources(REAL, tmp_path / "replay")


@pytest.mark.skipif(not (REAL / "receipt.json").exists(), reason="local Phase 9 artifacts absent")
def test_real_interface_smoke_has_no_synthetic_forecast_fallback(monkeypatch):
    from sklearn.pipeline import Pipeline
    from streamlit.testing.v1 import AppTest

    from f1ml.ingestion.provider import FastF1Provider

    def forbidden(*args, **kwargs):
        raise AssertionError("Interface attempted to download or fit")

    monkeypatch.setattr(FastF1Provider, "fetch_results", forbidden)
    monkeypatch.setattr(FastF1Provider, "fetch_timing", forbidden)
    monkeypatch.setattr(Pipeline, "fit", forbidden)
    before = read_json(REAL / "receipt.json")
    at = AppTest.from_file(str(ROOT / "src/f1ml_app/streamlit_app.py"), default_timeout=60).run()
    at.segmented_control(key="source").set_value("real").run()
    for page in ("overview", "qualifying", "race", "comparison", "methodology"):
        at.switch_page(f"views/{page}.py").run()
        assert not at.exception and not at.error, page
        text = "\n".join(
            str(e.value) for kind in ("markdown", "caption", "warning") for e in at.get(kind)
        )
        text += "\n".join(h.proto.body for h in at.get("html"))
        assert "sealed forecast (frozen)" not in text
        if page in ("overview", "comparison", "methodology"):
            assert "blocked_before_training" in text
        else:
            assert "No " + ("qualifying" if page == "qualifying" else "race") + " forecast" in text
    assert read_json(REAL / "receipt.json") == before
    verify_directory(REAL)
