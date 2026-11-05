from types import SimpleNamespace

import pandas as pd
import pytest

from f1ml.ingestion.provider import FastF1Provider


def fake_provider(status="Finalised", complete=True, timing=True):
    session = SimpleNamespace(
        t0_date=pd.Timestamp("2024-01-01T10:00:00") if timing else None,
        session_start_time=pd.Timedelta(hours=1),
        session_status=pd.DataFrame(
            {"Time": [pd.Timedelta(hours=1), pd.Timedelta(hours=2)], "Status": ["Started", status]}
        ),
        results=pd.DataFrame({"Position": [999]}),
        load=lambda **kwargs: None,
    )
    raw = pd.DataFrame({"driverId": ["a", "b", "c"], "position": [1, 2, 3]})
    response = SimpleNamespace(is_complete=complete, content=[raw])
    provider = FastF1Provider.__new__(FastF1Provider)
    provider.fastf1 = SimpleNamespace(get_session=lambda *args: session)
    provider.results_api = SimpleNamespace(
        get_qualifying_results=lambda **kwargs: response, get_race_results=lambda **kwargs: response
    )
    return provider


def test_completion_uses_actual_timing_and_official_results_not_timing_fallback():
    provider = fake_provider()
    event = {"year": 2024, "round": 1}
    spec = {"kind": "Qualifying", "scheduled_start": "2024-01-01T09:00:00Z"}
    meta, table = provider.session(event, spec, pd.Timestamp("2024-01-02T00:00:00Z"))
    assert meta["completed"] is True
    assert meta["started_at"] == "2024-01-01T11:00:00+00:00"
    assert meta["ended_at"] == "2024-01-01T12:00:00+00:00"
    assert table.official_position.tolist() == [1, 2, 3]


def test_finished_segment_is_not_a_finalised_session():
    meta, table = fake_provider(status="Finished").session(
        {"year": 2024, "round": 1}, {"kind": "Qualifying"}, pd.Timestamp("2024-01-02T00:00:00Z")
    )
    assert meta["completed"] is False
    assert table.empty


@pytest.mark.parametrize("kind,laps", [("Practice 1", True), ("Qualifying", True), ("Race", False)])
def test_public_load_flags_defer_raw_telemetry_and_race_laps(kind, laps):
    provider = fake_provider()
    session = provider.fastf1.get_session()
    calls = []
    session.load = lambda **kwargs: calls.append(kwargs)
    batch = provider.fetch_timing({"year": 2024, "round": 1}, {"kind": kind})
    assert calls == [{"laps": laps, "telemetry": False, "weather": True, "messages": laps}]
    assert batch.metadata["load_options"]["telemetry"] is False


def test_completion_state_does_not_require_fabricated_absolute_time():
    batch = fake_provider(timing=False).fetch_timing(
        {"year": 2024, "round": 1}, {"kind": "Qualifying"}
    )
    assert batch.metadata["completion_state"] == "completed"
    assert batch.metadata["completed_at"] is None
    assert batch.metadata["absolute_time_available"] is False


def test_public_result_pagination_covers_the_reported_total():
    provider = fake_provider()
    second = SimpleNamespace(
        is_complete=False,
        total_results=3,
        content=[pd.DataFrame({"driverId": ["c"], "position": [3]})],
    )
    first = SimpleNamespace(
        is_complete=False,
        total_results=3,
        content=[pd.DataFrame({"driverId": ["a", "b"], "position": [1, 2]})],
        get_next_result_page=lambda: second,
    )
    provider.results_api.get_qualifying_results = lambda **kwargs: first
    batch = provider.fetch_results({"year": 2024, "round": 1}, {"kind": "Qualifying"})
    assert batch.metadata["response_complete"] and batch.metadata["page_count"] == 2
    assert batch.frames["results"].driverId.tolist() == ["a", "b", "c"]


def test_restart_after_finalisation_requires_new_completion_evidence():
    provider = fake_provider()
    session = provider.fastf1.get_session()
    session.session_status = pd.DataFrame(
        {
            "Time": pd.to_timedelta([0, 60, 120, 180], unit="s"),
            "Status": ["Started", "Finalised", "Started", "Ends"],
        }
    )
    batch = provider.fetch_timing({"year": 2024, "round": 1}, {"kind": "Qualifying"})
    assert batch.metadata["completion_state"] == "in_progress"
    assert batch.metadata["completed_at"] is None


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"complete": False}, "incomplete official results"),
        ({"timing": False}, "absolute session timing"),
    ],
)
def test_incomplete_results_and_unprovable_timing_fail_closed(kwargs, message):
    with pytest.raises(ValueError, match=message):
        fake_provider(**kwargs).session(
            {"year": 2024, "round": 1}, {"kind": "Qualifying"}, pd.Timestamp("2024-01-02T00:00:00Z")
        )


def test_schedule_excludes_testing_and_keeps_session_types_distinct():
    rows = []
    for round_number in (0, 1):
        event = {
            "RoundNumber": round_number,
            "EventFormat": "testing" if not round_number else "sprint",
            "EventName": "Synthetic",
            "Location": "Synthetic",
        }
        for index, kind in enumerate(
            ["Practice 1", "Sprint Qualifying", "Sprint", "Qualifying", "Race"], 1
        ):
            event[f"Session{index}"] = kind
            event[f"Session{index}DateUtc"] = pd.Timestamp("2026-01-01") + pd.Timedelta(days=index)
        rows.append(event)
    provider = FastF1Provider.__new__(FastF1Provider)

    def schedule(year, include_testing):
        assert year == 2026 and include_testing is False
        return pd.DataFrame(rows)

    provider.fastf1 = SimpleNamespace(get_event_schedule=schedule)
    events = provider.schedule(2026)
    assert len(events) == 1 and events[0]["event_id"] == "2026-01"
    assert {s["kind"] for s in events[0]["sessions"]} == {
        "Practice 1",
        "Sprint Qualifying",
        "Sprint",
        "Qualifying",
        "Race",
    }
