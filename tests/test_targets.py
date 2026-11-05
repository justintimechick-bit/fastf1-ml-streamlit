from types import SimpleNamespace

import pandas as pd
import pytest

from f1ml.evaluation.metrics import score_event
from f1ml.ingestion.normalize import normalize_results, summarize_practice


def test_official_order_dsq_dns_nonclassified_and_unknown_are_distinct():
    raw = pd.DataFrame(
        {
            "driverId": list("abcdefg"),
            "position": [1, 2, 3, 4, None, 0, None],
            "positionText": ["1", "2", "3", "D", "W", "N", ""],
            "status": [
                "Finished",
                "+1 Lap",
                "Finished",
                "Disqualified",
                "Did not start",
                "Retired",
                "",
            ],
            "grid": [6, 5, 4, 3, 2, 1, 7],
            "laps": [10, 9, 10, 10, 0, 0, None],
        }
    )
    labels = normalize_results(raw, "Race").set_index("driver_id")
    assert labels.loc["a", "official_position"] == 1
    assert labels.loc["d", "official_position"] == 4  # Preserve provided administrative order.
    assert labels.loc["d", "outcome"] == "dsq"
    assert labels.loc["e", "outcome"] == "dns"
    assert labels.loc["f", "outcome"] == "non_classified"
    assert labels.loc[["d", "e", "f"], "podium"].eq(0).all()
    assert labels.loc[["a", "b", "c"], "podium"].eq(1).all()
    assert labels.loc[["e", "f", "g"], "official_position"].isna().all()
    assert pd.isna(labels.loc["g", "podium"])
    assert not {"grid", "laps", "status"} & set(labels.columns)


def test_qualifying_position_is_never_grid_position():
    raw = pd.DataFrame({"driverId": ["a", "b"], "position": [2, 1], "grid": [1, 2]})
    labels = normalize_results(raw, "Qualifying")
    assert labels.official_position.tolist() == [2, 1]
    assert labels.podium.isna().all()


@pytest.mark.parametrize("column,values", [("driverId", ["a", "a"]), ("position", [1, 1])])
def test_ambiguous_result_keys_fail(column, values):
    raw = pd.DataFrame({"driverId": ["a", "b"], "position": [1, 2]})
    raw[column] = values
    with pytest.raises(ValueError, match="Duplicate"):
        normalize_results(raw, "Race")


def test_practice_summary_excludes_deleted_inaccurate_pit_and_unknown_laps():
    laps = pd.DataFrame(
        {
            "DriverNumber": ["1"] * 6,
            "LapTime": pd.to_timedelta([90, 80, 70, 60, 50, 40], unit="s"),
            "IsAccurate": [True, True, False, True, True, True],
            "Deleted": [False, True, False, False, False, None],
            "PitInTime": pd.to_timedelta([None, None, None, 1, None, None], unit="s"),
            "PitOutTime": pd.to_timedelta([None, None, None, None, 1, None], unit="s"),
        }
    )
    session = SimpleNamespace(
        laps=laps, results=pd.DataFrame({"DriverNumber": ["1"], "DriverId": ["driver"]})
    )
    summary = summarize_practice(session)
    assert summary.iloc[0].best_lap_seconds == 90
    assert summary.iloc[0].clean_laps == 1


def test_partial_classification_reports_coverage_without_inventing_tail_order():
    predictions = pd.DataFrame(
        {"event_id": ["x"] * 3, "driver_id": list("abc"), "predicted_position": [1, 2, 3]}
    )
    labels = pd.DataFrame(
        {"event_id": ["x"] * 3, "driver_id": list("abc"), "official_position": [1, 2, None]}
    )
    metrics = score_event(predictions, labels, "race")
    assert metrics["label_coverage"] == 2 / 3
    assert metrics["complete_order"] is False
    assert metrics["position_mae"] is None
    assert metrics["exact_order"] is None
    assert metrics["pairwise_accuracy"] == 1
