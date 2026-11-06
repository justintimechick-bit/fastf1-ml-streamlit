"""Leakage checks perturb copies of real observations, not invented F1 fixtures."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from f1ml_real.lab import NUMERIC_FEATURES, ObservedPreprocessor


def test_history_uses_availability_not_calendar_position(real_dataset):
    mexico = real_dataset.history_events("2026-19")
    assert len(mexico) == 18
    assert "2026-19" not in set(mexico.event_id)
    boundary = pd.Timestamp(real_dataset.event("2026-19").weekend_start)
    assert (pd.to_datetime(mexico.race_available_at, utc=True) < boundary).all()
    # The selected final Monaco classification was revised after the Italian
    # weekend began. Earlier calendar position does not make this revision known.
    italy = real_dataset.history_events("2026-13")
    assert "2026-06" not in set(italy.event_id)
    assert "2026-06" in set(real_dataset.history_events("2026-14").event_id)


def test_observed_history_matches_field_normalized_real_classifications(real_dataset):
    frame, audit = real_dataset.feature_frame("2026-19", "qualifying")
    assert frame.driver_qualifying_rank_mean_3.notna().sum() >= 20
    assert frame.driver_qualifying_rank_count_3.max() == 3
    ids = audit["history_event_ids"][-3:]
    labels = real_dataset.labels
    selected = labels.loc[
        labels.event_id.isin(ids) & labels.task.eq("qualifying") & labels.driver_id.eq("norris")
    ]
    sizes = real_dataset.rosters.query("task == 'qualifying'").groupby("event_id").size()
    expected = ((selected.official_position - 1) / (selected.event_id.map(sizes) - 1)).mean()
    actual = frame.loc[frame.driver_id.eq("norris"), "driver_qualifying_rank_mean_3"].iloc[0]
    assert actual == pytest.approx(expected)
    # A substitute with no previous observations remains genuinely unknown.
    assert pd.isna(
        frame.loc[frame.driver_id.eq("ryo_hirakawa"), "driver_qualifying_rank_mean_3"].iloc[0]
    )


@pytest.mark.parametrize("task", ["qualifying", "race", "podium"])
def test_target_and_future_outcomes_cannot_change_features(real_dataset, task):
    original, audit = real_dataset.feature_frame("2026-17", task)
    changed = deepcopy(real_dataset)
    changed._history_cache = {}
    # Deliberately corrupt labels of this target's race and all later events.
    # Current qualifying is legitimately available to a race/podium forecast.
    target_or_future = changed.labels.event_id.ge("2026-17")
    if task != "qualifying":
        target_or_future &= ~(
            changed.labels.event_id.eq("2026-17") & changed.labels.task.eq("qualifying")
        )
    changed.labels.loc[target_or_future, "official_position"] = 999.0
    changed.labels.loc[target_or_future, "podium"] = 1.0
    changed.labels.loc[target_or_future, "outcome"] = "dsq"
    rebuilt, rebuilt_audit = changed.feature_frame("2026-17", task)
    pd.testing.assert_frame_equal(original, rebuilt)
    assert audit == rebuilt_audit
    assert audit["current_qualifying_used"] == (task != "qualifying")
    assert "2026-17" not in audit["history_event_ids"]


def test_target_qualifying_is_used_only_after_its_publication(real_dataset):
    qframe, _ = real_dataset.feature_frame("2026-19", "qualifying")
    rframe, _ = real_dataset.feature_frame("2026-19", "race")
    assert "current_qualifying_rank" not in qframe
    assert rframe.current_qualifying_rank.notna().sum() == 21
    changed = deepcopy(real_dataset)
    changed.events.loc[changed.events.event_id.eq("2026-19"), "qualifying_available_at"] = (
        changed.event("2026-19").race_cutoff
    )
    with pytest.raises(ValueError, match="qualifying|Qualifying"):
        changed.feature_frame("2026-19", "race")


def test_future_or_target_event_cannot_be_injected_as_history(real_dataset):
    with pytest.raises(ValueError, match="unavailable"):
        real_dataset.feature_frame("2026-17", "qualifying", history_ids=["2026-18"])
    with pytest.raises(ValueError, match="unavailable"):
        real_dataset.feature_frame("2026-17", "qualifying", history_ids=["2026-17"])


def test_preprocessing_uses_training_observations_only(real_dataset):
    training = pd.concat(
        [real_dataset.feature_frame(eid, "race")[0] for eid in ["2026-09", "2026-10", "2026-11"]],
        ignore_index=True,
    )
    empty = "driver_race_rank_mean_5"
    median_feature = "driver_qualifying_rank_mean_3"
    assert median_feature in NUMERIC_FEATURES
    training[empty] = np.nan
    fitted = ObservedPreprocessor().fit(training)
    assert fitted.excluded_features_[empty] == "entirely_missing_in_training"
    assert empty not in fitted.numeric_features_
    assert "field_size" in fitted.excluded_features_  # genuinely constant 22-car fields
    names = fitted.transformer_.get_feature_names_out()
    assert not any(empty in name for name in names)
    imputer = fitted.transformer_.named_transformers_["numeric"].named_steps["impute"]
    ix = fitted.numeric_features_.index(median_feature)
    expected_median = training[median_feature].median()
    assert imputer.statistics_[ix] == pytest.approx(expected_median)
    categories = fitted.transformer_.named_transformers_["categorical"].named_steps["encode"]
    vocabulary = [values.copy() for values in categories.categories_]
    heldout = real_dataset.feature_frame("2026-19", "race")[0].copy()
    heldout[empty] = 10_000.0
    heldout[median_feature] = 10_000.0
    heldout["team_id"] = "deliberately_unseen_category"
    before = fitted.transform(training).shape[1]
    assert fitted.transform(heldout).shape[1] == before
    assert np.isfinite(fitted.transform(heldout)).all()
    assert imputer.statistics_[ix] == expected_median
    for before_values, after_values in zip(vocabulary, categories.categories_, strict=True):
        assert np.array_equal(before_values, after_values)


def test_late_revision_masks_its_calendar_slot_without_extending_window(real_dataset):
    frame, audit = real_dataset.feature_frame("2026-07", "qualifying")
    assert audit["calendar_window_event_ids"]["3"] == ["2026-04", "2026-05", "2026-06"]
    antonelli = frame.loc[frame.driver_id.eq("antonelli")].iloc[0]
    assert antonelli.driver_race_rank_count_3 == 2
    assert antonelli.driver_qualifying_rank_count_3 == 3
    changed = deepcopy(real_dataset)
    changed._history_cache = {}
    revised = changed.labels.event_id.eq("2026-06") & changed.labels.task.eq("race")
    changed.labels.loc[revised, "official_position"] = 999.0
    changed.labels.loc[revised, "podium"] = 1.0
    rebuilt, _ = changed.feature_frame("2026-07", "qualifying")
    pd.testing.assert_frame_equal(frame, rebuilt)
