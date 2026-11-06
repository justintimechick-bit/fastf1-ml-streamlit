"""Train all three families on real FIA history and verify their saved outputs."""

import json

import joblib
import pandas as pd
import pytest

from f1ml.storage import file_digest, verify_directory
from f1ml_real import lab


def test_all_nine_real_estimators_and_baselines_have_identical_evaluation_fields(
    replay, replay_path, real_dataset
):
    summary = replay["summary"]
    assert summary["data_kind"] == "real"
    assert summary["model_count"] == 9
    assert summary["target_event_count"] == 1
    assert summary["target_event_id"] == "2026-14"
    assert summary["training_event_count"] == 12
    assert len(list((replay_path / "models").glob("*.joblib"))) == 9
    forecasts, metrics = replay["forecasts"], replay["metrics"]
    assert set(forecasts.family) == {*lab.FAMILIES, "baseline"}
    assert set(forecasts.task) == set(lab.TASKS)
    for task in lab.TASKS:
        source_task = "race" if task == "podium" else task
        roster = real_dataset.rosters.loc[
            real_dataset.rosters.event_id.eq("2026-14") & real_dataset.rosters.task.eq(source_task)
        ]
        for family in (*lab.FAMILIES, "baseline"):
            group = forecasts.loc[forecasts.task.eq(task) & forecasts.family.eq(family)]
            assert set(group.driver_id) == set(roster.driver_id)
            assert not group.driver_id.duplicated().any()
            assert set(group.predicted_rank) == set(range(1, len(roster) + 1))
        task_metrics = metrics[metrics.task.eq(task)]
        assert task_metrics.event_count.eq(1).all()
        assert task_metrics.groupby("family").known_label_count.first().nunique() == 1
        assert task_metrics.groupby("family").total_driver_count.first().nunique() == 1
        assert task_metrics.value.notna().all()
    podium = forecasts[forecasts.task.eq("podium")]
    assert podium.podium_probability.between(0, 1).all()


def test_development_fits_and_calibration_obey_their_own_cutoffs(replay, real_dataset):
    config = replay["config"]
    dev, calibration = config["development_event_ids"], config["calibration_event_ids"]
    assert len(dev) >= 2
    assert len(calibration) == 2
    assert not set(dev) & set(calibration)
    assert "2026-14" not in config["training_event_ids"]
    for event_id, trained_on in config["fold_training_event_ids"].items():
        assert event_id not in trained_on
        assert set(trained_on) <= set(real_dataset.history_events(event_id).event_id)
        assert set(trained_on) <= set(config["training_event_ids"])
    # Real late revision: a June race cannot train a July validation fold when
    # this snapshot's selected FIA final was published in September.
    earlier_folds = [e for e in dev if e < "2026-13"]
    assert earlier_folds
    for event_id in earlier_folds:
        assert "2026-06" not in config["fold_training_event_ids"][event_id]
    development = replay["development_metrics"]
    assert set(development.stage) == {
        "development_selection",
        "development_calibration_fit_diagnostic",
    }
    selected = development[development.stage.eq("development_selection")]
    assert selected.event_count.eq(len(dev)).all()
    assert not replay["metrics"].stage.eq("development_selection").any()


def test_target_forecasts_are_sealed_and_saved_models_reproduce_them(
    replay, replay_path, real_dataset
):
    lock = replay["prediction_lock"]
    assert lock["sealed_before_target_scoring"]
    assert lock["config_sha256"] == file_digest(replay_path / "config.json")
    assert lock["forecasts_sha256"] == file_digest(replay_path / "forecasts.parquet")
    assert not {"actual_rank", "actual_podium", "label_source"} & set(replay["forecasts"])
    config = replay["config"]
    for task in lab.TASKS:
        frame, _ = real_dataset.feature_frame("2026-14", task)
        for family in lab.FAMILIES:
            bundle = joblib.load(replay_path / "models" / f"{task}-{family}.joblib")
            assert bundle["source_snapshot_id"] == real_dataset.receipt["id"]
            assert bundle["training_event_ids"] == config["training_event_ids"]
            recalculated = lab._forecast(
                bundle["model"],
                frame,
                task,
                family,
                bundle["calibration"],
                bundle["threshold"] or 0.5,
            )
            saved = replay["forecasts"].query("task == @task and family == @family")
            pd.testing.assert_frame_equal(
                recalculated[saved.columns].reset_index(drop=True),
                saved.reset_index(drop=True),
                check_dtype=False,
            )
            excluded = bundle["model"].named_steps["preprocess"].excluded_features_
            assert excluded["driver_qualifying_podium_rate_3"] == "entirely_missing_in_training"


def test_cached_run_reuses_verified_real_artifacts_without_refitting(
    replay_path, dataset_root, monkeypatch
):
    before = verify_directory(replay_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("An identical saved real run must not refit")

    monkeypatch.setattr(lab, "run_backtest", forbidden)
    loaded = lab.cached_run(dataset_root, "2026-14", replay_path.parent, last_n=12)
    assert loaded == replay_path
    assert verify_directory(replay_path) == before


def test_an_existing_run_is_never_overwritten(replay_path, dataset_root):
    before = json.loads((replay_path / "receipt.json").read_text())
    with pytest.raises(FileExistsError):
        lab.run_backtest(dataset_root, "2026-14", replay_path, last_n=12)
    assert verify_directory(replay_path) == before
