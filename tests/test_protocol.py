import shutil

import pandas as pd
import pytest

from f1ml.cli import main
from f1ml.config import Config
from f1ml.domain import utc
from f1ml.evaluation.splits import freeze, load_manifest
from f1ml.ingestion.pipeline import Dataset
from f1ml.models.baseline import fit_baseline
from f1ml.prediction.service import load_model, predict_checked, predict_event
from f1ml.storage import read_json, verify_directory


def test_latest_four_completed_2026_weekends_are_sealed_and_folds_never_mix_weekends(manifest):
    current = [e["event_id"] for e in manifest["eligible"] if e["year"] == 2026]
    assert manifest["final_test"] == current[-4:]
    assert "2026-08" not in current  # Future event is discovered but not assumed complete.
    assert "2026-08" in {e["event_id"] for e in manifest["excluded"]}
    dates = {e["event_id"]: utc(e["race_start"]) for e in manifest["eligible"]}
    for fold in manifest["folds"]:
        assert fold["validate"].startswith("2026-")
        assert fold["validate"] not in fold["train"]
        assert not set(fold["train"]) & set(manifest["final_test"])
        assert all(dates[event] < dates[fold["validate"]] for event in fold["train"])


def test_infeasible_test_block_is_not_silently_shrunk(dataset, tmp_path):
    with pytest.raises(ValueError, match="no automatic shrinking"):
        freeze(dataset, Config(final_test_weekends=7), tmp_path / "manifest.json")
    smaller = freeze(dataset, Config(final_test_weekends=2), tmp_path / "smaller.json")
    assert len(smaller["final_test"]) == 2
    with pytest.raises(FileExistsError):
        freeze(dataset, Config(final_test_weekends=2), tmp_path / "smaller.json")


def test_manifest_and_data_revisions_are_verified(workflow, tmp_path):
    manifest_path = tmp_path / "modified.json"
    text = (
        (workflow / "manifest.json")
        .read_text()
        .replace('"final_test_weekends": 4', '"final_test_weekends": 3')
    )
    manifest_path.write_text(text)
    with pytest.raises(ValueError, match="manifest changed"):
        load_manifest(manifest_path)
    copy = tmp_path / "dataset"
    shutil.copytree(workflow / "dataset", copy)
    (copy / "events.json").write_text("[]")
    with pytest.raises(ValueError, match="Snapshot contents changed"):
        Dataset(copy)


def test_offline_workflow_artifacts_cover_full_variable_fields(workflow, manifest):
    receipt = verify_directory(workflow / "artifacts")
    assert receipt["status"] == "development_only_final_test_sealed"
    metrics = pd.read_parquet(workflow / "artifacts/weekend_metrics.parquet")
    assert len(metrics) == len(manifest["folds"]) * 3
    assert set(metrics.task) == {"qualifying", "race", "podium"}
    assert set(metrics.event_id) == {f["validate"] for f in manifest["folds"]}
    sizes = []
    for task in ("qualifying", "race", "podium"):
        for fold in manifest["folds"]:
            root = workflow / "artifacts" / task / fold["validate"]
            output = pd.read_parquet(root / "predictions.parquet")
            assert not output.driver_id.duplicated().any()
            sizes.append(len(output))
            if task == "podium":
                assert output.podium_probability.between(0, 1).all()
            else:
                assert set(output.predicted_position) == set(range(1, len(output) + 1))
            metadata = read_json(root / "model.json")
            assert set(metadata["train_events"]) == set(fold["train"])
    assert len(set(sizes)) > 1


def test_unknown_driver_and_all_missing_columns_can_be_predicted(workflow, manifest):
    event = manifest["folds"][0]["validate"]
    artifact = load_model(workflow / "artifacts/qualifying" / event / "model.joblib")
    features = pd.read_parquet(workflow / "features/qualifying_features.parquet")
    features = features.loc[features.event_id.eq(event)].copy()
    features.loc[features.index[0], "driver_id"] = "unseen_rookie"
    features["practice_gap_pct"] = float("nan")
    prediction = predict_checked(artifact, features, manifest)
    assert len(prediction) == len(features)
    assert "unseen_rookie" in set(prediction.driver_id)
    with pytest.raises(ValueError, match="part of this model's training"):
        predict_checked(
            artifact, features.assign(event_id=artifact.metadata["train_events"][0]), manifest
        )


def test_podium_single_class_training_and_all_missing_inputs(workflow, manifest):
    features = pd.read_parquet(workflow / "features/race_features.parquet")
    labels = pd.read_parquet(workflow / "features/race_labels.parquet")
    fold = manifest["folds"][0]
    train = features.loc[features.event_id.isin(fold["train"])]
    targets = labels.loc[labels.event_id.isin(fold["train"])].copy()
    targets["podium"] = 0.0
    artifact = fit_baseline(train, targets, "podium", manifest, 42)
    validation = features.loc[features.event_id.eq(fold["validate"])]
    result = predict_checked(artifact, validation, manifest)
    assert result.podium_probability.eq(0).all()


def test_publication_evaluation_rejects_late_training_label_revisions(workflow, manifest):
    event = manifest["folds"][0]["validate"]
    artifact = load_model(workflow / "artifacts/race" / event / "model.joblib")
    features = pd.read_parquet(workflow / "features/race_features.parquet")
    features = features.loc[features.event_id.eq(event)].copy()
    features["mode"] = "publication_snapshot"
    with pytest.raises(ValueError, match="reconstructed model"):
        predict_checked(artifact, features, manifest)
    artifact.metadata["modes"] = ["publication_snapshot"]
    artifact.metadata["max_label_retrieved_at"] = "2026-07-01T00:00:00Z"
    with pytest.raises(ValueError, match="label revision was downloaded after"):
        predict_checked(artifact, features, manifest)


def test_inference_supports_discovered_future_events_without_reading_target_labels(
    workflow, dataset, manifest, event_inputs, tmp_path
):
    event, cutoff, entrants = event_inputs("2026-08")
    last_fold = manifest["folds"][-1]["validate"]
    artifact = load_model(workflow / "artifacts/qualifying" / last_fold / "model.joblib")
    # No target Q/R tables exist for this future event.
    receipt = predict_event(
        artifact,
        dataset,
        manifest,
        event["event_id"],
        cutoff,
        entrants,
        "historical_reconstruction",
        tmp_path / "prediction",
    )
    assert receipt["prediction_dataset_id"] == dataset.receipt["id"]
    assert len(pd.read_parquet(tmp_path / "prediction/predictions.parquet")) == len(entrants)


def test_cli_helps_and_expected_failures_do_not_need_network(tmp_path):
    assert main(["freeze", "--dataset", str(tmp_path / "missing"), "--output", "unused"]) == 2


@pytest.mark.parametrize(
    "values",
    [
        {"years": (2023, 2024, 2025)},
        {"final_test_weekends": 0},
        {"history_window": -1},
        {"evaluation_year": 2025},
    ],
)
def test_scope_and_split_configuration_are_validated(values):
    with pytest.raises(ValueError):
        Config(**values)
