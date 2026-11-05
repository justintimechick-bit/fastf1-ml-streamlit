"""Expanding-window development evaluation; never opens a final-test feature/label file."""

from pathlib import Path

import joblib
import pandas as pd

from f1ml.evaluation.metrics import score_event
from f1ml.evaluation.splits import assert_development, assert_disjoint_events
from f1ml.features.contract import model_matrix
from f1ml.models.baseline import fit_baseline
from f1ml.prediction.service import predict_checked
from f1ml.storage import new_directory, save_table, seal_directory, verify_directory, write_json


def run_backtest(manifest: dict, feature_root: Path, output: Path) -> dict:
    receipt = verify_directory(feature_root)
    if receipt["manifest_id"] != manifest["id"]:
        raise ValueError("Features belong to another frozen manifest")
    new_directory(output)
    scores = []
    for task in ("qualifying", "race", "podium"):
        source_task = "race" if task == "podium" else task
        features = pd.read_parquet(feature_root / f"{source_task}_features.parquet")
        labels = pd.read_parquet(feature_root / f"{source_task}_labels.parquet")
        assert_development(manifest, features.event_id)
        assert_development(manifest, labels.event_id)
        model_matrix(features, task)
        for fold in manifest["folds"]:
            assert_disjoint_events(fold["train"], [fold["validate"]], manifest["final_test"])
            train = features.loc[features.event_id.isin(fold["train"])]
            validation = features.loc[features.event_id.eq(fold["validate"])]
            training_labels = labels.loc[labels.event_id.isin(fold["train"])]
            validation_labels = labels.loc[labels.event_id.eq(fold["validate"])]
            if set(train.event_id) != set(fold["train"]) or validation.empty:
                raise ValueError("A complete manifest fold is missing from features")
            artifact = fit_baseline(
                train, training_labels, task, manifest, manifest["config"]["random_seed"]
            )
            predictions = predict_checked(artifact, validation, manifest)
            destination = output / task / fold["validate"]
            destination.mkdir(parents=True)
            joblib.dump(artifact, destination / "model.joblib")
            write_json(destination / "model.json", artifact.metadata)
            save_table(destination / "predictions.parquet", predictions)
            scores.append(score_event(predictions, validation_labels, task))
    frame = pd.DataFrame(scores)
    save_table(output / "weekend_metrics.parquet", frame)
    metric_names = [
        "position_mae",
        "spearman",
        "pairwise_accuracy",
        "top3_recall",
        "exact_order",
        "brier",
        "log_loss",
        "label_coverage",
    ]
    summary = {}
    for task, group in frame.groupby("task"):
        metrics = {}
        for name in metric_names:
            if name in group and group[name].notna().any():
                metrics[name] = {
                    "mean": float(group[name].dropna().astype(float).mean()),
                    "weekends": int(group[name].notna().sum()),
                }
        summary[task] = {"validation_weekends": len(group), "metrics": metrics}
    write_json(output / "summary.json", summary)
    return seal_directory(
        output,
        {
            "manifest_id": manifest["id"],
            "feature_snapshot_id": receipt["id"],
            "status": "development_only_final_test_sealed",
        },
    )
