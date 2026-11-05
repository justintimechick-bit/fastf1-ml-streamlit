"""Fail closed on training, tuning and model-selection information at prediction time."""

from pathlib import Path

from f1ml.domain import utc
from f1ml.features.build import build_event
from f1ml.models.qualifying import predict_scores
from f1ml.storage import new_directory, save_table, seal_directory


def predict_checked(artifact, features, manifest, weekend_start=None):
    metadata = artifact.metadata
    if metadata["manifest_id"] != manifest["id"]:
        raise ValueError("Qualifying model belongs to a different frozen manifest")
    targets = set(features.event_id)
    if targets & set(manifest["final_test"]):
        raise ValueError("Final test block is sealed during model development")
    if targets & set(metadata["dependency_events"]):
        raise ValueError("Prediction event was used for training, tuning or model selection")
    earliest = min(utc(t) for t in features.cutoff)
    for name in ("max_feature_cutoff", "max_label_end", "max_weekend_end"):
        if utc(metadata[name]) >= earliest:
            raise ValueError(f"Model {name} is not strictly earlier than prediction cutoff")
    if weekend_start is not None and utc(metadata["max_weekend_end"]) >= utc(weekend_start):
        raise ValueError("Model dependencies are not earlier complete weekends")
    if features["mode"].eq("publication_snapshot").any():
        if metadata["modes"] != ["publication_snapshot"]:
            raise ValueError("A reconstructed model cannot claim publication snapshot prediction")
        if utc(metadata["max_label_retrieved_at"]) > earliest:
            raise ValueError(
                "Model dependency label revision was retrieved after prediction cutoff"
            )
    return predict_scores(artifact, features)


def predict_event(artifact, dataset, manifest, event_id, cutoff, entrants, mode, output: Path):
    features, audit = build_event(
        dataset, manifest, event_id, "qualifying", cutoff, entrants, mode, inference=True
    )
    first_session = min(utc(s["scheduled_start"]) for s in dataset.event(event_id)["sessions"])
    predictions = predict_checked(artifact, features, manifest, first_session)
    new_directory(output)
    save_table(output / "predictions.parquet", predictions)
    save_table(output / "features.parquet", features)
    return seal_directory(
        output,
        {
            "manifest_id": manifest["id"],
            "prediction_dataset_id": dataset.receipt["id"],
            "task": "qualifying",
            "model": artifact.metadata,
            "audit": audit,
        },
    )
