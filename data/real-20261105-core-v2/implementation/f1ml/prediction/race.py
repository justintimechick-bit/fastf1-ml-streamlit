"""After-qualifying predictions with model/feature/cutoff lineage and two explicit top threes."""

from pathlib import Path

from f1ml.domain import utc
from f1ml.features.build import build_event
from f1ml.models.qualifying import validate_permutation
from f1ml.models.race import predict_model
from f1ml.storage import new_directory, save_table, seal_directory


def predict_checked(artifact, features, manifest, weekend_start=None):
    meta = artifact.metadata
    if meta["manifest_id"] != manifest["id"]:
        raise ValueError("Race/podium model belongs to another frozen manifest")
    targets = set(features.event_id)
    if targets & set(manifest["final_test"]):
        raise ValueError("Final test block is sealed during model development")
    if targets & set(meta["dependency_events"]):
        raise ValueError("Prediction event was used for training, tuning, calibration or selection")
    cutoff = min(utc(t) for t in features.cutoff)
    for name in ("max_feature_cutoff", "max_label_end", "max_weekend_end"):
        if utc(meta[name]) >= cutoff:
            raise ValueError(f"Model {name} is not strictly earlier than prediction cutoff")
    if weekend_start is not None and utc(meta["max_weekend_end"]) >= utc(weekend_start):
        raise ValueError("Model dependencies are not earlier complete weekends")
    if features["mode"].eq("publication_snapshot").any():
        if meta["modes"] != ["publication_snapshot"]:
            raise ValueError("A reconstructed model cannot claim publication snapshot prediction")
        if utc(meta["max_label_retrieved_at"]) > cutoff:
            raise ValueError("A dependency label revision was retrieved after prediction cutoff")
    return predict_model(artifact, features)


def combine_predictions(race, podium):
    validate_permutation(race, podium)
    keys = ["event_id", "driver_id", "cutoff", "mode"]
    result = race.merge(podium, on=keys, how="outer", validate="one_to_one", indicator=True)
    if not result._merge.eq("both").all():
        raise ValueError("Race and podium predictions must share the same field and cutoff")
    result = result.drop(columns="_merge")
    ordered = result.sort_values(
        ["event_id", "podium_probability", "driver_id"],
        ascending=[True, False, True],
        kind="stable",
    )
    ordered["podium_probability_rank"] = ordered.groupby("event_id").cumcount() + 1
    ordered["race_order_top_three"] = ordered.predicted_position.le(3)
    ordered["classifier_top_three"] = ordered.podium_probability_rank.le(3)
    result = ordered.sort_values(["event_id", "predicted_position"]).reset_index(drop=True)
    disagreements = []
    for event_id, group in result.groupby("event_id"):
        order_top = sorted(group.loc[group.race_order_top_three, "driver_id"])
        probability_top = sorted(group.loc[group.classifier_top_three, "driver_id"])
        disagreements.append(
            {
                "event_id": event_id,
                "race_order_top_three": order_top,
                "classifier_top_three": probability_top,
                "disagree": order_top != probability_top,
                "independent_probability_sum": float(group.podium_probability.sum()),
            }
        )
    return result, disagreements


def predict_event(
    race_model,
    podium_model,
    dataset,
    manifest,
    event_id,
    cutoff,
    entrants,
    mode,
    output: Path,
    model_ids=None,
):
    if race_model.task != "race" or podium_model.task != "podium":
        raise ValueError("A race model and a podium model are required")
    if race_model.metadata["feature_snapshot_id"] != podium_model.metadata["feature_snapshot_id"]:
        raise ValueError("Race/podium models must share a feature training snapshot")
    features, audit = build_event(
        dataset, manifest, event_id, "race", cutoff, entrants, mode, inference=True
    )
    first = min(utc(s["scheduled_start"]) for s in dataset.event(event_id)["sessions"])
    race = predict_checked(race_model, features, manifest, first)
    podium = predict_checked(podium_model, features, manifest, first)
    predictions, disagreements = combine_predictions(race, podium)
    new_directory(output)
    save_table(output / "predictions.parquet", predictions)
    save_table(output / "features.parquet", features)
    return seal_directory(
        output,
        {
            "task": "race_and_podium",
            "manifest_id": manifest["id"],
            "prediction_dataset_id": dataset.receipt["id"],
            "audit": audit,
            "models": {"race": race_model.metadata, "podium": podium_model.metadata},
            "model_ids": model_ids or {},
            "top_three_comparison": disagreements,
            "classification_meaning": "predicted official classification, not pure pace",
            "probability_policy": "independent marginals; no normalization to sum to three",
        },
    )
