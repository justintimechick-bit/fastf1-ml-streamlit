"""Full-event official order metrics and marginal-probability reliability diagnostics."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss

from f1ml.evaluation.qualifying import complete_target_reason
from f1ml.models.qualifying import align_labels, validate_permutation
from f1ml.models.race import aligned_targets

RACE_METRICS = ("position_mae", "spearman", "winner_accuracy", "top3_overlap")
PODIUM_METRICS = (
    "log_loss",
    "brier",
    "average_precision",
    "ece",
    "precision",
    "recall",
    "f1",
    "raw_log_loss",
    "raw_brier",
    "raw_ece",
)


def target_reason(features, labels, task):
    if task == "race":
        reason = complete_target_reason(features, labels)
        return reason.replace("qualifying", "race") if reason else None
    try:
        aligned = aligned_targets(features, labels, "podium")
    except ValueError as exc:
        return str(exc)
    if aligned.podium.isna().any():
        return "missing_podium_targets"
    positive = aligned.loc[aligned.podium.eq(1), "official_position"]
    if positive.isna().any() or sorted(positive.tolist()) != list(
        range(1, min(3, len(features)) + 1)
    ):
        return "incomplete_or_contradictory_official_podium"
    return None


def race_score(predictions, labels, field):
    validate_permutation(predictions, field)
    aligned = align_labels(predictions, labels)
    joined = predictions.merge(aligned, on=["event_id", "driver_id"], validate="one_to_one")
    known = joined.official_position.notna() & joined.rank_target.notna()
    actual = joined.loc[known, "official_position"].astype(float)
    predicted = joined.loc[known, "predicted_position"].astype(float)
    complete = target_reason(field, labels, "race") is None
    top = joined.loc[joined.podium.eq(1), "driver_id"]
    predicted_top = joined.loc[joined.predicted_position.le(3), "driver_id"]
    winner = joined.loc[joined.official_position.eq(1), "driver_id"]
    return {
        "event_id": str(field.event_id.iloc[0]),
        "drivers": len(field),
        "label_coverage": float(known.mean()),
        "complete_order": complete,
        "position_mae": float((actual - predicted).abs().mean()) if complete else None,
        "spearman": float(actual.corr(predicted, method="spearman")) if known.sum() > 1 else None,
        "winner_accuracy": float(
            predictions.loc[predictions.predicted_position.eq(1), "driver_id"].iloc[0]
            == winner.iloc[0]
        )
        if len(winner) == 1
        else None,
        "top3_overlap": len(set(top) & set(predicted_top)) / len(top)
        if len(top) == min(3, len(field))
        else None,
    }


def reliability(probability, target, bins, weights=None):
    p, y = np.asarray(probability, float), np.asarray(target, float)
    weights = np.ones(len(p)) if weights is None else np.asarray(weights, float)
    if not len(p) or not np.isfinite(p).all() or not ((0 <= p) & (p <= 1)).all():
        raise ValueError("Reliability requires finite probabilities in [0, 1]")
    indices = np.minimum((p * bins).astype(int), bins - 1)
    rows, ece = [], 0.0
    for index in range(bins):
        mask = indices == index
        mass = float(weights[mask].sum())
        mean = float(np.average(p[mask], weights=weights[mask])) if mask.any() else None
        observed = float(np.average(y[mask], weights=weights[mask])) if mask.any() else None
        if mass:
            ece += mass / weights.sum() * abs(mean - observed)
        rows.append(
            {
                "bin": index,
                "lower": index / bins,
                "upper": (index + 1) / bins,
                "driver_rows": int(mask.sum()),
                "weight": mass,
                "mean_probability": mean,
                "observed_rate": observed,
            }
        )
    return float(ece), rows


def binary_scores(probability, target, bins):
    p, y = np.asarray(probability, float), np.asarray(target, float)
    ece, _ = reliability(p, y, bins)
    return {
        "log_loss": float(log_loss(y, np.clip(p, 1e-12, 1 - 1e-12), labels=[0, 1])),
        "brier": float(np.mean((p - y) ** 2)),
        "average_precision": float(average_precision_score(y, p)) if y.sum() else None,
        "ece": ece,
    }


def threshold_scores(probability, target, threshold):
    positive = np.asarray(probability, float) >= threshold
    y = np.asarray(target, float) == 1
    tp = int((positive & y).sum())
    precision = tp / positive.sum() if positive.any() else 0.0
    recall = tp / y.sum() if y.any() else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "predicted_positive_count": int(positive.sum()),
        "actual_positive_count": int(y.sum()),
    }


def podium_score(predictions, labels, field, bins):
    aligned = aligned_targets(predictions, labels, "podium")
    joined = predictions.merge(aligned, on=["event_id", "driver_id"], validate="one_to_one")
    known = joined.podium.notna()
    if not known.any():
        raise ValueError("No known podium labels to evaluate")
    joined = joined.loc[known]
    p, raw, y = (
        joined.podium_probability,
        joined.raw_podium_probability,
        joined.podium.astype(float),
    )
    result = {
        "event_id": str(field.event_id.iloc[0]),
        "drivers": len(field),
        "label_coverage": float(known.mean()),
        **binary_scores(p, y, bins),
    }
    result.update(
        {
            f"raw_{key}": value
            for key, value in binary_scores(raw, y, bins).items()
            if key != "average_precision"
        }
    )
    threshold = float(joined.podium_threshold.iloc[0])
    if not joined.podium_threshold.eq(threshold).all():
        raise ValueError("One development-selected threshold is required per event/model")
    result.update(
        threshold=threshold, probability_sum=float(p.sum()), **threshold_scores(p, y, threshold)
    )
    return result


def target_coverage(features, labels):
    aligned = align_labels(features, labels)
    columns = [
        c
        for c in (
            "event_id",
            "driver_id",
            "official_position",
            "rank_target",
            "podium",
            "outcome",
            "classification_code",
            "is_retired",
            "is_classified",
            "source_position",
            "source_positionText",
            "source_status",
        )
        if c in aligned
    ]
    result = aligned[columns].copy()
    result["race_training_eligible"] = aligned.rank_target.notna()
    result["podium_training_eligible"] = aligned.podium.notna()
    result["race_exclusion_reason"] = np.where(
        aligned.rank_target.isna(), "missing_or_ambiguous_official_position", ""
    )
    result["podium_exclusion_reason"] = np.where(aligned.podium.isna(), "unknown_podium_target", "")
    return result


def common_reliability(predictions, labels, event_ids, bins):
    rows = []
    subset = predictions.loc[predictions.task.eq("podium") & predictions.event_id.isin(event_ids)]
    for (family, strategy), group in subset.groupby(["family", "strategy"]):
        joined = group.merge(
            labels[["event_id", "driver_id", "podium"]],
            on=["event_id", "driver_id"],
            validate="one_to_one",
        )
        joined = joined.loc[joined.podium.notna()]
        weights = 1 / joined.groupby("event_id").driver_id.transform("size")
        for source in ("raw_podium_probability", "podium_probability"):
            ece, cells = reliability(joined[source], joined.podium, bins, weights)
            rows.extend(
                {
                    "family": family,
                    "strategy": strategy,
                    "probability_source": source,
                    "pooled_equal_event_ece": ece,
                    **cell,
                }
                for cell in cells
            )
    return pd.DataFrame(rows)
