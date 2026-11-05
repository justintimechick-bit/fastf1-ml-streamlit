"""Metrics are calculated per weekend; summaries weight weekends equally."""

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss


def score_event(predictions: pd.DataFrame, labels: pd.DataFrame, task: str) -> dict:
    if predictions.event_id.nunique() != 1 or labels.event_id.nunique() != 1:
        raise ValueError("Metrics require exactly one complete weekend")
    joined = predictions.merge(
        labels, on=["event_id", "driver_id"], how="outer", validate="one_to_one", indicator=True
    )
    if not joined._merge.eq("both").all():
        raise ValueError("Prediction/label fields differ")
    result = {"event_id": str(joined.event_id.iloc[0]), "task": task, "field_size": len(joined)}
    if task == "podium":
        known = joined.podium.notna()
        y = joined.loc[known, "podium"].astype(float).to_numpy()
        p = joined.loc[known, "podium_probability"].astype(float).to_numpy()
        if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise ValueError("Invalid podium probability")
        result.update(
            label_coverage=float(known.mean()),
            brier=float(brier_score_loss(y, p)) if len(y) else None,
            log_loss=float(log_loss(y, np.clip(p, 1e-12, 1 - 1e-12), labels=[0, 1]))
            if len(y)
            else None,
        )
        return result
    known = joined.official_position.notna()
    subset = joined.loc[known]
    truth = subset.official_position.to_numpy(dtype=float)
    predicted = subset.predicted_position.to_numpy(dtype=float)
    n = len(truth)
    pairs = np.triu_indices(n, k=1)
    concordant = np.sign(truth[pairs[0]] - truth[pairs[1]]) == np.sign(
        predicted[pairs[0]] - predicted[pairs[1]]
    )
    complete = bool(known.all() and set(truth) == set(range(1, len(joined) + 1)))
    actual_top = set(subset.loc[subset.official_position.le(3), "driver_id"])
    predicted_top = set(joined.loc[joined.predicted_position.le(3), "driver_id"])
    result.update(
        label_coverage=float(known.mean()),
        complete_order=complete,
        position_mae=float(np.abs(truth - predicted).mean()) if complete else None,
        spearman=float(pd.Series(truth).corr(pd.Series(predicted), method="spearman"))
        if n > 1
        else None,
        pairwise_accuracy=float(concordant.mean()) if len(concordant) else None,
        top3_recall=len(actual_top & predicted_top) / len(actual_top) if actual_top else None,
        exact_order=bool(np.array_equal(truth, predicted)) if complete else None,
    )
    return result
