"""Fit sigmoid calibration and thresholds on earlier chronological OOF predictions only."""

import numpy as np
from sklearn.linear_model import LogisticRegression

from f1ml.evaluation.race_metrics import threshold_scores
from f1ml.models.tabular import logits, positive_probability


def fit_postprocessing(artifact, oof, config, seed):
    events = [] if oof.empty else list(oof.event_id.unique())
    calibration = {
        "method": "identity",
        "events": events,
        "status": "insufficient_earlier_oof_events",
    }
    threshold = {
        "value": 0.5,
        "events": events,
        "status": "fixed_default_insufficient_oof_events",
        "objective": "maximum equal-event F1; closest to 0.5, then higher threshold",
    }
    if not oof.empty:
        if oof.duplicated(["event_id", "driver_id"]).any():
            raise ValueError("Duplicate postprocessing prediction keys")
        y = oof.podium.astype(float)
        raw = oof.raw_podium_probability.to_numpy(float)
        if len(events) >= config.min_calibration_events and y.nunique() == 2:
            weights = 1 / oof.groupby("event_id").driver_id.transform("size")
            weights = weights / weights.mean()
            calibrator = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
            calibrator.fit(logits(raw), y, sample_weight=weights.to_numpy())
            artifact.calibrator = calibrator
            calibration.update(
                method="sigmoid_of_raw_logit",
                status="fitted_earlier_oof",
                parameters=calibrator.get_params(),
                coefficient=calibrator.coef_.tolist(),
                intercept=calibrator.intercept_.tolist(),
            )
        elif y.nunique() < 2:
            calibration["status"] = "identity_single_class_oof"
        calibrated = (
            positive_probability(artifact.calibrator, logits(raw))
            if artifact.calibrator is not None
            else raw
        )
        if len(events) >= config.min_threshold_events:
            choices = []
            for value in config.thresholds:
                event_scores = [
                    threshold_scores(calibrated[group.index], group.podium, value)["f1"]
                    for _, group in oof.reset_index(drop=True).groupby("event_id")
                ]
                choices.append({"threshold": value, "mean_event_f1": float(np.mean(event_scores))})
            best = min(
                choices,
                key=lambda r: (-r["mean_event_f1"], abs(r["threshold"] - 0.5), -r["threshold"]),
            )
            artifact.threshold = float(best["threshold"])
            threshold.update(
                value=artifact.threshold, status="selected_earlier_oof", candidates=choices
            )
    artifact.metadata.update(calibration=calibration, threshold=threshold)
    return artifact
