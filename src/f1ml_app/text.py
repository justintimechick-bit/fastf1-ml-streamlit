"""Explanatory copy. Definitions follow the scoring code in f1ml.evaluation.* exactly."""

METRICS = {
    "position_mae": {
        "name": "Position MAE",
        "better": "lower",
        "what": "Mean absolute error in places between predicted and official position, "
        "averaged over drivers in one weekend.",
        "formula": r"\mathrm{MAE}=\frac{1}{N}\sum_{i=1}^{N}\lvert \hat p_i - p_i\rvert",
        "note": "Only computed when the official order is complete; a 2-place miss on every "
        "driver gives MAE 2.",
    },
    "spearman": {
        "name": "Spearman rank correlation",
        "better": "higher",
        "what": "Agreement between predicted and official orderings: 1 = identical order, "
        "0 = unrelated, −1 = reversed.",
        "formula": r"\rho = 1-\frac{6\sum_i d_i^2}{N(N^2-1)},\quad d_i=\hat p_i-p_i",
        "note": "Rewards getting the order right even when exact positions are off.",
    },
    "pole_accuracy": {
        "name": "Pole accuracy",
        "better": "higher",
        "what": "1 if the driver predicted P1 took official pole, else 0. Averaged over "
        "weekends it is the fraction of correct pole calls.",
        "formula": None,
        "note": "A hard, all-or-nothing metric; noisy with few weekends.",
    },
    "winner_accuracy": {
        "name": "Winner accuracy",
        "better": "higher",
        "what": "1 if the driver predicted P1 won the race, else 0.",
        "formula": None,
        "note": "Race-order model only.",
    },
    "top3_overlap": {
        "name": "Top-3 overlap",
        "better": "higher",
        "what": "Share of the official top three that the forecast also placed in its top "
        "three (order ignored).",
        "formula": r"\frac{\lvert \widehat{T}_3 \cap T_3\rvert}{3}",
        "note": "Top-10 overlap is the same with ten places (or the full field if smaller).",
    },
    "log_loss": {
        "name": "Log loss",
        "better": "lower",
        "what": "Average negative log-likelihood of the observed podium outcomes. Confident "
        "wrong probabilities are punished heavily.",
        "formula": r"-\frac{1}{N}\sum_i \big[y_i\log q_i+(1-y_i)\log(1-q_i)\big]",
        "note": "Used to select the podium model. Probabilities are clipped to [1e-12, 1−1e-12].",
    },
    "brier": {
        "name": "Brier score",
        "better": "lower",
        "what": "Mean squared difference between podium probability and the 0/1 outcome.",
        "formula": r"\frac{1}{N}\sum_i (q_i-y_i)^2",
        "note": "Bounded in [0, 1]; less sensitive than log loss to a single confident miss.",
    },
    "average_precision": {
        "name": "Average precision",
        "better": "higher",
        "what": "Area under the precision–recall curve: how well probabilities rank the "
        "actual podium finishers above everyone else.",
        "formula": None,
        "note": "Depends only on the ranking of probabilities, not their calibration.",
    },
    "ece": {
        "name": "Expected calibration error (ECE)",
        "better": "lower",
        "what": "Probabilities are grouped into 5 equal-width bins; ECE is the "
        "driver-weighted average gap between mean probability and observed podium rate.",
        "formula": r"\sum_b \frac{n_b}{N}\,\big\lvert \bar q_b-\bar y_b\big\rvert",
        "note": "With a handful of weekends most bins are nearly empty, so ECE cannot "
        "establish calibration quality here.",
    },
    "precision": {
        "name": "Precision / recall / F1",
        "better": "higher",
        "what": "Treat probability ≥ the development-selected threshold as a podium call. "
        "Precision = correct calls / calls; recall = correct calls / actual podiums; "
        "F1 is their harmonic mean.",
        "formula": None,
        "note": "The threshold was chosen on earlier development weekends, never on the test.",
    },
}


def _metric_spec(value: str) -> dict | None:
    text = value.casefold().replace("_", " ")
    for key, spec in METRICS.items():
        if key.replace("_", " ") == text or spec["name"].casefold() == text:
            return spec
    for term, key in (
        ("mae", "position_mae"),
        ("spearman", "spearman"),
        ("brier", "brier"),
        ("log loss", "log_loss"),
    ):
        if term in text:
            return METRICS[key]
    return None


def metric_label(value: str) -> str:
    spec = _metric_spec(value)
    if spec and "is better" not in value:
        value = value.replace(" ↓", "").replace(" ↑", "")
        return f"{value} {'↓ lower' if spec['better'] == 'lower' else '↑ higher'} is better"
    return value


def metric_help(value: str) -> str | None:
    spec = _metric_spec(value)
    return f"{spec['what']} {spec['better'].capitalize()} is better." if spec else None


STRATEGIES = {
    "2026_only": (
        "A · 2026-only",
        "Train only on earlier 2026 weekends. Needs at least two of them, so the first two "
        "2026 validation folds are ineligible rather than silently shrunk.",
    ),
    "equal": (
        "B · Pooled 2024–26 (equal)",
        "Train on all earlier 2024–2026 weekends with equal weight per weekend.",
    ),
    "weighted": (
        "C · Weighted history (2026-favored)",
        "All earlier weekends, weighted by season (2024 = 0.5, 2025 = 1, 2026 = tuned 2 or 4) "
        "times recency decay 2^(−age / half-life), half-life tuned over 6 or 12 weekends.",
    ),
}

FAMILIES = {
    "ridge": (
        "Ridge regression",
        "Linear model with an L2 penalty (alpha). Predicts the normalized official rank; "
        "drivers are then ordered by the raw score.",
    ),
    "random_forest": (
        "Random forest",
        "Average of 80 bootstrapped decision trees (depth 4 or 8, min leaf 2). Captures "
        "nonlinear interactions; regressor for order, classifier for podium.",
    ),
    "gradient_boosting": (
        "Histogram gradient boosting",
        "100 boosting iterations (learning rate 0.05 or 0.1) of small trees (15 leaves, "
        "min leaf 5, L2 = 1), each fitting the previous ensemble's errors. No random "
        "internal holdout and no early stopping.",
    ),
    "logistic": (
        "Logistic regression",
        "Linear classifier (C = 0.1, 1 or 10) giving a podium probability per driver.",
    ),
}

PREPROCESSING = (
    "Every learned model shares one fold-local pipeline: numeric median imputation with "
    "missing-value indicators and standard scaling, plus one-hot encoding of driver, team "
    "history and circuit (unknown categories are ignored). These transforms are fitted on "
    "training weekends only."
)

CUTOFF_RULES = [
    {
        "Predictor": "Qualifying",
        "Cutoff": "Strictly before the scheduled (and any observed) GP qualifying start",
        "Allowed weekend inputs": "Reviewed entrants known by the cutoff; completed practice "
        "available before the cutoff; history from earlier weekends",
        "Forbidden": "Qualifying results/times/classification; any race or Sprint observation",
    },
    {
        "Predictor": "Race order + podium",
        "Cutoff": "After official GP qualifying is available, strictly before race start",
        "Allowed weekend inputs": "Everything above plus official GP qualifying classification "
        "and pace; a grid only if independently evidenced before the cutoff",
        "Forbidden": "Race laps, telemetry, weather, pit stops, results, status; "
        "Race.results grid; Sprint observations",
    },
]

LIMITATIONS = [
    "The only frozen experiment uses **synthetic fixtures**. Its scores validate the "
    "software and protocol, not real Formula 1 accuracy. The fixture deliberately makes "
    "race order equal qualifying order, which explains near-perfect race and podium scores.",
    "The real FastF1 catalog holds official Q/R results for part of 2024–2026, but "
    "practice/qualifying laps and weather were unavailable (timing archive HTTP 401, result "
    "API rate limits), and it lacks reviewed cutoff rosters. No real-data model has been "
    "trained or evaluated.",
    "Backtests are **historical reconstructions**: data downloaded later may include "
    "corrections that were not public at the original cutoff.",
    "Only one development weekend (2026-03) is common to all strategies, so family and "
    "weighting selection rests on very weak evidence.",
    "Four final-test weekends are too few for confidence intervals; event ranges and "
    "leave-one-event-out ranges are sensitivity summaries only.",
    "The final-test block has been consumed once. Any change evaluated on it is exploratory "
    "and needs a fresh future holdout.",
]
