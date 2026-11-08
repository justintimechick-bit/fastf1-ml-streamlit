"""Cutoff-safe, real-only historical experiments backed by reviewed FIA evidence.

This is a retrospective teaching laboratory. Previously consulted results are
never represented as an untouched final holdout. The existing evaluation ledger
and its sealed software/demo experiment are not opened or changed here.
"""

from __future__ import annotations

import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from f1ml.storage import (
    digest,
    file_digest,
    new_directory,
    read_json,
    save_table,
    seal_directory,
    verify_directory,
    write_json,
)

FAMILIES = ("linear", "random_forest", "gradient_boosting")
TASKS = ("qualifying", "race", "podium")
CATEGORICAL_FEATURES = ("driver_id", "team_id")
CORE_NUMERIC_FEATURES = (
    ("field_size", "season_round", "history_event_count")
    + tuple(
        f"{entity}_{session}_{stat}_{window}"
        for entity in ("driver", "team")
        for session in ("qualifying", "race")
        for stat in ("rank_mean", "rank_count", "podium_rate", "nonclassification_rate")
        for window in (3, 5)
    )
    + ("driver_qualifying_previous_rank", "driver_race_previous_rank")
)
RACE_NUMERIC_FEATURES = (
    "current_qualifying_rank",
    "current_qualifying_rank_known",
    "current_qualifying_nonclassified",
)
NUMERIC_FEATURES = CORE_NUMERIC_FEATURES + RACE_NUMERIC_FEATURES
PROTOCOL = {
    "version": "real-historical-lab-v1",
    "mode": "historical_reconstruction_exploratory",
    "exposure": "Historical outcomes have been consulted; no untouched holdout claim",
    "training_scope": "2026-only, earlier FIA evidence available before target weekend",
    "eligibility": (
        "independent cutoff-valid entry lists and reviewed final labels; unknown labels masked"
    ),
    "history": "strict earlier available calendar-event windows, not last driver appearances",
    "target": (
        "explicit official numeric rank normalized by independent field size; "
        "unknown/NC/DSQ ranks excluded"
    ),
    "features": list(CATEGORICAL_FEATURES + NUMERIC_FEATURES),
    "missing_groups_excluded": ["practice_laps", "qualifying_laps", "weather", "pace"],
    "preprocessing": (
        "observed/nonconstant feature selection, median imputation, "
        "scale, one-hot fitted on training only"
    ),
    "training_weight": "each event has equal total sample weight within each task",
    "selection": (
        "at most last three pre-calibration development events, expanding "
        "chronological training; minimum three earlier training events"
    ),
    "calibration": (
        "last two earlier events reserved for out-of-fold sigmoid calibration "
        "and threshold when at least eight earlier events; otherwise identity and 0.5"
    ),
    "selection_metric": "event-mean rank MAE for ranks; event-mean Brier for podium",
    "threshold": "calibration-OOF F1 on 0.10..0.90 by 0.05; ties nearest 0.5",
    "baseline_qualifying": (
        "last-three-event driver qualifying form, then team form, then "
        "neutral; deterministic driver-id tie break"
    ),
    "baseline_race": (
        "published target qualifying rank, then prior driver race form, "
        "then team form, then neutral"
    ),
    "baseline_podium": (
        "0.65 for race-baseline top three; 1.05/(field_size-3) otherwise, fixed before evaluation"
    ),
    "evaluation": (
        "rank metrics use only explicitly numbered targets; all field drivers "
        "predicted; unknown podium labels masked; metrics macro-average events"
    ),
    "prediction_seal": (
        "config and forecast hashes frozen before target labels are joined or scored"
    ),
    "search": {
        "linear_rank_alpha": [1.0, 10.0],
        "linear_podium_C": [0.1, 1.0],
        "random_forest_max_depth": [4, None],
        "gradient_boosting_max_leaf_nodes": [7, 15],
    },
}


def _utc(value):
    return pd.Timestamp(value).tz_convert("UTC")


class RealDataset:
    """Verified facts, independent fields and conservative document timestamps."""

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.receipt = verify_directory(self.root)
        if self.receipt.get("data_kind") != "real":
            raise ValueError("The historical lab accepts only verified real-data snapshots")
        for name in ("events", "rosters", "labels", "documents"):
            setattr(self, name, pd.read_parquet(self.root / f"{name}.parquet"))
        self.events = self.events.sort_values(["year", "round"]).reset_index(drop=True)
        if self.events.event_id.duplicated().any():
            raise ValueError("Repeated event identity")
        if self.rosters.duplicated(["event_id", "task", "driver_id"]).any():
            raise ValueError("Repeated independent roster identity")
        if self.labels.duplicated(["event_id", "task", "driver_id"]).any():
            raise ValueError("Repeated target identity")
        if (
            pd.to_datetime(self.rosters.known_at, utc=True, format="ISO8601")
            > pd.to_datetime(self.rosters.cutoff, utc=True, format="ISO8601")
        ).any():
            raise ValueError("An independent roster was published after its cutoff")
        known = self.labels.official_position.dropna()
        if ((known < 1) | (known % 1 != 0)).any():
            raise ValueError("Official rank must be an explicitly printed positive integer")
        self._history_cache = {}

    def event(self, event_id: str) -> pd.Series:
        found = self.events.loc[self.events.event_id.eq(event_id)]
        if len(found) != 1:
            raise ValueError(f"Event {event_id!r} is absent from this reviewed real snapshot")
        return found.iloc[0]

    def history_events(self, event_id: str) -> pd.DataFrame:
        """Prior same-season events whose latest selected labels were available."""
        target = self.event(event_id)
        boundary = min(_utc(target.weekend_start), _utc(target.qualifying_cutoff))
        mask = (
            self.events.year.eq(target.year)
            & (pd.to_datetime(self.events.weekend_start, utc=True) < _utc(target.weekend_start))
            & (pd.to_datetime(self.events.race_available_at, utc=True) < boundary)
            & (pd.to_datetime(self.events.qualifying_available_at, utc=True) < boundary)
        )
        return self.events.loc[mask].copy().reset_index(drop=True)

    def _calendar_events(self, event_id):
        target = self.event(event_id)
        return self.events.loc[
            self.events.year.eq(target.year)
            & (pd.to_datetime(self.events.weekend_start, utc=True) < _utc(target.weekend_start))
        ].copy()

    def _history_observations(self, ids: list[str], task: str, boundary) -> pd.DataFrame:
        key = (tuple(ids), task, str(boundary))
        if key not in self._history_cache:
            roster = self.rosters.loc[
                self.rosters.event_id.isin(ids) & self.rosters.task.eq(task)
            ].copy()
            labels = self.labels.loc[
                self.labels.event_id.isin(ids)
                & self.labels.task.eq(task)
                & (pd.to_datetime(self.labels.available_at, utc=True) < boundary)
            ]
            frame = roster.merge(
                labels[["event_id", "driver_id", "official_position", "outcome", "podium"]],
                on=["event_id", "driver_id"],
                how="left",
                validate="one_to_one",
            )
            field = frame.groupby("event_id").driver_id.transform("size")
            frame["rank_fraction"] = (frame.official_position - 1) / (field - 1)
            frame["nonclassification"] = frame.outcome.map(
                lambda x: np.nan if pd.isna(x) or x == "unknown" else float(x != "ordered")
            )
            self._history_cache[key] = frame
        return self._history_cache[key]

    def feature_frame(self, event_id: str, task: str, history_ids=None):
        """Build features without reading the target task's outcomes.

        A race/podium forecast may read this event's final qualifying label only
        after its official document publication and before the race cutoff.
        """
        if task not in TASKS:
            raise ValueError(f"Unknown task {task}")
        source_task = "race" if task == "podium" else task
        event = self.event(event_id)
        cutoff = _utc(event[f"{source_task}_cutoff"])
        eligible = self.history_events(event_id)
        calendar_ids = self._calendar_events(event_id).event_id.tolist()
        ids = calendar_ids
        boundary = min(_utc(event.weekend_start), cutoff)
        if history_ids is not None:
            requested = list(history_ids)
            if len(requested) != len(set(requested)) or not set(requested).issubset(ids):
                raise ValueError(
                    "Feature history includes an event unavailable before this weekend"
                )
            ids = [eid for eid in ids if eid in requested]
        frame = (
            self.rosters.loc[self.rosters.event_id.eq(event_id) & self.rosters.task.eq(source_task)]
            .sort_values("driver_id")
            .copy()
            .reset_index(drop=True)
        )
        if frame.empty:
            raise ValueError("No independent pre-cutoff field")
        if (pd.to_datetime(frame.known_at, utc=True) > cutoff).any():
            raise ValueError("Roster information crosses prediction cutoff")
        frame["field_size"] = len(frame)
        frame["season_round"] = int(event["round"])
        frame["history_event_count"] = len(ids)
        for session in ("qualifying", "race"):
            for window in (3, 5):
                # Choose the calendar window before masking unavailable components.
                # A later document revision must not extend it to older races.
                window_ids = [eid for eid in calendar_ids[-window:] if eid in ids]
                observations = self._history_observations(window_ids, session, boundary)
                for entity in ("driver", "team"):
                    key = f"{entity}_id"
                    if observations.empty:
                        grouped = pd.DataFrame()
                    else:
                        grouped = observations.groupby(key).agg(
                            rank_mean=("rank_fraction", "mean"),
                            rank_count=("rank_fraction", "count"),
                            podium_rate=("podium", "mean"),
                            nonclassification_rate=("nonclassification", "mean"),
                        )
                    for stat in (
                        "rank_mean",
                        "rank_count",
                        "podium_rate",
                        "nonclassification_rate",
                    ):
                        name = f"{entity}_{session}_{stat}_{window}"
                        frame[name] = frame[key].map(grouped[stat]) if not grouped.empty else np.nan
                        if stat == "rank_count":
                            frame[name] = frame[name].fillna(0.0)
            previous = self._history_observations(
                [eid for eid in calendar_ids[-1:] if eid in ids], session, boundary
            )
            rank = previous.set_index("driver_id").rank_fraction if not previous.empty else {}
            frame[f"driver_{session}_previous_rank"] = frame.driver_id.map(rank)
        current_q_used = False
        if source_task == "race":
            if _utc(event.qualifying_available_at) >= cutoff:
                raise ValueError(
                    "Final qualifying was unavailable at the post-qualifying race cutoff"
                )
            q = self.labels.loc[
                self.labels.event_id.eq(event_id) & self.labels.task.eq("qualifying")
            ]
            if (pd.to_datetime(q.available_at, utc=True) >= cutoff).any():
                raise ValueError("Qualifying labels cross race cutoff")
            qfield = self.rosters.loc[
                self.rosters.event_id.eq(event_id) & self.rosters.task.eq("qualifying")
            ]
            q = q.set_index("driver_id")
            frame["current_qualifying_rank"] = frame.driver_id.map(q.official_position) / max(
                len(qfield), 1
            )
            frame["current_qualifying_rank_known"] = frame.current_qualifying_rank.notna().astype(
                float
            )
            frame["current_qualifying_nonclassified"] = frame.driver_id.map(q.outcome).map(
                lambda x: np.nan if pd.isna(x) or x == "unknown" else float(x != "ordered")
            )
            current_q_used = True
        audit = {
            "event_id": event_id,
            "task": task,
            "cutoff": cutoff.isoformat(),
            "history_event_ids": eligible.event_id.tolist(),
            "calendar_history_event_ids": calendar_ids,
            "feature_history_event_ids": ids,
            "calendar_window_event_ids": {str(w): calendar_ids[-w:] for w in (3, 5)},
            "history_boundary": _utc(event.weekend_start).isoformat(),
            "history_max_race_available_at": (
                eligible.loc[eligible.event_id.isin(ids), "race_available_at"].max()
                if not eligible.empty
                else None
            ),
            "current_qualifying_used": current_q_used,
            "current_qualifying_available_at": event.qualifying_available_at
            if current_q_used
            else None,
            "roster_sources": sorted(frame.source.unique().tolist()),
            "roster_known_at": sorted(frame.known_at.unique().tolist()),
            "field_size": len(frame),
            "feature_schema": list(CATEGORICAL_FEATURES)
            + [c for c in NUMERIC_FEATURES if c in frame.columns],
            "unavailable_feature_groups_excluded": PROTOCOL["missing_groups_excluded"],
            "history_windows": (
                "last three/five prior calendar events; absent drivers do not extend window"
            ),
        }
        return frame, audit


class ObservedPreprocessor(BaseEstimator, TransformerMixin):
    """Select observed, varying columns using *only* the supplied training rows."""

    def fit(self, X, y=None):
        self.numeric_features_, self.categorical_features_, self.excluded_features_ = [], [], {}
        for feature in NUMERIC_FEATURES:
            if feature not in X or X[feature].notna().sum() == 0:
                self.excluded_features_[feature] = "entirely_missing_in_training"
            elif X[feature].dropna().nunique() <= 1:
                self.excluded_features_[feature] = "constant_observed_training_values"
            else:
                self.numeric_features_.append(feature)
        for feature in CATEGORICAL_FEATURES:
            if feature not in X or X[feature].dropna().nunique() <= 1:
                self.excluded_features_[feature] = "missing_or_constant_in_training"
            else:
                self.categorical_features_.append(feature)
        transforms = []
        if self.numeric_features_:
            transforms.append(
                (
                    "numeric",
                    Pipeline(
                        [
                            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                            ("scale", StandardScaler()),
                        ]
                    ),
                    self.numeric_features_,
                )
            )
        if self.categorical_features_:
            transforms.append(
                (
                    "categorical",
                    Pipeline(
                        [
                            ("impute", SimpleImputer(strategy="most_frequent")),
                            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                        ]
                    ),
                    self.categorical_features_,
                )
            )
        if not transforms:
            raise ValueError("No observed, varying training features")
        self.transformer_ = ColumnTransformer(transforms, remainder="drop")
        self.transformer_.fit(X)
        return self

    def transform(self, X):
        return self.transformer_.transform(X)


def _grid(task, family):
    if family == "linear":
        return (
            [{"C": c} for c in (0.1, 1.0)]
            if task == "podium"
            else [{"alpha": a} for a in (1.0, 10.0)]
        )
    if family == "random_forest":
        return [{"max_depth": d} for d in (4, None)]
    return [{"max_leaf_nodes": n} for n in (7, 15)]


def _estimator(task, family, params, seed):
    classifier = task == "podium"
    if family == "linear":
        return (
            LogisticRegression(**params, max_iter=1500, random_state=seed)
            if classifier
            else Ridge(**params)
        )
    if family == "random_forest":
        cls = RandomForestClassifier if classifier else RandomForestRegressor
        return cls(
            **params,
            n_estimators=100,
            min_samples_leaf=3,
            max_features=0.8,
            n_jobs=1,
            random_state=seed,
        )
    cls = HistGradientBoostingClassifier if classifier else HistGradientBoostingRegressor
    return cls(
        **params,
        learning_rate=0.06,
        max_iter=100,
        min_samples_leaf=8,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=seed,
    )


def _weights(frame):
    # Mean weight one, with identical total event mass despite masked labels.
    sizes = frame.groupby("event_id").driver_id.transform("size")
    return len(frame) / (frame.event_id.nunique() * sizes.to_numpy())


def _fit(frame, task, family, params, seed):
    target = "_podium_target" if task == "podium" else "_rank_target"
    observed = frame.loc[frame[target].notna()].copy()
    if observed.empty or (task == "podium" and observed[target].nunique() != 2):
        raise ValueError(f"Insufficient observed {task} targets")
    model = Pipeline(
        [
            ("preprocess", ObservedPreprocessor()),
            ("model", _estimator(task, family, params, seed)),
        ]
    )
    model.fit(observed, observed[target], model__sample_weight=_weights(observed))
    return model


def _ordered(scores, identities):
    order = np.lexsort((np.asarray(identities), np.asarray(scores)))
    rank = np.empty(len(order), dtype=float)
    rank[order] = np.arange(1, len(order) + 1)
    return rank


def _forecast(model, frame, task, family, calibration=None, threshold=0.5):
    columns = ["event_id", "driver_id", "driver", "team_id", "abbreviation"]
    result = frame[columns].copy()
    result["task"], result["family"] = task, family
    if task == "podium":
        raw = model.predict_proba(frame)[:, 1]
        calibrated = _calibrate(raw, calibration)
        result["raw_podium_probability"] = raw
        result["podium_probability"] = calibrated
        result["podium_prediction"] = (calibrated >= threshold).astype(int)
        result["threshold"] = threshold
        result["predicted_rank"] = _ordered(-calibrated, result.driver_id)
        result["predicted_position"] = np.nan
    else:
        scores = np.clip(model.predict(frame), 0, 1)
        result["predicted_position"] = 1 + scores * (len(frame) - 1)
        result["predicted_rank"] = _ordered(scores, result.driver_id)
        result["raw_podium_probability"] = np.nan
        result["podium_probability"] = np.nan
        result["podium_prediction"] = np.nan
        result["threshold"] = np.nan
    return result


def _baseline(frame, task):
    result = frame[["event_id", "driver_id", "driver", "team_id", "abbreviation"]].copy()
    result["task"], result["family"] = task, "baseline"
    source = "qualifying" if task == "qualifying" else "race"
    scores = (
        frame[f"driver_{source}_rank_mean_3"]
        .fillna(frame[f"team_{source}_rank_mean_3"])
        .fillna(0.5)
    )
    if task != "qualifying":
        # Qualifying is normalized to [1/N, 1], historical ranks to [0, 1].
        q = (frame.current_qualifying_rank * len(frame) - 1) / (len(frame) - 1)
        scores = q.fillna(scores)
    result["predicted_rank"] = _ordered(scores, result.driver_id)
    result["predicted_position"] = 1 + scores.to_numpy() * (len(frame) - 1)
    result["raw_podium_probability"] = np.nan
    result["podium_probability"] = np.nan
    result["podium_prediction"] = np.nan
    result["threshold"] = np.nan
    if task == "podium":
        result["podium_probability"] = np.where(
            result.predicted_rank <= 3, 0.65, 1.05 / (len(frame) - 3)
        )
        result["raw_podium_probability"] = result.podium_probability
        result["podium_prediction"] = (result.podium_probability >= 0.5).astype(int)
        result["threshold"] = 0.5
    return result


def _label_training(dataset, frame, task):
    source = "race" if task == "podium" else task
    labels = dataset.labels.loc[
        dataset.labels.task.eq(source),
        [
            "event_id",
            "driver_id",
            "official_position",
            "podium",
            "outcome",
            "available_at",
            "source",
        ],
    ]
    joined = frame.merge(
        labels,
        on=["event_id", "driver_id"],
        how="left",
        validate="one_to_one",
        suffixes=("_roster", "_label"),
    )
    joined["_rank_target"] = (joined.official_position - 1) / (joined.field_size - 1)
    joined["_podium_target"] = joined.podium
    return joined


def _attach_actuals(dataset, forecasts):
    pieces = []
    for task in forecasts.task.unique():
        labels = dataset.labels.loc[
            dataset.labels.task.eq("race" if task == "podium" else task),
            [
                "event_id",
                "driver_id",
                "official_position",
                "podium",
                "outcome",
                "available_at",
                "source",
            ],
        ].rename(
            columns={
                "official_position": "actual_rank",
                "podium": "actual_podium",
                "available_at": "label_available_at",
                "source": "label_source",
            }
        )
        part = forecasts.loc[forecasts.task.eq(task)].merge(
            labels, on=["event_id", "driver_id"], how="left", validate="many_to_one"
        )
        pieces.append(part)
    return pd.concat(pieces, ignore_index=True)


def _metric_rows(predictions, stage):
    """Long metrics, averaging per event, with known/total label counts."""
    rows = []
    if predictions.empty:
        return pd.DataFrame(
            columns=[
                "task",
                "family",
                "metric",
                "value",
                "event_count",
                "known_label_count",
                "total_driver_count",
                "coverage",
                "stage",
            ]
        )
    for (task, family), block in predictions.groupby(["task", "family"], sort=False):
        event_metrics, known_count = [], 0
        for _, frame in block.groupby("event_id", sort=False):
            if task == "podium":
                observed = frame.loc[frame.actual_podium.notna()]
                known_count += len(observed)
                if observed.empty:
                    continue
                y = observed.actual_podium.to_numpy()
                p = np.clip(observed.podium_probability.to_numpy(), 1e-8, 1 - 1e-8)
                decision = observed.podium_prediction.to_numpy()
                bins = np.minimum((p * 5).astype(int), 4)
                ece = sum(
                    abs(y[bins == b].mean() - p[bins == b].mean()) * (bins == b).mean()
                    for b in range(5)
                    if (bins == b).any()
                )
                event_metrics.append(
                    {
                        "brier": brier_score_loss(y, p),
                        "log_loss": log_loss(y, p, labels=[0, 1]),
                        "ece": ece,
                        "precision": precision_score(y, decision, zero_division=0),
                        "recall": recall_score(y, decision, zero_division=0),
                        "f1": f1_score(y, decision, zero_division=0),
                    }
                )
            else:
                observed = frame.loc[frame.actual_rank.notna()]
                known_count += len(observed)
                if observed.empty:
                    continue
                errors = observed.predicted_rank - observed.actual_rank
                actual_top3 = set(observed.loc[observed.actual_rank.le(3), "driver_id"])
                predicted_top3 = set(frame.loc[frame.predicted_rank.le(3), "driver_id"])
                event_metrics.append(
                    {
                        "mae": errors.abs().mean(),
                        "rmse": np.sqrt(np.mean(errors**2)),
                        "spearman": observed.predicted_rank.corr(
                            observed.actual_rank, method="spearman"
                        )
                        if len(observed) > 1
                        else np.nan,
                        "top3_recall": len(actual_top3 & predicted_top3) / 3
                        if len(actual_top3) == 3
                        else np.nan,
                    }
                )
        if not event_metrics:
            continue
        for metric, value in pd.DataFrame(event_metrics).mean().items():
            rows.append(
                {
                    "task": task,
                    "family": family,
                    "metric": metric,
                    "value": float(value),
                    "event_count": len(event_metrics),
                    "known_label_count": known_count,
                    "total_driver_count": len(block),
                    "coverage": known_count / len(block),
                    "stage": stage,
                }
            )
    return pd.DataFrame(rows)


def _logits(probabilities):
    p = np.clip(np.asarray(probabilities), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def _calibrate(probabilities, calibration):
    if calibration is None or calibration.get("method") != "sigmoid":
        return np.asarray(probabilities)
    z = _logits(probabilities).ravel() * calibration["coefficient"] + calibration["intercept"]
    return 1 / (1 + np.exp(-np.clip(z, -35, 35)))


def _fit_calibration(predictions):
    observed = predictions.loc[predictions.actual_podium.notna()]
    if observed.event_id.nunique() < 2 or observed.actual_podium.nunique() < 2:
        return {
            "method": "identity",
            "reason": "fewer than two independent calibration events or classes",
        }, 0.5
    calibrator = LogisticRegression(C=1e6, max_iter=1500, random_state=42)
    calibrator.fit(
        _logits(observed.raw_podium_probability),
        observed.actual_podium,
        sample_weight=_weights(observed),
    )
    config = {
        "method": "sigmoid",
        "coefficient": float(calibrator.coef_[0, 0]),
        "intercept": float(calibrator.intercept_[0]),
        "event_ids": observed.event_id.drop_duplicates().tolist(),
    }
    probabilities = _calibrate(observed.raw_podium_probability, config)
    candidates = [
        (float(t), f1_score(observed.actual_podium, probabilities >= t, zero_division=0))
        for t in np.arange(0.1, 0.901, 0.05)
    ]
    threshold = min(candidates, key=lambda item: (-item[1], abs(item[0] - 0.5), item[0]))[0]
    return config, threshold


def _configuration(dataset, event_id, last_n, seed):
    return {
        "protocol": PROTOCOL,
        "source_snapshot_id": dataset.receipt["id"],
        "target_event_id": event_id,
        "last_n": last_n,
        "seed": seed,
        "implementation_id": file_digest(Path(__file__)),
        "runtime": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }


def _fold_training_ids(dataset, selected_ids, event_id):
    """Availability, including later classification revisions, applies per fold."""
    eligible = set(dataset.history_events(event_id).event_id)
    return [eid for eid in selected_ids if eid in eligible]


def run_backtest(
    dataset_root: Path, event_id: str, output: Path, *, last_n: int | None = None, seed=42
):
    """Fit and seal one explicit retrospective run; never overwrite an old run."""
    with threadpool_limits(limits=1):
        return _run_backtest(Path(dataset_root), event_id, Path(output), last_n=last_n, seed=seed)


def _run_backtest(dataset_root, event_id, output, *, last_n, seed, live_tasks=None):
    dataset = RealDataset(dataset_root)
    event = dataset.event(event_id)
    tasks = tuple(live_tasks) if live_tasks is not None else TASKS
    if live_tasks is not None:
        from f1ml_real.refresh import forecast_tasks

        if not tasks or not set(tasks).issubset(forecast_tasks(dataset, event_id)):
            raise ValueError("Forecast inputs are unavailable or the target session has started")
    elif pd.isna(event.race_available_at) or pd.isna(event.qualifying_available_at):
        raise ValueError("Historical tests require completed qualifying and race classifications")
    if last_n is not None and (isinstance(last_n, bool) or last_n < 1):
        raise ValueError("last_n must be a positive number of earlier eligible events")
    training = dataset.history_events(event_id)
    if last_n is not None:
        training = training.tail(last_n).copy().reset_index(drop=True)
    ids = training.event_id.tolist()
    if not ids:
        raise ValueError("No earlier published real events are available for training")
    base_config = _configuration(dataset, event_id, last_n, seed)
    if live_tasks is not None:
        base_config = {**base_config, "mode": "live_forecast", "forecast_tasks": list(tasks)}
    run_id = digest(base_config)
    new_directory(output)
    write_json(output / "protocol.json", {**base_config, "id": run_id})
    save_table(output / "training_events.parquet", training)
    matrices, target_frames, audits = {}, {}, {}
    for task in tasks:
        parts = []
        for eid in ids:
            frame, audit = dataset.feature_frame(eid, task)
            parts.append(_label_training(dataset, frame, task))
            audits[f"{eid}/{task}"] = audit
        matrices[task] = pd.concat(parts, ignore_index=True)
        target_frames[task], audits[f"{event_id}/{task}"] = dataset.feature_frame(event_id, task)
        save_table(output / f"training_{task}.parquet", matrices[task])
    # Reserve these later events before searching; all choices depend only on chronology.
    calibration_ids = ids[-2:] if len(ids) >= 8 else []
    tuning_pool = ids[:-2] if calibration_ids else ids
    development_ids = [
        eid for eid in tuning_pool if len(_fold_training_ids(dataset, ids, eid)) >= 3
    ][-3:]
    selection, models, calibration, thresholds = {}, {}, {}, {}
    development_predictions, calibration_predictions, search_results = [], [], []
    exclusions = {}
    for task in tasks:
        matrix = matrices[task]
        for family in FAMILIES:
            candidates = []
            candidate_predictions = []
            for candidate in _grid(task, family):
                pieces = []
                for dev_id in development_ids:
                    earlier = _fold_training_ids(dataset, ids, dev_id)
                    model = _fit(
                        matrix.loc[matrix.event_id.isin(earlier)], task, family, candidate, seed
                    )
                    forecast = _forecast(
                        model, matrix.loc[matrix.event_id.eq(dev_id)], task, family
                    )
                    pieces.append(_attach_actuals(dataset, forecast))
                if pieces:
                    predictions = pd.concat(pieces, ignore_index=True)
                    metrics = _metric_rows(predictions, "development_selection")
                    metric = "brier" if task == "podium" else "mae"
                    loss = float(metrics.loc[metrics.metric.eq(metric), "value"].iloc[0])
                else:
                    predictions, loss = pd.DataFrame(), None
                candidates.append(loss)
                candidate_predictions.append(predictions)
                search_results.append(
                    {
                        "task": task,
                        "family": family,
                        "params": candidate,
                        "development_event_ids": development_ids,
                        "fold_training_event_ids": {
                            eid: _fold_training_ids(dataset, ids, eid) for eid in development_ids
                        },
                        "loss": loss,
                        "metric": "brier" if task == "podium" else "mae",
                    }
                )
            winner = int(np.argmin(candidates)) if development_ids else 0
            params = _grid(task, family)[winner]
            key = f"{task}/{family}"
            selection[key] = params
            if not candidate_predictions[winner].empty:
                development_predictions.append(candidate_predictions[winner])
            if task == "podium":
                oof = []
                for cal_id in calibration_ids:
                    earlier = _fold_training_ids(dataset, ids, cal_id)
                    cal_model = _fit(
                        matrix.loc[matrix.event_id.isin(earlier)], task, family, params, seed
                    )
                    forecast = _forecast(
                        cal_model, matrix.loc[matrix.event_id.eq(cal_id)], task, family
                    )
                    oof.append(_attach_actuals(dataset, forecast))
                cal_frame = pd.concat(oof, ignore_index=True) if oof else pd.DataFrame()
                if not cal_frame.empty:
                    calibration[key], thresholds[key] = _fit_calibration(cal_frame)
                    # Calibration diagnostics are in-sample to the calibration map,
                    # and explicitly distinct from selection and target evaluation.
                    raw_frame = cal_frame.copy()
                    raw_frame["family"] += "_uncalibrated"
                    calibration_predictions.append(raw_frame)
                    cal_frame["podium_probability"] = _calibrate(
                        cal_frame.raw_podium_probability, calibration[key]
                    )
                    cal_frame["podium_prediction"] = (
                        cal_frame.podium_probability >= thresholds[key]
                    ).astype(int)
                    cal_frame["threshold"] = thresholds[key]
                    calibration_predictions.append(cal_frame)
                else:
                    calibration[key] = {
                        "method": "identity",
                        "reason": (
                            "fewer than eight earlier events for separate chronological calibration"
                        ),
                    }
                    thresholds[key] = 0.5
            model = _fit(matrix, task, family, params, seed)
            models[key] = model
            exclusions[key] = model.named_steps["preprocess"].excluded_features_
    # Baseline development predictions use the same event membership as model selection.
    for task in tasks:
        for dev_id in development_ids:
            frame = matrices[task].loc[matrices[task].event_id.eq(dev_id)]
            development_predictions.append(_attach_actuals(dataset, _baseline(frame, task)))
    dev_predictions = (
        pd.concat(development_predictions, ignore_index=True)
        if development_predictions
        else pd.DataFrame()
    )
    cal_predictions = (
        pd.concat(calibration_predictions, ignore_index=True)
        if calibration_predictions
        else pd.DataFrame()
    )
    dev_metrics = _metric_rows(dev_predictions, "development_selection")
    cal_metrics = _metric_rows(cal_predictions, "development_calibration_fit_diagnostic")
    save_table(output / "development_predictions.parquet", dev_predictions)
    save_table(output / "calibration_predictions.parquet", cal_predictions)
    save_table(
        output / "development_metrics.parquet",
        pd.concat([dev_metrics, cal_metrics], ignore_index=True),
    )
    write_json(output / "feature_audits.json", audits)
    write_json(output / "search_results.json", search_results)
    config = {
        **base_config,
        "run_id": run_id,
        "training_event_ids": ids,
        "development_event_ids": development_ids,
        "calibration_event_ids": calibration_ids,
        "fold_training_event_ids": {
            eid: _fold_training_ids(dataset, ids, eid) for eid in development_ids + calibration_ids
        },
        "feature_history_scope": (
            "all earlier calendar windows with component availability "
            "masking; last_n restricts fitted events only"
        ),
        "selected_parameters": selection,
        "calibration": calibration,
        "thresholds": thresholds,
        "excluded_features": exclusions,
        "insufficient_development_history": not bool(development_ids),
        "frozen_at": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    write_json(output / "config.json", config)
    for key, model in models.items():
        path = output / "models" / f"{key.replace('/', '-')}.joblib"
        path.parent.mkdir(exist_ok=True)
        with path.open("xb") as stream:
            joblib.dump(
                {
                    "model": model,
                    "task_family": key,
                    "parameters": selection[key],
                    "calibration": calibration.get(key),
                    "threshold": thresholds.get(key),
                    "training_event_ids": ids,
                    "source_snapshot_id": dataset.receipt["id"],
                    "run_id": run_id,
                },
                stream,
            )
    # No target labels are joined before this whole-field forecast is frozen.
    forecasts = []
    for task in tasks:
        frame = target_frames[task]
        expected = set(frame.driver_id)
        for family in FAMILIES:
            key = f"{task}/{family}"
            forecast = _forecast(
                models[key], frame, task, family, calibration.get(key), thresholds.get(key, 0.5)
            )
            if set(forecast.driver_id) != expected or forecast.driver_id.duplicated().any():
                raise ValueError(
                    "Forecast field differs from the independent pre-cutoff entry list"
                )
            forecasts.append(forecast)
        forecasts.append(_baseline(frame, task))
    forecasts = pd.concat(forecasts, ignore_index=True)
    if live_tasks is not None:
        # A long fit must not be published as a pre-session prediction after the start.
        if not set(tasks).issubset(forecast_tasks(dataset, event_id)):
            raise ValueError("Target session started during training; forecast was not sealed")
    save_table(output / "forecasts.parquet", forecasts)
    write_json(
        output / "prediction_lock.json",
        {
            "run_id": run_id,
            "source_snapshot_id": dataset.receipt["id"],
            "config_sha256": file_digest(output / "config.json"),
            "forecasts_sha256": file_digest(output / "forecasts.parquet"),
            "sealed_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "sealed_before_target_scoring": True,
            "field_source": "independent FIA pre-cutoff entry lists",
            "evaluation_status": "unscored live forecast"
            if live_tasks is not None
            else "exploratory historical replay; outcomes previously consulted",
        },
    )
    # Target-label access/scoring boundary. A complete field remains in every task.
    predictions = (
        forecasts.copy() if live_tasks is not None else _attach_actuals(dataset, forecasts)
    )
    metrics = (
        pd.DataFrame() if live_tasks is not None else _metric_rows(predictions, "historical_target")
    )
    save_table(output / "predictions.parquet", predictions)
    save_table(output / "metrics.parquet", metrics)
    summary = {
        "run_id": run_id,
        "target_event_id": event_id,
        "target_event_name": event["name"],
        "training_event_count": len(ids),
        "training_event_ids": ids,
        "training_first_event_id": ids[0],
        "training_last_event_id": ids[-1],
        "training_last_available_at": training.race_available_at.max(),
        "qualifying_cutoff": event.qualifying_cutoff,
        "race_cutoff": event.race_cutoff,
        "development_event_ids": development_ids,
        "calibration_event_ids": calibration_ids,
        "data_kind": "real",
        "mode": "live_forecast" if live_tasks is not None else PROTOCOL["mode"],
        "evaluation_status": "unscored live forecast"
        if live_tasks is not None
        else "exploratory historical target; not an untouched final test",
        "source": dataset.receipt.get("source"),
        "source_snapshot_id": dataset.receipt["id"],
        "source_catalog_id": dataset.receipt.get("source_catalog_id"),
        "excluded_features": exclusions,
        "missing_feature_groups": PROTOCOL["missing_groups_excluded"],
        "training_history": "2026-only",
        "last_n": last_n,
        "target_event_count": 1,
        "model_count": len(models),
        "rank_evaluation": (
            "printed numeric positions only; partial field where NC/DSQ/unknown has no rank"
        ),
        "calibration_diagnostics": (
            "calibration-fit diagnostics are not independent validation findings"
        ),
        "not_comparable": (
            "2026-only replay does not establish pooled/weighted 2024-2025 "
            "superiority; those source/session groups remain unavailable"
        ),
    }
    write_json(output / "summary.json", summary)
    seal_directory(
        output,
        {
            "data_kind": "real",
            "kind": "live_forecast" if live_tasks is not None else "historical_backtest",
            "mode": summary["mode"],
            "run_id": run_id,
            "source_snapshot_id": dataset.receipt["id"],
            "target_event_id": event_id,
        },
    )
    return summary


def cached_run(dataset_root: Path, event_id: str, output_root: Path, *, last_n: int | None = None):
    """Explicit-action cache; existing runs must verify, never silently regenerate."""
    dataset = RealDataset(dataset_root)
    config = _configuration(dataset, event_id, last_n, 42)
    run_id = digest(config)
    suffix = "all" if last_n is None else f"last{last_n}"
    path = Path(output_root) / f"replay-{event_id}-{suffix}-{run_id[:12]}"
    if path.exists():
        receipt = verify_directory(path)
        if receipt.get("run_id") != run_id:
            raise ValueError("An existing cached run has a different immutable configuration")
        load_run(path)
    else:
        run_backtest(Path(dataset_root), event_id, path, last_n=last_n)
    return path


def cached_forecast(dataset_root, event_id, output_root, *, last_n=None):
    """Explicitly fit unstarted tasks, with no target-label join or target scoring."""
    from f1ml_real.refresh import forecast_tasks

    dataset = RealDataset(dataset_root)
    tasks = forecast_tasks(dataset, event_id)
    if not tasks:
        raise ValueError("Refresh data first; no unstarted target has the required inputs")
    config = {
        **_configuration(dataset, event_id, last_n, 42),
        "mode": "live_forecast",
        "forecast_tasks": list(tasks),
    }
    run_id = digest(config)
    path = Path(output_root) / f"forecast-{event_id}-{run_id[:12]}"
    if path.exists():
        run = load_run(path)
        if run["receipt"].get("run_id") != run_id:
            raise ValueError("Saved forecast configuration differs")
    else:
        with threadpool_limits(limits=1):
            _run_backtest(
                Path(dataset_root), event_id, path, last_n=last_n, seed=42, live_tasks=tasks
            )
    return path


def load_run(path: Path | str):
    """Read reviewed tables only; UI reruns neither deserialize nor fit models."""
    path = Path(path)
    receipt = verify_directory(path)
    if receipt.get("data_kind") != "real":
        raise ValueError("Historical lab cannot load a non-real experiment")
    lock = read_json(path / "prediction_lock.json")
    if lock["config_sha256"] != file_digest(path / "config.json") or lock[
        "forecasts_sha256"
    ] != file_digest(path / "forecasts.parquet"):
        raise ValueError("Forecast/configuration prediction lock does not verify")
    result = {name: read_json(path / f"{name}.json") for name in ("summary", "config")}
    for name in ("forecasts", "predictions", "metrics", "development_metrics", "training_events"):
        result[name] = pd.read_parquet(path / f"{name}.parquet")
    result.update({"receipt": receipt, "prediction_lock": lock, "path": path})
    return result
