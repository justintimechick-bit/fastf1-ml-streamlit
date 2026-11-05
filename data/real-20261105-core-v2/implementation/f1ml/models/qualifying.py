"""Qualifying regressors, fixed ranking rules and auditable local model bundles."""

import platform
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from f1ml.domain import utc
from f1ml.evaluation.splits import assert_development
from f1ml.features.contract import model_matrix, schema_version
from f1ml.models.tabular import model_input
from f1ml.storage import new_directory, seal_directory, verify_directory, write_json

FAMILIES = ("ridge", "random_forest", "gradient_boosting")
STRATEGIES = ("2026_only", "equal", "weighted")
TIE_RULE = "ascending raw_score, then ascending stable driver_id (exact scores, no rounding)"
WEIGHT_RULE = (
    "event_weight / labeled_driver_count, normalized to mean row weight 1; "
    "weighted event_weight = season_weight * 2**(-age_in_earlier_eligible_events / half_life); "
    "season_weight: 2024=0.5, 2025=1, 2026=tuned; equal/2026_only event_weight=1"
)


@dataclass
class QualifyingArtifact:
    pipeline: Pipeline
    metadata: dict


def validate_field(features: pd.DataFrame) -> None:
    if features.empty or features.duplicated(["event_id", "driver_id"]).any():
        raise ValueError("Qualifying field must be nonempty with unique event/driver keys")
    if (
        features.driver_id.isna().any()
        or not features.driver_id.map(lambda v: isinstance(v, str) and bool(v)).all()
    ):
        raise ValueError("Stable nonempty string driver IDs are required")
    for _, group in features.groupby("event_id"):
        if not pd.to_numeric(group.field_size).eq(len(group)).all():
            raise ValueError("Feature rows must contain the complete actual event field")
        if group.cutoff.nunique() != 1 or group["mode"].nunique() != 1:
            raise ValueError("Each event must have one cutoff and provenance mode")


def order_scores(features: pd.DataFrame, scores) -> pd.DataFrame:
    validate_field(features)
    values = np.asarray(scores, dtype=float)
    if values.shape != (len(features),) or not np.isfinite(values).all():
        raise ValueError("One finite raw score is required per eligible driver")
    result = features[["event_id", "driver_id", "cutoff", "mode"]].copy()
    result["raw_score"] = values
    result = result.sort_values(["event_id", "raw_score", "driver_id"], kind="stable")
    result["predicted_position"] = result.groupby("event_id").cumcount() + 1
    result = result.reset_index(drop=True)
    validate_permutation(result, features)
    return result


def validate_permutation(predictions: pd.DataFrame, field: pd.DataFrame) -> None:
    keys = ["event_id", "driver_id"]
    if predictions.duplicated(keys).any() or field.duplicated(keys).any():
        raise ValueError("Duplicate event/driver keys in prediction field")
    if set(map(tuple, predictions[keys].to_numpy())) != set(map(tuple, field[keys].to_numpy())):
        raise ValueError("Predictions must contain each eligible driver exactly once")
    for _, group in predictions.groupby("event_id"):
        if sorted(group.predicted_position.tolist()) != list(range(1, len(group) + 1)):
            raise ValueError("Predicted positions must be a permutation of 1 through N")


def align_labels(features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    if labels.duplicated(["event_id", "driver_id"]).any():
        raise ValueError("Duplicate qualifying label keys")
    aligned = features[["event_id", "driver_id"]].merge(
        labels, on=["event_id", "driver_id"], how="left", validate="one_to_one", indicator=True
    )
    if len(aligned) != len(labels) or not aligned._merge.eq("both").all():
        raise ValueError("Qualifying labels must exactly match the complete feature field")
    target = pd.to_numeric(aligned.rank_target, errors="raise")
    if target.notna().any() and not target.dropna().between(0, 1).all():
        raise ValueError("Normalized qualifying rank targets must be between 0 and 1")
    return aligned.drop(columns="_merge")


def training_events(features, labels, available, strategy, config):
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown training strategy: {strategy}")
    part = features.loc[features.event_id.isin(available)]
    if strategy == "2026_only":
        part = part.loc[part.season.eq(2026)]
    usable = labels.loc[labels.rank_target.notna(), "event_id"]
    ids = set(part.event_id) & set(usable)
    ordered = [e for e in available if e in ids]
    minimum = (
        config.min_2026_training_events if strategy == "2026_only" else config.min_training_events
    )
    reason = None if len(ordered) >= minimum else f"needs_{minimum}_usable_training_events"
    return ordered, reason


def sample_weights(aligned, features, manifest, strategy, profile):
    """Equal event mass before optional temporal weights; never normalize away the profile."""
    chronological = [e["event_id"] for e in manifest["eligible"]]
    latest = max(chronological.index(e) for e in aligned.event_id.unique())
    seasons = features.groupby("event_id").season.first().to_dict()
    event_weights = {}
    for event_id in aligned.event_id.unique():
        weight = 1.0
        if strategy == "weighted":
            year = int(seasons[event_id])
            multiplier = {2024: 0.5, 2025: 1.0, 2026: profile["season_2026_weight"]}[year]
            age = latest - chronological.index(event_id)
            weight = multiplier * 2 ** (-age / profile["half_life_events"])
        event_weights[event_id] = weight
    weights = aligned.event_id.map(event_weights) / aligned.groupby("event_id").event_id.transform(
        "size"
    )
    return weights.to_numpy(dtype=float) / float(weights.mean()), event_weights


def add_dependencies(metadata, features, labels, event_ids, manifest):
    """Selection labels are model dependencies as well as rows used for fitting."""
    ids = set(event_ids)
    assert_development(manifest, ids)
    f = features.loc[features.event_id.isin(ids)]
    y = labels.loc[labels.event_id.isin(ids)]
    if f.empty or y.empty:
        raise ValueError("Missing model dependency provenance")
    metadata.update(
        dependency_events=[e for e in manifest["development"] if e in ids],
        max_feature_cutoff=max(utc(t) for t in f.cutoff).isoformat(),
        max_label_end=max(utc(t) for t in y.label_session_end).isoformat(),
        max_label_retrieved_at=max(utc(t) for t in y.label_retrieved_at).isoformat(),
        max_weekend_end=max(
            utc(e["race_end"]) for e in manifest["eligible"] if e["event_id"] in ids
        ).isoformat(),
        modes=sorted(f["mode"].unique()),
    )


def fit_qualifying(
    features, labels, manifest, family, strategy, parameters, profile, seed, experiment=None
):
    features = features.reset_index(drop=True)
    assert_development(manifest, features.event_id)
    validate_field(features)
    schema = schema_version(features)
    if schema != "3.0.0":
        raise ValueError("Qualifying comparison requires Phase 3 features")
    if strategy not in STRATEGIES or family not in FAMILIES:
        raise ValueError("Unknown qualifying model or strategy")
    if strategy == "2026_only" and not features.season.eq(2026).all():
        raise ValueError("2026-only training includes an older season")
    matrix, numeric_features, categorical_features = model_input(features, "qualifying", experiment)
    aligned = align_labels(features, labels)
    mask = aligned.rank_target.notna()
    if not mask.any():
        raise ValueError("No known normalized qualifying labels")
    transform = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        (
                            "impute",
                            SimpleImputer(
                                strategy="median", add_indicator=True, keep_empty_features=True
                            ),
                        ),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "category",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
        ]
    )
    if family == "ridge":
        estimator = Ridge(**parameters)
    elif family == "random_forest":
        estimator = RandomForestRegressor(random_state=seed, n_jobs=1, **parameters)
    else:
        # No random internal holdout: all validation is by complete chronological weekends.
        estimator = HistGradientBoostingRegressor(
            random_state=seed, early_stopping=False, categorical_features=None, **parameters
        )
    weights, event_weights = sample_weights(
        aligned.loc[mask], features, manifest, strategy, profile
    )
    pipeline = Pipeline([("transform", transform), ("estimator", estimator)])
    pipeline.fit(
        matrix.loc[mask],
        aligned.loc[mask, "rank_target"].astype(float),
        estimator__sample_weight=weights,
    )
    metadata = {
        "artifact_version": "qualifying-1",
        "task": "qualifying",
        "target": "(official qualifying position - 1) / (official field size - 1)",
        "score_meaning": "unclipped normalized-rank estimate; lower ranks ahead",
        "tie_rule": TIE_RULE,
        "manifest_id": manifest["id"],
        "family": family,
        "strategy": strategy,
        "parameters": parameters,
        "weight_profile": profile,
        "weight_formula": WEIGHT_RULE,
        "training_event_weights": event_weights,
        "train_events": [e for e in manifest["development"] if e in set(features.event_id)],
        "train_rows": int(mask.sum()),
        "missing_targets": int((~mask).sum()),
        "seed": seed,
        "python": platform.python_version(),
        "sklearn": sklearn_version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "joblib": joblib.__version__,
        "feature_version": schema,
        "feature_code_id": features.feature_code_id.iloc[0],
        "model_feature_columns": list(matrix.columns),
    }
    add_dependencies(metadata, features, labels, features.event_id, manifest)
    if experiment is not None:
        metadata["experiment"] = experiment
    return QualifyingArtifact(pipeline, metadata)


def predict_scores(artifact, features):
    validate_field(features)
    matrix, _, _ = model_input(features, "qualifying", artifact.metadata.get("experiment"))
    if not features.feature_code_id.eq(artifact.metadata["feature_code_id"]).all():
        raise ValueError("Feature implementation differs from the qualifying model")
    if list(matrix.columns) != artifact.metadata["model_feature_columns"]:
        raise ValueError("Qualifying model feature schema differs")
    return order_scores(features, artifact.pipeline.predict(matrix))


def rule_predictions(features, rule):
    """Fixed baselines; no fitted transforms, outcomes or tuning."""
    validate_field(features)
    model_matrix(features, "qualifying")
    scores = pd.Series(np.nan, index=features.index, dtype=float)
    sources = pd.Series("neutral_0.5", index=features.index)
    for column in ("qualifying_rank_mean_3", "qualifying_rank_mean_5", "previous_qualifying_rank"):
        use = scores.isna() & features[column].notna()
        scores.loc[use] = features.loc[use, column]
        sources.loc[use] = column
    scores = scores.fillna(0.5)
    if rule == "practice":
        for _, group in features.groupby("event_id"):
            gaps = group.practice_gap_pct.dropna()
            if len(gaps) >= 2:
                # Convert comparable pace gaps to rank units before mixing in historical fallbacks.
                scores.loc[gaps.index] = (gaps.rank(method="min") - 1) / (len(gaps) - 1)
                sources.loc[gaps.index] = "clean_practice_observed_field_rank"
    elif rule != "recent":
        raise ValueError("Unknown qualifying ranking rule")
    result = order_scores(features, scores.to_numpy())
    diagnostics = features[["event_id", "driver_id", "practice_gap_pct"]].copy()
    diagnostics["score_source"] = sources
    return result.merge(diagnostics, on=["event_id", "driver_id"], validate="one_to_one")


def save_model(root: Path, artifact: QualifyingArtifact):
    new_directory(root)
    joblib.dump(artifact, root / "model.joblib")
    write_json(root / "model.json", artifact.metadata)
    return seal_directory(
        root, {"kind": "qualifying_model", "manifest_id": artifact.metadata["manifest_id"]}
    )


def load_model(root: Path):
    # Integrity is checked before unpickling. Only load bundles from a trusted local source.
    verify_directory(root)
    artifact = joblib.load(root / "model.joblib")
    if not isinstance(artifact, QualifyingArtifact):
        raise ValueError("Expected a qualifying model bundle")
    return artifact
