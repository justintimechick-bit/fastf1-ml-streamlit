"""Official-classification regression and independent marginal podium classifiers."""

import platform
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline

from f1ml.evaluation.splits import assert_development
from f1ml.features.contract import model_matrix, schema_version
from f1ml.models.qualifying import (
    STRATEGIES,
    TIE_RULE,
    WEIGHT_RULE,
    add_dependencies,
    align_labels,
    order_scores,
    sample_weights,
    training_events,
    validate_field,
)
from f1ml.models.tabular import logits, model_input, positive_probability, preprocessing
from f1ml.storage import new_directory, save_table, seal_directory, verify_directory, write_json

RACE_FAMILIES = ("ridge", "random_forest", "gradient_boosting")
PODIUM_FAMILIES = ("logistic", "random_forest", "gradient_boosting")


@dataclass
class RaceArtifact:
    task: str
    pipeline: Pipeline
    metadata: dict
    calibrator: LogisticRegression | None = None
    threshold: float = 0.5


def eligible_training(features, labels, available, strategy, config, task):
    targets = labels.assign(rank_target=labels.podium) if task == "podium" else labels
    return training_events(features, targets, available, strategy, config)


def aligned_targets(features, labels, task):
    aligned = align_labels(features, labels)
    if task == "podium":
        y = pd.to_numeric(aligned.podium, errors="raise")
        if not y.dropna().isin([0, 1]).all():
            raise ValueError("Podium targets must be binary or missing")
        special = aligned.outcome.isin(
            [
                "dns",
                "dsq",
                "excluded",
                "withdrawn",
                "dnq",
                "non_classified",
                "retired_non_classified",
            ]
        )
        if (special & y.eq(1)).any():
            raise ValueError("Special non-podium status contradicts podium target")
        positioned = aligned.official_position.notna() & y.notna() & ~special
        if (
            y.loc[positioned] != aligned.loc[positioned, "official_position"].le(3).astype(float)
        ).any():
            raise ValueError("Podium target contradicts official race position")
    return aligned


def fit_model(
    features, labels, manifest, task, family, strategy, parameters, profile, seed, experiment=None
):
    features = features.reset_index(drop=True)
    assert_development(manifest, features.event_id)
    validate_field(features)
    schema = schema_version(features)
    if schema != "3.0.0" or task not in ("race", "podium"):
        raise ValueError("Race/podium models require Phase 3 race-cutoff features")
    families = RACE_FAMILIES if task == "race" else PODIUM_FAMILIES
    if family not in families or strategy not in STRATEGIES:
        raise ValueError("Unknown race/podium model or strategy")
    if strategy == "2026_only" and not features.season.eq(2026).all():
        raise ValueError("2026-only training includes older seasons")
    matrix, numeric_features, categorical_features = model_input(features, task, experiment)
    aligned = aligned_targets(features, labels, task)
    target = aligned.rank_target if task == "race" else aligned.podium
    mask = target.notna()
    if not mask.any():
        raise ValueError(f"No known {task} targets")
    x, y = matrix.loc[mask], target.loc[mask].astype(float)
    single_class = task == "podium" and y.nunique() == 1
    if single_class:
        estimator = DummyClassifier(strategy="prior")
    elif family == "ridge":
        estimator = Ridge(**parameters)
    elif family == "logistic":
        estimator = LogisticRegression(max_iter=2000, random_state=seed, **parameters)
    elif family == "random_forest":
        cls = RandomForestRegressor if task == "race" else RandomForestClassifier
        estimator = cls(random_state=seed, n_jobs=1, **parameters)
    else:
        cls = HistGradientBoostingRegressor if task == "race" else HistGradientBoostingClassifier
        estimator = cls(
            random_state=seed, early_stopping=False, categorical_features=None, **parameters
        )
    weights, event_weights = sample_weights(
        aligned.loc[mask], features, manifest, strategy, profile
    )
    pipeline = Pipeline(
        [
            ("transform", preprocessing(task, schema, numeric_features, categorical_features)),
            ("estimator", estimator),
        ]
    )
    pipeline.fit(x, y, estimator__sample_weight=weights)
    metadata = {
        "artifact_version": "race-podium-1",
        "task": task,
        "manifest_id": manifest["id"],
        "target": "normalized official race position" if task == "race" else "official race podium",
        "output_meaning": "predicted official classification, including ordered administrative tail"
        if task == "race"
        else "independent marginal podium probabilities; no sum constraint",
        "family": family,
        "strategy": strategy,
        "parameters": parameters,
        "estimator_parameters": estimator.get_params(),
        "single_class_fallback": single_class,
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
        "tie_rule": TIE_RULE,
        "grid_policy": "only Phase 3 externally evidenced known_grid_*; never Race.results grid",
    }
    add_dependencies(metadata, features, labels, features.event_id, manifest)
    if experiment is not None:
        metadata["experiment"] = experiment
    return RaceArtifact(task, pipeline, metadata)


def predict_model(artifact, features):
    validate_field(features)
    matrix, _, _ = model_input(features, artifact.task, artifact.metadata.get("experiment"))
    if not features.feature_code_id.eq(artifact.metadata["feature_code_id"]).all():
        raise ValueError("Feature implementation differs from race/podium model")
    if list(matrix.columns) != artifact.metadata["model_feature_columns"]:
        raise ValueError("Race/podium feature schema differs")
    if artifact.task == "race":
        return order_scores(features, artifact.pipeline.predict(matrix))
    raw = positive_probability(artifact.pipeline, matrix)
    probability = (
        positive_probability(artifact.calibrator, logits(raw))
        if artifact.calibrator is not None
        else raw.copy()
    )
    if not np.isfinite(probability).all() or not ((0 <= probability) & (probability <= 1)).all():
        raise ValueError("Invalid podium probabilities")
    result = features[["event_id", "driver_id", "cutoff", "mode"]].copy()
    result["raw_podium_probability"] = raw
    result["podium_probability"] = probability
    result["podium_threshold"] = artifact.threshold
    result["predicted_podium"] = probability >= artifact.threshold
    return result.sort_values(["event_id", "driver_id"]).reset_index(drop=True)


def rule_predictions(features, task, rule):
    validate_field(features)
    model_matrix(features, task)
    if rule not in ("qualifying", "recent"):
        raise ValueError("Unknown race/podium rule")
    scores = features.qualifying_rank_norm.copy().astype(float)
    sources = pd.Series("qualifying_rank_norm", index=features.index)
    if task == "race":
        if rule == "recent":
            scores[:] = np.nan
            for column in (
                "race_rank_mean_3",
                "race_rank_mean_5",
                "previous_race_rank",
                "qualifying_rank_norm",
            ):
                use = scores.isna() & features[column].notna()
                scores.loc[use] = features.loc[use, column]
                sources.loc[use] = column
        sources.loc[scores.isna()] = "neutral_0.5"
        result = order_scores(features, scores.fillna(0.5).to_numpy())
    else:
        # Hard qualifying top-three is deliberately a naive, untuned 0/1 probability reference.
        scores = (
            features.qualifying_position.le(3)
            .astype(float)
            .where(features.qualifying_position.notna(), 0.5)
        )
        sources[:] = "qualifying_top_three_indicator_or_0.5"
        if rule == "recent":
            for column in ("podium_rate_5", "podium_rate_3"):
                use = features[column].notna() & sources.eq("qualifying_top_three_indicator_or_0.5")
                scores.loc[use] = features.loc[use, column]
                sources.loc[use] = column
        result = features[["event_id", "driver_id", "cutoff", "mode"]].copy()
        result["raw_podium_probability"] = scores.to_numpy()
        result["podium_probability"] = scores.to_numpy()
        result["podium_threshold"] = 0.5
        result["predicted_podium"] = scores.to_numpy() >= 0.5
    return result.merge(
        features[["event_id", "driver_id"]].assign(score_source=sources),
        on=["event_id", "driver_id"],
        validate="one_to_one",
    )


def save_model(root: Path, artifact, postprocess_data=None):
    new_directory(root)
    joblib.dump(artifact, root / "model.joblib")
    write_json(root / "model.json", artifact.metadata)
    if postprocess_data is not None and not postprocess_data.empty:
        save_table(root / "postprocess_training.parquet", postprocess_data)
    return seal_directory(
        root,
        {
            "kind": "race_podium_model",
            "task": artifact.task,
            "manifest_id": artifact.metadata["manifest_id"],
        },
    )


def load_model(root: Path, task=None):
    verify_directory(root)
    artifact = joblib.load(root / "model.joblib")  # Trusted local bundles only.
    if not isinstance(artifact, RaceArtifact) or (task is not None and artifact.task != task):
        raise ValueError("Expected a matching race/podium model bundle")
    return artifact
