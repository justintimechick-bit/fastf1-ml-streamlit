"""Fixed, deliberately modest baselines. All learned transforms stay in the fold."""

import platform
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from f1ml.domain import utc
from f1ml.evaluation.splits import assert_development
from f1ml.features.contract import (
    categorical_columns,
    model_matrix,
    numeric_columns,
    schema_version,
)


@dataclass
class ModelArtifact:
    task: str
    pipeline: Pipeline
    metadata: dict


def fit_baseline(
    features: pd.DataFrame, labels: pd.DataFrame, task: str, manifest: dict, seed: int
) -> ModelArtifact:
    assert_development(manifest, features.event_id)
    matrix = model_matrix(features, task)
    schema = schema_version(features)
    if labels.duplicated(["event_id", "driver_id"]).any():
        raise ValueError("Duplicate label keys")
    aligned = features[["event_id", "driver_id"]].merge(
        labels, how="left", on=["event_id", "driver_id"], validate="one_to_one", indicator=True
    )
    if not aligned._merge.eq("both").all() or len(aligned) != len(labels):
        raise ValueError("Training labels must exactly match the complete feature field")
    target = (
        aligned.podium
        if task == "podium"
        else aligned.rank_target
        if schema != "legacy"
        else aligned.official_position
    )
    mask = target.notna()
    if not mask.any():
        raise ValueError("No usable training labels")
    x, y = matrix.loc[mask], target.loc[mask].astype(float)
    numeric = Pipeline(
        [
            (
                "impute",
                SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
            ),
            ("scale", StandardScaler()),
        ]
    )
    transform = ColumnTransformer(
        [
            ("numeric", numeric, list(numeric_columns(task, schema))),
            (
                "category",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(categorical_columns(task, schema)),
            ),
        ]
    )
    if task == "podium":
        estimator = (
            LogisticRegression(C=1.0, max_iter=2000, random_state=seed)
            if y.nunique() > 1
            else DummyClassifier(strategy="prior")
        )
    else:
        estimator = Ridge(alpha=10.0)
    pipeline = Pipeline([("transform", transform), ("estimator", estimator)])
    # Give each complete weekend the same total weight, including variable field sizes.
    weights = 1 / aligned.loc[mask].groupby("event_id").event_id.transform("size")
    # Normalize to mean 1 so regularization has its usual sample-size interpretation.
    weights = weights / weights.mean()
    pipeline.fit(x, y, estimator__sample_weight=weights.to_numpy())
    return ModelArtifact(
        task,
        pipeline,
        {
            "manifest_id": manifest["id"],
            "train_events": sorted(features.event_id.unique()),
            "train_rows": len(x),
            "missing_targets": int((~mask).sum()),
            "seed": seed,
            "max_feature_cutoff": max(utc(t) for t in features.cutoff).isoformat(),
            "max_label_end": max(utc(t) for t in labels.label_session_end).isoformat(),
            "max_label_retrieved_at": max(utc(t) for t in labels.label_retrieved_at).isoformat(),
            "modes": sorted(features["mode"].unique()),
            "python": platform.python_version(),
            "sklearn": sklearn_version,
            "baseline": "logistic_C1" if task == "podium" else "ridge_alpha10",
            "feature_version": schema,
            "feature_code_id": features.feature_code_id.iloc[0] if schema != "legacy" else None,
            "model_feature_columns": list(matrix.columns),
        },
    )


def predict(artifact: ModelArtifact, features: pd.DataFrame) -> pd.DataFrame:
    matrix = model_matrix(features, artifact.task)
    if (
        artifact.metadata.get("feature_code_id")
        and not features.feature_code_id.eq(artifact.metadata["feature_code_id"]).all()
    ):
        raise ValueError("Model feature implementation differs; regenerate features and retrain")
    expected = list(artifact.pipeline.feature_names_in_)
    if list(matrix.columns) != expected:
        raise ValueError("Model feature schema differs; retrain for the new feature version")
    result = features[["event_id", "driver_id", "cutoff", "mode"]].copy()
    if artifact.task == "podium":
        classes = list(artifact.pipeline.named_steps["estimator"].classes_)
        probabilities = artifact.pipeline.predict_proba(matrix)
        result["podium_probability"] = (
            probabilities[:, classes.index(1.0)] if 1.0 in classes else np.zeros(len(features))
        )
    else:
        result["score"] = artifact.pipeline.predict(matrix)
        result = result.sort_values(["event_id", "score", "driver_id"], kind="stable")
        result["predicted_position"] = result.groupby("event_id").cumcount() + 1
    return result.reset_index(drop=True)
