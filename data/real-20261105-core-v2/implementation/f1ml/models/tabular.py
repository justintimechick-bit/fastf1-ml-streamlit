"""Fold-local transforms and binary probability handling shared by race models."""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from f1ml.features.contract import categorical_columns, numeric_columns


def preprocessing(task, schema, numeric=None, categorical=None):
    return ColumnTransformer(
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
                list(numeric_columns(task, schema) if numeric is None else numeric),
            ),
            (
                "category",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(categorical_columns(task, schema) if categorical is None else categorical),
            ),
        ]
    )


def model_input(features, task, experiment=None):
    from f1ml.features.contract import model_matrix, schema_version

    if experiment is not None:
        from f1ml.experiments.contract import matrix

        return matrix(features, task, experiment)
    schema = schema_version(features)
    return (
        model_matrix(features, task),
        list(numeric_columns(task, schema)),
        list(categorical_columns(task, schema)),
    )


def positive_probability(classifier, matrix):
    classes = list(classifier.classes_)
    if 1.0 not in classes:
        return np.zeros(len(matrix))
    return np.asarray(classifier.predict_proba(matrix)[:, classes.index(1.0)], dtype=float)


def logits(probabilities):
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)
