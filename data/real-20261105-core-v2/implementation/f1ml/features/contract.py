"""Allowlisted model matrices: identifiers/targets/raw source columns cannot slip in."""

import numpy as np
import pandas as pd

from f1ml.features.dictionary import FEATURE_VERSION, METADATA, RACE_METADATA, columns

COMMON_NUMERIC = (
    "field_size",
    "history_count",
    "history_quali_mean",
    "history_race_mean",
    "history_podium_rate",
    "practice_gap_pct",
    "practice_clean_laps",
    "practice_sessions",
)
CATEGORICAL = ("driver_id", "team_id")
RACE_NUMERIC = ("qualifying_position", "known_grid_position")
IDENTIFIERS = ("event_id", "driver_id", "task", "cutoff", "mode")


def schema_version(frame):
    if "feature_version" not in frame:
        return "legacy"
    if not frame.feature_version.eq(FEATURE_VERSION).all():
        raise ValueError("Unsupported feature schema version")
    if (
        "feature_code_id" not in frame
        or frame.feature_code_id.isna().any()
        or frame.feature_code_id.nunique() != 1
    ):
        raise ValueError("Mixed or missing feature implementation fingerprints")
    return FEATURE_VERSION


def numeric_columns(task: str, schema="legacy") -> tuple[str, ...]:
    if task not in ("qualifying", "race", "podium"):
        raise ValueError(f"Unsupported task: {task}")
    return (
        columns(task)
        if schema == FEATURE_VERSION
        else COMMON_NUMERIC + (() if task == "qualifying" else RACE_NUMERIC)
    )


def categorical_columns(task, schema="legacy"):
    return columns(task, "categorical") if schema == FEATURE_VERSION else CATEGORICAL


def model_matrix(frame: pd.DataFrame, task: str) -> pd.DataFrame:
    schema = schema_version(frame)
    numeric = numeric_columns(task, schema)
    categorical = categorical_columns(task, schema)
    metadata = (
        METADATA + (() if task == "qualifying" else RACE_METADATA)
        if schema == FEATURE_VERSION
        else IDENTIFIERS
    )
    expected = set(numeric) | set(categorical) | set(metadata)
    if set(frame.columns) != expected:
        raise ValueError(f"Feature schema mismatch: {set(frame.columns) ^ expected}")
    if frame.event_id.isna().any() or frame.driver_id.isna().any():
        raise ValueError("Missing feature keys")
    if frame.duplicated(["event_id", "driver_id"]).any():
        raise ValueError("Duplicate driver/event feature rows")
    expected_task = "race" if task == "podium" else task
    if not frame.task.eq(expected_task).all():
        raise ValueError("Feature cutoff task differs from model task")
    matrix = frame[list(numeric) + list(categorical)].copy()
    for column in numeric:
        matrix[column] = pd.to_numeric(matrix[column], errors="raise").astype(float)
        if np.isinf(matrix[column]).any():
            raise ValueError("Infinite feature value")
    for column in categorical:
        matrix[column] = matrix[column].fillna("unknown").astype(str)
    return matrix
