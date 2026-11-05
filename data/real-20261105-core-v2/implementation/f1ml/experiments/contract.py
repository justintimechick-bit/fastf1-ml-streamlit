"""Explicit feature groups without modifying the frozen Phase 3 feature implementation."""

from pathlib import Path

import numpy as np
import pandas as pd

from f1ml.features.contract import categorical_columns, model_matrix, numeric_columns
from f1ml.storage import digest, file_digest

VERSION = "experiments-1"
GROUPS = ("history", "sessions", "practice", "telemetry", "weather")
SESSION = ("practice_sessions",)
PACE = (
    "practice_gap_pct",
    "practice_gap_pct_missing",
    "practice_clean_laps",
    "practice_comparable_laps",
    "practice_pace_sessions",
)
PRACTICE_EXTRA = (
    "practice_lap_cv",
    "practice_sector_cv",
    "practice_consistency_laps",
    "practice_detail_gap_pct",
    "practice_detail_laps",
    "practice_detail_comparable_laps",
)
RESULT_EXTRA = ("practice_result_rank_norm", "practice_result_cells")
TELEMETRY = (
    "telemetry_speed_mean_kph",
    "telemetry_speed_p95_kph",
    "telemetry_full_throttle_fraction",
    "telemetry_braking_fraction",
    "telemetry_laps",
    "telemetry_channel_coverage",
    "telemetry_teammate_speed_delta_kph",
    "telemetry_teammate_throttle_delta",
    "telemetry_teammate_braking_delta",
    "telemetry_teammate_comparisons",
)
WEATHER = (
    "observed_air_temp_c",
    "observed_track_temp_c",
    "observed_humidity_pct",
    "observed_wind_speed_mps",
    "observed_rain_fraction",
    "observed_weather_sessions",
    "observed_weather_samples",
)
EXTRA = RESULT_EXTRA + PRACTICE_EXTRA + TELEMETRY + WEATHER
KEYS = ("event_id", "driver_id", "task", "cutoff", "mode")


def implementation_id():
    root = Path(__file__).parent
    return digest({p.name: file_digest(p) for p in sorted(root.glob("*.py"))})


def specification(group):
    if group not in (*GROUPS, "core"):
        raise ValueError("Unknown experiment feature group")
    return {"version": VERSION, "group": group, "implementation_id": implementation_id()}


def feature_columns(task, group):
    core = list(numeric_columns(task, "3.0.0"))
    if group == "core":
        return core
    level = GROUPS.index(group)
    current = set(SESSION + PACE)
    current_q = (
        "qualifying_rank_norm",
        "qualifying_classification_count",
        "qualifying_q1_",
        "qualifying_q2_",
        "qualifying_q3_",
        "known_grid_",
        "known_pit_lane",
    )
    current |= {c for c in core if c.startswith(current_q)}
    result = [c for c in core if c not in current]
    if level >= 1:
        result += list(SESSION + RESULT_EXTRA)
        result += [
            c
            for c in core
            if c.startswith(current_q) and not any(s in c for s in ("_q1_", "_q2_", "_q3_"))
        ]
    if level >= 2:
        result += list(PACE + PRACTICE_EXTRA)
        result += [c for c in core if any(s in c for s in ("_q1_", "_q2_", "_q3_"))]
    if level >= 3:
        result += list(TELEMETRY)
    if level >= 4:
        result += list(WEATHER)
    if len(result) != len(set(result)):
        raise ValueError("Duplicate experimental feature columns")
    return result


def matrix(features, task, spec):
    if spec != specification(spec.get("group")):
        raise ValueError("Experiment implementation/version differs; regenerate and retrain")
    if not set(EXTRA).issubset(features) or set(features) & {"source_status", "target_weather"}:
        raise ValueError("Missing or forbidden experimental columns")
    base = model_matrix(features.drop(columns=list(EXTRA)), task)
    numeric = feature_columns(task, spec["group"])
    categories = list(categorical_columns(task, "3.0.0"))
    extra = features[list(EXTRA)].apply(pd.to_numeric, errors="raise").astype(float)
    if np.isinf(extra.to_numpy()).any():
        raise ValueError("Infinite experiment feature")
    return pd.concat([base, extra], axis=1)[numeric + categories], numeric, categories


def merge_features(core, extra):
    if set(extra) != set(KEYS + EXTRA) or extra.duplicated(list(KEYS)).any():
        raise ValueError("Experiment supplement requires exact allowlist and unique cutoff keys")
    result = core.merge(extra, on=list(KEYS), how="outer", validate="one_to_one", indicator=True)
    if not result._merge.eq("both").all() or len(result) != len(core):
        raise ValueError("Experiment supplement must retain the complete core field")
    return result.drop(columns="_merge")
