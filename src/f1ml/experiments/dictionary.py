"""Documented optional inputs, separate from the immutable core feature registry."""

from f1ml.experiments.contract import EXTRA, GROUPS, feature_columns

FORMULAS = {
    "practice_result_rank_norm": (
        "Mean within-cell best clean-lap normalized rank; actual cell participants"
    ),
    "practice_result_cells": "Number of observed comparable practice cells",
    "practice_lap_cv": "Mean population std/mean of lap seconds, >=2 driver laps per matched cell",
    "practice_sector_cv": (
        "Mean population std/mean of observed positive sector seconds within matched cells"
    ),
    "practice_consistency_laps": "Count of driver laps contributing to lap CV",
    "practice_detail_gap_pct": (
        "Mean within-comparable-cell deficit percent over clean practice laps"
    ),
    "practice_detail_laps": "Clean practice lap count across eligible sessions",
    "practice_detail_comparable_laps": "Count with verified comparable practice conditions",
    "telemetry_speed_mean_kph": (
        "Equal-lap mean of time-integrated speed (trapezoid on valid adjacent samples)"
    ),
    "telemetry_speed_p95_kph": (
        "Equal-lap mean of interval-duration-weighted left-sample speed 95th percentile"
    ),
    "telemetry_full_throttle_fraction": (
        "Equal-lap mean sum(dt*[left Throttle>=99%])/valid throttle seconds"
    ),
    "telemetry_braking_fraction": "Equal-lap mean sum(dt*[left Brake=True])/valid brake seconds",
    "telemetry_laps": "Selected lap count with at least one usable aggregate channel",
    "telemetry_channel_coverage": (
        "Mean valid adjacent seconds/lap duration across selected laps/channels"
    ),
    "telemetry_teammate_speed_delta_kph": (
        "Own minus same-event team peers' mean speed in matching cells"
    ),
    "telemetry_teammate_throttle_delta": (
        "Own minus team peers' full-throttle time fraction in matching cells"
    ),
    "telemetry_teammate_braking_delta": (
        "Own minus team peers' braking time fraction in matching cells"
    ),
    "telemetry_teammate_comparisons": (
        "Number of matched team comparison cells across eligible sessions"
    ),
    "observed_air_temp_c": (
        "Equal-session mean of observed time-weighted air temperature, degrees C"
    ),
    "observed_track_temp_c": (
        "Equal-session mean of observed time-weighted track temperature, degrees C"
    ),
    "observed_humidity_pct": "Equal-session mean of observed time-weighted humidity percent",
    "observed_wind_speed_mps": "Equal-session mean of observed time-weighted wind speed, m/s",
    "observed_rain_fraction": (
        "Equal-session mean of rainy valid weather interval seconds/valid seconds"
    ),
    "observed_weather_sessions": "Eligible completed sessions with weather samples",
    "observed_weather_samples": "Number of observed weather records in eligible sessions",
}


def dictionary():
    return {
        "features": {
            name: {
                "formula": FORMULAS[name],
                "source": (
                    "eligible current-weekend GP practice; GP Q also for race telemetry/weather"
                ),
                "cutoff": "completed/available strictly before explicit task cutoff",
                "missing": "NaN measurement; observed counts zero; training-fold imputation only",
                "regulation_era": (
                    "no DRS/energy/aero equivalence; Speed/Throttle/Brake observed only"
                ),
            }
            for name in EXTRA
        },
        "groups": {
            task: {group: feature_columns(task, group) for group in ("core", *GROUPS)}
            for task in ("qualifying", "race", "podium")
        },
        "limitations": [
            "Fuel load and run plans unknown",
            "No archived pre-cutoff forecasts obtained",
            "No sample resampling/edge interpolation; left-held discrete states, trapezoidal speed",
            "Lap selection and thresholds are recorded in options; no target-session weather",
        ],
    }
