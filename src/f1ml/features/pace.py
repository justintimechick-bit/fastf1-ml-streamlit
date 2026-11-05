"""Relative pace only within a session, qualifying part, time and condition cell."""

import numpy as np
import pandas as pd

from f1ml.config import FeatureConfig


def values(frame, *names, default=np.nan):
    for name in names:
        if name in frame:
            return frame[name]
    return pd.Series(default, index=frame.index)


def numbers(frame, *names):
    return pd.to_numeric(values(frame, *names), errors="coerce")


def seconds(frame, canonical, *source):
    if canonical in frame:
        return pd.to_numeric(frame[canonical], errors="coerce")
    return pd.to_timedelta(values(frame, *source), errors="coerce").dt.total_seconds()


def boolean(series):
    return series.map(
        lambda v: (
            True
            if str(v).strip().casefold() in ("true", "1", "1.0")
            else False
            if str(v).strip().casefold() in ("false", "0", "0.0")
            else pd.NA
        )
    ).astype("boolean")


def clean_laps(raw: pd.DataFrame):
    if raw.empty:
        return pd.DataFrame()
    frame = raw.copy().reset_index(drop=True)
    frame["driver_id"] = values(frame, "driver_id", "DriverId")
    frame["seconds"] = seconds(frame, "lap_time_seconds", "LapTime", "source_LapTime")
    frame["end"] = seconds(frame, "end_time_seconds", "Time", "source_Time")
    frame["start"] = seconds(frame, "lap_start_seconds", "LapStartTime", "source_LapStartTime")
    frame["start"] = frame.start.fillna(frame.end - frame.seconds)
    accurate = boolean(values(frame, "is_accurate", "IsAccurate", "source_IsAccurate"))
    deleted = boolean(values(frame, "deleted", "Deleted", "source_Deleted"))
    generated = boolean(
        values(frame, "fastf1_generated", "FastF1Generated", "source_FastF1Generated")
    )
    pit_known = any(n in frame for n in ("PitInTime", "source_PitInTime")) and any(
        n in frame for n in ("PitOutTime", "source_PitOutTime")
    )
    pit_in = pd.to_timedelta(values(frame, "PitInTime", "source_PitInTime"), errors="coerce")
    pit_out = pd.to_timedelta(values(frame, "PitOutTime", "source_PitOutTime"), errors="coerce")
    clear = values(frame, "TrackStatus", "source_TrackStatus").fillna("").astype(str).eq("1")
    mask = (
        accurate.eq(True)
        & deleted.eq(False)
        & ~generated.fillna(False)
        & pit_in.isna()
        & pit_out.isna()
        & clear
        & pit_known
        & frame.driver_id.notna()
        & np.isfinite(frame.seconds)
        & frame.seconds.gt(0)
    )
    return frame.loc[mask.fillna(False)].copy()


def comparable_pace(raw, weather, kind, options: FeatureConfig):
    """Return clean laps with optional matched-cell deficits; unknowns stay missing."""
    clean = clean_laps(raw)
    if clean.empty:
        return clean
    clean["gap_pct"] = np.nan
    clean["segment"] = (
        values(clean, "qualifying_segment", "source_QualifyingSegment", "QualifyingSegment")
        .fillna("")
        .astype(str)
        .str.upper()
        if kind == "Qualifying"
        else "PRACTICE"
    )
    clean["compound"] = (
        values(clean, "Compound", "source_Compound").fillna("").astype(str).str.upper()
    )
    if weather.empty:
        return clean
    observed = pd.DataFrame(
        {
            "weather_time": seconds(weather, "session_time_seconds", "Time", "source_Time"),
            "rainfall": boolean(values(weather, "rainfall", "Rainfall", "source_Rainfall")),
            "temperature": numbers(weather, "track_temp_c", "TrackTemp", "source_TrackTemp"),
        }
    )
    observed = observed.loc[np.isfinite(observed.weather_time)].sort_values("weather_time")
    if observed.weather_time.duplicated().any():
        raise ValueError("Duplicate weather timestamps need source review")
    candidates = clean.loc[np.isfinite(clean.start)].copy()
    if candidates.empty or observed.empty:
        return clean
    candidates["source_index"] = candidates.index
    matched = pd.merge_asof(
        candidates.sort_values("start"),
        observed,
        left_on="start",
        right_on="weather_time",
        direction="backward",
        tolerance=options.weather_max_age_seconds,
    )
    matched["time_bin"] = np.floor(matched.start / options.pace_time_bin_seconds)
    matched["temperature_bin"] = np.floor(matched.temperature / options.track_temperature_bin_c)
    known = (
        matched.rainfall.notna()
        & np.isfinite(matched.temperature_bin)
        & matched.compound.isin(["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"])
    )
    if kind == "Qualifying":
        known &= matched.segment.isin(["Q1", "Q2", "Q3"])
    cells = ["segment", "compound", "time_bin", "rainfall", "temperature_bin"]
    # The caller passes one GP session at a time, so no session or circuit is mixed.
    for _, group in matched.loc[known].groupby(cells, dropna=False, sort=True):
        if group.driver_id.nunique() < options.min_comparable_drivers:
            continue
        reference = group.seconds.min()
        clean.loc[group.source_index, "gap_pct"] = 100 * (group.seconds.to_numpy() / reference - 1)
    return clean


def summarize_driver(laps, driver, segment=None):
    if laps.empty:
        return {"gap": np.nan, "clean": 0, "comparable": 0}
    selected = laps.loc[laps.driver_id.eq(driver)]
    if segment is not None:
        selected = selected.loc[selected.segment.eq(segment)]
    known = selected.gap_pct.dropna()
    return {
        "gap": float(known.median()) if len(known) else np.nan,
        "clean": len(selected),
        "comparable": len(known),
    }
