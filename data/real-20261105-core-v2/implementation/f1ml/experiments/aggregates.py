"""Small observed-channel aggregates; no merged/resampled telemetry or inferred car state."""

import numpy as np
import pandas as pd

from f1ml.features.pace import boolean, clean_laps, comparable_pace, numbers, seconds, values

CELL = ["segment", "compound", "time_bin", "rain", "temperature_bin"]


def cells(raw, weather, kind, options):
    clean = comparable_pace(raw, weather, kind, options)
    if clean.empty or weather.empty:
        return pd.DataFrame()
    observed = (
        pd.DataFrame(
            {
                "weather_time": seconds(weather, "session_time_seconds", "Time", "source_Time"),
                "rain": boolean(values(weather, "rainfall", "Rainfall", "source_Rainfall")),
                "temperature": numbers(weather, "track_temp_c", "TrackTemp", "source_TrackTemp"),
            }
        )
        .dropna()
        .sort_values("weather_time")
    )
    if observed.empty:
        return pd.DataFrame()
    if observed.weather_time.duplicated().any():
        raise ValueError("Duplicate weather timestamps")
    clean = clean.loc[np.isfinite(clean.start)].sort_values("start")
    matched = pd.merge_asof(
        clean,
        observed,
        left_on="start",
        right_on="weather_time",
        direction="backward",
        tolerance=options.weather_max_age_seconds,
    )
    matched["time_bin"] = np.floor(matched.start / options.pace_time_bin_seconds)
    matched["temperature_bin"] = np.floor(matched.temperature / options.track_temperature_bin_c)
    matched = matched.loc[matched[CELL].notna().all(axis=1) & matched.gap_pct.notna()].copy()
    for number in (1, 2, 3):
        matched[f"sector{number}"] = seconds(
            matched, f"sector{number}_seconds", f"Sector{number}Time", f"source_Sector{number}Time"
        )
    return matched


def pace_summary(comparable, driver):
    ranks, lap_cvs, sector_cvs, laps = [], [], [], 0
    if comparable.empty:
        return {"ranks": [], "lap_cvs": [], "sector_cvs": [], "laps": 0}
    for _, cell in comparable.groupby(CELL, sort=True):
        best = cell.groupby("driver_id").seconds.min()
        if driver in best.index and len(best) >= 2:
            ranks.append(float((best.rank(method="min").loc[driver] - 1) / (len(best) - 1)))
        own = cell.loc[cell.driver_id.eq(driver)]
        if len(own) < 2:
            continue
        lap_cvs.append(float(own.seconds.std(ddof=0) / own.seconds.mean()))
        laps += len(own)
        for name in ("sector1", "sector2", "sector3"):
            known = own[name].dropna()
            known = known.loc[known.gt(0)]
            if len(known) >= 2:
                sector_cvs.append(float(known.std(ddof=0) / known.mean()))
    return {"ranks": ranks, "lap_cvs": lap_cvs, "sector_cvs": sector_cvs, "laps": laps}


def lap_telemetry(raw, lap, options):
    """Integrate valid adjacent intervals only, with independent channel denominators."""
    result = {
        "speed_mean_kph": np.nan,
        "speed_p95_kph": np.nan,
        "full_throttle_fraction": np.nan,
        "braking_fraction": np.nan,
        "speed_coverage": 0.0,
        "throttle_coverage": 0.0,
        "brake_coverage": 0.0,
    }
    if raw.empty:
        return result
    frame = raw.copy()
    if "SessionTime" not in frame:
        return result
    frame["t"] = pd.to_timedelta(frame.SessionTime, errors="coerce").dt.total_seconds()
    if "Source" in frame:
        frame = frame.loc[frame.Source.eq("car")]
    frame = frame.loc[frame.t.between(float(lap["start"]), float(lap["end"]))].sort_values("t")
    if frame.t.duplicated().any():
        raise ValueError("Duplicate telemetry timestamps require source review")
    t = frame.t.to_numpy(float)
    dt = np.diff(t)
    duration = float(lap["seconds"])
    if duration <= 0 or len(dt) == 0:
        return result
    valid_interval = (dt > 0) & (dt <= options.max_sample_gap_seconds)
    for channel, name in (("Speed", "speed"), ("Throttle", "throttle"), ("Brake", "brake")):
        source = values(frame, channel)
        data = (
            boolean(source).astype("Float64").to_numpy(dtype=float, na_value=np.nan)
            if channel == "Brake"
            else pd.to_numeric(source, errors="coerce").to_numpy(float)
        )
        valid = np.isfinite(data) & (data >= 0)
        if channel == "Throttle":
            valid &= data <= 100  # e.g. FastF1's observed 104 sentinel is not full throttle.
        mask = valid_interval & valid[:-1] & valid[1:]
        observed = float(dt[mask].sum())
        result[f"{name}_coverage"] = min(observed / duration, 1.0)
        if observed <= 0 or observed / duration < options.min_channel_coverage:
            continue
        weights, left, right = dt[mask], data[:-1][mask], data[1:][mask]
        if channel == "Speed":
            result["speed_mean_kph"] = float(np.sum((left + right) / 2 * weights) / observed)
            order = np.argsort(left, kind="stable")
            index = np.searchsorted(np.cumsum(weights[order]), 0.95 * observed)
            result["speed_p95_kph"] = float(left[order[min(index, len(order) - 1)]])
        elif channel == "Throttle":
            result["full_throttle_fraction"] = float(
                weights[left >= options.full_throttle_percent].sum() / observed
            )
        else:
            result["braking_fraction"] = float(weights[left == 1].sum() / observed)
    return result


def telemetry_summaries(frames, comparable, options):
    raw, laps = frames.get("telemetry", pd.DataFrame()), frames.get("laps", pd.DataFrame())
    rows = []
    if raw.empty or laps.empty:
        return pd.DataFrame()
    laps = clean_laps(laps)
    for lap in laps.to_dict("records"):
        selected = raw.loc[
            raw.driver_id.eq(lap["driver_id"]) & raw.lap_number.eq(lap["lap_number"])
        ]
        aggregate = lap_telemetry(selected, lap, options)
        aggregate.update(driver_id=lap["driver_id"], lap_number=lap["lap_number"])
        if not comparable.empty and "lap_number" in comparable:
            match = comparable.loc[
                comparable.driver_id.eq(lap["driver_id"])
                & comparable.lap_number.eq(lap["lap_number"])
            ]
            if len(match) == 1:
                aggregate.update({name: match.iloc[0][name] for name in CELL})
        rows.append(aggregate)
    return pd.DataFrame(rows)


def teammate_deltas(aggregates, entrants, driver):
    keys = ("speed_mean_kph", "full_throttle_fraction", "braking_fraction")
    result = {key: [] for key in keys}
    if aggregates.empty or not set(CELL).issubset(aggregates):
        return result, 0
    own_team = entrants.loc[entrants.driver_id.eq(driver), "team_id"].iloc[0]
    teammates = entrants.loc[
        entrants.team_id.eq(own_team) & entrants.driver_id.ne(driver), "driver_id"
    ]
    count = 0
    for _, group in aggregates.dropna(subset=CELL).groupby(CELL, sort=True):
        own, other = (
            group.loc[group.driver_id.eq(driver)],
            group.loc[group.driver_id.isin(teammates)],
        )
        if own.empty or other.empty:
            continue
        contributed = False
        for key in keys:
            if own[key].notna().any() and other[key].notna().any():
                result[key].append(float(own[key].mean() - other[key].mean()))
                contributed = True
        count += contributed
    return result, count


def weather_summary(raw):
    if raw.empty:
        return {}
    result = {}
    time = seconds(raw, "session_time_seconds", "Time", "source_Time")
    order = np.argsort(time.to_numpy(float), kind="stable")
    raw = raw.iloc[order]
    elapsed = time.iloc[order].to_numpy(float)
    dt = np.diff(elapsed)
    weights = np.where((dt > 0) & (dt <= 120), dt, 0.0)

    def weighted(values, allow_single=False):
        data = values.to_numpy(dtype=float, na_value=np.nan)
        if len(data) == 1 and allow_single:
            return float(data[0])
        known = np.isfinite(data[:-1]) & np.isfinite(data[1:]) & (weights > 0)
        return (
            float(np.average(data[:-1][known], weights=weights[known])) if known.any() else np.nan
        )

    for target, canonical, source in (
        ("observed_air_temp_c", "air_temp_c", "AirTemp"),
        ("observed_track_temp_c", "track_temp_c", "TrackTemp"),
        ("observed_humidity_pct", "humidity_pct", "Humidity"),
        ("observed_wind_speed_mps", "wind_speed_mps", "WindSpeed"),
    ):
        result[target] = weighted(
            numbers(raw, canonical, source, f"source_{source}"), allow_single=True
        )
    rain = boolean(values(raw, "rainfall", "Rainfall", "source_Rainfall")).astype("Float64")
    result["observed_rain_fraction"] = weighted(rain)
    result["observed_weather_samples"] = len(raw)
    return result
