"""Normalized table contracts, source preservation, and validation."""

import json
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from f1ml.ingestion.identity import TeamMapping
from f1ml.ingestion.normalize import normalize_results, positive_position
from f1ml.ingestion.types import TRANSFORM_VERSION, SourceBatch, session_id

TABLE_KEYS = {
    "events": ["event_id"],
    "sessions": ["session_id"],
    "drivers": ["driver_id"],
    "driver_entries": ["session_id", "entry_source", "driver_number"],
    "results": ["session_id", "driver_id"],
    "practice_laps": ["session_id", "driver_number", "lap_key"],
    "qualifying_laps": ["session_id", "driver_number", "lap_key"],
    "session_weather": ["session_id", "sample_key"],
    "session_status": ["session_id", "status_index"],
    "team_aliases": ["year", "source_team_id", "source_team_name"],
}


EMPTY_COLUMNS = {
    "driver_entries": [
        "event_id",
        "year",
        "kind",
        "driver_id",
        "team_id",
        "team_history_id",
        "identity_state",
        "team_mapping_state",
        "source_team_id",
        "source_team_name",
    ],
    "results": [
        "event_id",
        "year",
        "kind",
        "driver_number",
        "official_position",
        "classification_code",
        "outcome",
        "is_classified",
        "is_retired",
        "podium",
        "position_missing",
        "position_ambiguous",
        "official_field_complete",
    ],
    "practice_laps": [
        "event_id",
        "year",
        "kind",
        "driver_id",
        "lap_number",
        "lap_time_seconds",
        "end_time_seconds",
        "lap_start_seconds",
        "lap_time_missing",
        "lap_number_missing",
        "is_accurate",
        "deleted",
        "fastf1_generated",
        "absolute_time_missing",
        "identity_missing",
    ],
    "qualifying_laps": [
        "event_id",
        "year",
        "kind",
        "driver_id",
        "lap_number",
        "lap_time_seconds",
        "end_time_seconds",
        "lap_start_seconds",
        "lap_time_missing",
        "lap_number_missing",
        "is_accurate",
        "deleted",
        "fastf1_generated",
        "absolute_time_missing",
        "identity_missing",
    ],
    "session_weather": [
        "event_id",
        "year",
        "kind",
        "session_time_seconds",
        "time_missing",
        "air_temp_c",
        "track_temp_c",
        "humidity_pct",
        "pressure_mbar",
        "wind_speed_mps",
        "wind_direction_deg",
        "rainfall",
    ],
    "session_status": ["event_id", "year", "kind", "session_time_seconds", "status"],
}


def empty_table(name):
    return pd.DataFrame(columns=list(dict.fromkeys(TABLE_KEYS[name] + EMPTY_COLUMNS.get(name, []))))


def text_value(value):
    return None if value is None or pd.isna(value) else str(value)


def json_value(value):
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, pd.Timedelta, datetime, date, timedelta)):
        return str(value)
    return value


def source_columns(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.reset_index(drop=True)
    # Stable string representation avoids mixed Timedelta/string columns when the
    # official-results and Session interfaces expose the same field differently.
    result = raw.map(text_value).add_prefix("source_")
    result["source_record_json"] = [
        json.dumps(json_value(row), sort_keys=True, default=str) for row in raw.to_dict("records")
    ]
    return result


def assert_keys(table: pd.DataFrame, name: str) -> None:
    keys = TABLE_KEYS[name]
    if table.empty:
        return
    if not set(keys).issubset(table.columns) or table[keys].isna().any().any():
        raise ValueError(f"Missing {name} key: {keys}")
    if table.duplicated(keys).any():
        raise ValueError(f"Duplicate {name} keys: {keys}")
    allowed = {
        "results": {"Qualifying", "Race"},
        "practice_laps": {"Practice 1", "Practice 2", "Practice 3"},
        "qualifying_laps": {"Qualifying"},
        "session_weather": {"Practice 1", "Practice 2", "Practice 3", "Qualifying", "Race"},
    }
    if name in allowed and "kind" in table and not table.kind.isin(allowed[name]).all():
        raise ValueError(f"Wrong session kind in {name}")
    if {"event_id", "kind", "session_id"}.issubset(table.columns):
        expected = [
            session_id(event, kind) for event, kind in zip(table.event_id, table.kind, strict=True)
        ]
        if table.session_id.tolist() != expected:
            raise ValueError(f"Session identity/kind mismatch in {name}")


def entries(raw: pd.DataFrame, year: int, mapping: TeamMapping, entry_source: str):
    if raw.empty:
        return pd.DataFrame()
    rows = []
    for source in raw.to_dict("records"):

        def get(*names, source=source):
            return next(
                (
                    text_value(source[n])
                    for n in names
                    if n in source and text_value(source[n]) not in (None, "")
                ),
                None,
            )

        driver = get("driverId", "DriverId")
        number = get("number", "DriverNumber")
        if number is None:
            raise ValueError("Entry has no driver number; cannot associate session observations")
        constructor = get("constructorId", "TeamId")
        name = get("constructorName", "TeamName")
        rows.append(
            {
                "driver_id": driver,
                "driver_number": number,
                "entry_source": entry_source,
                "identity_state": "resolved" if driver else "unresolved",
                "source_team_id": constructor,
                "source_team_name": name,
                "first_name": get("givenName", "FirstName"),
                "last_name": get("familyName", "LastName"),
                "abbreviation": get("driverCode", "Abbreviation"),
                **mapping.resolve(year, constructor, name),
            }
        )
    return pd.concat([pd.DataFrame(rows), source_columns(raw)], axis=1)


def seconds(raw: pd.DataFrame, column: str) -> pd.Series:
    if column not in raw:
        return pd.Series(np.nan, index=raw.index)
    return pd.to_timedelta(raw[column], errors="coerce").dt.total_seconds()


def normalize_laps(raw: pd.DataFrame, driver_entries: pd.DataFrame) -> pd.DataFrame:
    raw = raw.reset_index(drop=True)
    if raw.empty:
        return pd.DataFrame()
    if "DriverNumber" not in raw:
        raise ValueError("Lap source is missing DriverNumber")
    frame = pd.DataFrame({"driver_number": raw.DriverNumber.map(text_value)})
    frame["lap_number"] = positive_position(
        raw.get("LapNumber", pd.Series(np.nan, index=raw.index))
    )
    frame["lap_key"] = [
        str(int(n)) if pd.notna(n) else f"missing:{index}"
        for index, n in enumerate(frame.lap_number)
    ]
    for target, source in (
        ("lap_time_seconds", "LapTime"),
        ("end_time_seconds", "Time"),
        ("lap_start_seconds", "LapStartTime"),
    ):
        frame[target] = seconds(raw, source)
    frame["lap_time_missing"] = frame.lap_time_seconds.isna()
    frame["lap_number_missing"] = frame.lap_number.isna()
    for target, source in (
        ("is_accurate", "IsAccurate"),
        ("deleted", "Deleted"),
        ("fastf1_generated", "FastF1Generated"),
    ):
        frame[target] = raw.get(source, pd.Series(pd.NA, index=raw.index)).astype("boolean")
    frame["absolute_time_missing"] = raw.LapStartDate.isna() if "LapStartDate" in raw else True
    if not driver_entries.empty:
        lookup = driver_entries[["driver_number", "driver_id"]]
        frame = frame.merge(
            lookup, on="driver_number", how="left", validate="many_to_one", sort=False
        )
    else:
        frame["driver_id"] = None
    frame["identity_missing"] = frame.driver_id.isna()
    return pd.concat([frame.reset_index(drop=True), source_columns(raw)], axis=1)


def normalize_weather(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.reset_index(drop=True)
    if raw.empty:
        return pd.DataFrame()
    elapsed = seconds(raw, "Time")
    keys = [str(float(t)) if pd.notna(t) else f"missing:{index}" for index, t in enumerate(elapsed)]
    frame = pd.DataFrame(
        {"sample_key": keys, "session_time_seconds": elapsed, "time_missing": elapsed.isna()}
    )
    for source, target in (
        ("AirTemp", "air_temp_c"),
        ("TrackTemp", "track_temp_c"),
        ("Humidity", "humidity_pct"),
        ("Pressure", "pressure_mbar"),
        ("WindSpeed", "wind_speed_mps"),
        ("WindDirection", "wind_direction_deg"),
    ):
        frame[target] = pd.to_numeric(
            raw.get(source, pd.Series(np.nan, index=raw.index)), errors="coerce"
        )
        frame[f"{target}_missing"] = frame[target].isna()
    frame["rainfall"] = raw.get("Rainfall", pd.Series(pd.NA, index=raw.index)).astype("boolean")
    return pd.concat([frame, source_columns(raw)], axis=1)


def normalize_batch(
    batch: SourceBatch, event: dict, spec: dict, mapping: TeamMapping, component: str
) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    output, errors = {}, []
    metadata = batch.metadata

    def add(name, operation):
        try:
            frame = operation()
            if frame.empty:
                return
            for column, value in {
                "event_id": event["event_id"],
                "year": event["year"],
                "session_id": session_id(event["event_id"], spec["kind"]),
                "kind": spec["kind"],
                "retrieved_at": metadata["retrieved_at"],
                "fastf1_version": metadata["fastf1_version"],
                "transform_version": TRANSFORM_VERSION,
                "source": metadata["source"],
            }.items():
                frame[column] = value
            assert_keys(frame, name)
            output[name] = frame
        except (ValueError, KeyError, TypeError) as exc:
            errors.append({"component": name, "type": type(exc).__name__, "message": str(exc)})

    raw_entries = batch.frames.get(
        "results" if component == "results" else "entries", pd.DataFrame()
    )
    source = "official_results" if component == "results" else "session_participation"
    add("driver_entries", lambda: entries(raw_entries, event["year"], mapping, source))
    if component == "results":

        def results():
            raw = batch.frames.get("results", pd.DataFrame())
            result = normalize_results(raw, spec["kind"], allow_ambiguous_positions=True)
            if result.empty:
                return result
            result["source_team_id"] = result.team_id
            for index, row in result.iterrows():
                resolved = mapping.resolve(
                    event["year"],
                    text_value(row.team_id),
                    text_value(row.get("source_constructorName")),
                )
                for key, value in resolved.items():
                    result.loc[index, key] = value
            result["official_field_complete"] = metadata.get("response_complete", False)
            return result

        add("results", results)
    else:
        if metadata.get("completion_state") == "completed":
            name = "qualifying_laps" if spec["kind"] == "Qualifying" else "practice_laps"
            if spec["kind"] != "Race":
                add(
                    name,
                    lambda: normalize_laps(
                        batch.frames.get("laps", pd.DataFrame()),
                        output.get("driver_entries", pd.DataFrame()),
                    ),
                )
        add(
            "session_weather",
            lambda: normalize_weather(batch.frames.get("weather", pd.DataFrame())),
        )

        def statuses():
            raw = batch.frames.get("status", pd.DataFrame()).reset_index(drop=True)
            if raw.empty:
                return raw
            return pd.concat(
                [
                    pd.DataFrame(
                        {
                            "status_index": range(len(raw)),
                            "session_time_seconds": seconds(raw, "Time"),
                            "status": raw.Status,
                        }
                    ),
                    source_columns(raw),
                ],
                axis=1,
            )

        add("session_status", statuses)
    return output, errors
