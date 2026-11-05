"""Materialize every discovered weekend and publish season/event/session coverage."""

import json
from pathlib import Path
from uuid import uuid4

import pandas as pd

from f1ml.domain import GP_SESSIONS, PRACTICES, utc
from f1ml.ingestion.tables import TABLE_KEYS, assert_keys, empty_table
from f1ml.ingestion.types import SCHEMA_VERSION, TRANSFORM_VERSION, session_id
from f1ml.storage import read_json, save_table, seal_directory, write_json


def materialize(root: Path, state: dict, stats: dict) -> dict:
    from f1ml.ingestion.catalog import atomic_json, load_record, now

    loaded = {}
    sources = {
        name: []
        for name in TABLE_KEYS
        if name not in ("events", "sessions", "drivers", "team_aliases")
    }
    failures = []
    for job in state["jobs"].values():
        relative = job["latest"]
        if relative not in loaded:
            loaded[relative] = load_record(root, relative)
        meta = loaded[relative]
        for error in (
            meta.get("errors", [])
            + meta.get("normalization_errors", [])
            + meta.get("attempt_failures", [])
        ):
            failures.append(
                {
                    "event_id": meta["event_id"],
                    "session_id": meta["session_id"],
                    "kind": meta["kind"],
                    "component": meta["component"],
                    "retrieved_at": meta["retrieved_at"],
                    "type": error.get("type", "SourceDiagnostic"),
                    "message": error.get("message", ""),
                    "revision": relative,
                }
            )
        for table, pointer in job["table_sources"].items():
            if pointer not in loaded:
                loaded[pointer] = load_record(root, pointer)
            frame = pd.read_parquet(root / pointer / "normalized" / f"{table}.parquet")
            for column in frame:
                if column.startswith("source_"):
                    frame[column] = frame[column].map(
                        lambda value: None if pd.isna(value) else str(value)
                    )
            frame["source_revision"] = pointer
            frame["stale_after_latest_attempt"] = pointer != relative
            sources[table].append(frame)
    tables = {
        name: pd.concat(parts, ignore_index=True) if parts else empty_table(name)
        for name, parts in sources.items()
    }
    entries = tables["driver_entries"]
    if not entries.empty:
        resolved = entries.loc[entries.driver_id.notna()].sort_values("retrieved_at")
        # Driver identity contains no team membership; observations keep their own teams.
        drivers = resolved.groupby("driver_id", as_index=False).agg(
            first_name=("first_name", "last"),
            last_name=("last_name", "last"),
            abbreviation=("abbreviation", "last"),
            last_observed_at=("retrieved_at", "max"),
            identity_observations=("driver_id", "size"),
        )
        aliases = entries[
            [
                "year",
                "source_team_id",
                "source_team_name",
                "team_id",
                "team_history_id",
                "team_mapping_state",
            ]
        ].drop_duplicates()
        aliases = aliases.fillna({"source_team_id": "", "source_team_name": ""})
    else:
        drivers = pd.DataFrame(columns=TABLE_KEYS["drivers"])
        aliases = pd.DataFrame(columns=TABLE_KEYS["team_aliases"])
    tables.update(drivers=drivers, team_aliases=aliases)

    session_rows, event_rows = [], []
    for event in state["events"]:
        event_rows.append(
            {k: v for k, v in event.items() if k not in ("sessions", "schedule_source_columns")}
        )
        for spec in event["sessions"]:
            identifier = session_id(event["event_id"], spec["kind"])
            future = utc(spec["scheduled_start"]) >= utc(state["as_of"])
            outside = spec["kind"] not in GP_SESSIONS
            results_job = state["jobs"].get(f"{identifier}|results", {})
            timing_job = state["jobs"].get(f"{identifier}|timing", {})
            result_meta = loaded.get(results_job.get("latest"), {})
            timing_meta = loaded.get(timing_job.get("latest"), {})
            table_counts = {}
            for name in (
                "results",
                "driver_entries",
                "practice_laps",
                "qualifying_laps",
                "session_weather",
            ):
                frame = tables[name]
                sub = frame.loc[frame.session_id.eq(identifier)] if not frame.empty else frame
                table_counts[name] = sub
            results = table_counts["results"]
            participation = table_counts["driver_entries"]
            result_count = len(results)
            driver_count = participation.driver_number.nunique() if not participation.empty else 0
            expected_target = spec["kind"] in ("Qualifying", "Race")
            entries_without_results = 0
            if expected_target and driver_count:
                observed = set(participation.driver_number.dropna())
                reported = set(results.driver_number.dropna()) if result_count else set()
                entries_without_results = len(observed - reported)
            missing_targets = (
                int(results.official_position.isna().sum()) + entries_without_results
                if result_count
                else driver_count
                if driver_count and expected_target
                else None
            )
            unknown_podium = (
                int(results.podium.isna().sum()) + entries_without_results
                if result_count and spec["kind"] == "Race"
                else None
            )
            lap_count = len(table_counts["practice_laps"]) + len(table_counts["qualifying_laps"])
            weather_count = len(table_counts["session_weather"])
            blocked = (
                not timing_job
                and state["timing_circuit"]["blocked_until"]
                and not outside
                and not future
            )
            rate_blocked = (
                state.get("results_circuit", {}).get("blocked_until") and not outside and not future
            )

            def availability(
                job,
                count,
                applicable=True,
                *,
                timing=True,
                outside=outside,
                future=future,
                blocked=blocked,
                rate_blocked=rate_blocked,
            ):
                if not applicable:
                    return "not_applicable"
                if outside:
                    return "out_of_scope"
                if future:
                    return "not_started"
                if count:
                    return "available" if job.get("status") == "complete" else "partial_or_stale"
                if not job:
                    if rate_blocked:
                        return "blocked_rate_limit"
                    return "blocked_timing_access" if blocked and timing else "not_requested"
                return job.get("status", "unknown")

            complete = timing_meta.get("completion_state", "unknown")
            if future:
                complete = "not_started"
            elif complete == "unknown" and result_count:
                complete = "results_available"
            unusual = (
                results.loc[~results.outcome.eq("ordered"), "outcome"].value_counts().to_dict()
                if result_count
                else {}
            )
            ambiguous = (
                int(results.position_ambiguous.sum())
                if result_count and "position_ambiguous" in results
                else 0
            )
            if ambiguous:
                unusual["ambiguous_official_position"] = ambiguous
            record = {
                "session_id": identifier,
                "event_id": event["event_id"],
                "year": event["year"],
                "event_name": event["name"],
                "event_format": event.get("event_format", "unknown"),
                "kind": spec["kind"],
                "source_kind": spec.get("source_kind", spec["kind"]),
                "scheduled_start": spec["scheduled_start"],
                "completion_state": complete,
                "completion_evidence": timing_meta.get(
                    "completion_evidence", result_meta.get("completion_evidence", "none")
                ),
                "completed_at": timing_meta.get("completed_at"),
                "absolute_completion_time_missing": not timing_meta.get(
                    "absolute_time_available", False
                ),
                "completion_observed_at": timing_meta.get("retrieved_at"),
                "driver_count": int(driver_count),
                "result_driver_count": result_count,
                "resolved_driver_count": int(participation.driver_id.nunique())
                if driver_count
                else 0,
                "unresolved_entries": int(participation.driver_id.isna().sum())
                if driver_count
                else 0,
                "unmapped_team_entries": int(participation.team_mapping_state.eq("unmapped").sum())
                if driver_count
                else 0,
                "results_rows": result_count,
                "laps_rows": lap_count,
                "weather_rows": weather_count,
                "results_availability": availability(
                    results_job, result_count, expected_target, timing=False
                ),
                "laps_availability": availability(
                    timing_job, lap_count, spec["kind"] in PRACTICES | {"Qualifying"}
                ),
                "weather_availability": availability(timing_job, weather_count),
                "official_results_complete": bool(
                    result_count and results.official_field_complete.all()
                ),
                "missing_rank_targets": missing_targets,
                "entries_without_results": entries_without_results,
                "ambiguous_rank_targets": ambiguous,
                "missing_podium_targets": unknown_podium,
                "target_missing_all": bool(expected_target and not result_count),
                "unusual_classifications": json.dumps(unusual, sort_keys=True),
                "unusual_count": int(sum(unusual.values())),
                "results_job_state": results_job.get(
                    "status", "blocked_rate_limit" if rate_blocked else "not_requested"
                ),
                "timing_job_state": timing_job.get(
                    "status", "blocked" if blocked else "not_requested"
                ),
                "failure_count": len(result_meta.get("errors", []))
                + len(timing_meta.get("errors", []))
                + len(result_meta.get("normalization_errors", []))
                + len(timing_meta.get("normalization_errors", []))
                + len(result_meta.get("attempt_failures", []))
                + len(timing_meta.get("attempt_failures", [])),
                "warning_count": len(timing_meta.get("warnings", [])),
                "results_source_columns": json.dumps(
                    result_meta.get("source_columns", {}), sort_keys=True
                ),
                "timing_source_columns": json.dumps(
                    timing_meta.get("source_columns", {}), sort_keys=True
                ),
                "results_retrieved_at": result_meta.get("retrieved_at"),
                "timing_retrieved_at": timing_meta.get("retrieved_at"),
                "fastf1_version": result_meta.get(
                    "fastf1_version", timing_meta.get("fastf1_version")
                ),
                "transform_version": TRANSFORM_VERSION,
            }
            session_rows.append(record)
    tables["events"] = (
        pd.DataFrame(event_rows) if event_rows else pd.DataFrame(columns=["event_id", "year"])
    )
    sessions = (
        pd.DataFrame(session_rows)
        if session_rows
        else pd.DataFrame(columns=["session_id", "year", "event_id"])
    )
    tables["sessions"] = sessions
    for name, frame in tables.items():
        assert_keys(frame, name)
    event_coverage = []
    for event in event_rows:
        group = sessions.loc[sessions.event_id.eq(event["event_id"])]
        practice = group.loc[group.kind.isin(PRACTICES)]
        gp_targets = group.loc[group.kind.isin(["Qualifying", "Race"])]
        event_coverage.append(
            {
                "event_id": event["event_id"],
                "year": event["year"],
                "name": event["name"],
                "event_format": event.get("event_format", "unknown"),
                "sessions": len(group),
                "gp_sessions": int(group.kind.isin(GP_SESSIONS).sum()),
                "driver_count": int(gp_targets.driver_count.max()) if not gp_targets.empty else 0,
                "result_sessions": int((gp_targets.results_rows > 0).sum()),
                "practice_expected": len(practice),
                "practice_with_laps": int((practice.laps_rows > 0).sum()),
                "qualifying_with_laps": int(
                    ((group.kind == "Qualifying") & (group.laps_rows > 0)).sum()
                ),
                "sessions_with_weather": int((group.weather_rows > 0).sum()),
                "sessions_missing_targets": int(
                    gp_targets.loc[
                        gp_targets.completion_state.ne("not_started"), "target_missing_all"
                    ].sum()
                ),
                "unusual_classifications": int(group.unusual_count.sum()),
                "failures": int(group.failure_count.sum()),
                "blocked_sessions": int(group.timing_job_state.eq("blocked").sum()),
                "future_sessions": int(group.completion_state.eq("not_started").sum()),
            }
        )
    event_coverage = pd.DataFrame(event_coverage)
    season_coverage = []
    for year in state["years"]:
        group = (
            event_coverage.loc[event_coverage.year.eq(year)]
            if not event_coverage.empty
            else pd.DataFrame()
        )
        row = {
            "year": year,
            "schedule_available": year in state["discovered_years"],
            "events": len(group),
        }
        row["driver_count"] = (
            int(entries.loc[entries.year.eq(year), "driver_id"].nunique())
            if not entries.empty
            else 0
        )
        fields = (
            sessions.loc[
                sessions.year.eq(year) & sessions.results_rows.gt(0), "result_driver_count"
            ]
            if not sessions.empty
            else pd.Series(dtype=int)
        )
        row["result_field_min"] = int(fields.min()) if not fields.empty else None
        row["result_field_max"] = int(fields.max()) if not fields.empty else None
        for column in (
            "result_sessions",
            "practice_expected",
            "practice_with_laps",
            "qualifying_with_laps",
            "sessions_with_weather",
            "sessions_missing_targets",
            "unusual_classifications",
            "failures",
            "blocked_sessions",
            "future_sessions",
        ):
            row[column] = int(group[column].sum()) if not group.empty else 0
        season_coverage.append(row)
    destination = root / "snapshots" / uuid4().hex
    destination.mkdir(parents=True)
    for name, frame in tables.items():
        save_table(destination / "tables" / f"{name}.parquet", frame)
    failure_frame = pd.DataFrame(
        failures,
        columns=[
            "event_id",
            "session_id",
            "kind",
            "component",
            "retrieved_at",
            "type",
            "message",
            "revision",
        ],
    )
    for name, frame in {
        "sessions": sessions,
        "events": event_coverage,
        "seasons": pd.DataFrame(season_coverage),
        "failures": failure_frame,
    }.items():
        save_table(destination / "coverage" / f"{name}.parquet", frame)
        frame.to_csv(destination / "coverage" / f"{name}.csv", index=False)
    write_json(destination / "discovery_failures.json", state["discovery_errors"])
    write_json(destination / "team_mapping.json", read_json(root / "team_mapping.json"))
    receipt = seal_directory(
        destination,
        {
            "schema_version": SCHEMA_VERSION,
            "created_at": now().isoformat(),
            "as_of": state["as_of"],
            "transform_version": TRANSFORM_VERSION,
            "team_mapping_id": state["team_mapping_id"],
            "job_revisions": {key: job["latest"] for key, job in state["jobs"].items()},
            "timing_circuit": state["timing_circuit"],
            "results_circuit": state.get("results_circuit", {"blocked_until": None}),
            "run": stats,
            "discovery_failures": len(state["discovery_errors"]),
            "failure_rows": len(failure_frame),
            "partial_jobs": sum(job["status"] != "complete" for job in state["jobs"].values()),
        },
    )
    atomic_json(
        root / "CURRENT.json", {"path": str(destination.relative_to(root)), "id": receipt["id"]}
    )
    return {**receipt, "snapshot": str(destination)}


def inspect_coverage(root: Path, level: str = "seasons") -> pd.DataFrame:
    from f1ml.ingestion.catalog import current_snapshot

    if level not in ("seasons", "events", "sessions", "failures"):
        raise ValueError("Coverage level must be seasons, events, sessions or failures")
    snapshot, _ = current_snapshot(root)
    return pd.read_parquet(snapshot / "coverage" / f"{level}.parquet")
