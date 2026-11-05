"""Discover schedules, then save an auditable local source revision."""

import logging
from pathlib import Path

import pandas as pd

from f1ml.config import Config
from f1ml.domain import GP_SESSIONS, event_key, session_kind, utc
from f1ml.storage import (
    new_directory,
    read_json,
    save_table,
    seal_directory,
    verify_directory,
    write_json,
)

log = logging.getLogger(__name__)


def discover(provider, config: Config, output: Path) -> dict:
    events = [event for year in config.years for event in provider.schedule(year)]
    validate_events(events)
    schedule = {
        "schema_version": 1,
        "discovered_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "events": events,
    }
    write_json(output, schedule)
    return schedule


def validate_events(events: list[dict]) -> None:
    seen = set()
    for event in events:
        if event_key(event["year"], event["round"]) != event["event_id"]:
            raise ValueError("Event identity disagrees with year/round")
        if event["event_id"] in seen:
            raise ValueError("Duplicate event in schedule")
        seen.add(event["event_id"])
        kinds = [session_kind(s["kind"]) for s in event["sessions"]]
        if len(kinds) != len(set(kinds)):
            raise ValueError("Duplicate session kind in event")
        for spec in event["sessions"]:
            utc(spec["scheduled_start"])


def ingest(provider, schedule: dict, output: Path, as_of, selected: list[str] | None = None):
    as_of = utc(as_of)
    validate_events(schedule["events"])
    selected_ids = set(selected) if selected else {e["event_id"] for e in schedule["events"]}
    if selected_ids - {e["event_id"] for e in schedule["events"]}:
        raise ValueError("Selected event is not in the discovered schedule")
    new_directory(output)
    events = []
    failures = []
    for event in schedule["events"]:
        if event["event_id"] not in selected_ids:
            continue
        loaded = {**event, "sessions": []}
        for spec in event["sessions"]:
            meta = {**spec, "completed": False, "official_results": False}
            kind = session_kind(spec["kind"])
            if kind not in GP_SESSIONS:
                meta["skipped"] = "sprint_out_of_scope"
            elif utc(spec["scheduled_start"]) >= as_of:
                meta["skipped"] = "not_started_at_as_of"
            else:
                try:
                    meta, table = provider.session(event, spec, as_of)
                    if meta["completed"] and not table.empty:
                        relative = f"{event['event_id']}/{kind.value.replace(' ', '_')}.parquet"
                        save_table(output / relative, table)
                        meta["table"] = relative
                except Exception as exc:
                    log.exception("Failed ingestion: %s %s", event["event_id"], kind)
                    meta["error"] = f"{type(exc).__name__}: {exc}"
                    failures.append(
                        {"event_id": event["event_id"], "kind": kind.value, "error": meta["error"]}
                    )
            loaded["sessions"].append(meta)
        events.append(loaded)
    write_json(output / "events.json", events)
    return seal_directory(
        output,
        {
            "schema_version": 1,
            "as_of": as_of.isoformat(),
            "source": "FastF1",
            "failures": failures,
            "schedule": schedule,
        },
    )


class Dataset:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.receipt = verify_directory(self.root)
        self.events = read_json(self.root / "events.json")
        validate_events(self.events)

    def event(self, event_id: str) -> dict:
        return next(e for e in self.events if e["event_id"] == event_id)

    def table(self, spec: dict) -> pd.DataFrame:
        return pd.read_parquet(self.root / spec["table"])
