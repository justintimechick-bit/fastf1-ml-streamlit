"""Resumable acquisition with atomic checkpoints and immutable component revisions."""

import fcntl
import json
import logging
import os
import time
from contextlib import contextmanager
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import pandas as pd
from fastf1.exceptions import ErgastInvalidRequestError, ErgastJsonError, RateLimitExceededError
from requests import ConnectionError, HTTPError, Timeout

from f1ml.config import Config
from f1ml.domain import GP_SESSIONS, utc
from f1ml.ingestion.identity import TeamMapping
from f1ml.ingestion.pipeline import validate_events
from f1ml.ingestion.tables import json_value, normalize_batch
from f1ml.ingestion.types import SCHEMA_VERSION, TRANSFORM_VERSION, SourceBatch, session_id
from f1ml.storage import read_json, save_table, seal_directory, verify_directory, write_json

log = logging.getLogger(__name__)


def now():
    return pd.Timestamp.now(tz="UTC")


def atomic_json(path: Path, value):
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    with temporary.open("x") as stream:
        json.dump(json_value(value), stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


@contextmanager
def writer_lock(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".writer.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Another ingestion process holds the catalog writer lock") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def transient(exc):
    if isinstance(exc, (ConnectionError, Timeout)):
        return True
    if isinstance(exc, ErgastJsonError):
        return True
    if isinstance(exc, ErgastInvalidRequestError):
        return any(
            reason in str(exc).casefold()
            for reason in (
                "internal server error",
                "bad gateway",
                "service unavailable",
                "gateway timeout",
            )
        )
    return (
        isinstance(exc, HTTPError)
        and exc.response is not None
        and (exc.response.status_code == 429 or exc.response.status_code >= 500)
    )


def rate_limited(error):
    if isinstance(error, dict):
        return (
            error.get("type") == "RateLimitExceededError"
            or "too many requests" in error.get("message", "").casefold()
            or "429 client error" in error.get("message", "").casefold()
        )
    return (
        isinstance(error, RateLimitExceededError)
        or isinstance(error, ErgastInvalidRequestError)
        and "too many requests" in str(error).casefold()
        or isinstance(error, HTTPError)
        and error.response is not None
        and error.response.status_code == 429
    )


class Catalog:
    def __init__(self, root: Path, config: Config, provider, *, sleeper=time.sleep):
        self.root = root.resolve()
        self.config = config
        self.options = config.ingestion
        self.provider = provider
        self.sleeper = sleeper
        self.mapping = TeamMapping(Path(self.options.team_mapping_path))
        self.state_path = self.root / "state.json"
        self.state = None

    def checkpoint(self):
        self.state["updated_at"] = now().isoformat()
        atomic_json(self.state_path, self.state)

    def load_or_create(self, as_of, create):
        if self.state_path.exists():
            self.state = read_json(self.state_path)
            if self.state["schema_version"] != SCHEMA_VERSION:
                raise ValueError("Unsupported ingestion catalog schema")
            if self.state["years"] != list(self.config.years):
                raise ValueError("Catalog scope differs from configured years")
            if self.state.get("reparse_plan"):
                raise ValueError(
                    "Offline reparse is incomplete; run reparse before resuming downloads"
                )
            if self.state["team_mapping_id"] != self.mapping.id:
                raise ValueError(
                    "Team mapping changed; run reparse to update normalized tables offline"
                )
            if as_of is not None:
                self.state["as_of"] = utc(as_of).isoformat()
        elif create:
            self.state = {
                "schema_version": SCHEMA_VERSION,
                "created_at": now().isoformat(),
                "years": list(self.config.years),
                "as_of": utc(as_of).isoformat(),
                "team_mapping_id": self.mapping.id,
                "events": [],
                "jobs": {},
                "discovered_years": [],
                "discovery_errors": [],
                "generations": {},
                "timing_circuit": {"failed_events": [], "blocked_until": None},
            }
            write_json(self.root / "team_mapping.json", self.mapping.document)
        else:
            raise FileNotFoundError(self.state_path)

    def attempt(self, operation):
        failures = []
        for attempt in range(1, self.options.max_attempts + 1):
            try:
                return operation(), failures, attempt
            except Exception as exc:
                failures.append(
                    {
                        "attempt": attempt,
                        "retrieved_at": now().isoformat(),
                        "type": type(exc).__name__,
                        "message": str(exc)[:4000],
                        "retryable": transient(exc) and not rate_limited(exc),
                        "rate_limited": rate_limited(exc),
                    }
                )
                if rate_limited(exc) or not transient(exc) or attempt == self.options.max_attempts:
                    return None, failures, attempt
                self.sleeper(
                    min(
                        self.options.retry_max_seconds,
                        self.options.retry_base_seconds * 2 ** (attempt - 1),
                    )
                )
        raise AssertionError("Unreachable retry state")

    def discover_missing(self):
        for year in self.config.years:
            if year in self.state["discovered_years"]:
                continue
            events, errors, _ = self.attempt(lambda year=year: self.provider.schedule(year))
            if events is None:
                self.state["discovery_errors"].append({"year": year, "errors": errors})
            else:
                validate_events(events)
                if not events:
                    self.state["discovery_errors"].append(
                        {
                            "year": year,
                            "errors": [
                                {"type": "EmptySchedule", "message": "No GP events returned"}
                            ],
                        }
                    )
                    continue
                self.state["events"].extend(events)
                self.state["discovered_years"].append(year)
                self.state["discovery_errors"] = [
                    e for e in self.state["discovery_errors"] if e["year"] != year
                ]
            self.checkpoint()
        validate_events(self.state["events"])

    def save_batch(self, key, event, spec, component, batch, failures, attempts, old_job):
        revision = uuid4().hex
        relative = (
            Path("records")
            / event["event_id"]
            / session_id(event["event_id"], spec["kind"]).split("/")[1]
            / component
            / revision
        )
        root = self.root / relative
        root.mkdir(parents=True)
        if batch is None:
            batch = SourceBatch(
                metadata={
                    "retrieved_at": now().isoformat(),
                    "fastf1_version": version("fastf1"),
                    "source": "fastf1_public_interface",
                    "errors": [],
                    "completion_state": "unknown",
                }
            )
        provenance = {
            **batch.metadata,
            "event_id": event["event_id"],
            "session_id": session_id(event["event_id"], spec["kind"]),
            "kind": spec["kind"],
            "component": component,
            "transform_version": TRANSFORM_VERSION,
            "team_mapping_id": self.mapping.id,
            "attempts": attempts,
            "attempt_failures": failures,
            "cache_generation": self.state["generations"].get(event["event_id"], 0),
            "source_columns": {},
            "source_missing_counts": {},
        }
        for name, frame in batch.frames.items():
            save_table(root / "source" / f"{name}.parquet", frame)
            provenance["source_columns"][name] = list(frame.columns)
            provenance["source_missing_counts"][name] = {
                str(k): int(v) for k, v in frame.isna().sum().items()
            }
        tables, parsing_errors = normalize_batch(batch, event, spec, self.mapping, component)
        provenance["normalization_errors"] = parsing_errors
        for name, frame in tables.items():
            save_table(root / "normalized" / f"{name}.parquet", frame)
        if component == "results":
            primary = "results" in tables
            full = primary and provenance.get("response_complete", False)
        else:
            weather = "session_weather" in tables
            laps = "practice_laps" in tables or "qualifying_laps" in tables
            primary = weather or laps or "session_status" in tables
            full = weather and (laps or spec["kind"] == "Race")
        errors = provenance.get("errors", []) + parsing_errors
        rate_deadline = None
        for error in provenance.get("errors", []) + failures:
            if rate_limited(error):
                observed = utc(error.get("retrieved_at", provenance["retrieved_at"]))
                deadline = observed + pd.Timedelta(hours=self.options.rate_limit_cooldown_hours)
                rate_deadline = deadline.isoformat()
                self.state["results_circuit"] = {
                    "blocked_until": deadline.isoformat(),
                    "reason": error["message"],
                }
        status = (
            "complete"
            if full and not errors
            else "partial"
            if primary
            else "failed"
            if failures or parsing_errors
            else "unavailable"
        )
        provenance["status"] = status
        provenance["normalized_tables"] = list(tables)
        write_json(root / "metadata.json", provenance)
        seal_directory(
            root,
            {
                "schema_version": SCHEMA_VERSION,
                "revision": revision,
                "transform_version": TRANSFORM_VERSION,
            },
        )
        pointers = dict(old_job.get("table_sources", {}))
        for name in tables:
            # Never supersede a complete official field with an interrupted page.
            if name == "results" and not provenance.get("response_complete") and name in pointers:
                continue
            pointers[name] = str(relative)
        retry_after = rate_deadline or (old_job.get("retry_after") if not attempts else None)
        if retry_after is None:
            retry_after = (
                now() + pd.Timedelta(hours=self.options.retry_cooldown_hours)
            ).isoformat()
        self.state["jobs"][key] = {
            "status": status,
            "latest": str(relative),
            "table_sources": pointers,
            "history": old_job.get("history", []) + [str(relative)],
            "attempts_total": old_job.get("attempts_total", 0) + attempts,
            "retry_after": retry_after,
            "event_id": event["event_id"],
            "kind": spec["kind"],
            "component": component,
        }
        if component == "timing" and attempts:
            circuit = self.state["timing_circuit"]
            if not primary and not parsing_errors:
                circuit["failed_events"] = sorted(
                    set(circuit["failed_events"]) | {event["event_id"]}
                )
                if len(circuit["failed_events"]) >= self.options.timing_circuit_event_threshold:
                    circuit["blocked_until"] = (
                        now() + pd.Timedelta(hours=self.options.timing_circuit_cooldown_hours)
                    ).isoformat()
            elif primary:
                circuit["failed_events"] = []
        self.checkpoint()
        return status

    def reparse(self):
        """Apply reviewed mappings/transforms to retained source frames without a provider."""
        with writer_lock(self.root):
            self.state = read_json(self.state_path)
            if self.state["schema_version"] != SCHEMA_VERSION:
                raise ValueError("Unsupported catalog schema")
            if not self.state.get("reparse_plan"):
                plan = []
                for key, job in self.state["jobs"].items():
                    pointers = sorted(set(job["table_sources"].values()) - {job["latest"]})
                    for pointer in pointers + [job["latest"]]:
                        plan.append({"key": key, "source_revision": pointer})
                self.state["reparse_plan"] = plan
                self.state["reparse_mapping_id"] = self.mapping.id
                self.state["team_mapping_id"] = self.mapping.id
                atomic_json(self.root / "team_mapping.json", self.mapping.document)
                self.checkpoint()
            if self.state["reparse_mapping_id"] != self.mapping.id:
                raise ValueError("Finish reparse with the mapping version that started it")
            processed = 0
            events = {e["event_id"]: e for e in self.state["events"]}
            while self.state["reparse_plan"]:
                item = self.state["reparse_plan"][0]
                job = self.state["jobs"][item["key"]]
                metadata = load_record(self.root, item["source_revision"])
                source_dir = self.root / item["source_revision"] / "source"
                frames = {p.stem: pd.read_parquet(p) for p in source_dir.glob("*.parquet")}
                batch = SourceBatch(
                    frames,
                    {
                        **metadata,
                        "reparsed_from": item["source_revision"],
                        "normalized_at": now().isoformat(),
                    },
                )
                event = events[job["event_id"]]
                spec = next(s for s in event["sessions"] if s["kind"] == job["kind"])
                self.save_batch(
                    item["key"],
                    event,
                    spec,
                    job["component"],
                    batch,
                    metadata.get("attempt_failures", []),
                    0,
                    job,
                )
                self.state["reparse_plan"].pop(0)
                processed += 1
                self.checkpoint()
            from f1ml.ingestion.coverage import materialize

            return materialize(
                self.root, self.state, {"attempted_jobs": 0, "reparsed_records": processed}
            )

    def run(self, *, as_of=None, selected=None, create=False, retry_failed=False, refresh=None):
        with writer_lock(self.root):
            self.load_or_create(as_of, create)
            self.discover_missing()
            results_circuit = self.state.setdefault("results_circuit", {"blocked_until": None})
            if results_circuit["blocked_until"] and now() >= utc(results_circuit["blocked_until"]):
                results_circuit["blocked_until"] = None
            events = {e["event_id"]: e for e in self.state["events"]}
            selected = set(selected or events)
            if selected - events.keys():
                raise ValueError(f"Events not in discovered scope: {selected - events.keys()}")
            if refresh:
                if refresh not in events:
                    raise ValueError("Refresh event is not in the catalog")
                selected = {refresh}
                self.state["generations"][refresh] = self.state["generations"].get(refresh, 0) + 1
                for job in self.state["jobs"].values():
                    if job["event_id"] == refresh:
                        job["status"] = "pending_refresh"
                retry_failed = True
            circuit = self.state["timing_circuit"]
            if retry_failed or (
                circuit["blocked_until"] and now() >= utc(circuit["blocked_until"])
            ):
                circuit.update(failed_events=[], blocked_until=None)
            self.checkpoint()
            stats = {"attempted_jobs": 0, "reused_jobs": 0, "deferred_jobs": 0}
            try:
                # Official results run first and never depend on timing availability.
                for component in ("results", "timing"):
                    for event_id in sorted(selected):
                        event = events[event_id]
                        for spec in event["sessions"]:
                            if (
                                spec["kind"] not in GP_SESSIONS
                                or component == "results"
                                and spec["kind"] not in ("Qualifying", "Race")
                                or utc(spec["scheduled_start"]) >= utc(self.state["as_of"])
                            ):
                                continue
                            key = f"{session_id(event_id, spec['kind'])}|{component}"
                            old = self.state["jobs"].get(key, {})
                            if (
                                old
                                and old["status"] != "pending_refresh"
                                and (
                                    old["status"] == "complete"
                                    or not retry_failed
                                    and now() < utc(old["retry_after"])
                                )
                            ):
                                stats["reused_jobs"] += 1
                                continue
                            if component == "timing" and circuit["blocked_until"]:
                                stats["deferred_jobs"] += 1
                                continue
                            if self.state["results_circuit"]["blocked_until"]:
                                stats["deferred_jobs"] += 1
                                continue
                            log.info("Loading %s %s %s", event_id, spec["kind"], component)
                            generation = self.state["generations"].get(event_id, 0)
                            with self.provider.event_cache(event_id, generation):
                                method = (
                                    self.provider.fetch_results
                                    if component == "results"
                                    else self.provider.fetch_timing
                                )
                                batch, failures, attempts = self.attempt(
                                    lambda method=method, event=event, spec=spec: method(
                                        event, spec
                                    )
                                )
                            status = self.save_batch(
                                key, event, spec, component, batch, failures, attempts, old
                            )
                            stats["attempted_jobs"] += 1
                            log.info("%s: %s", key, status)
            finally:
                self.checkpoint()
            from f1ml.ingestion.coverage import materialize

            return materialize(self.root, self.state, stats)


def load_record(root: Path, relative: str):
    path = root / relative
    verify_directory(path)
    return read_json(path / "metadata.json")


def current_snapshot(root: Path):
    pointer = read_json(root / "CURRENT.json")
    path = root / pointer["path"]
    receipt = verify_directory(path)
    if receipt["id"] != pointer["id"]:
        raise ValueError("Current snapshot pointer does not match receipt")
    return path, receipt


def catalog_configuration(config: Config):
    return asdict(config.ingestion)
