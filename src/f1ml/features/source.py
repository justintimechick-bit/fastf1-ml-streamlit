"""Read-only adapters for Phase 1 datasets and immutable Phase 2 snapshots."""

from pathlib import Path

import pandas as pd

from f1ml.domain import PRACTICES, utc
from f1ml.ingestion.pipeline import Dataset, validate_events
from f1ml.ingestion.tables import json_value
from f1ml.storage import digest, verify_directory


class LegacySource:
    def __init__(self, dataset):
        self.dataset = dataset
        self.receipt, self.events = dataset.receipt, dataset.events

    def event(self, event_id):
        return self.dataset.event(event_id)

    def table(self, spec):
        return self.dataset.table(spec)

    def component(self, spec, name):
        path = spec.get(f"{name}_table")
        if path:
            return self.dataset.table({**spec, "table": path})
        if name == "results" and spec["kind"] in ("Qualifying", "Race") and "table" in spec:
            return self.table(spec)
        if name == "practice_summary" and spec["kind"] in PRACTICES and "table" in spec:
            return self.table(spec)
        return pd.DataFrame()


class CatalogSource:
    """Availability reviews bind to a snapshot; a mutable catalog is never a feature source."""

    def __init__(
        self,
        root: Path,
        availability: pd.DataFrame | None = None,
        event_context: pd.DataFrame | None = None,
    ):
        self.root = root.resolve()
        receipt = verify_directory(self.root)
        availability = availability.copy() if availability is not None else pd.DataFrame()
        event_context = event_context.copy() if event_context is not None else pd.DataFrame()
        self.reviews = json_value(
            {
                "availability": availability.fillna("").to_dict("records"),
                "event_context": event_context.fillna("").to_dict("records"),
            }
        )
        self.receipt = {
            "id": digest({"snapshot_id": receipt["id"], **self.reviews}),
            "snapshot_id": receipt["id"],
            "reviews": self.reviews,
        }
        sessions = pd.read_parquet(self.root / "tables/sessions.parquet")
        events = pd.read_parquet(self.root / "tables/events.parquet")
        known = set(sessions.session_id)
        reviews = {}
        if not availability.empty:
            required = {"session_id", "completed_at", "available_at", "evidence"}
            if not required.issubset(availability) or availability.session_id.duplicated().any():
                raise ValueError(f"Availability review requires unique {sorted(required)}")
            for row in availability.to_dict("records"):
                if (
                    row["session_id"] not in known
                    or pd.isna(row["evidence"])
                    or not str(row["evidence"]).strip()
                ):
                    raise ValueError("Unknown session or missing availability evidence")
                if utc(row["completed_at"]) > utc(row["available_at"]):
                    raise ValueError("Availability cannot precede completion")
                reviews[row["session_id"]] = row
        contexts = {}
        if not event_context.empty:
            required = {"event_id", "circuit_id", "evidence"}
            if not required.issubset(event_context) or event_context.event_id.duplicated().any():
                raise ValueError("Event context requires unique event_id, circuit_id, evidence")
            if set(event_context.event_id) - set(events.event_id):
                raise ValueError("Event context contains an unknown event")
            for row in event_context.to_dict("records"):
                if (
                    pd.isna(row["circuit_id"])
                    or not str(row["circuit_id"]).strip()
                    or pd.isna(row["evidence"])
                    or not str(row["evidence"]).strip()
                ):
                    raise ValueError("Circuit context needs identity/evidence")
                contexts[row["event_id"]] = row
        self.events = []
        for event in events.to_dict("records"):
            event["sessions"] = []
            if event["event_id"] in contexts:
                event["circuit_id"] = contexts[event["event_id"]]["circuit_id"]
            for row in sessions.loc[sessions.event_id.eq(event["event_id"])].to_dict("records"):
                ended = row.get("completed_at")
                spec = {
                    "kind": row["kind"],
                    "scheduled_start": row["scheduled_start"],
                    "session_id": row["session_id"],
                    "completed": pd.notna(ended) and row["completion_state"] == "completed",
                    "official_results": bool(row.get("results_rows", 0)),
                    "ended_at": ended if pd.notna(ended) else None,
                    "started_at": row.get("started_at"),
                    "retrieved_at": row.get("results_retrieved_at")
                    if row["kind"] in ("Qualifying", "Race")
                    else row.get("timing_retrieved_at"),
                    "timing_retrieved_at": row.get("timing_retrieved_at"),
                    "result_field_complete": bool(row.get("official_results_complete", False)),
                }
                if row["session_id"] in reviews:
                    review = reviews[row["session_id"]]
                    if utc(review["completed_at"]) <= utc(row["scheduled_start"]):
                        raise ValueError("Reviewed completion must follow session start")
                    spec.update(
                        completed=True,
                        ended_at=utc(review["completed_at"]).isoformat(),
                        available_at=utc(review["available_at"]).isoformat(),
                        availability_evidence=review["evidence"],
                    )
                    if (
                        pd.notna(review.get("started_at"))
                        and str(review.get("started_at", "")).strip()
                    ):
                        if utc(review["started_at"]) >= utc(review["completed_at"]):
                            raise ValueError("Reviewed start must precede completion")
                        spec["started_at"] = utc(review["started_at"]).isoformat()
                if spec["official_results"] or row.get("laps_rows", 0):
                    spec["table"] = row["session_id"]
                event["sessions"].append(json_value(spec))
            self.events.append(event)
        validate_events(self.events)

    def event(self, event_id):
        return next(e for e in self.events if e["event_id"] == event_id)

    def component(self, spec, name):
        names = {
            "results": "results",
            "laps": "qualifying_laps" if spec["kind"] == "Qualifying" else "practice_laps",
            "weather": "session_weather",
        }
        if name not in names:
            return pd.DataFrame()
        if name == "laps" and spec["kind"] not in PRACTICES | {"Qualifying"}:
            raise ValueError("Target race/Sprint laps are not a feature source")
        path = self.root / "tables" / f"{names[name]}.parquet"
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path, filters=[("session_id", "==", spec["session_id"])])

    def table(self, spec):
        return self.component(spec, "results")


def as_source(dataset):
    return dataset if isinstance(dataset, (LegacySource, CatalogSource)) else LegacySource(dataset)


def load_source(root: Path, availability: Path | None = None, event_context: Path | None = None):
    if (root / "events.json").exists():
        if availability or event_context:
            raise ValueError(
                "Review overlays are for Phase 2 snapshots; legacy timing is already frozen"
            )
        return LegacySource(Dataset(root))
    return CatalogSource(
        root,
        pd.read_csv(availability, dtype=str) if availability else None,
        pd.read_csv(event_context, dtype=str) if event_context else None,
    )


def available(spec, cutoff, mode, *, timing=False):
    if not spec.get("completed") or not spec.get("ended_at"):
        return False
    if utc(spec["ended_at"]) >= cutoff:
        return False
    if spec.get("available_at") and utc(spec["available_at"]) >= cutoff:
        return False
    retrieved = spec.get("timing_retrieved_at") if timing else spec.get("retrieved_at")
    if timing and "timing_retrieved_at" not in spec:
        retrieved = spec.get("retrieved_at")
    if mode == "publication_snapshot" and (
        not retrieved or pd.isna(retrieved) or utc(retrieved) > cutoff
    ):
        return False
    return True


def audited_spec(event, spec):
    return json_value(
        {
            "event_id": event["event_id"],
            "kind": spec["kind"],
            "ended_at": spec.get("ended_at"),
            "retrieved_at": spec.get("retrieved_at"),
            "table": spec.get("table"),
            "available_at": spec.get("available_at"),
        }
    )
