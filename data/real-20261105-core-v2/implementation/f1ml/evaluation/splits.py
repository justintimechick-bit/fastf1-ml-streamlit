"""Freeze eligible weekends before modeling. Final-test access has no CLI switch."""

from pathlib import Path

import pandas as pd

from f1ml.config import Config
from f1ml.domain import utc
from f1ml.ingestion.pipeline import Dataset
from f1ml.storage import digest, read_json, write_json


def gp_session(event: dict, kind: str) -> dict:
    matches = [s for s in event["sessions"] if s["kind"] == kind]
    if len(matches) != 1:
        raise ValueError(f"Expected one GP {kind} session: {event['event_id']}")
    return matches[0]


def eligibility(dataset: Dataset, event: dict) -> str | None:
    for kind in ("Qualifying", "Race"):
        try:
            spec = gp_session(event, kind)
        except ValueError:
            return f"missing_gp_{kind.lower()}"
        if not spec.get("completed"):
            return f"gp_{kind.lower()}_not_confirmed_completed"
        if not spec.get("official_results") or "table" not in spec:
            return f"missing_official_{kind.lower()}_results"
        table = dataset.table(spec)
        if table.empty or table.official_position.notna().sum() == 0:
            return f"unresolved_{kind.lower()}_order"
        if table.outcome.eq("unknown").any():
            return f"unresolved_{kind.lower()}_entries"
        if kind == "Race" and (
            table.podium.isna().any()
            or set(table.loc[table.podium.eq(1), "official_position"]) != {1, 2, 3}
        ):
            return "incomplete_official_podium"
    return None


def freeze(dataset: Dataset, config: Config, output: Path) -> dict:
    eligible, excluded = [], []
    for event in dataset.events:
        reason = eligibility(dataset, event)
        if reason:
            excluded.append({"event_id": event["event_id"], "reason": reason})
            continue
        eligible.append(
            {
                "event_id": event["event_id"],
                "year": event["year"],
                "race_start": gp_session(event, "Race")["scheduled_start"],
                "race_end": gp_session(event, "Race")["ended_at"],
            }
        )
    eligible.sort(key=lambda event: utc(event["race_start"]))
    current = [e for e in eligible if e["year"] == config.evaluation_year]
    required = config.final_test_weekends + config.min_validation_weekends
    if len(current) < required:
        raise ValueError(
            f"Need at least {required} eligible completed 2026 weekends; found "
            f"{len(current)}. Final block remains {config.final_test_weekends}; "
            "no automatic shrinking. Discover/ingest more data or explicitly configure."
        )
    test = current[-config.final_test_weekends :]
    first_test = utc(test[0]["race_start"])
    development = [e for e in eligible if utc(e["race_start"]) < first_test]
    validation = [e for e in development if e["year"] == config.evaluation_year]
    folds = []
    for event in validation:
        # Earlier complete weekends only. A postponed/overlapping race cannot enter training.
        event_start = min(
            utc(s["scheduled_start"]) for s in dataset.event(event["event_id"])["sessions"]
        )
        train = [
            e["event_id"]
            for e in development
            if utc(e["race_end"]) < event_start and e["event_id"] != event["event_id"]
        ]
        if len(train) >= config.min_training_weekends:
            folds.append({"train": train, "validate": event["event_id"]})
    if len(folds) < config.min_validation_weekends:
        raise ValueError("Insufficient complete earlier training weekends for 2026 validation")
    payload = {
        "schema_version": 1,
        "dataset_id": dataset.receipt["id"],
        "config": config.to_dict(),
        "frozen_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "eligible": eligible,
        "excluded": excluded,
        "development": [e["event_id"] for e in development],
        "final_test": [e["event_id"] for e in test],
        "folds": folds,
    }
    manifest = {**payload, "id": digest(payload)}
    write_json(output, manifest)
    return manifest


def load_manifest(path: Path, dataset: Dataset | None = None) -> dict:
    manifest = read_json(path)
    if digest({k: v for k, v in manifest.items() if k != "id"}) != manifest["id"]:
        raise ValueError("Frozen manifest changed")
    if dataset is not None and dataset.receipt["id"] != manifest["dataset_id"]:
        raise ValueError("Dataset revision differs from frozen manifest")
    assert_disjoint_events(manifest["development"], (), manifest["final_test"])
    for fold in manifest["folds"]:
        assert_disjoint_events(fold["train"], [fold["validate"]], manifest["final_test"])
    return manifest


def assert_disjoint_events(train, validation=(), test=()):
    roles = [set(train), set(validation), set(test)]
    if roles[0] & roles[1] or roles[0] & roles[2] or roles[1] & roles[2]:
        raise ValueError("Train/validation/test event IDs must not overlap")


def assert_development(manifest: dict, event_ids) -> None:
    ids = set(event_ids)
    if ids & set(manifest["final_test"]):
        raise ValueError("Final test block is sealed during this phase")
    if ids - set(manifest["development"]):
        raise ValueError("Event is outside the frozen development manifest")
