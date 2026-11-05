"""Narrow holdout capability; ordinary feature/prediction commands remain sealed."""

from copy import deepcopy

from f1ml.domain import PRACTICES
from f1ml.features.build import build_event
from f1ml.features.source import LegacySource


class HoldoutSource(LegacySource):
    def __init__(self, source, manifest, event_id, task):
        if event_id not in manifest["final_test"] or task not in ("qualifying", "race"):
            raise ValueError("Holdout capability is restricted to the locked reserved event/task")
        self.source, self.manifest, self.target, self.task = source, manifest, event_id, task
        self.receipt = source.receipt
        self.events = [e for e in source.events if e["event_id"] in manifest["development"]]
        self.reads = []
        self.spec_events = {id(s): e["event_id"] for e in source.events for s in e["sessions"]}

    def event(self, event_id):
        if event_id != self.target and event_id not in self.manifest["development"]:
            raise ValueError("Other holdout weekends cannot enter prediction inputs")
        return self.source.event(event_id)

    def component(self, spec, name):
        event_id = self.spec_events.get(id(spec))
        if event_id is None:
            raise ValueError("Unrecognized source session identity")
        if event_id in self.manifest["development"]:
            permitted = name == "results" and spec["kind"] in ("Qualifying", "Race")
        else:
            permitted = event_id == self.target and (
                spec["kind"] in PRACTICES or (self.task == "race" and spec["kind"] == "Qualifying")
            )
        if not permitted:
            raise ValueError("Forbidden target/other-block observation read")
        self.reads.append({"event_id": event_id, "kind": spec["kind"], "component": name})
        return self.source.component(spec, name)

    def table(self, spec):
        return self.component(spec, "results")


def holdout_features(source, manifest, event_id, task, cutoff, entrants, mode):
    view = HoldoutSource(source, manifest, event_id, task)
    allowed = deepcopy(manifest)
    allowed["final_test"] = []  # Internal capability only; the original manifest is never changed.
    frame, audit = build_event(
        view, allowed, event_id, task, cutoff, entrants, mode, inference=True
    )
    if not set(audit["history_weekends"]) <= set(manifest["development"]):
        raise ValueError("Block outcomes entered feature history")
    audit["reads"] = view.reads
    audit["original_manifest_id"] = manifest["id"]
    return frame, audit, allowed
