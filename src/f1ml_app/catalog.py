"""Read-only view of the real FastF1 catalog through the Phase 2 coverage API.

The catalog is a source layer only: it has no reviewed cutoffs, rosters or frozen manifest,
so no model has been trained or evaluated on it. Nothing here triggers a download.
"""

from pathlib import Path

import pandas as pd

from f1ml.ingestion.catalog import current_snapshot
from f1ml.ingestion.coverage import inspect_coverage
from f1ml.storage import read_json, verify_directory

GP_KINDS = ("Practice 1", "Practice 2", "Practice 3", "Qualifying", "Race")
NO_MODEL_REASON = (
    "No model has been trained or evaluated on the real catalog. It lacks reviewed "
    "cutoff-known entrant rosters, verified session completion times and a frozen "
    "experiment manifest, so a forecast here would not be cutoff-safe."
)


class RealCatalog:
    def __init__(self, root: Path):
        self.root = root
        if (root / "CURRENT.json").exists():
            self.snapshot, self.receipt = current_snapshot(root)
            self.coverage = {
                level: inspect_coverage(root, level)
                for level in ("seasons", "events", "sessions", "failures")
            }
        else:
            self.snapshot, self.receipt = root, verify_directory(root)
            self.coverage = {
                level: pd.read_parquet(root / "coverage" / f"{level}.parquet")
                for level in ("seasons", "events", "sessions", "failures")
            }
        tables = self.snapshot / "tables"
        events = pd.read_parquet(tables / "events.parquet")
        self.events = (
            events[["event_id", "year", "round", "name", "location", "country", "event_format"]]
            .merge(
                self.coverage["events"].drop(columns=["year", "name", "event_format"]),
                on="event_id",
                how="left",
                validate="one_to_one",
            )
            .sort_values(["year", "round"])
            .reset_index(drop=True)
        )
        self._results = pd.read_parquet(tables / "results.parquet")
        self._drivers = pd.read_parquet(tables / "drivers.parquet")

    @property
    def as_of(self) -> str:
        return self.receipt.get("as_of", "")

    def sessions(self, event_id: str) -> pd.DataFrame:
        frame = self.coverage["sessions"]
        frame = frame.loc[frame.event_id.eq(event_id)].copy()
        frame["scheduled_start"] = pd.to_datetime(frame.scheduled_start, utc=True)
        frame["in_model_scope"] = frame.kind.isin(GP_KINDS)
        return frame.sort_values("scheduled_start").reset_index(drop=True)

    def failures(self, event_id: str | None = None) -> pd.DataFrame:
        frame = self.coverage["failures"]
        return frame if event_id is None else frame.loc[frame.event_id.eq(event_id)]

    def results(self, event_id: str, kind: str) -> pd.DataFrame:
        """Official classification as retrieved (Jolpica via FastF1); empty if not retrieved."""
        frame = self._results.loc[self._results.event_id.eq(event_id) & self._results.kind.eq(kind)]
        if frame.empty:
            return frame
        named = frame.merge(
            self._drivers[["driver_id", "first_name", "last_name", "abbreviation"]],
            on="driver_id",
            how="left",
            validate="many_to_one",
        )
        named["driver"] = (
            named.first_name.fillna("") + " " + named.last_name.fillna("")
        ).str.strip()
        columns = [
            "official_position",
            "abbreviation",
            "driver",
            "source_constructorName",
            "outcome",
            "source_status",
            "source_positionText",
        ]
        return (
            named[[c for c in columns if c in named]]
            .sort_values("official_position", na_position="last")
            .reset_index(drop=True)
        )


class RealExperiment(RealCatalog):
    """Prepared Phase 9 review artifacts; an incomplete experiment has no forecast fallback."""

    def __init__(self, root: Path):
        self.experiment_receipt = verify_directory(root)
        self.status = read_json(root / "status.json")
        if (
            self.experiment_receipt.get("data_kind") != "real"
            or self.status.get("data_kind") != "real"
            or self.status["experiment_id"] != self.experiment_receipt["experiment_id"]
        ):
            raise ValueError("Real experiment identity/provenance mismatch")
        super().__init__(root / "dataset")
        if self.status["source_snapshot_id"] != self.receipt["id"]:
            raise ValueError("Real experiment belongs to a different source snapshot")
        self.experiment_root = root
        self.exclusions = pd.read_csv(root / "event_exclusions.csv")
        self.group_coverage = pd.read_csv(root / "year_feature_coverage.csv")
        self.evaluation_records = pd.read_csv(root / "evaluation_records.csv")
        self.strategy_eligibility = pd.read_csv(root / "strategy_eligibility.csv")
        self.report = (root / "report.md").read_text()

    def forecast(self, event_id, task):
        if event_id not in set(self.events.event_id) or task not in (
            "qualifying",
            "race",
            "podium",
        ):
            raise ValueError("Unknown real experiment event/task")
        return None  # No fitted real model exists; never invoke a demo model.
