"""Deterministic, visibly synthetic end-to-end example; never represents real F1 data."""

from pathlib import Path

import pandas as pd

from f1ml.config import Config
from f1ml.domain import PRACTICES, event_key, utc
from f1ml.evaluation.backtest import run_backtest
from f1ml.evaluation.splits import freeze, gp_session
from f1ml.features.build import build_development
from f1ml.ingestion.normalize import normalize_results
from f1ml.ingestion.pipeline import Dataset, discover, ingest
from f1ml.storage import new_directory


class SyntheticProvider:
    """Small variable fields, one Sprint format, and an unfinished 2026 event."""

    def schedule(self, year: int) -> list[dict]:
        count = 8 if year == 2026 else 2
        events = []
        for number in range(1, count + 1):
            date = pd.Timestamp(f"{year}-03-01T10:00:00Z") + pd.Timedelta(weeks=number)
            if year == 2026 and number == count:
                date = pd.Timestamp("2026-12-01T10:00:00Z")
            kinds = [
                ("Practice 1", 0),
                ("Practice 2", 4),
                ("Practice 3", 24),
                ("Qualifying", 28),
                ("Race", 52),
            ]
            if number == 2:
                kinds = [
                    ("Practice 1", 0),
                    ("Sprint Qualifying", 4),
                    ("Sprint", 24),
                    ("Qualifying", 28),
                    ("Race", 52),
                ]
            events.append(
                {
                    "event_id": event_key(year, number),
                    "year": year,
                    "round": number,
                    "name": f"SYNTHETIC GP {year}/{number}",
                    "location": "synthetic",
                    "circuit_id": f"synthetic_circuit_{number % 2}",
                    "sessions": [
                        {
                            "kind": kind,
                            "scheduled_start": (date + pd.Timedelta(hours=offset)).isoformat(),
                        }
                        for kind, offset in kinds
                    ],
                }
            )
        return events

    @staticmethod
    def drivers(event: dict) -> list[str]:
        # Fixture parameters only; the actual package never assumes a field size.
        return [f"synthetic_driver_{i}" for i in range(6 + event["round"] % 2)]

    def session(self, event: dict, spec: dict, as_of) -> tuple[dict, pd.DataFrame]:
        start = utc(spec["scheduled_start"])
        end = start + pd.Timedelta(hours=2 if spec["kind"] == "Race" else 1)
        drivers = self.drivers(event)
        meta = {
            **spec,
            "started_at": start.isoformat(),
            "ended_at": end.isoformat(),
            "retrieved_at": utc(as_of).isoformat(),
            "completed": bool(end < as_of),
            "official_results": spec["kind"] not in PRACTICES,
            "completion_evidence": "synthetic_fixture",
            "results_source": "synthetic_fixture",
            "availability_basis": "synthetic_fixture",
        }
        if spec["kind"] in PRACTICES:
            return meta, pd.DataFrame(
                {
                    "driver_id": drivers,
                    "best_lap_seconds": [80.0 + i * 0.2 for i in range(len(drivers))],
                    "clean_laps": [8 + i for i in range(len(drivers))],
                }
            )
        order = drivers[event["round"] % len(drivers) :] + drivers[: event["round"] % len(drivers)]
        raw = pd.DataFrame(
            {
                "driverId": order,
                "number": range(1, len(order) + 1),
                "constructorId": [
                    f"synthetic_team_{drivers.index(driver) // 2}" for driver in order
                ],
                "position": range(1, len(order) + 1),
                "positionText": [str(i + 1) for i in range(len(order))],
                "status": "Finished",
            }
        )
        return meta, normalize_results(raw, spec["kind"])


def run_demo(output: Path) -> dict:
    new_directory(output)
    config = Config()
    provider = SyntheticProvider()
    schedule = discover(provider, config, output / "schedule.json")
    ingest(provider, schedule, output / "dataset", "2026-07-01T00:00:00Z")
    dataset = Dataset(output / "dataset")
    manifest = freeze(dataset, config, output / "manifest.json")
    rosters, cutoffs = [], []
    for event_id in manifest["development"]:
        event = dataset.event(event_id)
        quali = gp_session(event, "Qualifying")
        for task in ("qualifying", "race"):
            cutoff = (
                utc(quali["scheduled_start"]) - pd.Timedelta(seconds=1)
                if task == "qualifying"
                else utc(quali["ended_at"]) + pd.Timedelta(hours=1)
            )
            cutoffs.append(
                {
                    "event_id": event_id,
                    "task": task,
                    "cutoff": cutoff.isoformat(),
                    "mode": "historical_reconstruction",
                }
            )
            for index, driver in enumerate(provider.drivers(event)):
                rosters.append(
                    {
                        "event_id": event_id,
                        "task": task,
                        "driver_id": driver,
                        "team_id": f"synthetic_team_{index // 2}",
                        "known_at": (cutoff - pd.Timedelta(days=1)).isoformat(),
                        "source": "synthetic_fixture",
                    }
                )
    entrants, cutoff_frame = pd.DataFrame(rosters), pd.DataFrame(cutoffs)
    entrants.to_csv(output / "entrants.csv", index=False)
    cutoff_frame.to_csv(output / "cutoffs.csv", index=False)
    build_development(dataset, manifest, entrants, cutoff_frame, output / "features")
    return run_backtest(manifest, output / "features", output / "artifacts")
