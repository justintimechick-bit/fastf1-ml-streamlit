"""Explicit aliases; team identity and performance-history continuity are separate."""

import json
from pathlib import Path

from f1ml.storage import digest


class TeamMapping:
    def __init__(self, path: Path):
        self.document = json.loads(path.read_text())
        self.id = digest(self.document)
        self.rules = self.document["rules"]
        occupied = set()
        for rule in self.rules:
            for year in range(rule["first_year"], rule["last_year"] + 1):
                for name in rule["source_names"]:
                    key = (year, rule["source_id"], name.strip().casefold())
                    if key in occupied:
                        raise ValueError(f"Duplicate team mapping: {key}")
                    occupied.add(key)

    def resolve(self, year: int, source_id, source_name) -> dict:
        identifier = "" if source_id is None else str(source_id).strip()
        name = "" if source_name is None else str(source_name).strip()
        for rule in self.rules:
            if (
                rule["first_year"] <= year <= rule["last_year"]
                and identifier == rule["source_id"]
                and name.casefold() in {n.strip().casefold() for n in rule["source_names"]}
            ):
                return {
                    "team_id": rule["team_id"],
                    "team_history_id": rule["history_id"],
                    "team_mapping_state": "mapped",
                }
        # Unknown aliases are retained for review, with no invented cross-year continuity.
        token = digest({"year": year, "source_id": identifier, "source_name": name})[:16]
        return {
            "team_id": f"unmapped:{token}",
            "team_history_id": f"unmapped:{token}",
            "team_mapping_state": "unmapped",
        }
