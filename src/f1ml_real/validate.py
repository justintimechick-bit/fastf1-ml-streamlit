"""Offline replay of real result normalization from retained source frames."""

from pathlib import Path

import pandas as pd

from f1ml.ingestion.identity import TeamMapping
from f1ml.ingestion.tables import normalize_batch
from f1ml.ingestion.types import SourceBatch
from f1ml.storage import new_directory, read_json, seal_directory, verify_directory


def validate_sources(experiment: Path, output: Path):
    receipt = verify_directory(experiment)
    dataset = experiment / "dataset"
    targets = pd.read_parquet(dataset / "tables/results.parquet")
    mapping = TeamMapping(dataset / "team_mapping.json")
    rows = []
    columns = [
        "driver_number",
        "team_id",
        "team_history_id",
        "official_position",
        "classification_code",
        "outcome",
        "is_classified",
        "is_retired",
        "podium",
        "position_ambiguous",
        "official_field_complete",
    ]
    for (eid, kind, revision), saved in targets.groupby(["event_id", "kind", "source_revision"]):
        record = experiment / "source_records" / revision
        verify_directory(record)
        metadata = read_json(record / "metadata.json")
        frames = {p.stem: pd.read_parquet(p) for p in (record / "source").glob("*.parquet")}
        event = {"event_id": eid, "year": int(saved.year.iloc[0])}
        tables, errors = normalize_batch(
            SourceBatch(frames, metadata), event, {"kind": kind}, mapping, "results"
        )
        replay = tables.get("results", pd.DataFrame())
        problem = ""
        try:
            if errors:
                raise ValueError(str(errors))
            a = saved.set_index("driver_id")[columns].sort_index()
            b = replay.set_index("driver_id")[columns].sort_index()
            pd.testing.assert_frame_equal(a, b, check_dtype=False)
        except (AssertionError, KeyError, ValueError) as exc:
            problem = str(exc)
        rows.append(
            {
                "event_id": eid,
                "kind": kind,
                "source_revision": revision,
                "driver_rows": len(saved),
                "matches_source_replay": not problem,
                "problem": problem,
            }
        )
    new_directory(output)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "source_replay.csv", index=False)
    return seal_directory(
        output,
        {
            "experiment_receipt_id": receipt["id"],
            "data_kind": "real",
            "result_sessions": len(frame),
            "driver_rows": len(targets),
            "mismatched_sessions": int((~frame.matches_source_replay).sum()),
            "scope": "source-to-normalized label/identity/team/status replay; no fit or score",
        },
    )
