"""One row per reviewed cutoff entrant; predictors and targets are built separately."""

import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from f1ml.config import FeatureConfig
from f1ml.domain import MODES, PRACTICES, utc
from f1ml.evaluation.splits import assert_development, gp_session
from f1ml.features.dictionary import FEATURE_VERSION, implementation_id, write_dictionary
from f1ml.features.history import (
    historical_observations,
    normalized_results,
    statistics,
    team_history,
)
from f1ml.features.pace import comparable_pace, summarize_driver
from f1ml.features.source import as_source, audited_spec, available
from f1ml.ingestion.normalize import validate_ids
from f1ml.storage import new_directory, save_table, seal_directory, write_json

ENTRANT_COLUMNS = {"event_id", "task", "driver_id", "team_id", "known_at", "source"}
CUTOFF_COLUMNS = {"event_id", "task", "cutoff", "mode"}


def validate_cutoff(event, task, cutoff, mode):
    if mode not in MODES or task not in ("qualifying", "race"):
        raise ValueError("Invalid cutoff task or provenance mode")
    target = gp_session(event, "Qualifying" if task == "qualifying" else "Race")
    starts = [utc(target["scheduled_start"])]
    if target.get("started_at"):
        starts.append(utc(target["started_at"]))
    if cutoff >= min(starts):
        raise ValueError(f"{task} cutoff must be strictly before target session starts")
    if task == "race":
        quali = gp_session(event, "Qualifying")
        if not quali.get("official_results") or not available(quali, cutoff, mode):
            raise ValueError("Race cutoff requires completed, available official GP qualifying")


def validate_roster(entrants, event_id, task, cutoff):
    if not ENTRANT_COLUMNS.issubset(entrants.columns) or entrants.empty:
        raise ValueError(f"Entrants require {sorted(ENTRANT_COLUMNS)}")
    if not entrants.event_id.eq(event_id).all() or not entrants.task.eq(task).all():
        raise ValueError("Roster belongs to a different event/cutoff task")
    validate_ids(entrants)
    if any(entrants[c].fillna("").str.strip().eq("").any() for c in ("team_id", "source")):
        raise ValueError("Roster needs team identities and provenance")
    if any(utc(t) > cutoff for t in entrants.known_at):
        raise ValueError("Roster was not known at cutoff")


def grid_values(entrant, cutoff, n):
    raw = getattr(entrant, "grid_position", np.nan)
    if pd.isna(raw):
        return np.nan, np.nan, 0
    if not hasattr(entrant, "grid_known_at") or utc(entrant.grid_known_at) > cutoff:
        raise ValueError("Grid position was not known at race cutoff")
    if not getattr(entrant, "grid_source", None) or pd.isna(entrant.grid_source):
        raise ValueError("Known grid position needs an evidence source")
    raw = float(raw)
    if not np.isfinite(raw) or raw < 0 or raw % 1 or raw > n:
        raise ValueError("Invalid known grid position (0 denotes pit-lane start)")
    return raw, (raw - 1) / (n - 1) if raw > 0 and n > 1 else np.nan, int(raw == 0)


def build_event(
    dataset,
    manifest,
    event_id,
    task,
    cutoff,
    entrants,
    mode="historical_reconstruction",
    *,
    inference=False,
):
    from f1ml.features.contract import model_matrix

    source = as_source(dataset)
    if event_id in manifest["final_test"]:
        raise ValueError("Final test block is sealed during this phase")
    if not inference:
        assert_development(manifest, [event_id])
        if source.receipt["id"] != manifest["dataset_id"]:
            raise ValueError("Feature source differs from frozen dataset")
    cutoff = utc(cutoff)
    event = source.event(event_id)
    validate_cutoff(event, task, cutoff, mode)
    validate_roster(entrants, event_id, task, cutoff)
    options = FeatureConfig(**manifest["config"].get("features", {}))
    weekends, history, audit = historical_observations(source, manifest, event, cutoff, mode)
    practices, skipped = [], []
    for spec in sorted(event["sessions"], key=lambda s: utc(s["scheduled_start"])):
        if spec["kind"] not in PRACTICES:
            continue
        if not available(spec, cutoff, mode, timing=True):
            skipped.append(
                {
                    "kind": spec["kind"],
                    "reason": "completion/availability_not_verified_before_cutoff",
                }
            )
            continue
        laps = source.component(spec, "laps")
        summary = source.component(spec, "practice_summary") if laps.empty else pd.DataFrame()
        weather = source.component(spec, "weather") if not laps.empty else pd.DataFrame()
        pace = comparable_pace(laps, weather, spec["kind"], options)
        observed = set(laps.driver_id.dropna()) if "driver_id" in laps else set()
        if not summary.empty:
            observed |= set(summary.driver_id.dropna())
        practices.append((pace, observed))
        audit.append(audited_spec(event, spec))
    quali, qpace = pd.DataFrame(), pd.DataFrame()
    if task == "race":
        spec = gp_session(event, "Qualifying")
        quali = normalized_results(source.component(spec, "results")).set_index("driver_id")
        audit.append(audited_spec(event, spec))
        if available(spec, cutoff, mode, timing=True):
            qpace = comparable_pace(
                source.component(spec, "laps"),
                source.component(spec, "weather"),
                "Qualifying",
                options,
            )
    rows = []
    code_id = implementation_id()
    circuit = event.get("circuit_id") or "unknown"
    for entrant in entrants.sort_values("driver_id").itertuples(index=False):
        driver = entrant.driver_id
        era = team_history(
            event["year"], entrant.team_id, getattr(entrant, "team_history_id", None)
        )
        row = {
            "event_id": event_id,
            "driver_id": driver,
            "team_id": entrant.team_id,
            "team_history_id": era,
            "circuit_id": circuit,
            "task": task,
            "cutoff": cutoff.isoformat(),
            "mode": mode,
            "feature_version": FEATURE_VERSION,
            "feature_code_id": code_id,
            "season": event["year"],
            "round": event["round"],
            "regulation_2026": int(event["year"] == 2026),
            "field_size": len(entrants),
            "circuit_identity_verified": int(circuit != "unknown"),
        }
        row.update(statistics(weekends, history, driver, event, era, circuit))
        practice = [summarize_driver(pace, driver) for pace, _ in practices]
        gaps = [p["gap"] for p in practice if pd.notna(p["gap"])]
        row.update(
            practice_sessions=sum(driver in observed for _, observed in practices),
            practice_clean_laps=sum(p["clean"] for p in practice),
            practice_comparable_laps=sum(p["comparable"] for p in practice),
            practice_pace_sessions=len(gaps),
            practice_gap_pct=float(np.mean(gaps)) if gaps else np.nan,
            practice_gap_pct_missing=int(not gaps),
        )
        if task == "race":
            qrow = quali.loc[driver] if driver in quali.index else None
            qrank = qrow.rank_norm if qrow is not None else np.nan
            raw_position = qrow.official_position if qrow is not None else np.nan
            row.update(
                qualifying_position=raw_position,
                qualifying_rank_norm=qrank,
                qualifying_rank_norm_missing=int(pd.isna(qrank)),
                qualifying_classification_count=int(pd.notna(qrank)),
            )
            for segment in ("q1", "q2", "q3"):
                pace = summarize_driver(qpace, driver, segment.upper())
                row[f"qualifying_{segment}_gap_pct"] = pace["gap"]
                row[f"qualifying_{segment}_gap_pct_missing"] = int(pd.isna(pace["gap"]))
                row[f"qualifying_{segment}_lap_count"] = pace["comparable"]
            raw_grid, rank_grid, pit_lane = grid_values(entrant, cutoff, len(entrants))
            row.update(
                known_grid_position=raw_grid,
                known_grid_rank_norm=rank_grid,
                known_grid_rank_norm_missing=int(pd.isna(rank_grid)),
                known_pit_lane_start=pit_lane,
            )
        rows.append(row)
    features = pd.DataFrame(rows)
    model_matrix(features, task)
    return features, {
        "event_id": event_id,
        "task": task,
        "cutoff": cutoff.isoformat(),
        "mode": mode,
        "sources": audit,
        "history_weekends": [e["event_id"] for e in weekends],
        "skipped_optional": skipped,
        "feature_version": FEATURE_VERSION,
    }


def labels_for(dataset, event_id, task, drivers, *, strict=True):
    """Label-side join only. The v3 build retains missing labels for announced drivers."""
    source = as_source(dataset)
    spec = gp_session(source.event(event_id), "Qualifying" if task == "qualifying" else "Race")
    labels = normalized_results(source.component(spec, "results"))
    if strict and set(labels.driver_id) != set(drivers) or set(labels.driver_id) - set(drivers):
        raise ValueError(
            f"{event_id} {task}: roster/result field differs; review entries and substitutions"
        )
    labels = pd.DataFrame({"driver_id": list(drivers)}).merge(
        labels, how="left", on="driver_id", validate="one_to_one"
    )
    labels["event_id"], labels["task"] = event_id, task
    labels["rank_target"] = labels.rank_norm
    labels["target_available"] = labels.official_position.notna()
    labels["label_session_end"] = spec["ended_at"]
    labels["label_retrieved_at"] = spec["retrieved_at"]
    return labels


def build_development(dataset, manifest, entrants, cutoffs, output: Path):
    source = as_source(dataset)
    if not CUTOFF_COLUMNS.issubset(cutoffs) or not ENTRANT_COLUMNS.issubset(entrants):
        raise ValueError("Missing required cutoff/entrant columns")
    assert_development(manifest, cutoffs.event_id)
    assert_development(manifest, entrants.event_id)
    expected = {(e, t) for e in manifest["development"] for t in ("qualifying", "race")}
    keys = list(zip(cutoffs.event_id, cutoffs.task, strict=True))
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("Provide exactly one cutoff per development event and task")
    if set(zip(entrants.event_id, entrants.task, strict=True)) != expected:
        raise ValueError("Provide entrants for every development event and task")
    new_directory(output)
    audits, coverage = [], []
    for task in ("qualifying", "race"):
        frames, targets = [], []
        # Cutoff timestamps determine order; input CSV order has no effect on row order.
        ordered_cutoffs = cutoffs.assign(cutoff_order=[utc(value) for value in cutoffs.cutoff])
        for row in (
            ordered_cutoffs.loc[ordered_cutoffs.task.eq(task)]
            .sort_values(["cutoff_order", "event_id"])
            .itertuples(index=False)
        ):
            roster = entrants.loc[entrants.event_id.eq(row.event_id) & entrants.task.eq(task)]
            features, audit = build_event(
                source, manifest, row.event_id, task, row.cutoff, roster, row.mode
            )
            labels = labels_for(source, row.event_id, task, features.driver_id, strict=False)
            frames.append(features)
            targets.append(labels)
            audits.append(audit)
            coverage.append(
                {
                    "event_id": row.event_id,
                    "task": task,
                    "core_rows": len(features),
                    "practice_pace_rows": int(features.practice_gap_pct.notna().sum()),
                    "rank_targets": int(labels.rank_target.notna().sum()),
                    "qualifying_pace_rows": int(features.qualifying_q1_gap_pct.notna().sum())
                    if task == "race"
                    else 0,
                }
            )
        save_table(output / f"{task}_features.parquet", pd.concat(frames, ignore_index=True))
        save_table(output / f"{task}_labels.parquet", pd.concat(targets, ignore_index=True))
    save_table(
        output / "entrants.parquet",
        entrants.sort_values(["event_id", "task", "driver_id"]).reset_index(drop=True),
    )
    save_table(
        output / "cutoffs.parquet", cutoffs.sort_values(["event_id", "task"]).reset_index(drop=True)
    )
    save_table(output / "coverage.parquet", pd.DataFrame(coverage))
    dictionary_id = write_dictionary(output)
    write_json(output / "source_binding.json", source.receipt)
    return seal_directory(
        output,
        {
            "manifest_id": manifest["id"],
            "source_id": source.receipt["id"],
            "audit": audits,
            "feature_version": FEATURE_VERSION,
            "dictionary_id": dictionary_id,
            "implementation_id": implementation_id(),
            "dependency_versions": {
                "python": platform.python_version(),
                **{name: version(name) for name in ("pandas", "numpy", "pyarrow", "fastf1")},
            },
        },
    )
