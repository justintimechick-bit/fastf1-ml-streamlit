"""Freeze fitted artifacts and the complete scoring protocol before holdout access."""

import re
import shutil
from pathlib import Path

import pandas as pd

from f1ml.domain import utc
from f1ml.evaluation.splits import assert_development, assert_disjoint_events, gp_session
from f1ml.features.build import validate_cutoff, validate_roster
from f1ml.features.dictionary import implementation_id
from f1ml.models.qualifying import FAMILIES, STRATEGIES
from f1ml.models.qualifying import load_model as load_qualifying
from f1ml.models.race import PODIUM_FAMILIES, RACE_FAMILIES
from f1ml.models.race import load_model as load_race
from f1ml.storage import (
    digest,
    file_digest,
    new_directory,
    read_json,
    save_table,
    seal_directory,
    verify_directory,
    write_json,
)


def code_id():
    root = Path(__file__).parents[1]
    return digest({str(p.relative_to(root)): file_digest(p) for p in sorted(root.rglob("*.py"))})


def event_references(value):
    if isinstance(value, str):
        return {value} if re.fullmatch(r"202[456]-\d{2}", value) else set()
    if isinstance(value, dict):
        return set().union(*(event_references(k) | event_references(v) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return set().union(*(event_references(v) for v in value))
    return set()


def audit_temporal_folds(value, source, manifest):
    if isinstance(value, dict):
        if {"train", "validate"} <= set(value):
            assert_disjoint_events(value["train"], [value["validate"]], manifest["final_test"])
            start = min(
                utc(s["scheduled_start"]) for s in source.event(value["validate"])["sessions"]
            )
            if any(
                utc(gp_session(source.event(e), "Race")["ended_at"]) >= start
                for e in value["train"]
            ):
                raise ValueError("A tuning/calibration fold uses a later or overlapping weekend")
        for item in value.values():
            audit_temporal_folds(item, source, manifest)
    elif isinstance(value, list):
        for item in value:
            audit_temporal_folds(item, source, manifest)


def dataset_coverage(source, manifest):
    rows = []
    for event in source.events:
        role = (
            "development"
            if event["event_id"] in manifest["development"]
            else ("test" if event["event_id"] in manifest["final_test"] else "excluded")
        )
        for spec in event["sessions"]:
            rows.append(
                {
                    "year": event["year"],
                    "event_id": event["event_id"],
                    "role": role,
                    "kind": str(spec["kind"]),
                    "completed": bool(spec.get("completed")),
                    "official_results": bool(spec.get("official_results")),
                    "lap_table_available": bool(spec.get("laps_table")),
                    "weather_table_available": bool(spec.get("weather_table")),
                }
            )
    sessions = pd.DataFrame(rows)
    seasons = sessions.groupby(["year", "role"], as_index=False).agg(
        events=("event_id", "nunique"),
        sessions=("kind", "size"),
        completed_sessions=("completed", "sum"),
        result_sessions=("official_results", "sum"),
        lap_sessions=("lap_table_available", "sum"),
        weather_sessions=("weather_table_available", "sum"),
    )
    return sessions, seasons


def synthetic_inputs(source, manifest, output: Path):
    """Fixture input generation only: no target-result or participant-table lookup."""
    if source.receipt["id"] != manifest["dataset_id"] or not all(
        str(e.get("name", "")).startswith("SYNTHETIC GP ") for e in source.events
    ):
        raise ValueError("Synthetic final inputs require the explicit SYNTHETIC demo source")
    from f1ml.demo import SyntheticProvider

    rosters, cutoffs = [], []
    for event_id in manifest["final_test"]:
        event = source.event(event_id)
        q = gp_session(event, "Qualifying")
        for task in ("qualifying", "race"):
            cutoff = (
                utc(q["scheduled_start"]) - pd.Timedelta(seconds=1)
                if task == "qualifying"
                else utc(q["ended_at"]) + pd.Timedelta(hours=1)
            )
            cutoffs.append(
                {
                    "event_id": event_id,
                    "task": task,
                    "cutoff": cutoff.isoformat(),
                    "mode": "historical_reconstruction",
                }
            )
            for index, driver in enumerate(SyntheticProvider.drivers(event)):
                rosters.append(
                    {
                        "event_id": event_id,
                        "task": task,
                        "driver_id": driver,
                        "team_id": f"synthetic_team_{index // 2}",
                        "known_at": (cutoff - pd.Timedelta(days=1)).isoformat(),
                        "source": "SYNTHETIC independent fixture roster generator",
                    }
                )
    new_directory(output)
    pd.DataFrame(rosters).to_csv(output / "entrants.csv", index=False, mode="x")
    pd.DataFrame(cutoffs).to_csv(output / "cutoffs.csv", index=False, mode="x")
    return seal_directory(
        output,
        {
            "kind": "synthetic_test_inputs",
            "source_id": source.receipt["id"],
            "manifest_id": manifest["id"],
            "target_results_read": False,
        },
    )


def validate_inputs(source, manifest, entrants, cutoffs):
    expected = {(e, t) for e in manifest["final_test"] for t in ("qualifying", "race")}
    if not {"event_id", "task", "cutoff", "mode"} <= set(cutoffs):
        raise ValueError("Test cutoffs require event_id, task, cutoff, mode")
    if (
        cutoffs.duplicated(["event_id", "task"]).any()
        or set(zip(cutoffs.event_id, cutoffs.task, strict=True)) != expected
    ):
        raise ValueError("Provide exactly one cutoff for every reserved event/task")
    if not {"event_id", "task", "driver_id"} <= set(entrants):
        raise ValueError("Test entrants require event/task/driver identity")
    if set(zip(entrants.event_id, entrants.task, strict=True)) != expected:
        raise ValueError("Provide independent rosters for exactly the reserved block")
    for row in cutoffs.itertuples():
        roster = entrants.loc[entrants.event_id.eq(row.event_id) & entrants.task.eq(row.task)]
        validate_cutoff(source.event(row.event_id), row.task, utc(row.cutoff), row.mode)
        validate_roster(roster, row.event_id, row.task, utc(row.cutoff))


def audit_development(source, manifest, core_root, qualifying_root, race_root, experiments_root):
    if source.receipt["id"] != manifest["dataset_id"]:
        raise ValueError("Finalization source differs from the frozen manifest")
    dev, final = set(manifest["development"]), set(manifest["final_test"])
    assert_disjoint_events(dev, (), final)
    eligible_2026 = [e["event_id"] for e in manifest["eligible"] if e["year"] == 2026]
    if manifest["final_test"] != eligible_2026[-manifest["config"]["final_test_weekends"] :]:
        raise ValueError("Reserved block is not the configured latest eligible 2026 weekends")
    boundary = min(utc(s["scheduled_start"]) for e in final for s in source.event(e)["sessions"])
    receipts = {
        name: verify_directory(path)
        for name, path in (
            ("core", core_root),
            ("qualifying", qualifying_root),
            ("race", race_root),
            ("experiments", experiments_root),
        )
    }
    core = receipts["core"]
    if (
        core["source_id"] != source.receipt["id"]
        or core["implementation_id"] != implementation_id()
    ):
        raise ValueError("Frozen core source/implementation differs")
    for receipt in receipts.values():
        if receipt["manifest_id"] != manifest["id"]:
            raise ValueError("Development artifacts belong to another manifest")
    for name in ("qualifying", "race"):
        for suffix in ("features", "labels"):
            frame = pd.read_parquet(core_root / f"{name}_{suffix}.parquet")
            assert_development(manifest, frame.event_id)
            if set(frame.event_id) != dev:
                raise ValueError("Core development snapshot is incomplete")
    for entry in core["audit"]:
        assert_development(manifest, [entry["event_id"], *entry["history_weekends"]])
        for observation in entry["sources"]:
            assert_development(manifest, [observation["event_id"]])
            if observation["event_id"] == entry["event_id"] and (
                observation["kind"] == "Race"
                or (entry["task"] == "qualifying" and observation["kind"] == "Qualifying")
            ):
                raise ValueError("Forbidden target-session input in development audit")
    checked = 0
    for root in (qualifying_root, race_root, experiments_root):
        for path in root.rglob("model.json"):
            meta = read_json(path)
            if event_references(meta) & final:
                raise ValueError(f"Reserved outcome dependency found: {path}")
            assert_development(manifest, meta["train_events"])
            audit_temporal_folds(meta, source, manifest)
            validation_ids = event_references(list(path.relative_to(root).parts))
            for event_id in validation_ids:
                start = min(utc(s["scheduled_start"]) for s in source.event(event_id)["sessions"])
                if utc(meta["max_weekend_end"]) >= start or event_id in meta["train_events"]:
                    raise ValueError("Outer validation model uses its own or later weekend")
            for key in ("max_feature_cutoff", "max_label_end", "max_weekend_end"):
                if utc(meta[key]) >= boundary:
                    raise ValueError("Model dependency is not before the final block")
            if (
                meta["manifest_id"] != manifest["id"]
                or meta.get("feature_snapshot_id") != core["id"]
            ):
                raise ValueError("Model lineage differs from the frozen core")
            verify_directory(path.parent)
            checked += 1
        for file in ("event_metrics.parquet", "predictions.parquet", "evaluation_labels.parquet"):
            if not (root / file).exists():
                continue
            frame = pd.read_parquet(root / file)
            assert_development(manifest, frame.event_id)
    retention = read_json(experiments_root / "retention.json")
    if any(r["decision"] != "do_not_promote" for r in retention):
        raise ValueError("Resolve optional feature candidates before locking core-only models")
    return {
        "status": "no_recorded_test_outcome_dependencies",
        "models_checked": checked,
        "receipts": {k: r["id"] for k, r in receipts.items()},
        "first_test_weekend": boundary.isoformat(),
        "feature_policy": "core; optional groups not promoted under development evidence gates",
        "limits": [
            "Integrity/lineage audit cannot prove absence of unlogged human access.",
            (
                "Manifest eligibility inspected outcome availability before freezing, "
                "not model quality."
            ),
            "Synthetic fixtures share a generator; they do not establish real-world independence.",
        ],
    }


def choose_models(manifest, qualifying_root, race_root):
    choices = {}
    for task, root, families in (
        ("qualifying", qualifying_root, FAMILIES),
        ("race", race_root, RACE_FAMILIES),
        ("podium", race_root, PODIUM_FAMILIES),
    ):
        metrics = pd.read_parquet(root / "event_metrics.parquet")
        if "task" in metrics:
            metrics = metrics.loc[metrics.task.eq(task)]
        cohort = metrics.loc[metrics.common_comparison]
        events = set(cohort.event_id)
        if not events:
            raise ValueError("No common development events for all historical strategies")
        losses = []
        key = "log_loss" if task == "podium" else "position_mae"
        for family in families:
            for strategy in STRATEGIES:
                group = cohort.loc[cohort.family.eq(family) & cohort.strategy.eq(strategy)]
                if set(group.event_id) != events or group.event_id.duplicated().any():
                    raise ValueError("Historical strategies do not share the same complete events")
                losses.append((float(group[key].mean()), family, strategy))
        _, family, strategy = min(losses, key=lambda r: r[0])
        path = root / "selected_model" if task == "qualifying" else root / "selected_models" / task
        model = load_qualifying(path) if task == "qualifying" else load_race(path, task)
        if (model.metadata["family"], model.metadata["strategy"]) != (family, strategy):
            raise ValueError("Selected refit does not match the development-only winner")
        if model.metadata.get("experiment"):
            raise ValueError("Primary lock currently requires the validated core matrix")
        choices[task] = {
            "source": path,
            "family": family,
            "strategy": strategy,
            "common_events": [e for e in manifest["development"] if e in events],
            "metadata": model.metadata,
        }
    return choices


def lock_models(
    source,
    manifest,
    core_root: Path,
    qualifying_root: Path,
    race_root: Path,
    experiments_root: Path,
    entrants: Path,
    cutoffs: Path,
    output: Path,
    ledger: Path,
    exploratory=False,
):
    if (ledger / manifest["id"]).exists() and not exploratory:
        raise ValueError("Primary holdout already claimed; further locks must be exploratory")
    roster = pd.read_csv(entrants, dtype=str)
    times = pd.read_csv(cutoffs, dtype=str)
    validate_inputs(source, manifest, roster, times)
    audit = audit_development(
        source, manifest, core_root, qualifying_root, race_root, experiments_root
    )
    choices = choose_models(manifest, qualifying_root, race_root)
    new_directory(output)
    roster.to_csv(output / "entrants.csv", index=False, mode="x")
    times.to_csv(output / "cutoffs.csv", index=False, mode="x")
    write_json(output / "manifest.json", manifest)
    write_json(output / "audit.json", audit)
    sessions, seasons = dataset_coverage(source, manifest)
    for name, frame in (("dataset_sessions", sessions), ("dataset_coverage", seasons)):
        save_table(output / f"{name}.parquet", frame)
        frame.to_csv(output / f"{name}.csv", index=False, mode="x")
    for task, choice in choices.items():
        shutil.copytree(choice["source"], output / "models" / task)
    for name, root in (
        ("qualifying", qualifying_root),
        ("race", race_root),
        ("features", experiments_root),
    ):
        destination = output / "development" / name
        destination.mkdir(parents=True)
        for filename in (
            "event_metrics.parquet",
            "predictions.parquet",
            "summary.json",
            "summary.parquet",
            "eligibility.json",
            "tuning.json",
            "search_config.json",
            "config.json",
            "paired_differences.parquet",
            "retention.json",
            "missingness.parquet",
            "report.md",
            "evaluation_labels.parquet",
        ):
            if (root / filename).exists():
                shutil.copyfile(root / filename, destination / filename)
    shutil.copyfile(core_root / "coverage.parquet", output / "development/core_coverage.parquet")
    kind = (
        "synthetic_validation"
        if all(str(e.get("name", "")).startswith("SYNTHETIC") for e in source.events)
        else "observed_source"
    )
    protocol = {
        "version": 1,
        "manifest_id": manifest["id"],
        "source_id": source.receipt["id"],
        "core_snapshot_id": audit["receipts"]["core"],
        "data_kind": kind,
        "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "code_id": code_id(),
        "exploratory": exploratory,
        "fresh_future_holdout_required": exploratory or kind != "observed_source",
        "history_policy": "pre-block development weekends only; no block outcomes enter history",
        "training_policy": (
            "fixed copied pre-block refits; no fitting or calibration during final evaluation"
        ),
        "prediction_order": (
            "seal all qualifying forecasts, then race/podium forecasts, then score labels"
        ),
        "selection_policy": (
            "best learned family/strategy by equal-event common development loss; "
            "fixed baselines are references"
        ),
        "features": audit["feature_policy"],
        "baselines": {
            "qualifying": ["recent", "practice"],
            "race": ["qualifying", "recent"],
            "podium": ["qualifying", "recent"],
        },
        "calibration_bins": 5,
        "subgroup_min_events": 3,
        "bootstrap_min_events": 8,
        "bootstrap_replicates": 4000,
        "random_seed": 42,
        "uncertainty": (
            "event blocks only; below eight events report range/SD/leave-one-out "
            "sensitivity, no confidence interval"
        ),
        "models": {
            task: {k: v for k, v in choice.items() if k != "source"}
            for task, choice in choices.items()
        },
        "exclusions": manifest["excluded"],
        "dataset_coverage": seasons.to_dict("records"),
    }
    write_json(output / "protocol.json", protocol)
    return seal_directory(
        output,
        {
            "kind": "final_model_lock",
            "manifest_id": manifest["id"],
            "source_id": source.receipt["id"],
            "data_kind": kind,
            "exploratory": exploratory,
            "locked_before_test_outcomes": True,
        },
    )
