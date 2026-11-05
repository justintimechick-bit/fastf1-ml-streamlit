"""Immutable real-data inventory and fail-closed experiment feasibility review.

This module never downloads, fits, creates a final lock, or opens an evaluation
ledger. A source inventory is not a modeling dataset or a final-test manifest.
"""

import shutil
import sys
import tomllib
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from f1ml.config import load_config
from f1ml.domain import PRACTICES, utc
from f1ml.features.build import validate_cutoff, validate_roster
from f1ml.features.pace import comparable_pace
from f1ml.features.source import CatalogSource
from f1ml.finalization.locking import code_id
from f1ml.ingestion.catalog import current_snapshot
from f1ml.storage import (
    digest,
    file_digest,
    new_directory,
    read_json,
    seal_directory,
    verify_directory,
    write_json,
)


def failure_category(row):
    message = str(row.get("message", "")).lower()
    kind = str(row.get("type", ""))
    if "too many requests" in message or "429" in message:
        return "source_rate_limit"
    if "401" in message or "unauthorized" in message:
        return "source_authentication"
    if "404" in message or "not found" in message:
        return "source_endpoint_missing"
    if any(t in message for t in ("resolve", "connection", "timed out", "ssl")):
        return "transport_failure"
    if kind in ("DataNotLoadedError", "SessionNotAvailableError"):
        return "timing_not_loaded_root_cause_needs_endpoint_diagnostic"
    if "normaliz" in kind.lower() or "parsing" in kind.lower():
        return "parsing_or_normalization_error"
    return "unclassified_requires_review"


def target_checks(frame, kind):
    """Strict complete-order cohort; preserve the original labels and special statuses."""
    reasons = []
    if frame.empty:
        return ["missing_official_results"]
    if not frame.source.eq("jolpica_via_fastf1_ergast").all():
        reasons.append("unapproved_result_provenance")
    if frame.driver_id.isna().any() or frame.driver_id.duplicated().any():
        reasons.append("unresolved_or_duplicate_driver_identity")
    if frame.team_id.isna().any() or frame.team_mapping_state.ne("mapped").any():
        reasons.append("unresolved_team_mapping")
    rank = pd.to_numeric(frame.official_position, errors="coerce")
    if rank.isna().any():
        reasons.append("missing_numeric_classification")
    if rank.duplicated().any() or frame.position_ambiguous.fillna(True).any():
        reasons.append("ambiguous_official_positions")
    if sorted(rank.dropna().tolist()) != list(range(1, len(frame) + 1)):
        reasons.append("incomplete_official_order")
    if not frame.official_field_complete.fillna(False).all():
        reasons.append("incomplete_provider_response")
    if frame.outcome.eq("unknown").any():
        reasons.append("unknown_classification_status")
    if kind == "Race":
        expected = rank.between(1, 3).astype(float)
        prohibited = frame.outcome.isin(
            [
                "dns",
                "dsq",
                "excluded",
                "withdrawn",
                "dnq",
                "non_classified",
                "retired_non_classified",
            ]
        )
        expected.loc[prohibited] = 0
        if (
            frame.podium.isna().any()
            or not np.array_equal(frame.podium.to_numpy(), expected.to_numpy())
            or frame.podium.sum() != 3
        ):
            reasons.append("incomplete_or_contradictory_podium")
    return reasons


def review_event(source, event, results, entrants, cutoffs, as_of):
    """Eligibility uses source validity and review evidence, never prediction loss."""
    eid = event["event_id"]
    reasons, qids, rids, rosters_reviewed = [], set(), set(), []
    label_ok = True
    for kind in ("Qualifying", "Race"):
        rows = results.loc[results.event_id.eq(eid) & results.kind.eq(kind)]
        checks = target_checks(rows, kind)
        label_ok &= not checks
        reasons.extend(f"{kind}:{r}" for r in checks)
        ids = set(rows.driver_id.dropna()) if not rows.empty else set()
        if kind == "Qualifying":
            qids = ids
        else:
            rids = ids
        spec = next(s for s in event["sessions"] if s["kind"] == kind)
        if utc(spec["scheduled_start"]) >= as_of:
            reasons.append(f"{kind}:not_started_as_of")
        if not spec.get("completed") or not spec.get("ended_at"):
            reasons.append(f"{kind}:completion_time_unverified")
        elif utc(spec["ended_at"]) > as_of:
            reasons.append(f"{kind}:not_completed_as_of")
        task = "qualifying" if kind == "Qualifying" else "race"
        c = cutoffs.loc[cutoffs.event_id.eq(eid) & cutoffs.task.eq(task)]
        roster = entrants.loc[entrants.event_id.eq(eid) & entrants.task.eq(task)]
        if len(c) != 1:
            reasons.append(f"{kind}:missing_or_duplicate_reviewed_cutoff")
            rosters_reviewed.append(False)
            continue
        try:
            cutoff = utc(c.iloc[0].cutoff)
            mode = c.iloc[0]["mode"]
            validate_cutoff(event, task, cutoff, mode)
            validate_roster(roster, eid, task, cutoff)
            if set(roster.driver_id) != ids:
                raise ValueError("independent roster and official result field differ")
            teams = rows.set_index("driver_id").team_id
            if any(teams.get(r.driver_id) != r.team_id for r in roster.itertuples()):
                raise ValueError("independent roster and result team membership differ")
            if mode != "historical_reconstruction":
                raise ValueError("this experiment is explicitly historical_reconstruction")
            rosters_reviewed.append(True)
        except (ValueError, KeyError, TypeError) as exc:
            reasons.append(f"{kind}:review_invalid:{exc}")
            rosters_reviewed.append(False)
    if qids and rids and qids != rids and not all(rosters_reviewed):
        # A real substitution may explain this; it still requires independent task rosters.
        reasons.append("Q_R_field_difference_requires_independent_roster_review")
    return {
        "event_id": eid,
        "year": event["year"],
        "round": event["round"],
        "name": event["name"],
        "qualifying_drivers": len(qids),
        "race_drivers": len(rids),
        "complete_label_pair": bool(label_ok),
        "eligible": not reasons,
        "reasons": " | ".join(reasons),
        "outcome_exposure": "source_QA_previously_accessible; untouched_status_not_established",
    }


def session_groups(tables, sessions, options):
    rows = []
    for s in sessions.itertuples():
        for group, table in (
            ("official_results", "results"),
            ("practice_laps", "practice_laps"),
            ("qualifying_laps", "qualifying_laps"),
            ("weather_observations", "session_weather"),
            ("comparable_pace", None),
        ):
            applicable = (
                s.kind in ("Qualifying", "Race")
                if group == "official_results"
                else s.kind in PRACTICES
                if group == "practice_laps"
                else s.kind == "Qualifying"
                if group == "qualifying_laps"
                else s.kind in PRACTICES | {"Qualifying"}
                if group == "comparable_pace"
                else s.kind in PRACTICES | {"Qualifying", "Race"}
            )
            count, drivers = 0, 0
            if applicable:
                if table:
                    frame = tables[table]
                    frame = frame.loc[frame.session_id.eq(s.session_id)]
                else:
                    name = "qualifying_laps" if s.kind == "Qualifying" else "practice_laps"
                    laps = tables[name].loc[tables[name].session_id.eq(s.session_id)]
                    weather = tables["session_weather"]
                    weather = weather.loc[weather.session_id.eq(s.session_id)]
                    frame = comparable_pace(laps, weather, s.kind, options)
                    if not frame.empty:
                        frame = frame.loc[frame.gap_pct.notna()]
                count = len(frame)
                drivers = frame.driver_id.nunique() if "driver_id" in frame else 0
            rows.append(
                {
                    "year": s.year,
                    "event_id": s.event_id,
                    "session_id": s.session_id,
                    "kind": s.kind,
                    "completion_state": s.completion_state,
                    "feature_group": group,
                    "applicable": applicable,
                    "observed_rows": count,
                    "observed_drivers": drivers,
                    "observed": count > 0,
                    "cutoff_verified": bool(pd.notna(s.completed_at)),
                }
            )
    return pd.DataFrame(rows)


def planned_results():
    rows = []
    for task, families, rules in (
        (
            "qualifying",
            ["ridge", "random_forest", "gradient_boosting"],
            ["baseline_recent", "baseline_practice"],
        ),
        (
            "race",
            ["ridge", "random_forest", "gradient_boosting"],
            ["baseline_qualifying", "baseline_recent"],
        ),
        (
            "podium",
            ["logistic", "random_forest", "gradient_boosting"],
            ["baseline_qualifying", "baseline_recent"],
        ),
    ):
        for phase in ("development", "final_test"):
            for family in families + rules:
                strategies = (
                    ["fixed_rule"] if family in rules else ["2026_only", "equal", "weighted"]
                )
                for strategy in strategies:
                    rows.append(
                        {
                            "phase": phase,
                            "task": task,
                            "family": family,
                            "strategy": strategy,
                            "events": 0,
                            "value": None,
                            "status": "not_run_ineligible_dataset",
                        }
                    )
    return pd.DataFrame(rows)


def audit(config_path: Path, output: Path, root=Path(".")):
    root = root.resolve()
    config_bytes = config_path.read_bytes()
    cfg = tomllib.loads(config_bytes.decode())
    training = load_config(root / cfg["training_config"])
    as_of = utc(cfg["as_of"])
    snapshot, snapshot_receipt = current_snapshot(root / cfg["catalog"])
    if utc(snapshot_receipt["as_of"]) < as_of:
        raise ValueError("Catalog horizon is older than the requested experiment horizon")
    tables = {p.stem: pd.read_parquet(p) for p in (snapshot / "tables").glob("*.parquet")}
    if tables["events"].name.str.contains("synthetic", case=False).any():
        raise ValueError("Real experiment rejects synthetic events")
    if set(tables["events"].year) - {2024, 2025, 2026}:
        raise ValueError("Only real 2024–2026 data is supported")
    new_directory(output)
    write_json(
        output / "environment.json",
        {
            "python": sys.version,
            "packages": {
                name: version(name)
                for name in (
                    "fastf1",
                    "pandas",
                    "numpy",
                    "scikit-learn",
                    "pyarrow",
                    "joblib",
                    "streamlit",
                )
            },
        },
    )
    for package in ("f1ml_real", "f1ml"):
        src = Path(__file__).parents[1] / package
        shutil.copytree(
            src,
            output / "implementation" / package,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    # A pinned independent copy is safe from future CURRENT.json changes or resumes.
    shutil.copytree(snapshot, output / "dataset")
    verify_directory(output / "dataset")
    # Keep raw frames and failed attempts as well as normalized snapshot tables.
    catalog = root / cfg["catalog"]
    diagnostics = []
    for metadata_path in sorted((catalog / "records").rglob("metadata.json")):
        record = metadata_path.parent
        verify_directory(record)
        metadata = read_json(metadata_path)
        shutil.copytree(record, output / "source_records" / record.relative_to(catalog))
        diagnostics.append(
            {
                "event_id": metadata["event_id"],
                "session_id": metadata["session_id"],
                "component": metadata["component"],
                "revision": str(record.relative_to(catalog)),
                "normalization_errors": len(metadata.get("normalization_errors", [])),
                "source_errors": len(metadata.get("errors", [])),
                "attempt_failures": len(metadata.get("attempt_failures", [])),
            }
        )
    pd.DataFrame(diagnostics).to_csv(output / "component_diagnostics.csv", index=False)
    shutil.copy2(catalog / "state.json", output / "acquisition_state.json")
    (output / "config.toml").write_bytes(config_bytes)
    (output / "training-config.toml").write_bytes((root / cfg["training_config"]).read_bytes())
    for name in ("availability", "entrants", "cutoffs"):
        src = root / cfg[name]
        if src.exists():
            (output / "reviews").mkdir(exist_ok=True)
            shutil.copy2(src, output / "reviews" / f"{name}.csv")

    def review(name, columns):
        path = output / "reviews" / f"{name}.csv"
        frame = pd.read_csv(path, dtype=str) if path.exists() else pd.DataFrame(columns=columns)
        if not set(columns) <= set(frame):
            raise ValueError(f"Review {name} requires {columns}")
        return frame

    availability = review(
        "availability", ["session_id", "completed_at", "available_at", "evidence"]
    )
    entrants = review(
        "entrants", ["event_id", "task", "driver_id", "team_id", "known_at", "source"]
    )
    cutoffs = review("cutoffs", ["event_id", "task", "cutoff", "mode"])
    source = CatalogSource(output / "dataset", availability)
    events = pd.DataFrame(
        [
            review_event(source, e, tables["results"], entrants, cutoffs, as_of)
            for e in source.events
        ]
    )
    events.to_csv(output / "event_exclusions.csv", index=False)
    groups = session_groups(tables, tables["sessions"], training.features)
    groups.to_csv(output / "session_feature_coverage.csv", index=False)
    for level in ("year", "event_id"):
        summary = (
            groups.loc[groups.applicable]
            .groupby([level, "feature_group"], as_index=False)
            .agg(
                sessions=("session_id", "size"),
                observed_sessions=("observed", "sum"),
                observed_rows=("observed_rows", "sum"),
                cutoff_verified_sessions=("cutoff_verified", "sum"),
            )
        )
        summary.to_csv(output / f"{level}_feature_coverage.csv", index=False)
    failures = pd.read_parquet(snapshot / "coverage/failures.parquet")
    failures["diagnosis"] = failures.apply(failure_category, axis=1)
    failures.to_csv(output / "source_failures.csv", index=False)
    for name in ("endpoint_checks", "preservation"):
        path = root / cfg[name]
        if path.exists():
            shutil.copy2(path, output / f"{name}.json")
    preservation = (
        read_json(output / "preservation.json") if (output / "preservation.json").exists() else {}
    )
    changed = [
        p for p, h in preservation.items() if not (root / p).is_file() or file_digest(root / p) != h
    ]
    if changed:
        raise ValueError(f"Preserved synthetic experiment changed: {changed}")
    # Never read synthetic predictions as real model evidence.
    synthetic = root / cfg["synthetic_features"]
    missing = []
    if synthetic.exists():
        verify_directory(synthetic)
        for task in ("qualifying", "race"):
            frame = pd.read_parquet(synthetic / f"{task}_features.parquet")
            for col in frame:
                if "gap_pct" in col and not col.endswith("_missing"):
                    missing.append(
                        {
                            "data_kind": "synthetic_validation",
                            "task": task,
                            "feature": col,
                            "observed_rows": int(frame[col].notna().sum()),
                            "total_rows": len(frame),
                        }
                    )
    pd.DataFrame(missing).to_csv(output / "synthetic_pace_diagnostic.csv", index=False)
    # Record all result classifications without changing or dropping any official row.
    targets = tables["results"]
    columns = [
        "event_id",
        "year",
        "kind",
        "driver_id",
        "driver_number",
        "team_id",
        "official_position",
        "classification_code",
        "outcome",
        "podium",
        "source_revision",
        "team_mapping_state",
        "position_ambiguous",
    ]
    targets[[c for c in columns if c in targets]].to_csv(
        output / "target_identity_audit.csv", index=False
    )
    outcomes = targets.groupby(["year", "kind", "outcome"], dropna=False).size().rename("rows")
    outcomes.reset_index().to_csv(output / "classification_statuses.csv", index=False)
    planned_results().to_csv(output / "evaluation_records.csv", index=False)
    eligible = events.loc[events.eligible]
    current = eligible.loc[eligible.year.eq(2026)].sort_values("round")
    strategy_rows = []
    for event in events.loc[events.year.eq(2026)].sort_values("round").itertuples():
        weekend_start = min(
            utc(s["scheduled_start"]) for s in source.event(event.event_id)["sessions"]
        )
        earlier_ids = [
            e.event_id
            for e in eligible.itertuples()
            if e.event_id != event.event_id
            and utc(
                next(s for s in source.event(e.event_id)["sessions"] if s["kind"] == "Race")[
                    "ended_at"
                ]
            )
            < weekend_start
        ]
        earlier = eligible.loc[eligible.event_id.isin(earlier_ids)]
        for strategy in ("2026_only", "equal", "weighted"):
            n = len(earlier.loc[earlier.year.eq(2026)]) if strategy == "2026_only" else len(earlier)
            minimum = (
                cfg["protocol"]["min_2026_training_events"]
                if strategy == "2026_only"
                else training.qualifying.min_training_events
            )
            strategy_rows.append(
                {
                    "event_id": event.event_id,
                    "strategy": strategy,
                    "earlier_reviewed_events": n,
                    "minimum": minimum,
                    "eligible": bool(event.eligible and n >= minimum),
                    "reason": "target_dataset_ineligible"
                    if not event.eligible
                    else "insufficient_earlier_history"
                    if n < minimum
                    else "reviewed_candidate",
                }
            )
    pd.DataFrame(strategy_rows).to_csv(output / "strategy_eligibility.csv", index=False)
    ntest = cfg["protocol"]["final_test_weekends"]
    ndev = cfg["protocol"]["min_development_events"]
    ready = len(current) >= ntest + ndev
    blockers = []
    if not ready:
        blockers.append(
            f"Need {ntest + ndev} eligible 2026 events (including {ntest} final and "
            f"{ndev} development); found {len(current)}. No automatic block shrinking."
        )
    if availability.empty:
        blockers.append("No independently evidenced completion/availability review exists.")
    if entrants.empty or cutoffs.empty:
        blockers.append("No independently sourced cutoff entrants and explicit task cutoffs exist.")
    limitations = []
    if not groups.loc[groups.feature_group.eq("comparable_pace"), "observed"].any():
        limitations.append(
            "Optional pace unavailable; core history/classification remains planned."
        )
    results_by_season = targets.groupby("year").session_id.nunique().to_dict()
    status = {
        "schema_version": 1,
        "experiment_id": cfg["experiment_id"],
        "data_kind": "real",
        "state": "review_ready_not_trained" if ready else "blocked_before_training",
        "source_snapshot_id": snapshot_receipt["id"],
        "source_catalog": cfg["catalog"],
        "as_of": cfg["as_of"],
        "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "eligible_events": len(eligible),
        "eligible_2026_events": len(current),
        "result_sessions_by_season": {str(k): int(v) for k, v in results_by_season.items()},
        "development_evaluated_events": 0,
        "final_evaluated_events": 0,
        "models_trained": 0,
        "training_cutoffs": [],
        "final_manifest_frozen": False,
        "final_lock_created": False,
        "evaluation_ledger_claimed": False,
        "untouched_holdout_claim": False,
        "blockers": blockers,
        "limitations": limitations,
        "normalization_error_records": sum(d["normalization_errors"] > 0 for d in diagnostics),
        "source_code_id": code_id(),
        "preserved_files_verified": len(preservation),
        "protocol": cfg["protocol"],
        "search_configuration": training.to_dict(),
        "prior_consultation": "Existing synthetic models are not real-data evidence. Real outcomes "
        "were accessible during ingestion/source QA, including this audit. Unlogged previous "
        "consultation cannot be ruled out. Any retrospective block must disclose this exposure; "
        "fresh prospective events are required for an untouched confirmation claim.",
    }
    write_json(output / "status.json", status)
    write_json(
        output / "protocol.json", {"status": "planned_not_final_manifest", **cfg["protocol"]}
    )
    report = [
        f"# {cfg['experiment_id']}",
        "",
        f"State: **{status['state']}**. Real observations only.",
        "",
        "No model was fitted or scored. Development: 0 events. Final test: 0 events.",
        "No final manifest, model lock or ledger claim was created.",
        "",
        "## Blockers",
        "",
        *[f"- {b}" for b in blockers],
        "",
        "## Source inventory",
        "",
    ]
    for year, n in status["result_sessions_by_season"].items():
        report.append(f"- {year}: {n} official Q/R result sessions.")
    report += [
        "",
        "The dataset directory is an immutable source snapshot, not a validated modeling dataset.",
        "Session/event/season group CSVs distinguish observations from cutoff eligibility.",
        "Repeated positions and field differences remain in event_exclusions.csv; "
        "no tail order or entrant is invented.",
        "",
        "## Evaluation and provenance",
        "",
        status["prior_consultation"],
        "",
        "The protocol and full search grid are saved before model selection. "
        "Eligibility uses source validity only.",
        "Final membership remains unfrozen until enough independently reviewed events exist.",
        "Missing optional timing must not disqualify an otherwise reviewed core dataset.",
        "Entirely missing columns must be excluded per training fold before fitting a real model.",
        "evaluation_records.csv lists planned methods with zero events and no metric value.",
        f"Verified {len(preservation)} preserved files; legacy src/f1ml fingerprint: {code_id()}.",
        "",
    ]
    (output / "report.md").write_text("\n".join(report))
    package = Path(__file__).parent
    implementation = digest({p.name: file_digest(p) for p in sorted(package.glob("*.py"))})
    return seal_directory(
        output,
        {
            "experiment_id": cfg["experiment_id"],
            "data_kind": "real",
            "state": status["state"],
            "implementation_id": implementation,
        },
    )
