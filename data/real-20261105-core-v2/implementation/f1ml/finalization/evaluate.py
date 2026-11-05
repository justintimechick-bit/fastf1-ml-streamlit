"""Consume a frozen holdout once: no fit/selection/calibration APIs are called here."""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from f1ml.evaluation.qualifying import complete_target_reason, score_event
from f1ml.evaluation.race_metrics import (
    PODIUM_METRICS,
    RACE_METRICS,
    podium_score,
    race_score,
    target_reason,
)
from f1ml.evaluation.splits import gp_session, load_manifest
from f1ml.features.history import normalized_results
from f1ml.finalization.locking import code_id, validate_inputs
from f1ml.finalization.reporting import (
    QUALIFYING_METRICS,
    aggregate,
    calibration_figure,
    paired_differences,
    report,
    session_condition,
    subgroup_errors,
    table,
)
from f1ml.finalization.source import holdout_features
from f1ml.ingestion.tables import json_value
from f1ml.models.qualifying import (
    load_model as load_qualifying,
)
from f1ml.models.qualifying import (
    rule_predictions as qualifying_rule,
)
from f1ml.models.race import load_model as load_race
from f1ml.models.race import rule_predictions as race_rule
from f1ml.prediction.qualifying import predict_checked as predict_qualifying
from f1ml.prediction.race import combine_predictions
from f1ml.prediction.race import predict_checked as predict_race
from f1ml.storage import new_directory, read_json, seal_directory, verify_directory, write_json


def labels_after_prediction(source, event_id, task, drivers):
    spec = gp_session(source.event(event_id), "Qualifying" if task == "qualifying" else "Race")
    labels = normalized_results(source.component(spec, "results"))
    labels["rank_target"] = labels.rank_norm
    labels["label_session_end"], labels["label_retrieved_at"] = (
        spec["ended_at"],
        spec["retrieved_at"],
    )
    labels["event_id"] = event_id
    # The outer join exposes roster mismatches; no driver silently disappears.
    labels = pd.DataFrame({"driver_id": list(drivers)}).merge(
        labels, on="driver_id", how="outer", validate="one_to_one", indicator="roster_match"
    )
    labels["event_id"] = event_id
    return labels


def score(predicted, labels, field, task, protocol):
    mismatch = labels.roster_match.ne("both").any()
    reason = (
        "roster_result_field_mismatch"
        if mismatch
        else (
            complete_target_reason(field, labels)
            if task == "qualifying"
            else target_reason(field, labels, task)
        )
    )
    if reason:
        names = (
            QUALIFYING_METRICS
            if task == "qualifying"
            else RACE_METRICS
            if task == "race"
            else PODIUM_METRICS
        )
        return {
            "event_id": str(field.event_id.iloc[0]),
            "drivers": len(field),
            "primary_eligible": False,
            "exclusion_reason": reason,
            **{name: None for name in names},
        }
    result = (
        score_event(predicted, labels, field)
        if task == "qualifying"
        else (
            race_score(predicted, labels, field)
            if task == "race"
            else podium_score(predicted, labels, field, protocol["calibration_bins"])
        )
    )
    return {**result, "primary_eligible": True, "exclusion_reason": None}


def development_calibration(lock_root, protocol):
    meta = protocol["models"]["podium"]
    frame = pd.read_parquet(lock_root / "development/race/predictions.parquet")
    selected = frame.loc[
        frame.task.eq("podium")
        & frame.event_id.isin(meta["common_events"])
        & (
            (frame.family.eq(meta["family"]) & frame.strategy.eq(meta["strategy"]))
            | frame.family.isin(["baseline_qualifying", "baseline_recent"])
        )
    ].copy()
    selected["method"] = np.where(
        selected.family.isin(["baseline_qualifying", "baseline_recent"]), selected.family, "model"
    )
    # Labels are preserved from development, not re-read from final sessions.
    labels_path = lock_root / "development/race/evaluation_labels.parquet"
    labels = pd.read_parquet(labels_path)
    return selected.merge(
        labels[["event_id", "driver_id", "podium"]],
        on=["event_id", "driver_id"],
        validate="many_to_one",
    )


def evaluate_once(lock_root: Path, source, output: Path, ledger: Path, exploratory=False):
    locked = verify_directory(lock_root)
    protocol = read_json(lock_root / "protocol.json")
    manifest = load_manifest(lock_root / "manifest.json")
    if protocol["code_id"] != code_id():
        raise ValueError(
            "Code changed after lock; create a new lock, exploratory after holdout use"
        )
    if source.receipt["id"] != protocol["source_id"] or manifest["id"] != protocol["manifest_id"]:
        raise ValueError("Evaluation source/manifest differs from locked inputs")
    entrants, cutoffs = (
        pd.read_csv(lock_root / name, dtype=str) for name in ("entrants.csv", "cutoffs.csv")
    )
    validate_inputs(source, manifest, entrants, cutoffs)
    models = {
        "qualifying": load_qualifying(lock_root / "models/qualifying"),
        "race": load_race(lock_root / "models/race", "race"),
        "podium": load_race(lock_root / "models/podium", "podium"),
    }
    model_ids = {task: verify_directory(lock_root / "models" / task)["id"] for task in models}
    if output.exists():
        raise FileExistsError(output)
    exploratory = exploratory or protocol["exploratory"]
    effective_protocol = {
        **protocol,
        "exploratory": exploratory,
        "fresh_future_holdout_required": exploratory or protocol["fresh_future_holdout_required"],
    }
    primary = ledger / manifest["id"]
    if not exploratory and primary.exists():
        raise ValueError("Primary holdout already claimed; reuse must be explicitly exploratory")
    reservation = (
        primary
        if not exploratory
        else primary / "exploratory" / str(pd.Timestamp.now(tz="UTC").value)
    )
    new_directory(reservation)  # Exclusive claim is durable before any held-out observation read.
    write_json(
        reservation / "claim.json",
        {
            "lock_id": locked["id"],
            "output": str(output.resolve()),
            "claimed_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "exploratory": exploratory,
            "data_kind": protocol["data_kind"],
        },
    )
    try:
        new_directory(output)
        shutil.copytree(lock_root / "development", output / "development")
        for name in ("dataset_sessions", "dataset_coverage"):
            for suffix in ("csv", "parquet"):
                shutil.copyfile(lock_root / f"{name}.{suffix}", output / f"{name}.{suffix}")
        write_json(output / "protocol.json", effective_protocol)
        write_json(output / "pretest_audit.json", read_json(lock_root / "audit.json"))
        predictions, features, audits = {}, {}, []
        stages = {}
        # All Q forecasts are durable before any held-out Q result is used as a race input.
        for task in ("qualifying", "race"):
            stage = output / f"{task}_prediction_seal"
            new_directory(stage)
            task_features, task_predictions = [], []
            for event_id in manifest["final_test"]:
                row = cutoffs.loc[cutoffs.event_id.eq(event_id) & cutoffs.task.eq(task)].iloc[0]
                roster = entrants.loc[entrants.event_id.eq(event_id) & entrants.task.eq(task)]
                frame, audit, allowed = holdout_features(
                    source, manifest, event_id, task, row.cutoff, roster, row["mode"]
                )
                audits.append(audit)
                task_features.append(frame)
                beginning = min(s["scheduled_start"] for s in source.event(event_id)["sessions"])
                tasks = ("qualifying",) if task == "qualifying" else ("race", "podium")
                for prediction_task in tasks:
                    predicted = (
                        predict_qualifying(models[prediction_task], frame, allowed, beginning)
                        if task == "qualifying"
                        else predict_race(models[prediction_task], frame, allowed, beginning)
                    )
                    task_predictions.append(
                        predicted.assign(
                            task=prediction_task,
                            method="model",
                            model_id=model_ids[prediction_task],
                        )
                    )
                    for rule in protocol["baselines"][prediction_task]:
                        reference = (
                            qualifying_rule(frame, rule)
                            if task == "qualifying"
                            else race_rule(frame, prediction_task, rule)
                        )
                        task_predictions.append(
                            reference.assign(
                                task=prediction_task,
                                method=f"baseline_{rule}",
                                model_id=f"fixed_rule:{rule}",
                            )
                        )
            features[task] = pd.concat(task_features, ignore_index=True)
            predictions[task] = pd.concat(task_predictions, ignore_index=True)
            table(stage, "features", features[task])
            table(stage, "predictions", predictions[task])
            stages[task] = seal_directory(
                stage,
                {
                    "lock_id": locked["id"],
                    "kind": "forecasts_before_scoring",
                    "sealed_at": pd.Timestamp.now(tz="UTC").isoformat(),
                },
            )["id"]
        write_json(output / "feature_audit.json", audits)
        write_json(output / "prediction_seals.json", stages)
        # This is the first scoring-label/target-weather access in the evaluation.
        scores, coverage, tables, conditions = [], [], {}, []
        for task in ("qualifying", "race", "podium"):
            core_task = "race" if task == "podium" else task
            rows = []
            for event_id in manifest["final_test"]:
                field = features[core_task].loc[features[core_task].event_id.eq(event_id)]
                truth = labels_after_prediction(source, event_id, task, field.driver_id)
                condition = session_condition(source, event_id, core_task)
                conditions.append(condition)
                coverage.append(
                    {
                        "task": task,
                        "event_id": event_id,
                        "expected_entrants": len(field),
                        "source_rows": int(truth.roster_match.ne("left_only").sum()),
                        "field_mismatches": int(truth.roster_match.ne("both").sum()),
                        "missing_positions": int(truth.rank_target.isna().sum()),
                        "missing_podium_targets": int(truth.podium.isna().sum()),
                        "outcomes": truth.outcome.fillna("unknown").value_counts().to_dict(),
                        "condition": condition["condition"],
                    }
                )
                candidates = predictions[core_task].loc[
                    predictions[core_task].event_id.eq(event_id)
                    & predictions[core_task].task.eq(task)
                ]
                for method, predicted in candidates.groupby("method", sort=False):
                    result = score(predicted, truth, field, task, protocol)
                    scores.append({**result, "task": task, "method": method})
                    joined = predicted.merge(
                        truth,
                        on=["event_id", "driver_id"],
                        how="outer",
                        validate="one_to_one",
                        indicator=True,
                    )
                    context = field[["event_id", "driver_id", "team_id"]].rename(
                        columns={"team_id": "team_at_cutoff"}
                    )
                    joined = joined.merge(
                        context, on=["event_id", "driver_id"], how="left", validate="one_to_one"
                    )
                    joined["condition"] = condition["condition"]
                    joined["data_kind"] = protocol["data_kind"]
                    joined["event_name"] = source.event(event_id)["name"]
                    joined["primary_eligible"] = result["primary_eligible"]
                    if task != "podium":
                        joined["absolute_position_error"] = (
                            joined.predicted_position - joined.official_position
                        ).abs()
                    rows.append(joined)
            tables[task] = pd.concat(rows, ignore_index=True)
            table(output, f"{task}_predicted_vs_actual", tables[task])
        metrics = pd.DataFrame(scores)
        summaries = aggregate(metrics, protocol)
        coverage_frame = pd.DataFrame(coverage)
        table(output, "event_metrics", metrics)
        table(output, "aggregate_metrics", summaries)
        table(output, "paired_baseline_differences", paired_differences(metrics, protocol))
        table(output, "coverage", coverage_frame)
        table(
            output,
            "conditions_after_prediction",
            pd.DataFrame(conditions).drop_duplicates(subset=["task", "event_id"]),
        )
        groups = subgroup_errors(tables, protocol)
        table(output, "subgroup_errors", groups)
        table(
            output,
            "development_calibration_bins",
            calibration_figure(
                development_calibration(lock_root, protocol),
                output / "calibration-development",
                f"{protocol['data_kind']} — outer development folds",
                protocol["calibration_bins"],
            ),
        )
        table(
            output,
            "final_calibration_bins",
            calibration_figure(
                tables["podium"].loc[tables["podium"].primary_eligible],
                output / "calibration-final",
                f"{protocol['data_kind']} — frozen holdout",
                protocol["calibration_bins"],
            ),
        )
        all_predictions = pd.concat(predictions.values(), ignore_index=True)
        model_race = all_predictions.loc[
            all_predictions.task.eq("race") & all_predictions.method.eq("model")
        ]
        model_podium = all_predictions.loc[
            all_predictions.task.eq("podium") & all_predictions.method.eq("model")
        ]
        race_cols = ["event_id", "driver_id", "cutoff", "mode", "raw_score", "predicted_position"]
        podium_cols = [
            "event_id",
            "driver_id",
            "cutoff",
            "mode",
            "raw_podium_probability",
            "podium_probability",
            "podium_threshold",
            "predicted_podium",
        ]
        combined, top_threes = combine_predictions(model_race[race_cols], model_podium[podium_cols])
        combined["data_kind"] = protocol["data_kind"]
        table(output, "race_and_podium_forecasts", combined)
        write_json(output / "top_three_comparison.json", json_value(top_threes))
        report(output, effective_protocol, metrics, summaries, coverage_frame, groups)
        receipt = seal_directory(
            output,
            {
                "lock_id": locked["id"],
                "manifest_id": manifest["id"],
                "data_kind": protocol["data_kind"],
                "exploratory": exploratory,
                "primary_evaluation": not exploratory,
                "retrained_in_test_block": False,
                "prediction_seals": stages,
                "fresh_future_holdout_required": effective_protocol[
                    "fresh_future_holdout_required"
                ],
            },
        )
        write_json(reservation / "result.json", {"receipt_id": receipt["id"], "status": "complete"})
        return receipt
    except Exception as exc:
        write_json(
            reservation / "failure.json",
            {
                "status": "claimed_attempt_failed",
                "type": type(exc).__name__,
                "message": str(exc),
                "policy": "No automatic primary retry; any reuse is exploratory",
            },
        )
        raise
