"""Controlled paired ablations: matched fixed settings and complete development weekends."""

import logging
import time
from pathlib import Path

import pandas as pd

from f1ml.config import Config
from f1ml.evaluation.qualifying import _check_inputs, score_event
from f1ml.evaluation.race import metric, subset
from f1ml.evaluation.race_metrics import target_reason
from f1ml.experiments.contract import (
    GROUPS,
    feature_columns,
    implementation_id,
    matrix,
    merge_features,
    specification,
)
from f1ml.models.postprocess import fit_postprocessing
from f1ml.models.qualifying import add_dependencies, fit_qualifying
from f1ml.models.qualifying import save_model as save_qualifying
from f1ml.models.race import eligible_training, fit_model
from f1ml.models.race import save_model as save_race
from f1ml.prediction.qualifying import predict_checked as predict_qualifying
from f1ml.prediction.race import predict_checked as predict_race
from f1ml.storage import new_directory, save_table, seal_directory, verify_directory, write_json

log = logging.getLogger(__name__)


def settings(task, family, strategy, config):
    search = config.qualifying if task == "qualifying" else config.race
    actual = ("logistic" if task == "podium" else "ridge") if family == "linear" else family
    if actual == "ridge":
        parameters = {"alpha": 10.0 if 10.0 in search.ridge_alphas else search.ridge_alphas[0]}
    elif actual == "logistic":
        parameters = {"C": 1.0 if 1.0 in search.logistic_cs else search.logistic_cs[0]}
    elif actual == "random_forest":
        parameters = {
            "n_estimators": search.forest_trees,
            "max_depth": search.forest_depths[0],
            "min_samples_leaf": 2,
        }
    else:
        parameters = {
            "max_iter": search.boosting_iterations,
            "learning_rate": search.boosting_rates[0],
            "max_leaf_nodes": 15,
            "min_samples_leaf": 5,
            "l2_regularization": 1.0,
        }
    profile = (
        {
            "half_life_events": search.recency_half_lives[0],
            "season_2026_weight": search.season_2026_weights[0],
        }
        if strategy == "weighted"
        else {}
    )
    return actual, parameters, profile


def fit_cell(features, labels, manifest, ids, task, family, strategy, group, config):
    actual, parameters, profile = settings(task, family, strategy, config)
    args = (subset(features, ids), subset(labels, ids), manifest)
    spec = specification(group)
    if task == "qualifying":
        return fit_qualifying(
            *args, actual, strategy, parameters, profile, config.random_seed, experiment=spec
        )
    return fit_model(
        *args, task, actual, strategy, parameters, profile, config.random_seed, experiment=spec
    )


def prediction(model, field, manifest, task):
    return (
        predict_qualifying(model, field, manifest)
        if task == "qualifying"
        else predict_race(model, field, manifest)
    )


def postprocess(model, features, labels, manifest, outer, family, strategy, group, config):
    oof, folds = [], []
    for fold in manifest["folds"]:
        if fold["validate"] not in outer["train"] or not set(fold["train"]) <= set(outer["train"]):
            continue
        ids, reason = eligible_training(
            features, labels, fold["train"], strategy, config.race, "podium"
        )
        if reason or target_reason(
            subset(features, [fold["validate"]]), subset(labels, [fold["validate"]]), "podium"
        ):
            continue
        folds.append({"train": ids, "validate": fold["validate"]})
    folds = folds[-config.race.postprocess_folds :]
    for fold in folds:
        base = fit_cell(
            features, labels, manifest, fold["train"], "podium", family, strategy, group, config
        )
        predicted = prediction(base, subset(features, [fold["validate"]]), manifest, "podium")
        predicted = predicted.merge(
            subset(labels, [fold["validate"]])[["event_id", "driver_id", "podium"]],
            on=["event_id", "driver_id"],
            validate="one_to_one",
        )
        oof.append(predicted)
    frame = pd.concat(oof, ignore_index=True) if oof else pd.DataFrame()
    fit_postprocessing(model, frame, config.race, config.random_seed)
    dependencies = set(model.metadata["train_events"]) | {
        e for f in folds for e in [*f["train"], f["validate"]]
    }
    add_dependencies(model.metadata, features, labels, dependencies, manifest)
    model.metadata["postprocess_folds"] = folds
    return frame


def observed_fraction(frame, columns):
    signals = [
        c
        for c in columns
        if not any(
            s in c
            for s in (
                "_count",
                "_laps",
                "_cells",
                "_samples",
                "_sessions",
                "_comparisons",
                "_coverage",
                "_missing",
            )
        )
    ]
    return float(frame[signals].notna().any(axis=1).mean()) if signals else 0.0


def retention(metrics, data_kind, options):
    paired, decisions = [], []
    for (task, family, strategy), cell in metrics.groupby(["task", "family", "strategy"]):
        loss = "log_loss" if task == "podium" else "position_mae"
        core = cell.loc[cell.group.eq("core")].set_index("event_id")
        for index, group in enumerate(GROUPS):
            current = cell.loc[cell.group.eq(group)].set_index("event_id")
            previous_name = GROUPS[index - 1] if index else "core"
            previous = cell.loc[cell.group.eq(previous_name)].set_index("event_id")
            if set(current.index) != set(core.index) or set(current.index) != set(previous.index):
                raise ValueError("Paired feature groups must evaluate identical complete events")
            for event_id in core.index:
                paired.append(
                    {
                        "task": task,
                        "family": family,
                        "strategy": strategy,
                        "event_id": event_id,
                        "group": group,
                        "previous_group": previous_name,
                        "loss": loss,
                        "improvement_vs_previous": float(
                            previous.loc[event_id, loss] - current.loc[event_id, loss]
                        ),
                        "improvement_vs_core": float(
                            core.loc[event_id, loss] - current.loc[event_id, loss]
                        ),
                    }
                )
            incremental = float(previous[loss].mean() - current[loss].mean())
            vs_core = float(core[loss].mean() - current[loss].mean())
            runtime_ratio = float(
                current.runtime_seconds.mean() / max(core.runtime_seconds.mean(), 1e-9)
            )
            minimum = (
                options.min_log_loss_improvement
                if task == "podium"
                else options.min_position_improvement
            )
            reasons = []
            if data_kind != "observed_source_snapshot":
                reasons.append("not_verified_live_evidence")
            if len(current) < options.min_paired_events:
                reasons.append("insufficient_paired_events")
            if incremental < minimum or vs_core < minimum:
                reasons.append("insufficient_loss_improvement")
            if float(current.observed_fraction.mean()) < options.min_observed_fraction:
                reasons.append("insufficient_optional_observations")
            if runtime_ratio > options.max_runtime_ratio:
                reasons.append("runtime_complexity_limit")
            decisions.append(
                {
                    "task": task,
                    "family": family,
                    "strategy": strategy,
                    "group": group,
                    "paired_events": len(current),
                    "mean_improvement_vs_previous": incremental,
                    "mean_improvement_vs_core": vs_core,
                    "runtime_ratio_vs_core": runtime_ratio,
                    "observed_fraction": float(current.observed_fraction.mean()),
                    "decision": "candidate_for_protocol_review"
                    if not reasons
                    else "do_not_promote",
                    "reasons": reasons,
                }
            )
    return pd.DataFrame(paired), decisions


def run_experiments(manifest, core_root: Path, optional_root: Path, output: Path, config=None):
    config = config or Config()
    core_receipt, optional_receipt = verify_directory(core_root), verify_directory(optional_root)
    if (
        core_receipt["manifest_id"] != manifest["id"]
        or optional_receipt["manifest_id"] != manifest["id"]
        or optional_receipt["core_feature_snapshot_id"] != core_receipt["id"]
    ):
        raise ValueError("Experiment inputs belong to different frozen core snapshots")
    if optional_receipt["implementation_id"] != implementation_id():
        raise ValueError("Optional feature implementation changed; rebuild supplement")
    new_directory(output)
    write_json(output / "config.json", config.to_dict())
    results, predictions, eligibility, missing = [], [], [], []
    for task in ("qualifying", "race", "podium"):
        source_task = "race" if task == "podium" else task
        core = pd.read_parquet(core_root / f"{source_task}_features.parquet")
        labels = pd.read_parquet(core_root / f"{source_task}_labels.parquet")
        _check_inputs(core, labels, manifest, task)
        extra = pd.read_parquet(optional_root / f"{source_task}_optional.parquet")
        features = merge_features(core, extra)
        for strategy in config.experiments.strategies:
            for fold in manifest["folds"]:
                search = config.qualifying if task == "qualifying" else config.race
                target_task = "race" if task == "qualifying" else task
                ids, reason = eligible_training(
                    features, labels, fold["train"], strategy, search, target_task
                )
                field, truth = (
                    subset(features, [fold["validate"]]),
                    subset(labels, [fold["validate"]]),
                )
                reason = target_reason(field, truth, target_task) or reason
                eligibility.append(
                    {
                        "task": task,
                        "event_id": fold["validate"],
                        "strategy": strategy,
                        "reason": reason,
                        "train_events": ids,
                    }
                )
                if reason:
                    continue
                for family in config.experiments.families:
                    for group in ("core", *GROUPS):
                        log.info(
                            "Experiment %s %s %s %s %s",
                            task,
                            fold["validate"],
                            family,
                            strategy,
                            group,
                        )
                        start = time.perf_counter()
                        model = fit_cell(
                            features, labels, manifest, ids, task, family, strategy, group, config
                        )
                        oof = (
                            postprocess(
                                model,
                                features,
                                labels,
                                manifest,
                                fold,
                                family,
                                strategy,
                                group,
                                config,
                            )
                            if task == "podium"
                            else pd.DataFrame()
                        )
                        predicted = prediction(model, field, manifest, task)
                        elapsed = time.perf_counter() - start
                        score = (
                            score_event(predicted, truth, field)
                            if task == "qualifying"
                            else metric(predicted, truth, field, task, config.race.calibration_bins)
                        )
                        selected = feature_columns(task, group)
                        previous = (
                            feature_columns(task, GROUPS[GROUPS.index(group) - 1])
                            if group in GROUPS[1:]
                            else []
                        )
                        novel = [c for c in selected if c not in previous]
                        x, _, _ = matrix(field, task, specification(group))
                        results.append(
                            {
                                **score,
                                "task": task,
                                "family": family,
                                "strategy": strategy,
                                "group": group,
                                "runtime_seconds": elapsed,
                                "model_features": len(x.columns),
                                "driver_rows": len(field),
                                "observed_fraction": observed_fraction(field, novel),
                                "numeric_missing_fraction": float(
                                    x[selected].isna().to_numpy().mean()
                                ),
                            }
                        )
                        for column in selected:
                            missing.append(
                                {
                                    "task": task,
                                    "event_id": fold["validate"],
                                    "family": family,
                                    "strategy": strategy,
                                    "group": group,
                                    "feature": column,
                                    "missing_rows": int(field[column].isna().sum()),
                                    "rows": len(field),
                                }
                            )
                        model.metadata.update(
                            optional_snapshot_id=optional_receipt["id"],
                            feature_snapshot_id=core_receipt["id"],
                            parameters_policy="fixed matched settings; no group-specific tuning",
                            role="experimental_development_fold",
                        )
                        destination = (
                            output / "models" / task / fold["validate"] / family / strategy / group
                        )
                        bundle = (
                            save_qualifying(destination, model)
                            if task == "qualifying"
                            else save_race(destination, model, oof)
                        )
                        predictions.append(
                            predicted.assign(
                                task=task,
                                family=family,
                                strategy=strategy,
                                group=group,
                                model_id=bundle["id"],
                            )
                        )
    frame = pd.DataFrame(results)
    if frame.empty:
        raise ValueError("No eligible development event for the requested controlled experiment")
    paired, decisions = retention(frame, optional_receipt["data_kind"], config.experiments)
    save_table(output / "event_metrics.parquet", frame)
    save_table(output / "paired_differences.parquet", paired)
    save_table(output / "missingness.parquet", pd.DataFrame(missing))
    save_table(output / "predictions.parquet", pd.concat(predictions, ignore_index=True))
    write_json(output / "retention.json", decisions)
    write_json(output / "eligibility.json", eligibility)
    summary = (
        frame.groupby(["task", "family", "strategy", "group"], sort=False)
        .agg(
            events=("event_id", "nunique"),
            mean_runtime_seconds=("runtime_seconds", "mean"),
            mean_missing_fraction=("numeric_missing_fraction", "mean"),
            observed_fraction=("observed_fraction", "mean"),
            position_mae=("position_mae", "mean"),
            log_loss=("log_loss", "mean"),
        )
        .reset_index()
    )
    save_table(output / "summary.parquet", summary)
    report = [
        "# Controlled feature experiments",
        "",
        f"Data kind: {optional_receipt['data_kind']}.",
        "Final test remains sealed. Core models and artifacts are preserved.",
        "",
        "Every group uses identical complete events within each task/family/strategy.",
        "Fixed settings; calibration and thresholds use earlier development OOF only.",
        "Runtime includes fitting, transforms and postprocessing; downloads are separate.",
        "Optional coverage and capture receipts record aggregation and acquisition costs.",
        "",
        "| Task | Family | Strategy | Group | Events | Position MAE | Log loss | "
        "Seconds | Missing | Observed |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.to_dict("records"):
        position = f"{row['position_mae']:.4f}" if pd.notna(row["position_mae"]) else "—"
        loss = f"{row['log_loss']:.4f}" if pd.notna(row["log_loss"]) else "—"
        report.append(
            f"| {row['task']} | {row['family']} | {row['strategy']} | {row['group']} | "
            f"{row['events']} | {position} | {loss} | "
            f"{row['mean_runtime_seconds']:.3f} | {row['mean_missing_fraction']:.3f} | "
            f"{row['observed_fraction']:.3f} |"
        )
    report += [
        "",
        "retention.json requires paired improvement versus both predecessor and core,",
        "observed coverage, live evidence and runtime budget. No group is auto-promoted.",
        "Synthetic metrics are software checks only. No pre-cutoff forecasts were obtained.",
        "",
    ]
    with (output / "report.md").open("x") as stream:
        stream.write("\n".join(report))
    return seal_directory(
        output,
        {
            "manifest_id": manifest["id"],
            "core_feature_snapshot_id": core_receipt["id"],
            "optional_snapshot_id": optional_receipt["id"],
            "status": "development_only",
            "final_test_accessed": False,
            "promoted_groups": [],
        },
    )
