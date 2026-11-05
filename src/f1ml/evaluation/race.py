"""Nested event-wise race and podium development; final-test rows are never loaded."""

import logging
from dataclasses import asdict, replace
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from f1ml.config import RaceConfig
from f1ml.evaluation.qualifying import _check_inputs, candidates
from f1ml.evaluation.race_metrics import (
    PODIUM_METRICS,
    RACE_METRICS,
    common_reliability,
    podium_score,
    race_score,
    target_coverage,
    target_reason,
)
from f1ml.evaluation.splits import assert_disjoint_events
from f1ml.models.postprocess import fit_postprocessing
from f1ml.models.qualifying import STRATEGIES, add_dependencies
from f1ml.models.race import (
    PODIUM_FAMILIES,
    RACE_FAMILIES,
    eligible_training,
    fit_model,
    rule_predictions,
    save_model,
)
from f1ml.prediction.race import combine_predictions, predict_checked
from f1ml.storage import (
    digest,
    new_directory,
    save_table,
    seal_directory,
    verify_directory,
    write_json,
)

log = logging.getLogger(__name__)


def subset(frame, ids):
    return frame.loc[frame.event_id.isin(ids)].reset_index(drop=True)


def options(task, family, strategy, config):
    if family != "logistic":
        return candidates(family, strategy, config)
    values = candidates("ridge", strategy, replace(config, ridge_alphas=config.logistic_cs))
    for value in values:
        value["parameters"] = {"C": value["parameters"]["alpha"]}
        value["id"] = digest(
            {"parameters": value["parameters"], "weight_profile": value["weight_profile"]}
        )
    return values


def fit(features, labels, manifest, ids, task, family, strategy, option, seed):
    return fit_model(
        subset(features, ids),
        subset(labels, ids),
        manifest,
        task,
        family,
        strategy,
        option["parameters"],
        option["weight_profile"],
        seed,
    )


def metric(prediction, truth, field, task, bins):
    return (
        race_score(prediction, truth, field)
        if task == "race"
        else podium_score(prediction, truth, field, bins)
    )


def temporal_search(features, labels, manifest, available, task, family, strategy, config, seed):
    folds = []
    for fold in manifest["folds"]:
        if fold["validate"] not in available or not set(fold["train"]) <= set(available):
            continue
        ids, reason = eligible_training(features, labels, fold["train"], strategy, config, task)
        field, truth = subset(features, [fold["validate"]]), subset(labels, [fold["validate"]])
        if reason or target_reason(field, truth, task):
            continue
        assert_disjoint_events(ids, [fold["validate"]], manifest["final_test"])
        folds.append({"train": ids, "validate": fold["validate"]})
    folds = folds[-config.max_inner_folds :]
    post_folds = folds[-config.postprocess_folds :] if task == "podium" else []
    tune_folds = folds[: -len(post_folds)] if post_folds else folds
    candidates_ = options(task, family, strategy, config)
    records = []
    key = "position_mae" if task == "race" else "raw_log_loss"
    for option in candidates_:
        event_scores = {}
        for fold in tune_folds:
            model = fit(
                features, labels, manifest, fold["train"], task, family, strategy, option, seed
            )
            field, truth = subset(features, [fold["validate"]]), subset(labels, [fold["validate"]])
            prediction = predict_checked(model, field, manifest)
            event_scores[fold["validate"]] = metric(
                prediction, truth, field, task, config.calibration_bins
            )[key]
        records.append(
            {
                **option,
                "event_scores": event_scores,
                "mean_loss": float(np.mean(list(event_scores.values()))) if event_scores else None,
            }
        )
    best = min(range(len(records)), key=lambda i: records[i]["mean_loss"]) if tune_folds else 0
    selected = candidates_[best]
    oof = []
    for fold in post_folds:
        model = fit(
            features, labels, manifest, fold["train"], task, family, strategy, selected, seed
        )
        field = subset(features, [fold["validate"]])
        predicted = predict_checked(model, field, manifest)
        predicted = predicted.merge(
            subset(labels, [fold["validate"]])[
                ["event_id", "driver_id", "podium", "label_session_end", "label_retrieved_at"]
            ],
            on=["event_id", "driver_id"],
            validate="one_to_one",
        )
        predicted["base_training_events"] = [fold["train"]] * len(predicted)
        predicted["base_max_weekend_end"] = model.metadata["max_weekend_end"]
        oof.append(predicted)
    dependency_ids = {e for fold in folds for e in [*fold["train"], fold["validate"]]}
    search = {
        "status": "chronologically_tuned" if tune_folds else "fixed_default_no_tuning_folds",
        "objective": key,
        "selected": selected,
        "candidates": records,
        "tuning_folds": tune_folds,
        "postprocess_folds": post_folds,
        "dependency_events": [e for e in manifest["development"] if e in dependency_ids],
        "tie_rule": "lowest equal-event inner loss, then predeclared candidate order",
        "postprocess_policy": "reserved later OOF events; none select model parameters or weights",
    }
    return selected, search, pd.concat(oof, ignore_index=True) if oof else pd.DataFrame()


def fitted_cell(
    features, labels, manifest, available, ids, task, family, strategy, config, seed, snapshot_id
):
    option, search, oof = temporal_search(
        features, labels, manifest, available, task, family, strategy, config, seed
    )
    model = fit(features, labels, manifest, ids, task, family, strategy, option, seed)
    dependencies = set(ids) | set(search["dependency_events"])
    add_dependencies(model.metadata, features, labels, dependencies, manifest)
    model.metadata.update(tuning=search, feature_snapshot_id=snapshot_id)
    if task == "podium":
        fit_postprocessing(model, oof, config, seed)
    return model, oof


def write_report(output, summary, metrics, eligibility):
    lines = [
        "# Race classification and podium development",
        "",
        "Final test remains sealed. Classification includes official ordered retirement/",
        "administrative outcomes; it is not a pure-pace forecast. Probabilities are independent",
        "and are never rescaled to sum to three.",
        "",
    ]
    for task, names in (("race", RACE_METRICS), ("podium", PODIUM_METRICS)):
        lines += [
            f"## {task.title()} common-event averages",
            "",
            "Events: " + (", ".join(summary["common_evaluation_events"][task]) or "none"),
            "",
            "| Model | Strategy | Events | " + " | ".join(names) + " |",
            "| --- | --- | ---: | " + " | ".join("---:" for _ in names) + " |",
        ]
        for row in summary["event_averages"][task]:
            values = " | ".join(
                f"{row[name]:.4f}" if row[name] is not None else "missing" for name in names
            )
            lines.append(f"| {row['family']} | {row['strategy']} | {row['events']} | {values} |")
        lines += [
            "",
            f"### {task.title()} per-event results",
            "",
            "| Event | Model | Strategy | Common | " + " | ".join(names) + " |",
            "| --- | --- | --- | --- | " + " | ".join("---:" for _ in names) + " |",
        ]
        for row in (
            metrics.loc[metrics.task.eq(task)].to_dict("records") if not metrics.empty else []
        ):
            values = " | ".join(
                f"{row[name]:.4f}" if pd.notna(row.get(name)) else "missing" for name in names
            )
            lines.append(
                f"| {row['event_id']} | {row['family']} | {row['strategy']} | "
                f"{row['common_comparison']} | {values} |"
            )
        lines.append("")
    lines += [
        "## Eligibility and exclusions",
        "",
        "| Task | Event | Strategy | Usable training events | Reason |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for row in eligibility:
        lines.append(
            f"| {row['task']} | {row['event_id']} | {row['strategy']} | "
            f"{row['training_event_count']} | {row['reason'] or 'eligible'} |"
        )
    lines += [
        "",
        "Actual status and target exclusions are separate in target_coverage.parquet and",
        "evaluation_labels.parquet; these fields never enter prediction inputs. Calibration",
        "bins (raw and delivered probabilities) are in calibration.parquet. Thresholds and",
        "calibration eligibility are recorded per model. Selected bundles are best learned",
        "models; fixed rules may outperform them. Development selection is not a final-test",
        "estimate. Small samples do not establish calibration or real F1 accuracy.",
        "",
    ]
    with (output / "report.md").open("x") as stream:
        stream.write("\n".join(lines))


def run_comparison(manifest, feature_root: Path, output: Path, config=None, seed=42):
    config = config or RaceConfig()
    receipt = verify_directory(feature_root)
    if receipt["manifest_id"] != manifest["id"]:
        raise ValueError("Race features belong to another frozen manifest")
    features = pd.read_parquet(feature_root / "race_features.parquet")
    labels = pd.read_parquet(feature_root / "race_labels.parquet")
    # The common checks are about weekend membership/chronology, not qualifying targets.
    _check_inputs(features, labels, manifest, task="race")
    new_directory(output)
    write_json(output / "search_config.json", {"race": asdict(config), "seed": seed})
    save_table(output / "evaluation_labels.parquet", labels)
    save_table(output / "target_coverage.parquet", target_coverage(features, labels))
    common, summaries, selected = {}, {}, {}
    scores, all_predictions, eligibility, searches = [], [], [], []
    for task, families, names in (
        ("race", RACE_FAMILIES, RACE_METRICS),
        ("podium", PODIUM_FAMILIES, PODIUM_METRICS),
    ):
        common[task] = []
        for fold in manifest["folds"]:
            event_id = fold["validate"]
            field, truth = subset(features, [event_id]), subset(labels, [event_id])
            reason_target = target_reason(field, truth, task)
            choices = {}
            for strategy in STRATEGIES:
                ids, reason_train = eligible_training(
                    features, labels, fold["train"], strategy, config, task
                )
                reason = reason_target or reason_train
                eligibility.append(
                    {
                        "task": task,
                        "event_id": event_id,
                        "strategy": strategy,
                        "eligible": reason is None,
                        "reason": reason,
                        "training_events": ids,
                        "training_event_count": len(ids),
                    }
                )
                if reason is None:
                    choices[strategy] = ids
            is_common = len(choices) == len(STRATEGIES)
            if is_common:
                common[task].append(event_id)
            log.info(
                "%s %s: strategies=%s common=%s target_exclusion=%s",
                task,
                event_id,
                list(choices),
                is_common,
                reason_target,
            )
            if reason_target:
                continue
            for rule in ("qualifying", "recent"):
                predicted = rule_predictions(field, task, rule)
                family = f"baseline_{rule}"
                scores.append(
                    {
                        **metric(predicted, truth, field, task, config.calibration_bins),
                        "task": task,
                        "family": family,
                        "strategy": "fixed_rule",
                        "common_comparison": is_common,
                    }
                )
                all_predictions.append(
                    predicted.assign(task=task, family=family, strategy="fixed_rule")
                )
            for family, strategy in product(families, STRATEGIES):
                if strategy not in choices:
                    continue
                model, oof = fitted_cell(
                    features,
                    labels,
                    manifest,
                    fold["train"],
                    choices[strategy],
                    task,
                    family,
                    strategy,
                    config,
                    seed,
                    receipt["id"],
                )
                model.metadata.update(role="outer_development_fold", validation_event=event_id)
                predicted = predict_checked(model, field, manifest)
                bundle = save_model(
                    output / "fold_models" / task / event_id / family / strategy, model, oof
                )
                scores.append(
                    {
                        **metric(predicted, truth, field, task, config.calibration_bins),
                        "task": task,
                        "family": family,
                        "strategy": strategy,
                        "common_comparison": is_common,
                        "tuning_status": model.metadata["tuning"]["status"],
                        "calibration_status": model.metadata.get("calibration", {}).get("status"),
                        "threshold_status": model.metadata.get("threshold", {}).get("status"),
                    }
                )
                all_predictions.append(
                    predicted.assign(
                        task=task, family=family, strategy=strategy, model_id=bundle["id"]
                    )
                )
                searches.append(
                    {
                        "task": task,
                        "event_id": event_id,
                        "family": family,
                        "strategy": strategy,
                        **model.metadata["tuning"],
                    }
                )
        metrics = pd.DataFrame(scores)
        means = []
        if common[task]:
            for family, strategy in [
                *product(families, STRATEGIES),
                ("baseline_qualifying", "fixed_rule"),
                ("baseline_recent", "fixed_rule"),
            ]:
                group = metrics.loc[
                    metrics.task.eq(task)
                    & metrics.common_comparison
                    & metrics.family.eq(family)
                    & metrics.strategy.eq(strategy)
                ]
                if set(group.event_id) != set(common[task]):
                    raise ValueError("Race/podium comparison methods must share the same events")
                means.append(
                    {
                        "family": family,
                        "strategy": strategy,
                        "events": len(group),
                        **{
                            name: float(group[name].mean()) if group[name].notna().any() else None
                            for name in names
                        },
                    }
                )
            loss = "position_mae" if task == "race" else "log_loss"
            winner = min([row for row in means if row["family"] in families], key=lambda r: r[loss])
            family, strategy = winner["family"], winner["strategy"]
            ids, reason = eligible_training(
                features, labels, manifest["development"], strategy, config, task
            )
            if reason:
                raise ValueError("Selected race/podium strategy ineligible for refit")
            model, oof = fitted_cell(
                features,
                labels,
                manifest,
                manifest["development"],
                ids,
                task,
                family,
                strategy,
                config,
                seed,
                receipt["id"],
            )
            dependencies = set(model.metadata["dependency_events"]) | set(common[task])
            dependencies.update(
                e
                for fold in manifest["folds"]
                if fold["validate"] in common[task]
                for e in fold["train"]
            )
            add_dependencies(model.metadata, features, labels, dependencies, manifest)
            model.metadata.update(
                role="selected_development_refit",
                selection_events=common[task],
                selection_metric=loss,
                selection_tie_rule="predeclared family order, then strategy order",
            )
            bundle = save_model(output / "selected_models" / task, model, oof)
            selected[task] = {
                "family": family,
                "strategy": strategy,
                "bundle": f"selected_models/{task}",
                "id": bundle["id"],
            }
        else:
            selected[task] = None
        summaries[task] = means
    metrics = pd.DataFrame(scores)
    predictions = (
        pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    )
    save_table(output / "event_metrics.parquet", metrics)
    save_table(output / "predictions.parquet", predictions)
    save_table(
        output / "calibration.parquet",
        common_reliability(predictions, labels, common["podium"], config.calibration_bins)
        if not predictions.empty
        else pd.DataFrame(),
    )
    comparisons = _top_three_comparisons(predictions)
    write_json(output / "top_three_comparison.json", comparisons)
    write_json(output / "eligibility.json", eligibility)
    write_json(output / "tuning.json", searches)
    summary = {
        "status": "development_only_final_test_sealed"
        if all(selected.values())
        else "infeasible_missing_common_events",
        "manifest_id": manifest["id"],
        "feature_snapshot_id": receipt["id"],
        "common_evaluation_events": common,
        "event_averages": summaries,
        "selected_models": selected,
        "final_test_accessed": False,
        "aggregation": "equal-event averages on each task's all-strategy common cohort",
        "probability_policy": "independent marginal probabilities, no sum-to-three normalization",
        "data_note": "Source lineage is in receipts; synthetic metrics establish no real accuracy.",
    }
    write_json(output / "summary.json", summary)
    write_report(output, summary, metrics, eligibility)
    return seal_directory(
        output,
        {
            "manifest_id": manifest["id"],
            "feature_snapshot_id": receipt["id"],
            "status": summary["status"],
            "selected_models": selected,
        },
    )


def _top_three_comparisons(predictions):
    comparisons = []
    if predictions.empty:
        return comparisons
    race = predictions.loc[predictions.task.eq("race")]
    podium = predictions.loc[predictions.task.eq("podium")]
    # Pair corresponding cells; Ridge pairs with logistic, and fixed references pair by name.
    for (event, family, strategy), group in race.groupby(["event_id", "family", "strategy"]):
        classifier = "logistic" if family == "ridge" else family
        other = podium.loc[
            podium.event_id.eq(event) & podium.family.eq(classifier) & podium.strategy.eq(strategy)
        ]
        if other.empty:
            continue
        race_columns = [
            "event_id",
            "driver_id",
            "cutoff",
            "mode",
            "raw_score",
            "predicted_position",
        ]
        podium_columns = [
            "event_id",
            "driver_id",
            "cutoff",
            "mode",
            "raw_podium_probability",
            "podium_probability",
            "podium_threshold",
            "predicted_podium",
        ]
        _, report = combine_predictions(group[race_columns], other[podium_columns])
        comparisons.extend(
            {"race_family": family, "podium_family": classifier, "strategy": strategy, **row}
            for row in report
        )
    return comparisons
