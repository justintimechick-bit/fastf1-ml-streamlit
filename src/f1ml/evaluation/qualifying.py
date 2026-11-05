"""Nested chronological qualifying comparison. No final-test data access path."""

import logging
from dataclasses import asdict
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from f1ml.config import QualifyingConfig
from f1ml.domain import utc
from f1ml.evaluation.splits import assert_development, assert_disjoint_events
from f1ml.features.contract import model_matrix
from f1ml.models.qualifying import (
    FAMILIES,
    STRATEGIES,
    TIE_RULE,
    add_dependencies,
    align_labels,
    fit_qualifying,
    rule_predictions,
    save_model,
    training_events,
    validate_field,
    validate_permutation,
)
from f1ml.prediction.qualifying import predict_checked
from f1ml.storage import (
    digest,
    new_directory,
    save_table,
    seal_directory,
    verify_directory,
    write_json,
)

log = logging.getLogger(__name__)
METRICS = ("position_mae", "spearman", "pole_accuracy", "top3_overlap", "top10_overlap")


def _write_report(path, summary, metrics, eligibility):
    lines = [
        "# Qualifying development comparison",
        "",
        f"Status: `{summary['status']}`. Final test remains sealed.",
        "",
        "Common evaluation events: " + (", ".join(summary["common_evaluation_events"]) or "none"),
        "",
        "## Event-level averages on the common cohort",
        "",
        "| Model | Strategy | Events | Position MAE | Spearman | Pole | Top 3 | Top 10 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["event_averages"]:
        values = " | ".join(f"{row[name]:.4f}" for name in METRICS)
        lines.append(f"| {row['family']} | {row['strategy']} | {row['events']} | {values} |")
    lines += [
        "",
        "MAE is in positions. Pole and overlaps are fractions. Top-k overlap uses min(k, N).",
        "The selected bundle is the best learned regressor; a fixed rule can outperform it.",
        "Development selection is not an unbiased final-test estimate.",
        "",
        "## Per-event results (including additional eligible development folds)",
        "",
        "| Event | Model | Strategy | Common cohort | Position MAE | "
        "Spearman | Pole | Top 3 | Top 10 |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in metrics.to_dict("records"):
        values = " | ".join(f"{row[name]:.4f}" for name in METRICS)
        lines.append(
            f"| {row['event_id']} | {row['family']} | {row['strategy']} | "
            f"{row['common_comparison']} | {values} |"
        )
    lines += [
        "",
        "## Eligibility",
        "",
        "| Event | Strategy | Training events | Reason |",
        "| --- | --- | ---: | --- |",
    ]
    for row in eligibility:
        lines.append(
            f"| {row['event_id']} | {row['strategy']} | {row['training_event_count']} | "
            f"{row['reason'] or 'eligible'} |"
        )
    lines += [
        "",
        "See tuning.json and each model.json for inner folds, fixed defaults, parameters,",
        "weights and information dependencies. Raw scores remain in predictions.parquet.",
        "Scores are ranking estimates, not probabilities or lap-time forecasts.",
        "",
    ]
    with path.open("x") as stream:
        stream.write("\n".join(lines))


def complete_target_reason(features, labels):
    """A full-order evaluation needs all N official positions, not a manufactured tail."""
    validate_field(features)
    aligned = align_labels(features, labels)
    n = len(features)
    if n < 2:
        return "fewer_than_two_drivers"
    if aligned.rank_target.isna().any() or aligned.official_position.isna().any():
        return "missing_official_qualifying_targets"
    positions = aligned.official_position.astype(float)
    if sorted(positions) != list(range(1, n + 1)):
        return "official_positions_not_complete_permutation"
    if not np.allclose(aligned.rank_target.astype(float), (positions - 1) / (n - 1)):
        return "normalized_target_does_not_match_official_field"
    return None


def score_event(predictions, labels, field):
    reason = complete_target_reason(field, labels)
    if reason:
        raise ValueError(f"Cannot score full qualifying order: {reason}")
    if field.event_id.nunique() != 1:
        raise ValueError("Score exactly one complete event at a time")
    validate_permutation(predictions, field)
    joined = predictions.merge(
        labels[["event_id", "driver_id", "official_position"]],
        on=["event_id", "driver_id"],
        validate="one_to_one",
    )
    n = len(joined)
    predicted = joined.predicted_position.to_numpy(dtype=float)
    actual = joined.official_position.to_numpy(dtype=float)
    result = {
        "event_id": str(field.event_id.iloc[0]),
        "drivers": n,
        "position_mae": float(np.abs(predicted - actual).mean()),
        "spearman": float(1 - 6 * ((predicted - actual) ** 2).sum() / (n * (n**2 - 1))),
        "pole_accuracy": float(
            joined.loc[joined.predicted_position.eq(1), "official_position"].iloc[0] == 1
        ),
    }
    for k in (3, 10):
        size = min(k, n)
        overlap = int(((predicted <= size) & (actual <= size)).sum())
        result[f"top{k}_overlap"] = overlap / size
        result[f"top{k}_overlap_count"] = overlap
        result[f"top{k}_denominator"] = size
    return result


def candidates(family, strategy, config):
    if family == "ridge":
        params = [{"alpha": value} for value in config.ridge_alphas]
    elif family == "random_forest":
        params = [
            {"n_estimators": config.forest_trees, "max_depth": depth, "min_samples_leaf": 2}
            for depth in config.forest_depths
        ]
    elif family == "gradient_boosting":
        params = [
            {
                "max_iter": config.boosting_iterations,
                "learning_rate": rate,
                "max_leaf_nodes": 15,
                "min_samples_leaf": 5,
                "l2_regularization": 1.0,
            }
            for rate in config.boosting_rates
        ]
    else:
        raise ValueError(f"Unknown qualifying family: {family}")
    profiles = [{}]
    if strategy == "weighted":
        profiles = [
            {"half_life_events": half_life, "season_2026_weight": season_weight}
            for half_life, season_weight in product(
                config.recency_half_lives, config.season_2026_weights
            )
        ]
    return [
        {"id": digest({"parameters": p, "weight_profile": w}), "parameters": p, "weight_profile": w}
        for p, w in product(params, profiles)
    ]


def _subset(frame, ids):
    return frame.loc[frame.event_id.isin(ids)].reset_index(drop=True)


def _fit(features, labels, manifest, ids, family, strategy, candidate, seed):
    return fit_qualifying(
        _subset(features, ids),
        _subset(labels, ids),
        manifest,
        family,
        strategy,
        candidate["parameters"],
        candidate["weight_profile"],
        seed,
    )


def tune(features, labels, manifest, available, family, strategy, config, seed):
    """Only folds entirely inside the outer training block may select a candidate."""
    folds = []
    for fold in manifest["folds"]:
        if fold["validate"] not in available or not set(fold["train"]) <= set(available):
            continue
        ids, reason = training_events(features, labels, fold["train"], strategy, config)
        validation = _subset(features, [fold["validate"]])
        if reason or complete_target_reason(validation, _subset(labels, [fold["validate"]])):
            continue
        assert_disjoint_events(ids, [fold["validate"]], manifest["final_test"])
        folds.append((ids, fold["validate"]))
    folds = folds[-config.max_inner_folds :]
    options = candidates(family, strategy, config)
    records = []
    dependencies = set()
    for option in options:
        scores = []
        for ids, validation_id in folds:
            artifact = _fit(features, labels, manifest, ids, family, strategy, option, seed)
            field = _subset(features, [validation_id])
            prediction = predict_checked(artifact, field, manifest)
            score = score_event(prediction, _subset(labels, [validation_id]), field)
            scores.append(score["position_mae"])
            dependencies.update([*ids, validation_id])
        records.append(
            {
                **option,
                "mean_position_mae": float(np.mean(scores)) if scores else None,
                "inner_event_mae": dict(zip([v for _, v in folds], scores, strict=True)),
            }
        )
    # Python min preserves the predeclared candidate order for exact metric ties.
    best_index = (
        min(range(len(records)), key=lambda i: records[i]["mean_position_mae"]) if folds else 0
    )
    result = {
        "status": "chronologically_tuned" if folds else "fixed_default_no_inner_folds",
        "selected": options[best_index],
        "candidates": records,
        "inner_folds": [{"train": ids, "validate": val} for ids, val in folds],
        "dependency_events": [e for e in manifest["development"] if e in dependencies],
        "tie_rule": "lowest mean inner-event position MAE; then predeclared candidate order",
    }
    return options[best_index], result


def _check_inputs(features, labels, manifest, task="qualifying"):
    assert_development(manifest, features.event_id)
    assert_development(manifest, labels.event_id)
    if set(features.event_id) != set(manifest["development"]):
        raise ValueError("Qualifying features must cover every frozen development event")
    model_matrix(features, task)
    validate_field(features)
    align_labels(features, labels)
    eligible = {e["event_id"]: e for e in manifest["eligible"]}
    for fold in manifest["folds"]:
        assert_development(manifest, [*fold["train"], fold["validate"]])
        assert_disjoint_events(fold["train"], [fold["validate"]], manifest["final_test"])
        cutoff = min(utc(t) for t in _subset(features, [fold["validate"]]).cutoff)
        if int(eligible[fold["validate"]]["year"]) != 2026:
            raise ValueError("Qualifying development validation must be on 2026 events")
        if any(utc(eligible[e]["race_end"]) >= cutoff for e in fold["train"]):
            raise ValueError("Training weekends must finish before validation cutoff")


def run_comparison(manifest, feature_root: Path, output: Path, config=None, seed=42):
    config = config or QualifyingConfig()
    receipt = verify_directory(feature_root)
    if receipt["manifest_id"] != manifest["id"]:
        raise ValueError("Features belong to another frozen manifest")
    features = pd.read_parquet(feature_root / "qualifying_features.parquet")
    labels = pd.read_parquet(feature_root / "qualifying_labels.parquet")
    _check_inputs(features, labels, manifest)
    new_directory(output)
    write_json(output / "search_config.json", {"qualifying": asdict(config), "seed": seed})
    eligibility, scores, predictions, tuning = [], [], [], []
    common = []
    for fold in manifest["folds"]:
        event_id = fold["validate"]
        field, truth = _subset(features, [event_id]), _subset(labels, [event_id])
        target_reason = complete_target_reason(field, truth)
        choices = {}
        for strategy in STRATEGIES:
            ids, reason = training_events(features, labels, fold["train"], strategy, config)
            reason = target_reason or reason
            eligibility.append(
                {
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
            common.append(event_id)
        log.info(
            "Qualifying %s: eligible strategies=%s, common=%s", event_id, list(choices), is_common
        )
        if target_reason:
            continue
        for rule in ("recent", "practice"):
            prediction = rule_predictions(field, rule)
            name = f"baseline_{rule}"
            scores.append(
                {
                    **score_event(prediction, truth, field),
                    "family": name,
                    "strategy": "fixed_rule",
                    "common_comparison": is_common,
                }
            )
            predictions.append(prediction.assign(family=name, strategy="fixed_rule"))
        for family, strategy in product(FAMILIES, STRATEGIES):
            if strategy not in choices:
                continue
            option, search = tune(
                features, labels, manifest, fold["train"], family, strategy, config, seed
            )
            artifact = _fit(
                features, labels, manifest, choices[strategy], family, strategy, option, seed
            )
            dependency_ids = set(choices[strategy]) | set(search["dependency_events"])
            add_dependencies(artifact.metadata, features, labels, dependency_ids, manifest)
            artifact.metadata.update(
                tuning=search,
                feature_snapshot_id=receipt["id"],
                role="outer_development_fold",
                validation_event=event_id,
            )
            prediction = predict_checked(artifact, field, manifest)
            save_model(output / "fold_models" / event_id / family / strategy, artifact)
            scores.append(
                {
                    **score_event(prediction, truth, field),
                    "family": family,
                    "strategy": strategy,
                    "common_comparison": is_common,
                    "tuning_status": search["status"],
                }
            )
            predictions.append(prediction.assign(family=family, strategy=strategy))
            tuning.append({"event_id": event_id, "family": family, "strategy": strategy, **search})
    metrics = pd.DataFrame(scores)
    save_table(output / "event_metrics.parquet", metrics)
    save_table(
        output / "predictions.parquet",
        pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame(),
    )
    write_json(output / "eligibility.json", eligibility)
    write_json(output / "tuning.json", tuning)
    means = []
    if common:
        for family, strategy in [
            *product(FAMILIES, STRATEGIES),
            ("baseline_recent", "fixed_rule"),
            ("baseline_practice", "fixed_rule"),
        ]:
            group = metrics.loc[
                metrics.common_comparison
                & metrics.family.eq(family)
                & metrics.strategy.eq(strategy)
            ]
            if set(group.event_id) != set(common):
                raise ValueError("Primary comparison methods do not share exactly the same events")
            means.append(
                {
                    "family": family,
                    "strategy": strategy,
                    "events": len(group),
                    **{name: float(group[name].mean()) for name in METRICS},
                }
            )
    selected = None
    if means:
        learned = [row for row in means if row["family"] in FAMILIES]
        winner = min(learned, key=lambda row: row["position_mae"])
        family, strategy = winner["family"], winner["strategy"]
        ids, reason = training_events(features, labels, manifest["development"], strategy, config)
        if reason:
            raise ValueError("Selected training strategy unexpectedly ineligible for refit")
        option, search = tune(
            features, labels, manifest, manifest["development"], family, strategy, config, seed
        )
        artifact = _fit(features, labels, manifest, ids, family, strategy, option, seed)
        # Winner selection also consumed common outer labels, including nonwinning models' scores.
        dependencies = set(ids) | set(search["dependency_events"]) | set(common)
        dependencies.update(
            e for fold in manifest["folds"] if fold["validate"] in common for e in fold["train"]
        )
        dependencies.update(
            e for row in tuning if row["event_id"] in common for e in row["dependency_events"]
        )
        add_dependencies(artifact.metadata, features, labels, dependencies, manifest)
        artifact.metadata.update(
            tuning=search,
            feature_snapshot_id=receipt["id"],
            role="selected_development_refit",
            selection_events=common,
            selection_metric="mean common-event position MAE",
            selection_tie_rule="predeclared family order, then strategy order",
        )
        selected = {"family": family, "strategy": strategy, "bundle": "selected_model"}
        save_model(output / "selected_model", artifact)
    summary = {
        "status": "development_only_final_test_sealed" if common else "infeasible_no_common_events",
        "common_evaluation_events": common,
        "event_averages": means,
        "selected_model": selected,
        "aggregation": "equal weight per complete event, common cohort only",
        "tie_rule": TIE_RULE,
        "final_test_accessed": False,
        "limitations": [
            "Development selection is not an unbiased final-test estimate.",
            "2026-only restricts training examples; features retain allowed older priors.",
            "Scores are ranking estimates, not probabilities or lap-time forecasts.",
        ],
    }
    write_json(output / "summary.json", summary)
    _write_report(output / "report.md", summary, metrics, eligibility)
    return seal_directory(
        output,
        {
            "manifest_id": manifest["id"],
            "feature_snapshot_id": receipt["id"],
            "status": summary["status"],
            "common_evaluation_events": common,
        },
    )
