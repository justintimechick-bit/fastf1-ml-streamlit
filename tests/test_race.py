"""Chronological classification, calibrated marginal probabilities and no race/grid leakage."""

import copy
import shutil
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from f1ml.cli import main
from f1ml.config import RaceConfig, load_config
from f1ml.domain import utc
from f1ml.evaluation.race import fitted_cell, run_comparison
from f1ml.evaluation.race_metrics import (
    binary_scores,
    podium_score,
    race_score,
    reliability,
    target_coverage,
    target_reason,
)
from f1ml.evaluation.splits import gp_session, load_manifest
from f1ml.features.build import build_event
from f1ml.features.contract import model_matrix, numeric_columns
from f1ml.models.postprocess import fit_postprocessing
from f1ml.models.qualifying import STRATEGIES, validate_permutation
from f1ml.models.race import (
    PODIUM_FAMILIES,
    RACE_FAMILIES,
    fit_model,
    load_model,
    predict_model,
    rule_predictions,
)
from f1ml.prediction.race import combine_predictions, predict_checked
from f1ml.storage import read_json, save_table, seal_directory, verify_directory


@pytest.fixture(scope="module")
def comparison(workflow, tmp_path_factory):
    config = RaceConfig(
        min_2026_training_events=1,
        ridge_alphas=(1.0, 10.0),
        logistic_cs=(0.1, 1.0),
        forest_depths=(3,),
        forest_trees=5,
        boosting_rates=(0.1,),
        boosting_iterations=8,
        recency_half_lives=(6.0,),
        season_2026_weights=(2.0,),
        postprocess_folds=1,
        min_calibration_events=1,
        min_threshold_events=1,
    )
    output = tmp_path_factory.mktemp("race-models") / "comparison"
    run_comparison(load_manifest(workflow / "manifest.json"), workflow / "features", output, config)
    return output, config


@pytest.fixture
def frames(workflow):
    return (
        pd.read_parquet(workflow / "features/race_features.parquet"),
        pd.read_parquet(workflow / "features/race_labels.parquet"),
    )


def test_comparison_complete_events_common_cohorts_and_status_is_label_side(comparison, manifest):
    root, _ = comparison
    verify_directory(root)
    summary = read_json(root / "summary.json")
    metrics = pd.read_parquet(root / "event_metrics.parquet")
    predictions = pd.read_parquet(root / "predictions.parquet")
    labels = pd.read_parquet(root / "evaluation_labels.parquet")
    assert "source_status" in labels and "outcome" in labels
    assert not {"source_status", "outcome", "official_position", "is_retired"} & set(predictions)
    for task in ("race", "podium"):
        assert summary["common_evaluation_events"][task] == ["2026-02", "2026-03"]
        assert len(summary["event_averages"][task]) == 11
        for row in summary["event_averages"][task]:
            group = metrics.loc[
                metrics.task.eq(task)
                & metrics.family.eq(row["family"])
                & metrics.strategy.eq(row["strategy"])
                & metrics.common_comparison
            ]
            assert set(group.event_id) == {"2026-02", "2026-03"}
            key = "position_mae" if task == "race" else "log_loss"
            assert row[key] == pytest.approx(group[key].mean())
    assert not set(metrics.event_id) & set(manifest["final_test"])
    assert summary["final_test_accessed"] is False
    calibration = pd.read_parquet(root / "calibration.parquet")
    assert set(calibration.probability_source) == {"raw_podium_probability", "podium_probability"}
    assert calibration.weight.sum() == pytest.approx(11 * 2 * 2)
    assert (root / "report.md").is_file()


def test_postprocess_events_follow_parameter_tuning_and_precede_outer_event(comparison, manifest):
    root, _ = comparison
    dates = {e["event_id"]: utc(e["race_start"]) for e in manifest["eligible"]}
    searches = read_json(root / "tuning.json")
    for search in searches:
        outer = next(f for f in manifest["folds"] if f["validate"] == search["event_id"])
        assert set(search["dependency_events"]) <= set(outer["train"])
        tune = search["tuning_folds"]
        post = search["postprocess_folds"]
        for fold in tune + post:
            assert fold["validate"] not in fold["train"]
            assert dates[fold["validate"]] < dates[search["event_id"]]
            assert all(dates[event] < dates[fold["validate"]] for event in fold["train"])
        if tune and post:
            assert max(dates[f["validate"]] for f in tune) < min(dates[f["validate"]] for f in post)
    model = load_model(root / "fold_models/podium/2026-03/logistic/equal")
    assert model.metadata["tuning"]["status"] == "chronologically_tuned"
    assert model.metadata["calibration"]["events"] == ["2026-02"]
    assert model.metadata["threshold"]["events"] == ["2026-02"]
    oof = pd.read_parquet(
        root / "fold_models/podium/2026-03/logistic/equal/postprocess_training.parquet"
    )
    for row in oof.itertuples():
        assert row.event_id not in row.base_training_events
        assert utc(row.base_max_weekend_end) < utc(row.cutoff)


@pytest.mark.parametrize(
    "task,family", [("race", f) for f in RACE_FAMILIES] + [("podium", f) for f in PODIUM_FAMILIES]
)
def test_models_support_unknown_categories_missing_practice_and_complete_fields(
    task, family, comparison, frames, manifest
):
    root, _ = comparison
    features, _ = frames
    model = load_model(root / "fold_models" / task / "2026-03" / family / "equal", task)
    field = features.loc[features.event_id.eq("2026-03")].copy()
    field.loc[field.index[0], ["driver_id", "team_history_id"]] = ["unseen_substitute", "new_team"]
    field["practice_gap_pct"] = np.nan
    predicted = predict_checked(model, field, manifest)
    assert set(predicted.driver_id) == set(field.driver_id)
    assert not predicted.driver_id.duplicated().any()
    if task == "race":
        validate_permutation(predicted, field)
        assert np.isfinite(predicted.raw_score).all()
    else:
        assert predicted.podium_probability.between(0, 1).all()
        assert predicted.raw_podium_probability.between(0, 1).all()
    pd.testing.assert_frame_equal(
        predicted, predict_checked(model, field.sample(frac=1, random_state=13), manifest)
    )


def test_changing_outer_outcomes_cannot_change_parameters_calibration_or_threshold(
    frames, manifest, comparison
):
    features, labels = frames
    _, config = comparison
    fold = manifest["folds"][-1]
    ids = fold["train"]
    first, _ = fitted_cell(
        features,
        labels,
        manifest,
        ids,
        ids,
        "podium",
        "logistic",
        "weighted",
        config,
        42,
        "synthetic_snapshot",
    )
    changed = labels.copy()
    use = ~changed.event_id.isin(ids)
    changed.loc[use, "podium"] = 1 - changed.loc[use, "podium"]
    changed.loc[use, "official_position"] = 999
    changed.loc[use, "source_status"] = "Disqualified"
    second, _ = fitted_cell(
        features,
        changed,
        manifest,
        ids,
        ids,
        "podium",
        "logistic",
        "weighted",
        config,
        42,
        "synthetic_snapshot",
    )
    assert first.metadata == second.metadata
    assert first.threshold == second.threshold
    np.testing.assert_array_equal(first.calibrator.coef_, second.calibrator.coef_)
    field = features.loc[features.event_id.eq(fold["validate"])]
    pd.testing.assert_frame_equal(predict_model(first, field), predict_model(second, field))


def test_target_race_observations_and_retrospective_grid_never_enter_inputs(
    dataset, manifest, event_inputs, monkeypatch, comparison
):
    event, cutoff, entrants = event_inputs("2026-03", "race")
    original = dataset.table
    read = []

    def guarded(spec):
        if spec["table"].startswith(event["event_id"] + "/"):
            assert spec["kind"] != "Race"
        read.append(spec["kind"])
        return original(spec)

    monkeypatch.setattr(dataset, "table", guarded)
    features, audit = build_event(dataset, manifest, event["event_id"], "race", cutoff, entrants)
    assert "Qualifying" in read
    assert features.qualifying_position.notna().all()
    assert features.known_grid_position.isna().all()
    assert not any(
        s["kind"] == "Race" and s["event_id"] == event["event_id"] for s in audit["sources"]
    )
    root, _ = comparison
    for task, family in (("race", "ridge"), ("podium", "logistic")):
        artifact = load_model(root / "fold_models" / task / "2026-03" / family / "equal")
        predict_checked(artifact, features, manifest)
        for forbidden in ("source_grid", "official_position", "source_status", "race_weather"):
            with pytest.raises(ValueError, match="schema mismatch"):
                predict_model(artifact, features.assign(**{forbidden: 1}))
    early = utc(gp_session(event, "Qualifying")["ended_at"])
    with pytest.raises(ValueError, match="available official GP qualifying"):
        build_event(dataset, manifest, event["event_id"], "race", early, entrants)


def test_grid_requires_evidence_and_is_distinct_from_qualifying(dataset, manifest, event_inputs):
    event, cutoff, entrants = event_inputs("2026-03", "race")
    entrants["grid_position"] = list(range(len(entrants), 0, -1))
    with pytest.raises(ValueError, match="Grid position was not known"):
        build_event(dataset, manifest, event["event_id"], "race", cutoff, entrants)
    entrants["grid_known_at"] = (cutoff - pd.Timedelta(seconds=1)).isoformat()
    entrants["grid_source"] = "explicit_synthetic_penalty_notice"
    features, _ = build_event(dataset, manifest, event["event_id"], "race", cutoff, entrants)
    assert features.known_grid_position.tolist() == entrants.grid_position.tolist()
    assert not features.known_grid_position.equals(features.qualifying_position)
    entrants["grid_known_at"] = (cutoff + pd.Timedelta(seconds=1)).isoformat()
    with pytest.raises(ValueError, match="Grid position was not known"):
        build_event(dataset, manifest, event["event_id"], "race", cutoff, entrants)


def test_status_policy_retains_ordered_retirement_dsq_dns_and_reports_missing(frames, manifest):
    features, labels = frames
    event = "2026-03"
    field = features.loc[features.event_id.eq(event)].reset_index(drop=True)
    truth = labels.loc[labels.event_id.eq(event)].reset_index(drop=True)
    tail = truth.official_position.idxmax()
    truth.loc[tail, ["outcome", "source_status", "is_retired"]] = [
        "retired_classified",
        "Engine",
        True,
    ]
    assert target_reason(field, truth, "race") is None
    for status in ("dsq", "dns", "non_classified"):
        truth.loc[tail, "outcome"] = status
        assert target_reason(field, truth, "race") is None
        coverage = target_coverage(field, truth)
        assert coverage.race_training_eligible.all()
        assert coverage.loc[tail, "outcome"] == status
    truth.loc[tail, ["official_position", "rank_target"]] = np.nan
    assert target_reason(field, truth, "race") == "missing_official_race_targets"
    assert target_reason(field, truth, "podium") is None  # Explicit special outcome is negative.
    coverage = target_coverage(field, truth)
    assert coverage.race_training_eligible.sum() == len(field) - 1
    assert coverage.podium_training_eligible.all()
    truth.loc[tail, "podium"] = np.nan
    assert target_reason(field, truth, "podium") == "missing_podium_targets"


def test_known_metric_values_and_reliability_bins(frames):
    features, labels = frames
    field = features.loc[features.event_id.eq("2026-03")].reset_index(drop=True)
    truth = labels.loc[labels.event_id.eq("2026-03")].reset_index(drop=True)
    # This synthetic source has identical qualifying and race order.
    ranked = rule_predictions(field, "race", "qualifying")
    score = race_score(ranked, truth, field)
    assert score["position_mae"] == 0 and score["spearman"] == 1
    assert score["winner_accuracy"] == 1 and score["top3_overlap"] == 1
    podium = rule_predictions(field, "podium", "qualifying")
    score = podium_score(podium, truth, field, 5)
    assert score["brier"] == 0 and score["average_precision"] == 1 and score["ece"] == 0
    assert score["precision"] == score["recall"] == 1
    binary = binary_scores([0.25, 0.75], [0, 1], 2)
    assert binary["brier"] == 0.0625
    assert binary["log_loss"] == pytest.approx(-np.log(0.75))
    assert binary["ece"] == 0.25
    ece, bins = reliability([0, 1], [0, 1], 5)
    assert ece == 0
    assert bins[0]["driver_rows"] == bins[-1]["driver_rows"] == 1


def test_independent_probabilities_not_rescaled_and_top_threes_can_disagree(frames, manifest):
    features, labels = frames
    train_ids = manifest["folds"][0]["train"]
    train = features.loc[features.event_id.isin(train_ids)]
    targets = labels.loc[labels.event_id.isin(train_ids)].copy()
    targets.loc[targets.podium.eq(0), "podium"] = np.nan
    model = fit_model(train, targets, manifest, "podium", "logistic", "equal", {"C": 1}, {}, 42)
    assert model.metadata["single_class_fallback"]
    field = features.loc[features.event_id.eq("2026-03")]
    podium = predict_model(model, field)
    assert podium.podium_probability.eq(1).all()
    assert podium.podium_probability.sum() == len(field)  # No arbitrary sum-to-three projection.
    race = rule_predictions(field, "race", "qualifying").drop(columns="score_source")
    combined, differences = combine_predictions(race, podium)
    assert combined.race_order_top_three.sum() == 3
    assert combined.classifier_top_three.sum() == 3
    assert combined.predicted_podium.sum() == len(field)
    assert differences[0]["disagree"]
    assert differences[0]["independent_probability_sum"] == len(field)


def test_missing_postprocessing_stays_identity_with_fixed_threshold(comparison):
    root, config = comparison
    model = load_model(root / "fold_models/podium/2026-01/logistic/equal")
    assert model.calibrator is None and model.threshold == 0.5
    assert model.metadata["calibration"]["status"] == "insufficient_earlier_oof_events"
    assert model.metadata["threshold"]["status"] == "fixed_default_insufficient_oof_events"
    # A one-class chronological postprocess block cannot fit sigmoid calibration.
    oof = pd.DataFrame(
        {
            "event_id": ["earlier"] * 4,
            "driver_id": list("abcd"),
            "podium": 0,
            "raw_podium_probability": [0.1, 0.2, 0.3, 0.4],
        }
    )
    fit_postprocessing(model, oof, config, 42)
    assert model.calibrator is None
    assert model.metadata["calibration"]["status"] == "identity_single_class_oof"


def test_no_threshold_or_calibration_uses_final_test_outcomes(comparison, manifest):
    root, _ = comparison
    for path in root.rglob("model.json"):
        meta = read_json(path)
        assert not set(meta["dependency_events"]) & set(manifest["final_test"])
        if meta["task"] == "podium":
            for section in ("calibration", "threshold"):
                assert set(meta[section]["events"]) <= set(manifest["development"])
    selected = load_model(root / "selected_models/podium")
    assert selected.metadata["selection_events"] == ["2026-02", "2026-03"]


def test_preprocessing_uses_only_training_rows(frames, comparison):
    features, _ = frames
    root, _ = comparison
    model = load_model(root / "fold_models/podium/2026-03/logistic/equal")
    train = features.loc[features.event_id.isin(model.metadata["train_events"])]
    numeric = model.pipeline.named_steps["transform"].named_transformers_["numeric"]
    values = model_matrix(train, "podium")[list(numeric_columns("podium", "3.0.0"))]
    assert np.allclose(numeric.named_steps["impute"].statistics_, values.median().fillna(0))
    assert np.allclose(
        numeric.named_steps["scale"].mean_,
        numeric.named_steps["impute"].transform(values).mean(axis=0),
    )


def test_combined_prediction_cli_preserves_cutoff_features_and_model_lineage(
    comparison, workflow, manifest, event_inputs, tmp_path
):
    root, _ = comparison
    event, cutoff, entrants = event_inputs("2026-03", "race")
    roster = tmp_path / "entrants.csv"
    entrants.to_csv(roster, index=False)
    output = tmp_path / "prediction"
    race_path = root / "fold_models/race/2026-03/ridge/equal"
    podium_path = root / "fold_models/podium/2026-03/logistic/equal"
    assert (
        main(
            [
                "predict-race",
                "--race-model",
                str(race_path),
                "--podium-model",
                str(podium_path),
                "--dataset",
                str(workflow / "dataset"),
                "--manifest",
                str(workflow / "manifest.json"),
                "--event",
                event["event_id"],
                "--entrants",
                str(roster),
                "--cutoff",
                cutoff.isoformat(),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    receipt = verify_directory(output)
    assert receipt["model_ids"] == {
        "race": verify_directory(race_path)["id"],
        "podium": verify_directory(podium_path)["id"],
    }
    predictions = pd.read_parquet(output / "predictions.parquet")
    features = pd.read_parquet(output / "features.parquet")
    validate_permutation(predictions, entrants)
    assert features.task.eq("race").all()
    assert features.known_grid_position.isna().all()
    assert receipt["classification_meaning"] == "predicted official classification, not pure pace"
    assert len(receipt["top_three_comparison"]) == 1


def test_cutoff_dependency_test_seal_and_publication_guards(frames, comparison, manifest):
    features, _ = frames
    root, _ = comparison
    model = load_model(root / "fold_models/podium/2026-03/logistic/equal")
    field = features.loc[features.event_id.eq("2026-03")]
    with pytest.raises(ValueError, match="Final test block"):
        predict_checked(model, field.assign(event_id=manifest["final_test"][0]), manifest)
    with pytest.raises(ValueError, match="training, tuning, calibration or selection"):
        predict_checked(
            model, field.assign(event_id=model.metadata["dependency_events"][0]), manifest
        )
    selected = load_model(root / "selected_models/podium")
    with pytest.raises(ValueError, match="training, tuning, calibration or selection"):
        predict_checked(selected, field, manifest)
    invalid = copy.deepcopy(model)
    invalid.metadata["max_label_end"] = field.cutoff.iloc[0]
    with pytest.raises(ValueError, match="not strictly earlier"):
        predict_checked(invalid, field, manifest)
    with pytest.raises(ValueError, match="reconstructed model"):
        predict_checked(model, field.assign(mode="publication_snapshot"), manifest)
    invalid = copy.deepcopy(model)
    invalid.metadata["modes"] = ["publication_snapshot"]
    with pytest.raises(ValueError, match="retrieved after"):
        predict_checked(invalid, field.assign(mode="publication_snapshot"), manifest)


def test_incomplete_targets_are_reported_and_not_in_primary_averages(
    workflow, frames, comparison, tmp_path
):
    _, config = comparison
    features, labels = frames
    event = "2026-03"
    index = labels.loc[labels.event_id.eq(event)].official_position.idxmax()
    labels.loc[index, ["official_position", "rank_target"]] = np.nan
    labels.loc[index, ["outcome", "source_status", "podium"]] = ["dns", "Did not start", 0.0]
    source = tmp_path / "features"
    source.mkdir()
    save_table(source / "race_features.parquet", features)
    save_table(source / "race_labels.parquet", labels)
    manifest = load_manifest(workflow / "manifest.json")
    seal_directory(source, {"manifest_id": manifest["id"]})
    output = tmp_path / "comparison"
    run_comparison(manifest, source, output, config)
    summary = read_json(output / "summary.json")
    assert event not in summary["common_evaluation_events"]["race"]
    assert event in summary["common_evaluation_events"]["podium"]
    coverage = pd.read_parquet(output / "target_coverage.parquet")
    row = coverage.loc[
        coverage.event_id.eq(event) & coverage.driver_id.eq(labels.loc[index, "driver_id"])
    ]
    assert row.race_exclusion_reason.iloc[0] == "missing_or_ambiguous_official_position"
    assert row.podium_training_eligible.iloc[0]


def test_ineligible_common_strategies_save_reports_without_selected_models(
    workflow, tmp_path, comparison
):
    _, config = comparison
    output = tmp_path / "infeasible"
    receipt = run_comparison(
        load_manifest(workflow / "manifest.json"),
        workflow / "features",
        output,
        replace(config, min_2026_training_events=10),
    )
    assert receipt["status"] == "infeasible_missing_common_events"
    assert receipt["selected_models"] == {"race": None, "podium": None}
    assert read_json(output / "summary.json")["event_averages"] == {"race": [], "podium": []}


def test_model_bundle_integrity_and_task_checks(comparison, tmp_path):
    root, _ = comparison
    with pytest.raises(ValueError, match="matching race/podium"):
        load_model(root / "selected_models/race", "podium")
    target = tmp_path / "bundle"
    shutil.copytree(root / "selected_models/race", target)
    (target / "model.joblib").write_bytes(b"invalid changed content")
    with pytest.raises(ValueError, match="Snapshot contents changed"):
        load_model(target)


def test_race_config_validation_and_default_loading():
    assert load_config("configs/default.toml").race == RaceConfig()
    for values in (
        {"logistic_cs": ()},
        {"thresholds": (0, 0.5)},
        {"thresholds": (1,)},
        {"postprocess_folds": 1},
        {"calibration_bins": 0},
    ):
        with pytest.raises(ValueError):
            RaceConfig(**values)


def test_recent_baseline_fallback_does_not_drop_rookies(frames):
    features, _ = frames
    field = features.loc[features.event_id.eq("2026-03")].copy()
    field[["race_rank_mean_3", "race_rank_mean_5", "previous_race_rank"]] = np.nan
    race = rule_predictions(field, "race", "recent")
    qualifying = rule_predictions(field, "race", "qualifying")
    pd.testing.assert_frame_equal(race, qualifying)
    field[["podium_rate_5", "podium_rate_3"]] = np.nan
    pd.testing.assert_frame_equal(
        rule_predictions(field, "podium", "recent"), rule_predictions(field, "podium", "qualifying")
    )


def test_all_training_strategies_keep_2026_training_rows_separate(comparison, manifest):
    root, _ = comparison
    for task, family in (("race", "ridge"), ("podium", "logistic")):
        for strategy in STRATEGIES:
            model = load_model(root / "fold_models" / task / "2026-03" / family / strategy)
            ids = model.metadata["train_events"]
            if strategy == "2026_only":
                assert all(e.startswith("2026-") for e in ids)
            else:
                assert any(e.startswith("2024-") for e in ids)
                assert any(e.startswith("2025-") for e in ids)
            if strategy == "weighted":
                weights = model.metadata["training_event_weights"]
                assert weights["2026-02"] > weights["2024-01"]
