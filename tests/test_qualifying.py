"""Qualifying ordering, chronological selection, preprocessing and artifact safeguards."""

import copy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from f1ml.cli import main
from f1ml.config import QualifyingConfig, load_config
from f1ml.evaluation.qualifying import (
    _check_inputs,
    complete_target_reason,
    run_comparison,
    score_event,
    tune,
)
from f1ml.evaluation.splits import load_manifest
from f1ml.features.contract import model_matrix, numeric_columns
from f1ml.models.qualifying import (
    FAMILIES,
    STRATEGIES,
    align_labels,
    fit_qualifying,
    load_model,
    order_scores,
    rule_predictions,
    sample_weights,
    training_events,
    validate_permutation,
)
from f1ml.prediction.qualifying import predict_checked
from f1ml.storage import read_json, verify_directory


@pytest.fixture(scope="module")
def comparison(workflow, tmp_path_factory):
    # Smaller fixed search exercises nested 2026-only tuning even in the tiny demo.
    config = QualifyingConfig(
        min_2026_training_events=1,
        ridge_alphas=(1.0, 10.0),
        forest_depths=(3,),
        forest_trees=5,
        boosting_rates=(0.1,),
        boosting_iterations=8,
        recency_half_lives=(6.0,),
        season_2026_weights=(2.0,),
    )
    manifest = load_manifest(workflow / "manifest.json")
    output = tmp_path_factory.mktemp("qualifying") / "comparison"
    run_comparison(manifest, workflow / "features", output, config)
    return output, config


@pytest.fixture
def frames(workflow):
    return (
        pd.read_parquet(workflow / "features/qualifying_features.parquet"),
        pd.read_parquet(workflow / "features/qualifying_labels.parquet"),
    )


def test_comparison_common_cohort_and_event_averages(comparison, manifest):
    root, _ = comparison
    receipt = verify_directory(root)
    summary = read_json(root / "summary.json")
    metrics = pd.read_parquet(root / "event_metrics.parquet")
    assert receipt["status"] == "development_only_final_test_sealed"
    assert summary["common_evaluation_events"] == ["2026-02", "2026-03"]
    common = metrics.loc[metrics.common_comparison]
    assert len(common) == 2 * (len(FAMILIES) * len(STRATEGIES) + 2)
    assert not set(metrics.event_id) & set(manifest["final_test"])
    for row in summary["event_averages"]:
        group = common.loc[common.family.eq(row["family"]) & common.strategy.eq(row["strategy"])]
        assert set(group.event_id) == set(summary["common_evaluation_events"])
        assert row["position_mae"] == pytest.approx(group.position_mae.mean())
        assert row["events"] == 2
    assert not summary["final_test_accessed"]


def test_nested_tuning_uses_only_earlier_outer_training_events(comparison, manifest):
    root, _ = comparison
    searches = read_json(root / "tuning.json")
    for search in searches:
        outer = next(f for f in manifest["folds"] if f["validate"] == search["event_id"])
        assert set(search["dependency_events"]) <= set(outer["train"])
        for inner in search["inner_folds"]:
            assert inner["validate"] in outer["train"]
            assert set(inner["train"]) <= set(outer["train"])
            assert inner["validate"] not in inner["train"]
    assert any(
        s["strategy"] == "2026_only" and s["status"] == "chronologically_tuned" for s in searches
    )
    assert any(s["status"] == "fixed_default_no_inner_folds" for s in searches)


def test_default_2026_minimum_and_explicit_ineligibility(frames, manifest):
    features, labels = frames
    for index, fold in enumerate(manifest["folds"]):
        ids, reason = training_events(
            features, labels, fold["train"], "2026_only", QualifyingConfig()
        )
        assert all(e.startswith("2026-") for e in ids)
        assert bool(reason) == (index < 2)


def test_outer_and_future_labels_do_not_select_inner_parameters(frames, manifest, comparison):
    features, labels = frames
    _, config = comparison
    outer = manifest["folds"][-1]
    before = tune(features, labels, manifest, outer["train"], "ridge", "weighted", config, 42)
    changed = labels.copy()
    rows = ~changed.event_id.isin(outer["train"])
    changed.loc[rows, "rank_target"] = 1 - changed.loc[rows, "rank_target"]
    changed.loc[rows, "official_position"] = 999
    after = tune(features, changed, manifest, outer["train"], "ridge", "weighted", config, 42)
    assert before == after


@pytest.mark.parametrize("family", FAMILIES)
def test_every_model_predicts_full_variable_field_with_unseen_categories(
    family, frames, manifest, comparison
):
    root, _ = comparison
    features, _ = frames
    artifact = load_model(root / "fold_models/2026-03" / family / "equal")
    validation = features.loc[features.event_id.eq("2026-03")].copy()
    validation.loc[validation.index[0], ["driver_id", "team_history_id"]] = [
        "new_substitute",
        "brand_new_team",
    ]
    validation["practice_gap_pct"] = np.nan
    result = predict_checked(artifact, validation, manifest)
    validate_permutation(result, validation)
    assert np.isfinite(result.raw_score).all()
    assert "new_substitute" in set(result.driver_id)
    assert "probability" not in " ".join(result.columns)
    again = predict_checked(artifact, validation.sample(frac=1, random_state=13), manifest)
    pd.testing.assert_frame_equal(result, again)


def test_raw_scores_and_deterministic_ties(frames):
    features, _ = frames
    field = features.loc[features.event_id.eq("2026-03")].sample(frac=1, random_state=3)
    result = order_scores(field, np.full(len(field), -0.125))
    assert result.driver_id.tolist() == sorted(field.driver_id)
    assert result.raw_score.eq(-0.125).all()  # Scores remain unclipped and separate from positions.
    assert result.predicted_position.tolist() == list(range(1, len(field) + 1))
    with pytest.raises(ValueError, match="finite"):
        order_scores(field, np.full(len(field), np.nan))


def test_permutation_rejects_missing_duplicate_or_repeated_positions(frames):
    features, _ = frames
    field = features.loc[features.event_id.eq("2026-03")]
    result = order_scores(field, np.arange(len(field)))
    for invalid in (
        result.iloc[:-1],
        pd.concat([result, result.iloc[:1]]),
        result.assign(predicted_position=1),
    ):
        with pytest.raises(ValueError):
            validate_permutation(invalid, field)
    with pytest.raises(ValueError, match="complete actual"):
        order_scores(field.iloc[:-1], np.zeros(len(field) - 1))


def test_metrics_have_known_values_and_small_field_denominators(frames):
    features, labels = frames
    field = features.loc[features.event_id.eq("2026-03")].reset_index(drop=True)
    truth = labels.loc[labels.event_id.eq("2026-03")]
    aligned = align_labels(field, truth)
    perfect = order_scores(field, aligned.official_position)
    score = score_event(perfect, truth, field)
    for name in ("spearman", "pole_accuracy", "top3_overlap", "top10_overlap"):
        assert score[name] == 1
    assert score["position_mae"] == 0
    assert score["top10_denominator"] == len(field)
    reverse = order_scores(field, -aligned.official_position)
    reversed_score = score_event(reverse, truth, field)
    assert reversed_score["spearman"] == -1
    assert reversed_score["pole_accuracy"] == 0
    assert reversed_score["position_mae"] == pytest.approx(
        np.abs(np.arange(1, len(field) + 1) - np.arange(len(field), 0, -1)).mean()
    )
    missing = truth.copy()
    missing.loc[missing.index[0], "rank_target"] = np.nan
    assert complete_target_reason(field, missing) == "missing_official_qualifying_targets"
    with pytest.raises(ValueError, match="Cannot score full"):
        score_event(perfect, missing, field)


def test_fixed_practice_rule_falls_back_without_dropping_missing_drivers(frames):
    features, _ = frames
    field = features.loc[features.event_id.eq("2026-03")].reset_index(drop=True)
    field["practice_gap_pct"] = np.nan
    recent = rule_predictions(field, "recent")
    pd.testing.assert_frame_equal(recent, rule_predictions(field, "practice"))
    field.loc[:2, "practice_gap_pct"] = [1.0, 0.2, 0.6]
    result = rule_predictions(field, "practice").set_index("driver_id")
    expected = [1.0, 0.0, 0.5]
    for index in range(3):
        assert result.loc[field.driver_id.iloc[index], "raw_score"] == expected[index]
    assert len(result) == len(field)
    assert result.score_source.eq("clean_practice_observed_field_rank").sum() == 3
    blank = field.copy()
    blank[["qualifying_rank_mean_3", "qualifying_rank_mean_5", "previous_qualifying_rank"]] = np.nan
    assert rule_predictions(blank, "recent").raw_score.eq(0.5).all()


def test_event_weights_do_not_cancel_season_weights_or_favor_large_fields(frames, manifest):
    features, labels = frames
    aligned = align_labels(features, labels)
    equal, _ = sample_weights(aligned, features, manifest, "equal", {})
    masses = pd.Series(equal).groupby(aligned.event_id).sum()
    assert np.allclose(masses, masses.iloc[0])
    weighted, profile = sample_weights(
        aligned, features, manifest, "weighted", {"half_life_events": 6, "season_2026_weight": 4}
    )
    masses = pd.Series(weighted).groupby(aligned.event_id).sum()
    assert masses["2026-03"] > masses["2026-01"] > masses["2025-02"] > masses["2024-01"]
    assert masses["2026-03"] / masses["2024-01"] == pytest.approx(
        profile["2026-03"] / profile["2024-01"]
    )
    assert weighted.mean() == pytest.approx(1)


def test_preprocessing_is_fitted_on_training_rows_only(frames, manifest, comparison):
    root, _ = comparison
    features, _ = frames
    artifact = load_model(root / "fold_models/2026-03/ridge/equal")
    train = features.loc[features.event_id.isin(artifact.metadata["train_events"])]
    numeric = artifact.pipeline.named_steps["transform"].named_transformers_["numeric"]
    expected = model_matrix(train, "qualifying")[list(numeric_columns("qualifying", "3.0.0"))]
    medians = expected.median().fillna(0).to_numpy()
    assert np.allclose(numeric.named_steps["impute"].statistics_, medians)
    imputed = numeric.named_steps["impute"].transform(expected)
    assert np.allclose(numeric.named_steps["scale"].mean_, imputed.mean(axis=0))
    validation = features.loc[features.event_id.eq("2026-03")].copy()
    validation["round"] = 1e6
    predict_checked(artifact, validation, manifest)
    assert np.allclose(numeric.named_steps["impute"].statistics_, medians)


def test_overlap_future_dependencies_and_final_test_are_rejected(frames, manifest, comparison):
    root, _ = comparison
    features, labels = frames
    artifact = load_model(root / "fold_models/2026-03/ridge/equal")
    validation = features.loc[features.event_id.eq("2026-03")].copy()
    with pytest.raises(ValueError, match="training, tuning or model selection"):
        predict_checked(
            artifact, validation.assign(event_id=artifact.metadata["train_events"][0]), manifest
        )
    with pytest.raises(ValueError, match="Final test"):
        predict_checked(artifact, validation.assign(event_id=manifest["final_test"][0]), manifest)
    changed = copy.deepcopy(artifact)
    changed.metadata["max_weekend_end"] = validation.cutoff.iloc[0]
    with pytest.raises(ValueError, match="not strictly earlier"):
        predict_checked(changed, validation, manifest)
    bad_manifest = copy.deepcopy(manifest)
    bad_manifest["folds"][0]["train"].append(bad_manifest["folds"][0]["validate"])
    with pytest.raises(ValueError, match="must not overlap"):
        _check_inputs(features, labels, bad_manifest)


def test_model_bundle_and_cli_future_prediction(
    comparison, workflow, dataset, manifest, event_inputs, tmp_path
):
    root, _ = comparison
    artifact = load_model(root / "selected_model")
    metadata = artifact.metadata
    assert metadata["selection_events"] == ["2026-02", "2026-03"]
    assert not set(metadata["dependency_events"]) & set(manifest["final_test"])
    assert metadata["feature_snapshot_id"] == verify_directory(workflow / "features")["id"]
    assert metadata["model_feature_columns"] == list(artifact.pipeline.feature_names_in_)
    event, cutoff, entrants = event_inputs("2026-08")
    roster = tmp_path / "entrants.csv"
    entrants.to_csv(roster, index=False)
    assert (
        main(
            [
                "predict-qualifying",
                "--model",
                str(root / "selected_model"),
                "--dataset",
                str(workflow / "dataset"),
                "--manifest",
                str(workflow / "manifest.json"),
                "--event",
                event["event_id"],
                "--cutoff",
                cutoff.isoformat(),
                "--entrants",
                str(roster),
                "--output",
                str(tmp_path / "prediction"),
            ]
        )
        == 0
    )
    receipt = verify_directory(tmp_path / "prediction")
    assert receipt["task"] == "qualifying"
    prediction = pd.read_parquet(tmp_path / "prediction/predictions.parquet")
    validate_permutation(prediction, entrants)
    # No target Q/R files exist for this discovered future event.
    assert not any("2026-08" in p for p in dataset.receipt["files"])


def test_snapshot_provenance_and_model_tampering_are_rejected(frames, comparison, manifest):
    root, _ = comparison
    features, _ = frames
    artifact = load_model(root / "fold_models/2026-03/ridge/equal")
    validation = features.loc[features.event_id.eq("2026-03")].assign(mode="publication_snapshot")
    with pytest.raises(ValueError, match="reconstructed model"):
        predict_checked(artifact, validation, manifest)
    artifact.metadata["modes"] = ["publication_snapshot"]
    with pytest.raises(ValueError, match="retrieved after"):
        predict_checked(artifact, validation, manifest)


def test_no_common_events_is_explicit_not_an_unequal_comparison(workflow, tmp_path, comparison):
    _, config = comparison
    output = tmp_path / "infeasible"
    run_comparison(
        load_manifest(workflow / "manifest.json"),
        workflow / "features",
        output,
        replace(config, min_2026_training_events=10),
    )
    summary = read_json(output / "summary.json")
    assert summary["status"] == "infeasible_no_common_events"
    assert summary["event_averages"] == []
    assert summary["selected_model"] is None
    assert not (output / "selected_model").exists()


def test_configuration_loads_search_and_rejects_invalid_settings():
    assert load_config("configs/default.toml").qualifying == QualifyingConfig()
    for values in (
        {"ridge_alphas": ()},
        {"season_2026_weights": (1,)},
        {"max_inner_folds": 0},
        {"forest_depths": (2.5,)},
        {"recency_half_lives": (np.nan,)},
    ):
        with pytest.raises(ValueError):
            QualifyingConfig(**values)


def test_bundle_integrity_is_verified_before_loading(comparison, tmp_path):
    import shutil

    root, _ = comparison
    target = tmp_path / "model"
    shutil.copytree(root / "selected_model", target)
    (target / "model.joblib").write_bytes(b"invalid untrusted content")
    with pytest.raises(ValueError, match="Snapshot contents changed"):
        load_model(target)


def test_missing_training_targets_are_masked_not_imputed(frames, manifest):
    features, labels = frames
    ids = manifest["folds"][0]["train"]
    train = features.loc[features.event_id.isin(ids)]
    targets = labels.loc[labels.event_id.isin(ids)].copy()
    targets.loc[targets.index[0], "rank_target"] = np.nan
    artifact = fit_qualifying(train, targets, manifest, "ridge", "equal", {"alpha": 1}, {}, 42)
    assert artifact.metadata["missing_targets"] == 1
    assert artifact.metadata["train_rows"] == len(train) - 1
