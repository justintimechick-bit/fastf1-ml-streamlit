"""Lock before access, seal forecasts before scoring, and never fit in the held-out block."""

import copy
import shutil
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from f1ml.config import Config, ExperimentConfig, QualifyingConfig, RaceConfig
from f1ml.evaluation.qualifying import run_comparison as compare_qualifying
from f1ml.evaluation.race import run_comparison as compare_race
from f1ml.evaluation.splits import load_manifest
from f1ml.experiments.build import build_supplements
from f1ml.experiments.evaluate import run_experiments
from f1ml.features.build import build_event
from f1ml.features.source import load_source
from f1ml.finalization.evaluate import evaluate_once
from f1ml.finalization.locking import (
    audit_development,
    audit_temporal_folds,
    lock_models,
    synthetic_inputs,
)
from f1ml.finalization.reporting import session_condition, uncertainty
from f1ml.finalization.source import HoldoutSource, holdout_features
from f1ml.models.qualifying import validate_permutation
from f1ml.storage import read_json, verify_directory


@pytest.fixture(scope="module")
def completed(workflow, tmp_path_factory):
    root = tmp_path_factory.mktemp("finalization")
    source = load_source(workflow / "dataset")
    manifest = load_manifest(workflow / "manifest.json", source)
    qconfig = QualifyingConfig(
        ridge_alphas=(1.0,),
        forest_depths=(3,),
        forest_trees=5,
        boosting_rates=(0.1,),
        boosting_iterations=5,
        recency_half_lives=(6.0,),
        season_2026_weights=(2.0,),
    )
    rconfig = RaceConfig(
        ridge_alphas=(1.0,),
        logistic_cs=(1.0,),
        forest_depths=(3,),
        forest_trees=5,
        boosting_rates=(0.1,),
        boosting_iterations=5,
        recency_half_lives=(6.0,),
        season_2026_weights=(2.0,),
    )
    config = replace(
        Config(),
        qualifying=qconfig,
        race=rconfig,
        experiments=ExperimentConfig(families=("linear",)),
    )
    compare_qualifying(manifest, workflow / "features", root / "qualifying", qconfig)
    compare_race(manifest, workflow / "features", root / "race", rconfig)
    build_supplements(
        source,
        manifest,
        workflow / "features",
        root / "capture",
        root / "optional",
        config.experiments,
    )
    run_experiments(
        manifest, workflow / "features", root / "optional", root / "experiments", config
    )
    synthetic_inputs(source, manifest, root / "inputs")
    return root


def make_lock(root, workflow, output, ledger, exploratory=False):
    source = load_source(workflow / "dataset")
    manifest = load_manifest(workflow / "manifest.json", source)
    lock_models(
        source,
        manifest,
        workflow / "features",
        root / "qualifying",
        root / "race",
        root / "experiments",
        root / "inputs/entrants.csv",
        root / "inputs/cutoffs.csv",
        output,
        ledger,
        exploratory,
    )
    return source, manifest


def test_lock_and_fixture_inputs_read_no_held_out_results(
    completed, workflow, tmp_path, monkeypatch
):
    source = load_source(workflow / "dataset")
    manifest = load_manifest(workflow / "manifest.json")

    def forbidden(*args):
        raise AssertionError("No held-out component reads before model lock")

    monkeypatch.setattr(source, "component", forbidden)
    synthetic_inputs(source, manifest, tmp_path / "inputs")
    receipt = lock_models(
        source,
        manifest,
        workflow / "features",
        completed / "qualifying",
        completed / "race",
        completed / "experiments",
        tmp_path / "inputs/entrants.csv",
        tmp_path / "inputs/cutoffs.csv",
        tmp_path / "lock",
        tmp_path / "ledger",
    )
    assert receipt["locked_before_test_outcomes"]
    protocol = read_json(tmp_path / "lock/protocol.json")
    assert protocol["data_kind"] == "synthetic_validation"
    for choice in protocol["models"].values():
        assert set(choice["metadata"]["train_events"]) <= set(manifest["development"])
        assert not set(choice["metadata"]["dependency_events"]) & set(manifest["final_test"])
    assert not (tmp_path / "ledger" / manifest["id"]).exists()


def test_primary_evaluation_seals_before_scoring_and_never_calls_fit(
    completed, workflow, tmp_path, monkeypatch
):
    from sklearn.pipeline import Pipeline

    ledger, lock, output = tmp_path / "ledger", tmp_path / "lock", tmp_path / "result"
    source, manifest = make_lock(completed, workflow, lock, ledger)
    original = source.component

    def guard(spec, name):
        if spec.get("table", "").split("/")[0] in manifest["final_test"]:
            if spec["kind"] == "Qualifying" and name == "results":
                assert (output / "qualifying_prediction_seal/receipt.json").exists()
            if spec["kind"] == "Race":
                assert (output / "race_prediction_seal/receipt.json").exists()
            assert (ledger / manifest["id"] / "claim.json").exists()
        return original(spec, name)

    monkeypatch.setattr(source, "component", guard)

    def no_fit(*args, **kwargs):
        raise AssertionError("No fitting inside the final block")

    monkeypatch.setattr(Pipeline, "fit", no_fit)
    receipt = evaluate_once(lock, source, output, ledger)
    assert receipt["primary_evaluation"] and not receipt["retrained_in_test_block"]
    assert verify_directory(output)["id"] == receipt["id"]
    metrics = pd.read_parquet(output / "event_metrics.parquet")
    assert len(metrics) == 4 * 3 * 3
    assert set(metrics.event_id) == set(manifest["final_test"])
    assert metrics.primary_eligible.all()
    for task in ("qualifying", "race"):
        frame = pd.read_parquet(output / f"{task}_prediction_seal/predictions.parquet")
        for (_, _, _), group in frame.loc[frame.task.ne("podium")].groupby(
            ["task", "method", "event_id"]
        ):
            validate_permutation(group, group)
    assert (output / "calibration-final.png").stat().st_size > 1000
    assert (output / "calibration-development.svg").is_file()
    coverage = pd.read_parquet(output / "coverage.parquet")
    assert coverage.condition.eq("unknown").all()
    source_coverage = pd.read_parquet(output / "dataset_coverage.parquet")
    assert source_coverage.events.sum() == 12
    exported = pd.read_csv(output / "race_predicted_vs_actual.csv")
    assert exported.data_kind.eq("synthetic_validation").all()
    assert exported.event_name.str.startswith("SYNTHETIC").all()
    aggregates = pd.read_parquet(output / "aggregate_metrics.parquet")
    assert aggregates.ci_low.isna().all() and aggregates.events.eq(4).all()
    with pytest.raises(ValueError, match="already claimed"):
        evaluate_once(lock, source, tmp_path / "another", ledger)
    with pytest.raises(ValueError, match="exploratory"):
        make_lock(completed, workflow, tmp_path / "new-lock", ledger)


def test_race_outcome_changes_do_not_change_forecasts_and_reuse_is_exploratory(
    completed, workflow, tmp_path, monkeypatch
):
    ledger, lock = tmp_path / "ledger", tmp_path / "lock"
    source, manifest = make_lock(completed, workflow, lock, ledger)
    evaluate_once(lock, source, tmp_path / "first", ledger)
    original = source.component

    def changed(spec, name):
        frame = original(spec, name)
        if (
            name == "results"
            and spec["kind"] == "Race"
            and spec.get("table", "").split("/")[0] in manifest["final_test"]
        ):
            frame = frame.copy()
            frame["official_position"] = len(frame) + 1 - frame.official_position
            frame["podium"] = frame.official_position.le(3).astype(float)
        return frame

    monkeypatch.setattr(source, "component", changed)
    receipt = evaluate_once(lock, source, tmp_path / "exploration", ledger, exploratory=True)
    assert receipt["exploratory"] and receipt["fresh_future_holdout_required"]
    for task in ("qualifying", "race"):
        before = pd.read_parquet(tmp_path / f"first/{task}_prediction_seal/predictions.parquet")
        after = pd.read_parquet(
            tmp_path / f"exploration/{task}_prediction_seal/predictions.parquet"
        )
        pd.testing.assert_frame_equal(before, after)


def test_holdout_feature_capability_blocks_all_other_block_outcomes(completed, workflow):
    source = load_source(workflow / "dataset")
    manifest = load_manifest(workflow / "manifest.json")
    event_id = manifest["final_test"][1]
    view = HoldoutSource(source, manifest, event_id, "qualifying")
    event = source.event(event_id)
    for spec in event["sessions"]:
        if spec["kind"] in ("Qualifying", "Race"):
            with pytest.raises(ValueError, match="Forbidden"):
                view.component(spec, "results")
    with pytest.raises(ValueError, match="Other holdout"):
        view.event(manifest["final_test"][0])
    entrants = pd.read_csv(completed / "inputs/entrants.csv", dtype=str)
    cutoffs = pd.read_csv(completed / "inputs/cutoffs.csv", dtype=str)
    for task in ("qualifying", "race"):
        row = cutoffs.loc[cutoffs.event_id.eq(event_id) & cutoffs.task.eq(task)].iloc[0]
        roster = entrants.loc[entrants.event_id.eq(event_id) & entrants.task.eq(task)]
        frame, audit, _ = holdout_features(
            source, manifest, event_id, task, row.cutoff, roster, row["mode"]
        )
        assert set(audit["history_weekends"]) == set(manifest["development"])
        assert not any(
            r["event_id"] in manifest["final_test"] and r["kind"] == "Race" for r in audit["reads"]
        )
        with pytest.raises(ValueError, match="sealed"):
            build_event(source, manifest, event_id, task, row.cutoff, roster, inference=True)
        assert set(frame.event_id) == {event_id}


def test_audit_rejects_recorded_test_outcome_dependencies(completed, workflow, tmp_path):
    source = load_source(workflow / "dataset")
    manifest = load_manifest(workflow / "manifest.json")
    target = tmp_path / "qualifying"
    shutil.copytree(completed / "qualifying", target)
    path = target / "selected_model/model.json"
    text = path.read_text().replace('"train_events": [', '"train_events": ["2026-04",')
    path.write_text(text)
    with pytest.raises(ValueError):
        audit_development(
            source,
            manifest,
            workflow / "features",
            target,
            completed / "race",
            completed / "experiments",
        )


def test_lock_and_inputs_are_immutable_and_bad_source_never_consumes_holdout(
    completed, workflow, tmp_path
):
    ledger, lock = tmp_path / "ledger", tmp_path / "lock"
    source, manifest = make_lock(completed, workflow, lock, ledger)
    changed = copy.copy(source)
    changed.receipt = {"id": "wrong_source"}
    with pytest.raises(ValueError, match="source/manifest"):
        evaluate_once(lock, changed, tmp_path / "result", ledger)
    assert not (ledger / manifest["id"]).exists()
    (lock / "protocol.json").write_text("{}")
    with pytest.raises(ValueError, match="Snapshot contents"):
        evaluate_once(lock, source, tmp_path / "result", ledger)


def test_failed_claim_cannot_be_retried_as_primary(completed, workflow, tmp_path, monkeypatch):
    source, manifest = make_lock(completed, workflow, tmp_path / "lock", tmp_path / "ledger")

    def unavailable(*args, **kwargs):
        raise ValueError("explicit fixture source failure")

    monkeypatch.setattr(source, "component", unavailable)
    with pytest.raises(ValueError, match="explicit fixture"):
        evaluate_once(tmp_path / "lock", source, tmp_path / "failed", tmp_path / "ledger")
    assert (tmp_path / "ledger" / manifest["id"] / "failure.json").exists()
    with pytest.raises(ValueError, match="already claimed"):
        evaluate_once(tmp_path / "lock", source, tmp_path / "retry", tmp_path / "ledger")


def test_event_uncertainty_is_not_a_driver_row_confidence_interval():
    protocol = {"bootstrap_min_events": 8, "bootstrap_replicates": 1000, "random_seed": 42}
    small = uncertainty([1, 2, 3, 4], protocol)
    assert small["events"] == 4 and small["mean"] == 2.5
    assert small["ci_low"] is None and small["ci_high"] is None
    assert small["loo_mean_min"] == 2 and small["loo_mean_max"] == 3
    larger = uncertainty(np.arange(10), protocol)
    assert larger["interval_status"].startswith("descriptive_event_bootstrap")
    assert larger == uncertainty(np.arange(10), protocol)


def test_missing_weather_is_unknown_not_dry(workflow):
    source = load_source(workflow / "dataset")
    assert session_condition(source, "2026-04", "race")["condition"] == "unknown"


def test_temporal_audit_rejects_later_calibration_weekends(workflow):
    source = load_source(workflow / "dataset")
    manifest = load_manifest(workflow / "manifest.json")
    with pytest.raises(ValueError, match="later or overlapping"):
        audit_temporal_folds(
            {"postprocess_folds": [{"train": ["2026-03"], "validate": "2026-01"}]}, source, manifest
        )
