"""Phase 8 interface: integration checks against the saved Phase 3–7 artifacts in data/.

These tests use the real saved experiment (synthetic fixtures) and the real FastF1 catalog
snapshot. They skip when those local artifacts are absent (data/ is not versioned).
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from f1ml.models.qualifying import load_model as load_qualifying_model
from f1ml.prediction.qualifying import predict_checked
from f1ml.storage import file_digest
from f1ml_app import forecasts as fc
from f1ml_app.settings import load_paths

ROOT = Path(__file__).resolve().parents[1]
PATHS = load_paths(root=ROOT)
APP = ROOT / "src" / "f1ml_app" / "streamlit_app.py"
pytestmark = pytest.mark.skipif(
    not (PATHS.final / "receipt.json").exists() or not (PATHS.catalog / "CURRENT.json").exists(),
    reason="saved Phase 3–7 artifacts and catalog snapshot are not present",
)
LABEL_COLUMNS = {"official_position", "rank_target", "podium", "outcome", "source_status"}
VALIDATION = ["2026-01", "2026-02", "2026-03"]
TRAINING_ONLY = ["2024-01", "2024-02", "2025-01", "2025-02"]
FINAL = ["2026-04", "2026-05", "2026-06", "2026-07"]


@pytest.fixture(scope="module")
def exp():
    from f1ml_app.experiment import Experiment

    return Experiment(PATHS)


def snapshot(*roots):
    return {
        str(p): file_digest(p)
        for root in roots
        for p in sorted(Path(root).rglob("*"))
        if p.is_file()
    }


def test_roles_follow_the_frozen_manifest(exp):
    assert exp.data_kind == "synthetic_validation"
    assert [e for e in exp.events.event_id if exp.role(e) == "final_test"] == FINAL
    assert [e for e in exp.events.event_id if exp.role(e) == "development_validation"] == VALIDATION
    assert exp.role("2026-08") == "excluded"
    assert {exp.role(e) for e in TRAINING_ONLY} == {"development_training"}


def test_interface_package_is_outside_the_locked_code_fingerprint(exp):
    import f1ml.finalization.locking as locking
    import f1ml_app

    hashed_tree = Path(locking.__file__).resolve().parents[1]
    assert not Path(f1ml_app.__file__).resolve().is_relative_to(hashed_tree)
    assert exp.final["code_matches_lock"]


@pytest.mark.parametrize("event_id", VALIDATION)
def test_every_saved_outer_fold_model_replays_its_saved_predictions(exp, event_id):
    for task in ("qualifying", "race", "podium"):
        options = [o for o in fc.candidates(exp, task, event_id) if o["available"]]
        assert options, task
        for option in options:
            choice = (option["family"], option["strategy"])
            forecast = (
                fc.qualifying_forecast(exp, event_id, choice)
                if task == "qualifying"
                else fc.race_forecast(
                    exp,
                    event_id,
                    race_choice=choice if task == "race" else None,
                    podium_choice=choice if task == "podium" else None,
                )
            )
            assert forecast.available and forecast.source == "development_replay"
            card = forecast.models[task]
            assert event_id not in card.dependency_events
            assert pd.Timestamp(card.training_end) < pd.Timestamp(forecast.cutoff)
            assert not [c for c in forecast.checks if "differs" in c], forecast.checks
            assert sum("matches the saved development predictions" in c for c in forecast.checks)


def test_ineligible_strategy_is_reported_not_substituted(exp):
    forecast = fc.qualifying_forecast(exp, "2026-01", ("ridge", "2026_only"))
    assert not forecast.available
    assert "needs 2 usable earlier 2026 training weekends" in forecast.explanation
    assert forecast.predictions is None


@pytest.mark.parametrize("event_id", TRAINING_ONLY)
def test_training_weekends_have_no_earlier_trained_model(exp, event_id):
    assert exp.eligible_cards("qualifying", event_id) == []
    for forecast in (fc.qualifying_forecast(exp, event_id), fc.race_forecast(exp, event_id)):
        assert not forecast.available and forecast.predictions is None
        assert "later-trained model would leak" in forecast.explanation


def test_prediction_guard_rejects_a_later_trained_model(exp):
    locked = exp.locked_card("qualifying")
    assert locked.lineage_problems(
        exp.manifest, "2026-02", *exp.cutoff("2026-02", "qualifying")[:1], None
    )
    with pytest.raises(ValueError, match="used for training"):
        predict_checked(
            load_qualifying_model(locked.path),
            exp.saved_inputs("2026-02", "qualifying"),
            exp.manifest,
            exp.weekend_start("2026-02"),
        )


@pytest.mark.parametrize("event_id", FINAL)
def test_final_test_shows_the_sealed_forecast_unchanged(exp, event_id):
    seals = json.loads((PATHS.final / "prediction_seals.json").read_text())
    quali, race = fc.qualifying_forecast(exp, event_id), fc.race_forecast(exp, event_id)
    assert quali.source == race.source == "frozen_final_test"
    assert (quali.seal_id, race.seal_id) == (seals["qualifying"], seals["race"])
    sealed = pd.read_parquet(PATHS.final / "qualifying_prediction_seal/predictions.parquet")
    sealed = sealed.loc[sealed.event_id.eq(event_id) & sealed.method.eq("model")]
    merged = quali.predictions.merge(sealed, on="driver_id", suffixes=("", "_sealed"))
    assert len(merged) == len(sealed)
    assert merged.predicted_position.eq(merged.predicted_position_sealed).all()
    combined = pd.read_parquet(PATHS.final / "race_and_podium_forecasts.parquet")
    combined = combined.loc[combined.event_id.eq(event_id)].sort_values("driver_id")
    shown = race.predictions.sort_values("driver_id")
    assert shown.podium_probability.tolist() == combined.podium_probability.tolist()
    with pytest.raises(ValueError, match="sealed"):  # never recomputed through the pipeline
        predict_checked(
            load_qualifying_model(exp.locked_card("qualifying").path),
            quali.inputs,
            exp.manifest,
        )


def test_post_qualifying_forecast_is_unavailable_before_qualifying(exp):
    for event_id in VALIDATION + FINAL:
        forecast = fc.race_forecast(exp, event_id, "pre_qualifying")
        assert not forecast.available and forecast.predictions is None
        assert fc.qualifying_forecast(exp, event_id).available
    future = fc.race_forecast(exp, "2026-08", "post_race")
    assert not future.available and "no official qualifying result" in future.headline
    roster = fc.qualifying_forecast(exp, "2026-08")
    assert not roster.available and "inventing the field" in roster.explanation


def test_outcomes_stay_out_of_inputs_and_hidden_until_their_session(exp):
    for event_id in ("2026-02", "2026-05"):
        for forecast in (fc.qualifying_forecast(exp, event_id), fc.race_forecast(exp, event_id)):
            assert not LABEL_COLUMNS & set(forecast.inputs.columns)
            assert not LABEL_COLUMNS & set(forecast.predictions.columns)
        assert not fc.actual_outcome(exp, event_id, "qualifying", "pre_qualifying").visible
        assert fc.actual_outcome(exp, event_id, "qualifying", "post_qualifying").visible
        assert not fc.actual_outcome(exp, event_id, "race", "post_qualifying").visible
        assert fc.actual_outcome(exp, event_id, "race", "post_race").visible
    assert not fc.actual_outcome(exp, "2026-08", "qualifying", "post_race").visible


def test_race_forecast_keeps_both_top_threes_and_unnormalized_probabilities(exp):
    for event_id in ("2026-03", "2026-06"):
        forecast = fc.race_forecast(exp, event_id)
        preds = forecast.predictions
        assert sorted(preds.predicted_position) == list(range(1, len(preds) + 1))
        assert preds.race_order_top_three.sum() == preds.classifier_top_three.sum() == 3
        assert preds.podium_probability.between(0, 1).all()
        assert forecast.top_three["independent_probability_sum"] == pytest.approx(
            preds.podium_probability.sum()
        )


@pytest.mark.parametrize(
    ("event_id", "task"), [("2026-02", "qualifying"), ("2026-02", "race"), ("2026-07", "race")]
)
def test_csv_export_has_lineage_cutoff_and_coverage(exp, event_id, task):
    forecast = (
        fc.qualifying_forecast(exp, event_id)
        if task == "qualifying"
        else fc.race_forecast(exp, event_id)
    )
    frame = fc.export_frame(exp, forecast)
    prefix = "" if task == "qualifying" else "race_"
    required = {
        "data_kind",
        "manifest_id",
        "event_id",
        "prediction_source",
        "cutoff_utc",
        "provenance_mode",
        "driver_id",
        "predicted_position",
        f"{prefix}model_id",
        f"{prefix}training_first_event",
        f"{prefix}training_last_event",
        f"{prefix}training_data_end_utc",
        "practice_sessions",
        "practice_pace_available",
        "history_weekends",
        "imputed_inputs",
    }
    assert required <= set(frame.columns)
    assert not LABEL_COLUMNS & set(frame.columns)
    assert len(frame) == len(forecast.predictions) and frame.driver_id.is_unique
    assert frame[f"{prefix}model_id"].eq(forecast.models[task].model_id).all()
    assert frame.cutoff_utc.eq(forecast.cutoff).all()
    assert (
        pd.to_datetime(frame[f"{prefix}training_data_end_utc"]) < pd.Timestamp(forecast.cutoff)
    ).all()
    score = (
        "ranking_score_not_probability"
        if task == "qualifying"
        else "race_order_score_not_probability"
    )
    assert score in frame and "raw_score" not in frame
    round_trip = pd.read_csv(pd.io.common.BytesIO(frame.to_csv(index=False).encode()))
    assert list(round_trip.columns) == list(frame.columns)


def test_real_catalog_coverage_loads_without_models():
    from f1ml_app.catalog import RealCatalog

    catalog = RealCatalog(PATHS.catalog)
    assert set(catalog.events.year) == {2024, 2025, 2026}
    assert len(catalog.events) == int(catalog.coverage["seasons"].events.sum())
    first = catalog.events.loc[catalog.events.result_sessions.gt(0)].event_id.iloc[0]
    assert not catalog.results(first, "Race").empty
    assert not catalog.sessions(first).empty


# ----- the running application ------------------------------------------------------------


def _text(at) -> str:
    parts = [
        e.value
        for kind in ("markdown", "caption", "subheader", "info", "warning", "success", "error")
        for e in getattr(at, kind)
    ]
    parts += [h.proto.body for h in at.get("html")]
    parts += [f"{m.label} {m.value}" for m in at.metric]
    return "\n".join(str(p) for p in parts)


@pytest.fixture
def no_fitting_or_network(monkeypatch):
    """Any fit, recalibration, lock/evaluation call or network request fails the test."""
    import requests
    import sklearn.linear_model
    import sklearn.pipeline

    import f1ml.finalization.evaluate
    import f1ml.finalization.locking
    import f1ml.ingestion.provider
    import f1ml.models.qualifying
    import f1ml.models.race

    def forbidden(*args, **kwargs):
        raise AssertionError("The interface attempted a forbidden fit/download/evaluation")

    for target, name in (
        (sklearn.pipeline.Pipeline, "fit"),
        (sklearn.linear_model.LogisticRegression, "fit"),
        (f1ml.models.qualifying, "fit_qualifying"),
        (f1ml.models.race, "fit_model"),
        (f1ml.finalization.evaluate, "evaluate_once"),
        (f1ml.finalization.locking, "lock_models"),
        (f1ml.ingestion.provider.FastF1Provider, "__init__"),
        (requests.Session, "request"),
    ):
        monkeypatch.setattr(target, name, forbidden)


def test_app_views_render_events_switch_and_artifacts_stay_frozen(no_fitting_or_network):
    from streamlit.testing.v1 import AppTest

    frozen = snapshot(PATHS.final, PATHS.lock, PATHS.ledger, PATHS.qualifying, PATHS.race)
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    pages = [
        "views/overview.py",
        "views/qualifying.py",
        "views/race.py",
        "views/comparison.py",
        "views/methodology.py",
    ]
    seen = {}
    for event_id in ("2026-07", "2026-02", "2026-08"):
        at.selectbox(key="event_experiment_2026").set_value(event_id).run()
        for page in pages:
            at.switch_page(page).run()
            assert not at.exception, [e.value for e in at.exception]
            assert not at.error, [e.value for e in at.error]
            seen[(event_id, page)] = _text(at)
    assert "Final test · sealed forecast (frozen)" in seen[("2026-07", "views/qualifying.py")]
    assert "Development replay · outer-fold model" in seen[("2026-02", "views/race.py")]
    assert "No reviewed entrant field" in seen[("2026-08", "views/qualifying.py")]
    assert "no official qualifying result yet" in seen[("2026-08", "views/race.py")]
    assert "Synthetic data" in seen[("2026-02", "views/overview.py")]

    at.radio(key="moment").set_value("pre_qualifying").run()
    at.selectbox(key="event_experiment_2026").set_value("2026-05").run()
    at.switch_page("views/race.py").run()
    assert "Race forecast opens after qualifying" in _text(at)
    at.switch_page("views/qualifying.py").run()
    text = _text(at)
    assert "Official result not shown" in text and "Predicted top three" in text

    at.segmented_control(key="season_experiment").set_value(2024).run()
    at.switch_page("views/qualifying.py").run()
    assert "No model trained only on earlier weekends" in _text(at)

    at.segmented_control(key="source").set_value("catalog").run()
    for page in pages:
        at.switch_page(page).run()
        assert not at.exception and not at.error, page
    assert "No qualifying forecast for real catalog events" in _text(
        at.switch_page("views/qualifying.py").run()
    )
    at.segmented_control(key="season_catalog").set_value(2025).run()
    at.switch_page("views/overview.py").run()
    assert "Session availability" in _text(at) and not at.exception
    assert snapshot(PATHS.final, PATHS.lock, PATHS.ledger, PATHS.qualifying, PATHS.race) == frozen
