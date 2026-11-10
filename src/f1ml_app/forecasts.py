"""Historical replay through saved models and the pipeline's own prediction guards.

* Development validation weekends replay the outer-fold bundle saved for that weekend,
  which was trained only on earlier weekends, through ``predict_checked``.
* Final-test weekends show the Phase 7 sealed forecasts exactly as saved. They are never
  recomputed here (``predict_checked`` would refuse them anyway: the block is sealed).
* Any other weekend has no eligible model and is reported as unavailable instead of
  borrowing a model that was trained on that weekend or later.

Outcomes are loaded by ``actual_outcome`` from label-side tables only, and only joined to a
forecast for display after the forecast exists.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from f1ml.features.dictionary import columns as feature_columns
from f1ml.models.qualifying import FAMILIES as QUALIFYING_FAMILIES
from f1ml.models.qualifying import STRATEGIES
from f1ml.models.qualifying import load_model as load_qualifying_model
from f1ml.models.race import PODIUM_FAMILIES, RACE_FAMILIES
from f1ml.models.race import load_model as load_race_model
from f1ml.prediction.qualifying import predict_checked as predict_qualifying
from f1ml.prediction.race import combine_predictions
from f1ml.prediction.race import predict_checked as predict_race
from f1ml_app.experiment import FAMILY_LABELS, STRATEGY_LABELS, Experiment, ModelCard

MOMENTS = {
    "pre_qualifying": "Before qualifying",
    "post_qualifying": "After qualifying",
    "post_race": "After the race",
}
MOMENT_ORDER = tuple(MOMENTS)
FAMILIES = {"qualifying": QUALIFYING_FAMILIES, "race": RACE_FAMILIES, "podium": PODIUM_FAMILIES}
KEYS = ["event_id", "driver_id", "cutoff", "mode"]


def default_loader(card: ModelCard):
    """Integrity-checked bundle load (``verify_directory`` runs before unpickling)."""
    if card.task == "qualifying":
        return load_qualifying_model(card.path)
    return load_race_model(card.path, card.task)


@dataclass
class Forecast:
    task: str
    event_id: str
    available: bool
    headline: str
    explanation: str | None = None
    source: str | None = None
    predictions: pd.DataFrame | None = None
    inputs: pd.DataFrame | None = None
    references: pd.DataFrame | None = None
    models: dict[str, ModelCard] = field(default_factory=dict)
    cutoff: str | None = None
    mode: str | None = None
    seal_id: str | None = None
    top_three: dict | None = None
    checks: list[str] = field(default_factory=list)
    short: str | None = None

    @property
    def status_label(self) -> str:
        return (
            self.source_label if self.available else f"Unavailable · {self.short or self.headline}"
        )

    @property
    def source_label(self) -> str:
        return {
            "development_replay": "Development replay · outer-fold model",
            "frozen_final_test": "Final test · sealed forecast (frozen)",
        }.get(self.source, "Unavailable")


@dataclass
class Outcome:
    task: str
    visible: bool
    message: str | None = None
    table: pd.DataFrame | None = None
    source: str | None = None


def _unavailable(task, event_id, headline, explanation, **extra) -> Forecast:
    return Forecast(task, event_id, False, headline, explanation, **extra)


def _humanize(reason: str | None, strategy: str, count=None) -> str:
    if not reason:
        return "no saved bundle for this weekend"
    if reason.startswith("needs_") and reason.endswith("_usable_training_events"):
        needed = reason.split("_")[1]
        scope = "earlier 2026" if strategy == "2026_only" else "earlier"
        have = "" if count is None else f"; only {int(count)} existed"
        return f"needs {needed} usable {scope} training weekends{have}"
    return reason.replace("_", " ")


# ----- model choices ---------------------------------------------------------------------


def candidates(exp: Experiment, task: str, event_id: str) -> list[dict]:
    """Every family × strategy for a weekend: its saved bundle, or why none exists."""
    saved = {(c.family, c.strategy): c for c in exp.fold_cards(task, event_id)}
    try:
        comparison = exp.qualifying_comparison if task == "qualifying" else exp.race_comparison
        eligibility = comparison["eligibility"]
        eligibility = eligibility.loc[
            eligibility.event_id.eq(event_id)
            & eligibility.task.eq("race" if task == "podium" else task)
        ]
    except Exception:  # noqa: BLE001 - explanation is optional; bundles remain authoritative
        eligibility = pd.DataFrame(columns=["strategy", "reason", "training_event_count"])
    rows = []
    for family in FAMILIES[task]:
        for strategy in STRATEGIES:
            card = saved.get((family, strategy))
            note = eligibility.loc[eligibility.strategy.eq(strategy)]
            reason = None
            if card is None:
                reason = _humanize(
                    note.reason.iloc[0] if len(note) else None,
                    strategy,
                    note.training_event_count.iloc[0] if len(note) else None,
                )
            rows.append(
                {
                    "family": family,
                    "strategy": strategy,
                    "label": f"{FAMILY_LABELS[family]} · {STRATEGY_LABELS[strategy]}",
                    "available": card is not None,
                    "reason": reason,
                    "card": card,
                }
            )
    return rows


def _pick(exp, task, event_id, choice):
    family, strategy = choice if choice and choice[0] else exp.selected_config(task)
    for option in candidates(exp, task, event_id):
        if option["family"] == family and option["strategy"] == strategy:
            return option["card"], option
    return None, {"label": f"{family} · {strategy}", "reason": "unknown model choice"}


def _check_against_saved(replayed, saved, columns) -> str:
    if saved.empty:
        return "No saved development prediction to cross-check"
    joined = replayed.merge(saved, on="driver_id", suffixes=("", "_saved"), validate="one_to_one")
    if len(joined) != len(replayed):
        return "Replay field differs from the saved development predictions"
    for column in columns:
        if not np.allclose(joined[column], joined[f"{column}_saved"], rtol=0, atol=1e-9):
            return f"Replay differs from saved development predictions in {column}"
    return "Replay matches the saved development predictions exactly"


# ----- qualifying ------------------------------------------------------------------------


def qualifying_forecast(
    exp: Experiment, event_id: str, choice=None, loader: Callable = default_loader
) -> Forecast:
    role = exp.role(event_id)
    if role == "final_test":
        return _frozen_qualifying(exp, event_id)
    if role == "development_validation":
        return _replay_qualifying(exp, event_id, choice, loader)
    return _no_eligible_model(exp, "qualifying", event_id)


def _replay_qualifying(exp, event_id, choice, loader) -> Forecast:
    card, option = _pick(exp, "qualifying", event_id, choice)
    cut = exp.cutoff(event_id, "qualifying")
    if card is None:
        return _unavailable(
            "qualifying",
            event_id,
            f"No eligible {option['label']} model for this weekend",
            f"Not trained: {option['reason']}. No later-trained model is substituted.",
            short="strategy not trained for this weekend",
        )
    problems = card.lineage_problems(exp.manifest, event_id, cut[0], exp.weekend_start(event_id))
    if problems:
        return _unavailable(
            "qualifying", event_id, "Saved model is not eligible", "; ".join(problems)
        )
    inputs = exp.saved_inputs(event_id, "qualifying")
    predictions = predict_qualifying(
        loader(card), inputs, exp.manifest, exp.weekend_start(event_id)
    )
    saved = exp.qualifying_comparison["predictions"]
    mine = saved.event_id.eq(event_id)
    match = mine & saved.family.eq(card.family) & saved.strategy.eq(card.strategy)
    check = _check_against_saved(
        predictions,
        saved.loc[match, ["driver_id", "predicted_position", "raw_score"]],
        ["predicted_position", "raw_score"],
    )
    references = saved.loc[mine & saved.strategy.eq("fixed_rule")].rename(
        columns={"family": "method"}
    )
    return Forecast(
        "qualifying",
        event_id,
        True,
        "Qualifying forecast",
        source="development_replay",
        predictions=predictions,
        inputs=inputs,
        references=references,
        models={"qualifying": card},
        cutoff=cut[0],
        mode=cut[1],
        checks=[
            f"Bundle {card.model_id[:12]} integrity-verified before loading",
            "predict_checked accepted it: no training, tuning or selection use of this "
            "weekend, and all training data ends before the cutoff",
            check,
        ],
    )


def _frozen_qualifying(exp, event_id) -> Forecast:
    ok, problem = exp.final_available()
    if not ok:
        return _unavailable("qualifying", event_id, "Final-test forecast unavailable", problem)
    final, card = exp.final, exp.locked_card("qualifying")
    seal = final["qualifying_seal"]
    here = seal.loc[seal.event_id.eq(event_id) & seal.task.eq("qualifying")]
    model = here.loc[here.method.eq("model")]
    if card is None or model.empty:
        return _unavailable(
            "qualifying", event_id, "No sealed forecast", "The seal has no rows for this weekend."
        )
    if set(model.model_id) != {card.model_id}:
        return _unavailable(
            "qualifying",
            event_id,
            "Sealed forecast does not match the locked model",
            "Refusing to display a forecast whose model ID differs from the locked bundle.",
        )
    cutoff, mode = model.cutoff.iloc[0], model["mode"].iloc[0]
    problems = card.lineage_problems(exp.manifest, event_id, cutoff, exp.weekend_start(event_id))
    if problems:
        return _unavailable(
            "qualifying", event_id, "Locked model is not eligible", "; ".join(problems)
        )
    seal_id = final["seals"]["qualifying"]["id"]
    inputs = final["qualifying_inputs"]
    return Forecast(
        "qualifying",
        event_id,
        True,
        "Qualifying forecast",
        source="frozen_final_test",
        predictions=model[KEYS + ["raw_score", "predicted_position"]]
        .sort_values("predicted_position")
        .reset_index(drop=True),
        inputs=inputs.loc[inputs.event_id.eq(event_id)].reset_index(drop=True),
        references=here.loc[here.method.ne("model")],
        models={"qualifying": card},
        cutoff=cutoff,
        mode=mode,
        seal_id=seal_id,
        checks=[
            f"Prediction seal {seal_id[:12]} verified and matches the evaluation record",
            f"Locked model {card.model_id[:12]} trained only on weekends ending by "
            f"{card.training_end[:16].replace('T', ' ')} UTC",
            "Shown as sealed before any held-out result was read; not recomputed",
        ],
    )


# ----- race and podium -------------------------------------------------------------------


def race_forecast(
    exp: Experiment,
    event_id: str,
    moment: str = "post_race",
    race_choice=None,
    podium_choice=None,
    loader: Callable = default_loader,
) -> Forecast:
    if moment not in MOMENTS:
        raise ValueError(f"Unknown replay moment: {moment}")
    if moment == "pre_qualifying":
        return _unavailable(
            "race",
            event_id,
            "Race forecast opens after qualifying",
            "This replay moment is the qualifying cutoff. Official GP qualifying results do "
            "not exist yet, and the race-order and podium models take them as inputs. "
            "Move the replay moment to “After qualifying” to see the post-qualifying forecast.",
            short="opens after qualifying",
        )
    quali = exp.gp_session(event_id, "Qualifying")
    if quali is None or not quali.get("completed") or not quali.get("official_results"):
        return _unavailable(
            "race",
            event_id,
            "Race forecast unavailable: no official qualifying result yet",
            "The post-qualifying forecast needs a completed GP qualifying session with an "
            "official classification. This weekend has none, so no race order or podium "
            "probability is shown.",
            short="no qualifying result yet",
        )
    role = exp.role(event_id)
    if role == "final_test":
        return _frozen_race(exp, event_id)
    if role == "development_validation":
        return _replay_race(exp, event_id, race_choice, podium_choice, loader)
    return _no_eligible_model(exp, "race", event_id)


def _replay_race(exp, event_id, race_choice, podium_choice, loader) -> Forecast:
    cut = exp.cutoff(event_id, "race")
    start = exp.weekend_start(event_id)
    cards = {}
    for task, choice in (("race", race_choice), ("podium", podium_choice)):
        card, option = _pick(exp, task, event_id, choice)
        if card is None:
            return _unavailable(
                "race",
                event_id,
                f"No eligible {option['label']} {task} model for this weekend",
                f"Not trained: {option['reason']}. No later-trained model is substituted.",
                short="strategy not trained for this weekend",
            )
        problems = card.lineage_problems(exp.manifest, event_id, cut[0], start)
        if problems:
            return _unavailable(
                "race", event_id, f"Saved {task} model is not eligible", "; ".join(problems)
            )
        cards[task] = card
    if (
        cards["race"].metadata["feature_snapshot_id"]
        != cards["podium"].metadata["feature_snapshot_id"]
    ):
        return _unavailable(
            "race",
            event_id,
            "Incompatible models",
            "Race/podium models use different feature snapshots.",
        )
    inputs = exp.saved_inputs(event_id, "race")
    race = predict_race(loader(cards["race"]), inputs, exp.manifest, start)
    podium = predict_race(loader(cards["podium"]), inputs, exp.manifest, start)
    combined, comparison = combine_predictions(race, podium)
    saved = exp.race_comparison["predictions"]
    mine = saved.event_id.eq(event_id)

    def saved_rows(task):
        card = cards[task]
        return saved.loc[
            mine
            & saved.task.eq(task)
            & saved.family.eq(card.family)
            & saved.strategy.eq(card.strategy)
        ]

    race_check = _check_against_saved(
        race,
        saved_rows("race")[["driver_id", "predicted_position", "raw_score"]],
        ["predicted_position", "raw_score"],
    )
    podium_check = _check_against_saved(
        podium, saved_rows("podium")[["driver_id", "podium_probability"]], ["podium_probability"]
    )
    return Forecast(
        "race",
        event_id,
        True,
        "Race and podium forecast",
        source="development_replay",
        predictions=combined,
        inputs=inputs,
        references=saved.loc[mine & saved.strategy.eq("fixed_rule")].rename(
            columns={"family": "method"}
        ),
        models=cards,
        cutoff=cut[0],
        mode=cut[1],
        top_three=comparison[0],
        checks=[
            f"Race bundle {cards['race'].model_id[:12]} and podium bundle "
            f"{cards['podium'].model_id[:12]} integrity-verified before loading",
            "predict_checked accepted both for this weekend and cutoff",
            f"Race order: {race_check.lower()}",
            f"Podium probabilities: {podium_check.lower()}",
        ],
    )


def _frozen_race(exp, event_id) -> Forecast:
    ok, problem = exp.final_available()
    if not ok:
        return _unavailable("race", event_id, "Final-test forecast unavailable", problem)
    final = exp.final
    cards = {task: exp.locked_card(task) for task in ("race", "podium")}
    seal = final["race_seal"]
    here = seal.loc[seal.event_id.eq(event_id)]
    for task, card in cards.items():
        ids = set(here.loc[here.task.eq(task) & here.method.eq("model"), "model_id"])
        if card is None or ids != {card.model_id}:
            return _unavailable(
                "race",
                event_id,
                "Sealed forecast does not match the locked models",
                f"Refusing to display {task} forecasts whose model ID differs from the lock.",
            )
    combined = final["race_and_podium"]
    combined = combined.loc[combined.event_id.eq(event_id)].sort_values("predicted_position")
    cutoff, mode = combined.cutoff.iloc[0], combined["mode"].iloc[0]
    for task, card in cards.items():
        problems = card.lineage_problems(
            exp.manifest, event_id, cutoff, exp.weekend_start(event_id)
        )
        if problems:
            return _unavailable(
                "race", event_id, f"Locked {task} model is not eligible", "; ".join(problems)
            )
    top = next((t for t in final["top_three"] if t["event_id"] == event_id), None)
    inputs = final["race_inputs"]
    seal_id = final["seals"]["race"]["id"]
    return Forecast(
        "race",
        event_id,
        True,
        "Race and podium forecast",
        source="frozen_final_test",
        predictions=combined.drop(columns=["data_kind"], errors="ignore").reset_index(drop=True),
        inputs=inputs.loc[inputs.event_id.eq(event_id)].reset_index(drop=True),
        references=here.loc[here.method.ne("model")],
        models=cards,
        cutoff=cutoff,
        mode=mode,
        seal_id=seal_id,
        top_three=top,
        checks=[
            f"Race/podium seal {seal_id[:12]} verified and matches the evaluation record",
            f"Locked models trained only on weekends ending by "
            f"{cards['race'].training_end[:16].replace('T', ' ')} UTC",
            "Shown as sealed before scoring; not recomputed",
        ],
    )


# ----- no eligible model -----------------------------------------------------------------


def _no_eligible_model(exp, task, event_id) -> Forecast:
    role = exp.role(event_id)
    eligible = exp.eligible_cards(task, event_id)
    folds = ", ".join(f["validate"] for f in exp.manifest["folds"])
    word = "qualifying" if task == "qualifying" else "race/podium"
    if role == "development_training":
        return _unavailable(
            task,
            event_id,
            "No model trained only on earlier weekends",
            f"{event_id} is a development training weekend. Every saved {word} model trained "
            f"on it or on a later weekend: outer-fold models exist only for the 2026 "
            f"validation weekends ({folds}), and the locked refit trained on all development "
            "weekends. Using a later-trained model would leak this weekend's own outcome, so "
            "no forecast is shown.",
            short="no earlier-trained model",
        )
    reason = exp.exclusion_reason(event_id)
    note = f" Manifest status: {reason.replace('_', ' ')}." if reason else ""
    if eligible and exp.saved_inputs(event_id, task).empty:
        locked = [c for c in eligible if c.role == "locked_refit"]
        card = (locked or eligible)[0]
        kind = "locked refit" if card.role == "locked_refit" else "saved model"
        return _unavailable(
            task,
            event_id,
            "No reviewed entrant field for this weekend",
            f"An eligible {word} model exists (the {kind} {card.model_id[:12]}, trained only "
            "on earlier weekends), but this weekend has no reviewed, cutoff-known entrant "
            "roster and no completed sessions. A forecast would require inventing the field, "
            f"so none is shown.{note}",
            short="no reviewed entrant field",
        )
    return _unavailable(
        task,
        event_id,
        "No eligible model",
        f"No saved {word} model is eligible.{note}",
        short="no eligible model",
    )


# ----- label-side outcomes (display only) ------------------------------------------------

OUTCOME_COLUMNS = ["driver_id", "official_position", "outcome", "podium", "source_status"]


def outcome_visible(task: str, moment: str) -> bool:
    needed = "post_qualifying" if task == "qualifying" else "post_race"
    return MOMENT_ORDER.index(moment) >= MOMENT_ORDER.index(needed)


def actual_outcome(exp: Experiment, event_id: str, task: str, moment: str) -> Outcome:
    if not outcome_visible(task, moment):
        when = "qualifying" if task == "qualifying" else "race"
        return Outcome(
            task,
            False,
            f"Hidden at this replay moment: the official {when} result is not known yet.",
        )
    spec = exp.gp_session(event_id, "Qualifying" if task == "qualifying" else "Race")
    if spec is None or not spec.get("completed") or not spec.get("official_results"):
        return Outcome(task, False, "No official result exists for this session yet.")
    if exp.role(event_id) == "final_test":
        ok, problem = exp.final_available()
        if not ok:
            return Outcome(task, False, problem)
        frame = exp.final[f"{task}_vs_actual"]
        frame = frame.loc[frame.event_id.eq(event_id) & frame.method.eq("model")]
        source = "Phase 7 frozen scoring record (read only after forecasts were sealed)"
    else:
        frame = exp.development_labels(event_id, task)
        source = "Saved Phase 3 label table (label side; never a model input)"
    table = frame[[c for c in OUTCOME_COLUMNS if c in frame]].copy()
    return Outcome(
        task, True, None, table.sort_values("official_position").reset_index(drop=True), source
    )


def compare_to_actual(predictions: pd.DataFrame, outcome: Outcome) -> pd.DataFrame:
    joined = predictions[["driver_id", "predicted_position"]].merge(
        outcome.table[["driver_id", "official_position", "outcome"]],
        on="driver_id",
        how="left",
        validate="one_to_one",
    )
    joined["absolute_error"] = (joined.predicted_position - joined.official_position).abs()
    joined["places_vs_forecast"] = joined.predicted_position - joined.official_position
    return joined.sort_values("predicted_position").reset_index(drop=True)


def presentation_frame(
    predictions: pd.DataFrame,
    *,
    task: str | None = None,
    family: str | None = None,
    outcome: Outcome | None = None,
) -> pd.DataFrame:
    """Copy a single event's saved field into a common display schema.

    Real-lab ``predicted_rank`` is the discrete order; its ``predicted_position``
    is a regression score. Legacy forecasts use ``predicted_position`` as order.
    Race probabilities always join the separate podium task by event/driver/model.
    An explicit Outcome enforces the legacy replay's result-visibility boundary.
    """
    data = predictions.copy()
    if task is not None and "task" in data:
        data = data.loc[data.task.eq(task)].copy()
    if family is not None and "family" in data:
        data = data.loc[data.family.eq(family)].copy()
    if task == "race" and "task" in predictions:
        podium = predictions.loc[predictions.task.eq("podium")].copy()
        if family is not None and "family" in podium:
            podium = podium.loc[podium.family.eq(family)]
        if not podium.empty:
            keys = [
                c
                for c in ("event_id", "driver_id", "family", "strategy", "cutoff", "mode")
                if c in data and c in podium
            ]
            columns = [
                c
                for c in (
                    "podium_probability",
                    "raw_podium_probability",
                    "podium_prediction",
                    "threshold",
                )
                if c in podium
            ]
            data = data.drop(columns=columns, errors="ignore").merge(
                podium[keys + columns],
                on=keys,
                how="left",
                validate="one_to_one",
            )
    if outcome is not None:
        data = data.drop(
            columns=[
                "actual_rank",
                "official_position",
                "actual_podium",
                "podium",
                "outcome",
                "absolute_error",
                "places_vs_forecast",
                "label_source",
                "label_available_at",
            ],
            errors="ignore",
        )
        if outcome.visible and outcome.table is not None:
            keys = [c for c in ("event_id", "driver_id") if c in data and c in outcome.table]
            cols = [c for c in ("official_position", "outcome", "podium") if c in outcome.table]
            data = data.merge(
                outcome.table[keys + cols], on=keys, how="left", validate="one_to_one"
            )
    for target, source in (
        ("predicted_rank", "predicted_position"),
        ("actual_rank", "official_position"),
        ("actual_podium", "podium"),
    ):
        if target not in data:
            data[target] = data[source] if source in data else float("nan")
    for column in ("predicted_rank", "actual_rank"):
        data[column] = pd.to_numeric(data[column], errors="coerce").where(lambda x: x.ge(1))
    if "driver_id" not in data:
        data["driver_id"] = data.index.astype(str)
    fallback = data.driver_id.astype(str).str.replace("_", " ").str.title()
    data["driver"] = data["driver"].fillna(fallback) if "driver" in data else fallback
    if "team_id" not in data:
        data["team_id"] = None
    if "podium_probability" not in data:
        data["podium_probability"] = float("nan")
    data["podium_probability"] = pd.to_numeric(data.podium_probability, errors="coerce")
    data["podium_probability"] = data.podium_probability.where(
        data.podium_probability.between(0, 1)
    )
    data["rank_delta"] = data.predicted_rank - data.actual_rank
    data["rank_error"] = data.rank_delta.abs()
    data["rank_delta_label"] = data.rank_delta.map(rank_delta_label)
    return data.sort_values(["predicted_rank", "driver_id"], kind="stable").reset_index(drop=True)


def rank_delta_label(value) -> str:
    """Places gained versus forecast, with a distinct label for unknown outcomes."""
    if pd.isna(value):
        return "—"
    return f"▲ +{int(value)}" if value > 0 else f"▼ {int(value)}" if value < 0 else "— 0"


def compare_fields(fields: dict[str, pd.DataFrame], primary: str) -> pd.DataFrame:
    """Align saved model outputs by driver, retaining missing outputs as unknown.

    Ranks are never recomputed after filtering. Consensus is a descriptive mean of
    available ranks, not a new forecast or a metric evaluated against target labels.
    """
    nonempty = {model: frame for model, frame in fields.items() if not frame.empty}
    if not nonempty:
        return presentation_frame(pd.DataFrame())
    primary = primary if primary in nonempty else next(iter(nonempty))
    base = nonempty[primary].copy()
    # Keep entrants absent from the primary output, with its prediction left unknown.
    for frame in nonempty.values():
        missing = frame.loc[~frame.driver_id.isin(base.driver_id)].copy()
        if not missing.empty:
            missing[["predicted_rank", "podium_probability", "rank_error"]] = float("nan")
            missing["rank_delta"] = float("nan")
            missing["rank_delta_label"] = "—"
            base = pd.concat([base, missing], ignore_index=True)
    if len(fields) > 1:
        for model, frame in fields.items():
            indexed = frame.drop_duplicates("driver_id").set_index("driver_id")
            for source, prefix in (
                ("predicted_rank", "rank"),
                ("podium_probability", "podium"),
                ("rank_error", "error"),
            ):
                base[f"{prefix}__{model}"] = base.driver_id.map(indexed[source])
        ranks = base[[f"rank__{model}" for model in fields]]
        base["Consensus / Mean Rank"] = ranks.mean(axis=1)
        base["Model Spread (Max - Min Rank)"] = ranks.max(axis=1) - ranks.min(axis=1)
        base["Models with rank"] = ranks.notna().sum(axis=1)
    base.attrs["models"] = list(fields)
    base.attrs["primary"] = primary
    return base.sort_values(["predicted_rank", "driver_id"], kind="stable").reset_index(drop=True)


def comparison_frame(
    predictions: pd.DataFrame, *, task: str, models: list[str], primary: str
) -> pd.DataFrame:
    return compare_fields(
        {model: presentation_frame(predictions, task=task, family=model) for model in models},
        primary,
    )


def comparison_long(field: pd.DataFrame) -> pd.DataFrame:
    """One row per driver and active model for grouped charts, including unknown actuals."""
    rows = []
    for column in field:
        if not column.startswith("rank__"):
            continue
        model = column.removeprefix("rank__")
        frame = field[["driver_id", "driver", "team_id", "actual_rank"]].copy()
        frame["family"] = model
        frame["predicted_rank"] = field[column]
        frame["podium_probability"] = field[f"podium__{model}"]
        frame["rank_error"] = field[f"error__{model}"]
        rows.append(frame)
    return pd.concat(rows, ignore_index=True) if rows else field.copy()


def spotlight_candidates(field: pd.DataFrame) -> pd.DataFrame:
    """Race-order P1–P3 and P4; never select cards using observed outcomes."""
    return (
        field.loc[field.predicted_rank.notna()]
        .sort_values(["predicted_rank", "driver_id"], kind="stable")
        .head(4)
        .copy()
    )


def saved_event_metrics(exp: Experiment, forecast: Forecast, task: str) -> dict | None:
    """Metrics from the saved evaluation record for this exact model, not recomputed."""
    card = forecast.models.get(task)
    if card is None:
        return None
    if forecast.source == "frozen_final_test":
        frame = exp.final["event_metrics"]
        rows = frame.loc[
            frame.event_id.eq(forecast.event_id) & frame.task.eq(task) & frame.method.eq("model")
        ]
    else:
        frame = (exp.qualifying_comparison if task == "qualifying" else exp.race_comparison)[
            "event_metrics"
        ]
        rows = frame.loc[
            frame.event_id.eq(forecast.event_id)
            & frame.family.eq(card.family)
            & frame.strategy.eq(card.strategy)
        ]
        if "task" in rows:
            rows = rows.loc[rows.task.eq(task)]
    return rows.iloc[0].dropna().to_dict() if len(rows) else None


# ----- input coverage and export ---------------------------------------------------------


def input_coverage(inputs: pd.DataFrame, task: str) -> pd.DataFrame:
    """Per-driver coverage: numeric model inputs that are missing get the training-fold median."""
    numeric = [c for c in feature_columns(task) if c in inputs]
    missing = inputs[numeric].isna()
    result = inputs[["driver_id", "team_id"]].copy()
    result["practice_sessions"] = inputs.practice_sessions
    result["practice_clean_laps"] = inputs.practice_clean_laps
    result["practice_pace_available"] = inputs.practice_gap_pct.notna()
    result["history_weekends"] = inputs.history_count
    result["imputed_inputs"] = missing.sum(axis=1).astype(int)
    result["model_inputs"] = len(numeric)
    result["imputed_input_names"] = missing.apply(lambda r: ", ".join(r.index[r]), axis=1)
    return result


def event_coverage(inputs: pd.DataFrame, task: str) -> dict:
    per = input_coverage(inputs, task)
    summary = {
        "drivers": len(inputs),
        "practice_sessions_used": int(inputs.practice_sessions.max()) if len(inputs) else 0,
        "drivers_with_practice_pace": int(per.practice_pace_available.sum()),
        "drivers_with_history": int((inputs.history_count > 0).sum()),
        "max_history_weekends": int(inputs.history_count.max()) if len(inputs) else 0,
        "mean_imputed_inputs": float(per.imputed_inputs.mean()) if len(inputs) else 0.0,
        "model_inputs": int(per.model_inputs.iloc[0]) if len(inputs) else 0,
    }
    if task == "race":
        summary.update(
            drivers_with_qualifying_classification=int(inputs.qualifying_rank_norm.notna().sum()),
            drivers_with_qualifying_pace=int(inputs.qualifying_q1_gap_pct.notna().sum()),
            drivers_with_known_grid=int(inputs.known_grid_rank_norm.notna().sum()),
        )
    return summary


def _training(card: ModelCard, prefix: str, order: list[str]) -> dict:
    events = sorted(card.train_events, key=order.index)
    return {
        f"{prefix}model_id": card.model_id,
        f"{prefix}model_family": card.family,
        f"{prefix}model_strategy": card.strategy,
        f"{prefix}model_bundle": card.role,
        f"{prefix}training_event_count": len(events),
        f"{prefix}training_first_event": events[0],
        f"{prefix}training_last_event": events[-1],
        f"{prefix}training_events": ";".join(events),
        f"{prefix}training_data_end_utc": card.training_end,
    }


def export_frame(exp: Experiment, forecast: Forecast) -> pd.DataFrame:
    """Prediction export with model lineage, cutoff and data coverage. No outcome columns."""
    if not forecast.available:
        raise ValueError("Only available forecasts can be exported")
    event = exp.events.set_index("event_id").loc[forecast.event_id]
    order = [e["event_id"] for e in exp.manifest["eligible"]]
    coverage = input_coverage(forecast.inputs, forecast.task).drop(columns=["model_inputs"])
    base = {
        "data_kind": exp.data_kind,
        "manifest_id": exp.manifest_id,
        "event_id": forecast.event_id,
        "event_name": event["name"],
        "season": int(event["year"]),
        "round": int(event["round"]),
        "event_role": event["role"],
        "prediction_source": forecast.source,
        "cutoff_utc": forecast.cutoff,
        "provenance_mode": forecast.mode,
        "seal_id": forecast.seal_id or "",
    }
    if forecast.task == "qualifying":
        frame = forecast.predictions[["driver_id", "predicted_position", "raw_score"]].rename(
            columns={"raw_score": "ranking_score_not_probability"}
        )
        meta = _training(forecast.models["qualifying"], "", order)
    else:
        columns = [
            "driver_id",
            "predicted_position",
            "raw_score",
            "podium_probability",
            "raw_podium_probability",
            "podium_threshold",
            "predicted_podium",
            "podium_probability_rank",
            "race_order_top_three",
            "classifier_top_three",
        ]
        frame = forecast.predictions[columns].rename(
            columns={
                "raw_score": "race_order_score_not_probability",
                "raw_podium_probability": "uncalibrated_podium_probability",
            }
        )
        inputs = forecast.inputs[["driver_id", "qualifying_position", "known_grid_position"]]
        frame = frame.merge(
            inputs.rename(
                columns={
                    "qualifying_position": "input_qualifying_position",
                    "known_grid_position": "input_known_grid_position",
                }
            ),
            on="driver_id",
            validate="one_to_one",
        )
        meta = {
            **_training(forecast.models["race"], "race_", order),
            **_training(forecast.models["podium"], "podium_", order),
        }
    frame = frame.merge(coverage, on="driver_id", validate="one_to_one")
    for column in ("predicted_position", "podium_probability_rank"):
        if column in frame:  # sealed parquet stores ranks as floats; export whole places
            frame[column] = frame[column].astype(int)
    for key, value in {**base, **meta}.items():
        frame[key] = value
    leading = list(base) + ["driver_id", "team_id", "predicted_position"]
    return (
        frame[leading + [c for c in frame if c not in leading]]
        .sort_values("predicted_position")
        .reset_index(drop=True)
    )
