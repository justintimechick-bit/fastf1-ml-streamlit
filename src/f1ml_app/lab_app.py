"""Real historical tests: reading a page never fits models or downloads data."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from f1ml_app import charts, theme, ui
from f1ml_app import forecasts as fc
from f1ml_app.settings import MODEL_LABELS, project_root
from f1ml_app.text import metric_help, metric_label

FAMILIES = {
    "linear": "Linear models",
    "random_forest": "Random forest",
    "gradient_boosting": "Gradient boosting",
    "baseline": "Simple baseline",
}
SEASONS = (2026, 2025, 2024)
WINDOWS = {
    "All earlier races that year": None,
    "Previous 12 races that year": 12,
    "Previous 6 races that year": 6,
}
TASKS = {"qualifying": "Qualifying", "race": "Race", "podium": "Podium"}
# Radio values are part of the saved session contract; only their display text changes.
CUTOFF_LABELS = {
    "Before qualifying": "Qualifying order",
    "After qualifying · race": "Race order & podium",
}
CUTOFF_HELP = (
    "Qualifying order is predicted before qualifying, from earlier form and the entry list. "
    "Race order and podium chances are predicted after qualifying, using the grid."
)
# Headline score per task. Saved values are displayed as stored, never recomputed.
HEADLINE_METRIC = {
    "qualifying": ("mae", "Avg. places off"),
    "race": ("mae", "Avg. places off"),
    "podium": ("brier", "Brier score"),
}
BASELINE_COPY = {
    "qualifying": "a fixed rule based on recent form",
    "race": "a fixed rule that follows the qualifying order",
    "podium": "a fixed rule based on the grid and recent form",
}
MODEL_COPY = {
    "linear": (
        "Ridge regression + logistic regression",
        "Linear and logistic regression, ridge regularization",
        "Combine recent driver and team form with learned weights. Ridge regularization "
        "limits extreme weights; logistic regression estimates each driver's chance "
        "of a race podium.",
    ),
    "random_forest": (
        "Random forest",
        "Classification and regression trees; ensembles",
        "Average many decision trees to learn interactions between driver form, "
        "team form and the information available at the prediction cutoff.",
    ),
    "gradient_boosting": (
        "Gradient-boosted trees",
        "Boosting trees; ensembles",
        "Build trees in sequence, with each tree correcting earlier errors. Separate "
        "regressors predict qualifying and race order; a classifier estimates "
        "race podium probabilities.",
    ),
}


def paths() -> tuple[Path, Path]:
    root = project_root()
    dataset = Path(
        os.environ.get(
            "F1ML_LAB_DATASET", os.environ.get("F1ML_REAL_DATASET", "data/real-core-20261105-v4")
        )
    )
    runs = Path(
        os.environ.get("F1ML_LAB_RUNS", os.environ.get("F1ML_REAL_RUNS", "artifacts/real-lab"))
    )
    return (
        dataset if dataset.is_absolute() else root / dataset,
        runs if runs.is_absolute() else root / runs,
    )


@st.cache_resource(show_spinner=False)
def _dataset(path: str, receipt_mtime: int):
    from f1ml_real.lab import RealDataset

    return RealDataset(Path(path))


@st.cache_data(show_spinner=False)
def _load_run(path: str, receipt_mtime: int) -> dict:
    from f1ml_real.lab import load_run

    run = load_run(Path(path))
    calibration = Path(path) / "calibration_predictions.parquet"
    if calibration.exists():
        run["calibration_predictions"] = pd.read_parquet(calibration)
    return run


def historical_paths(season: int) -> tuple[Path, Path]:
    if season == 2026:
        return paths()[0], live_data_path()
    root = Path(os.environ.get("F1ML_HISTORICAL_DATA", "data/historical"))
    if not root.is_absolute():
        root = project_root() / root
    return root / str(season), live_data_path() / str(season)


def _selected_season() -> int:
    return st.session_state.get("lab_season", st.session_state.get("lab_selected_season", 2026))


def _season_selector() -> int:
    if "lab_season" not in st.session_state:
        st.session_state["lab_season"] = _selected_season()
    season = st.selectbox("Season", SEASONS, key="lab_season")
    previous = st.session_state.get("_lab_filter_season")
    if previous != season:
        memory = st.session_state.setdefault("_lab_season_filters", {})
        keys = ("lab_target_event", "lab_history_window", "lab_replay_artifact", "lab_active_run")
        if previous is not None:
            memory[previous] = {
                key: st.session_state[key] for key in keys if key in st.session_state
            }
        if previous is not None:
            for key in keys:
                if key in memory.get(season, {}):
                    st.session_state[key] = memory[season][key]
                else:
                    st.session_state.pop(key, None)
        st.session_state["_lab_filter_season"] = season
    st.session_state["lab_selected_season"] = season
    return season


def _load_dataset(*, active: bool = False, season: int = 2026):
    path, state = historical_paths(season)
    receipt = path / "receipt.json"
    if not receipt.exists():
        st.warning(f"The reviewed {season} dataset is not installed in this app yet.")
        st.caption(
            "Prepare and verify the source snapshot outside Streamlit, "
            "then add it to the deployment."
        )
        return None
    try:
        if active:
            from f1ml_real.refresh import current_dataset

            path = current_dataset(path, state)
            receipt = path / "receipt.json"
        return _dataset(str(path), receipt.stat().st_mtime_ns)
    except (ImportError, ValueError, OSError, KeyError) as exc:
        st.error(f"The real dataset could not be verified: {exc}")
        return None


def _read_saved(path: Path | str) -> dict | None:
    path = Path(path)
    try:
        # Reverify if any saved file changes, including the prediction lock or tables.
        stamp = hash(
            tuple(
                (str(p.relative_to(path)), p.stat().st_mtime_ns, p.stat().st_size)
                for p in sorted(path.rglob("*"))
                if p.is_file()
            )
        )
        run = _load_run(str(path), stamp)
        run["_path"] = str(path)
        return run
    except (ValueError, OSError, KeyError) as exc:
        st.warning(f"A saved run could not be verified: {exc}")
        return None


def _default_saved(event_id: str, source_snapshot_id: str) -> dict | None:
    """Find a verified bundle without starting a run or mutating a ledger."""
    import json

    _, root = paths()
    if not root.is_dir():
        return None
    candidates = []
    for marker in root.glob("*/config.json"):
        if not (marker.parent / "receipt.json").is_file():
            continue
        try:
            config = json.loads(marker.read_text())
        except (OSError, ValueError):
            continue
        if (
            config.get("event_id", config.get("target_event_id")) == event_id
            and config.get("last_n") is None
            and config.get("source_snapshot_id") == source_snapshot_id
        ):
            candidates.append(marker.parent)
    for path in sorted(candidates, key=lambda p: p.stat().st_mtime_ns, reverse=True):
        if run := _read_saved(path):
            return run
    return None


def _event_label(events: pd.DataFrame, event_id: str) -> str:
    rows = events.loc[events.event_id.eq(event_id)]
    if rows.empty:
        return str(event_id)
    event = rows.iloc[0]
    return f"{event_id.split('-')[0]} · R{int(event['round']):02d} · {event['name']}"


def _frame(run: dict, key: str) -> pd.DataFrame:
    value = run.get(key)
    return value.copy() if isinstance(value, pd.DataFrame) else pd.DataFrame(value or [])


def _table(frame: pd.DataFrame, *, key: str, **kwargs):
    st.dataframe(frame, hide_index=True, width="stretch", key=key, **kwargs)


def _title(title: str, detail: str, slots=None):
    ui.persist_filters()
    if slots is None:
        header = st.empty()
        with st.container(key="f1_context_ribbon"):
            ribbon = st.empty()
    else:
        header, ribbon = slots
    with header:
        ui.brand_header(title, detail)
    _render_context(ribbon)
    return header, ribbon


def _sidebar_sections():
    """The sidebar holds app-wide preferences only: models, grid filters and detail level.

    Race context (season, event, history window) sits at the top of each page, next to
    the action it feeds, so the main workflow reads top to bottom without hunting.
    """
    with st.sidebar:
        st.markdown("### Models")
        model_controls = st.container()
        st.markdown("### Filter the grid")
        ui.grid_preferences()
    return model_controls


def _setup_panel(columns=(1, 2.4, 1.7)):
    """A bordered 'race setup' card at the top of the page with one row of selectors."""
    box = st.container(border=True, key="f1_setup")
    with box:
        cols = st.columns(list(columns), vertical_alignment="bottom")
    return box, cols


def _selected_task(*, live: bool = False) -> str:
    """Read the task switch's current value before the switch itself is drawn."""
    if live:
        value = st.session_state.get("lab_live_task", "all")
        return value if value in ("all", "qualifying", "race") else "all"
    cutoff = st.session_state.get("lab_forecast_tab", "Before qualifying")
    return "race" if cutoff == "After qualifying · race" else "qualifying"


def _task_switch(*, live: bool = False, available=None) -> str:
    """Main-canvas segmented switch between the two prediction moments."""
    if live:
        options = ["all", "qualifying", "race"]
        if available is not None:
            options = ["all", *[task for task in ("qualifying", "race") if task in available]]
        if st.session_state.get("lab_live_task") not in options:
            st.session_state["lab_live_task"] = "all"
        st.radio(
            "Show",
            options,
            format_func={"all": "Everything available", **TASKS}.get,
            key="lab_live_task",
            horizontal=True,
        )
        return _selected_task(live=True)
    st.radio(
        "Prediction",
        list(CUTOFF_LABELS),
        format_func=CUTOFF_LABELS.get,
        key="lab_forecast_tab",
        horizontal=True,
        help=CUTOFF_HELP,
    )
    return _selected_task()


def _model_filters(*, live: bool = False) -> tuple[str, str]:
    family = ui.model_preferences(live=live)
    st.session_state["_lab_family"] = family
    return family, _selected_task(live=live)


def _render_context(
    slot,
    *,
    run=None,
    dataset=None,
    season=None,
    event=None,
    family=None,
    task=None,
    pending=None,
    unverified_artifact=None,
):
    """The ribbon describes the verified tables on screen, never the form's draft."""
    artifact = None
    receipt_verified = False
    lock_status = "No prediction lock"
    if run is not None:
        config = run.get("config", {})
        event_id = config.get("event_id", config.get("target_event_id"))
        event_id = event_id or run.get("summary", {}).get("target_event_id")
        if event_id:
            season = int(event_id.split("-")[0])
            event = _event_label(dataset.events, event_id) if dataset else event_id
        path = run.get("_path", run.get("path"))
        artifact = Path(path).name if path else run.get("receipt", {}).get("id")
        receipt_verified = bool(run.get("receipt", {}).get("id"))
        lock = run.get("prediction_lock", {})
        lock_status = (
            "Predictions locked"
            if lock.get("sealed_at") and lock.get("sealed_before_target_scoring") is True
            else "Prediction lock unconfirmed"
        )
    elif unverified_artifact:
        artifact = f"{Path(unverified_artifact).name} (unavailable)"
        lock_status = "Prediction lock not verified"
    elif dataset is not None:
        artifact = f"Dataset {dataset.root.name} (no replay selected)"
        receipt_verified = bool(dataset.receipt.get("id"))
    with slot:
        ui.context_ribbon(
            artifact_id=artifact,
            season=season or _selected_season(),
            event=event,
            family=family or st.session_state.get("_lab_family", "linear"),
            models=ui.selected_models(),
            task={"all": "All available", **TASKS}.get(task, task),
            lock_status=lock_status,
            receipt_verified=receipt_verified,
            pending=pending,
        )


def _saved_options(dataset, season: int) -> list[str]:
    """Discover candidate replay headers. Selection still goes through load_run verification."""
    candidates = []
    for marker in paths()[1].glob("*/config.json"):
        if not (marker.parent / "receipt.json").is_file():
            continue
        try:
            config = json.loads(marker.read_text())
            event_id = config.get("event_id", config.get("target_event_id", ""))
            if (
                event_id.startswith(f"{season}-")
                and config.get("source_snapshot_id") == dataset.receipt["id"]
                and config.get("mode", "historical_replay") != "live_forecast"
            ):
                candidates.append(str(marker.parent))
        except (OSError, ValueError, TypeError):
            continue
    return sorted(candidates)


def _replay_selector(dataset, season: int, *, submitted: bool) -> dict | None:
    options = _saved_options(dataset, season)
    active = st.session_state.get("lab_active_run")
    if active and active not in options:
        # A just-submitted run can live in an explicitly configured output directory.
        options.append(active)
    key = "lab_replay_artifact"
    if (
        submitted
        or st.session_state.get(key) not in [None, *options]
        or key not in st.session_state
    ):
        st.session_state[key] = active if active in options else None
    choice = st.selectbox(
        "Saved run",
        [None, *options],
        key=key,
        format_func=_replay_label,
        help="Open a run that was already trained and sealed. Nothing is retrained.",
    )
    st.session_state["lab_active_run"] = choice
    st.session_state["lab_active_season"] = season
    run = _read_saved(choice) if choice else None
    if run:
        config = run.get("config", {})
        event_id = config.get("event_id", config.get("target_event_id"))
        if event_id:
            st.caption(f"Showing results for {_event_label(dataset.events, event_id)}.")
            if event_id != st.session_state.get("lab_target_event", event_id):
                st.caption("This saved run is for a different race than the one selected above.")
    return run


def _replay_label(path) -> str:
    if not path:
        return "None (choose a race and run a test)"
    parts = Path(path).name.split("-")
    if len(parts) >= 5 and parts[0] in ("replay", "forecast"):
        scope = "-".join(parts[3:-1])
        scope = {"all": "all earlier races"}.get(scope, f"previous {scope} races")
        return f"{parts[1]} · R{int(parts[2]):02d} · {scope} · {parts[-1][:8]}"
    return Path(path).name


def _event_option(events: pd.DataFrame):
    """Short dropdown text; the season is already chosen in the adjacent selector."""
    rows = events.set_index("event_id")

    def label(event_id) -> str:
        if event_id not in rows.index:
            return str(event_id)
        row = rows.loc[event_id]
        return f"R{int(row['round']):02d} · {row['name']}"

    return label


def _window_label(window: str) -> str:
    return {
        "All earlier races that year": "All earlier races",
        "Previous 12 races that year": "Last 12 races",
        "Previous 6 races that year": "Last 6 races",
    }.get(window, window)


def _has_saved(event_id: str, last_n, snapshot_id: str) -> bool:
    """Cheap header scan for a matching saved run; loading still verifies it."""
    root = paths()[1]
    for marker in root.glob("*/config.json") if root.is_dir() else []:
        if not (marker.parent / "receipt.json").is_file():
            continue
        try:
            config = json.loads(marker.read_text())
        except (OSError, ValueError):
            continue
        if (
            config.get("event_id", config.get("target_event_id")) == event_id
            and config.get("last_n") == last_n
            and config.get("source_snapshot_id") == snapshot_id
            and config.get("mode", "historical_replay") != "live_forecast"
        ):
            return True
    return False


def _shared_replay_controls(dataset, season: int, cols=None, saved_slot=None) -> dict | None:
    """Same global event and artifact selectors on the explanatory pages."""
    if dataset is None:
        return None
    events = dataset.events.loc[dataset.events.year.eq(season)].sort_values("round")
    ids = events.event_id.tolist()
    if not ids:
        return None
    if st.session_state.get("lab_target_event") not in ids:
        st.session_state["lab_target_event"] = ids[-1]
    event_slot, window_slot = cols if cols else (st.container(), st.container())
    with event_slot:
        st.selectbox(
            "Race",
            ids,
            key="lab_target_event",
            format_func=_event_option(events),
        )
    with window_slot:
        st.selectbox("Train on", list(WINDOWS), key="lab_history_window", format_func=_window_label)
    with saved_slot if saved_slot is not None else st.container():
        return _replay_selector(dataset, season, submitted=False)


def _cutoff(value) -> str:
    return (
        "Unavailable"
        if value is None or pd.isna(value)
        else pd.Timestamp(value).strftime("%d %b %Y, %H:%M UTC")
    )


def _metrics_table(metrics: pd.DataFrame, task: str, key: str, *, active_only: bool = True):
    view = (
        metrics.loc[metrics.task.eq(task)].copy()
        if not metrics.empty and "task" in metrics
        else pd.DataFrame()
    )
    if view.empty:
        st.info("No saved metrics are available for this task.")
        return
    if active_only:
        view = ui.filter_models(view)
    if view.empty:
        st.info("No saved metrics for the active models and this task.")
        return
    counts = [
        c
        for c in ("event_count", "known_label_count", "total_driver_count", "coverage")
        if c in view
    ]
    if {"metric", "value"}.issubset(view):
        values = view.pivot_table(index="family", columns="metric", values="value", aggfunc="first")
        # Label coverage is specific to a metric; preserve it when rank and podium tasks differ.
        metadata = view.groupby("family", sort=False)[counts].first() if counts else pd.DataFrame()
        view = values.join(metadata).reset_index()
    if "family" in view:
        score = "mae" if "mae" in view else "log_loss" if "log_loss" in view else None
        if active_only and score and view[score].notna().any():
            winner = view.loc[view[score].idxmin()]
            name = FAMILIES.get(winner.family, winner.family)
            st.markdown(
                f":green-badge[Active selection winner] {name}"
                f" · {metric_label(score)}: {winner[score]:.3f}"
            )
        view["Model"] = view.family.map(
            lambda family: FAMILIES.get(
                family,
                FAMILIES.get(family.removesuffix("_uncalibrated"), family) + " (uncalibrated)",
            )
        )
        order = {f: i for i, f in enumerate(FAMILIES)}
        view["_order"] = view.family.map(
            lambda family: (
                order.get(family.removesuffix("_uncalibrated"), 9)
                + (0.1 if family.endswith("_uncalibrated") else 0)
            )
        )
        view = view.sort_values("_order").drop(columns=["_order", "family"])
        view = view[["Model", *[c for c in view if c != "Model"]]]
    labels = {
        "mae": "Rank MAE ↓",
        "rmse": "Rank RMSE ↓",
        "spearman": "Spearman ↑",
        "brier": "Brier score ↓",
        "brier_score": "Brier score ↓",
        "log_loss": "Log loss ↓",
        "accuracy": "Accuracy ↑",
        "f1": "F1 ↑",
        "precision": "Precision ↑",
        "recall": "Recall ↑",
        "ece": "Calibration error ↓",
        "top3_recall": "Top-three recall ↑",
        "event_count": "Events",
        "known_label_count": "Scored drivers",
        "total_driver_count": "Rostered drivers",
        "coverage": "Label coverage",
    }
    view = view.drop(columns=["task", "stage"], errors="ignore").rename(columns=labels)
    configs = {
        c: st.column_config.NumberColumn(metric_label(c), format="%.3f", help=metric_help(c))
        for c in view.select_dtypes(include="number")
        if c not in ("Events", "Scored drivers", "Rostered drivers")
    }
    if "Label coverage" in view:
        configs["Label coverage"] = st.column_config.NumberColumn(format="percent")
    _table(view, key=key, column_config=configs)


def _prediction_table(predictions: pd.DataFrame, task: str, family: str, *, spotlight: bool = True):
    data = fc.comparison_frame(predictions, task=task, models=ui.selected_models(), primary=family)
    if data.empty:
        st.info("This saved run has no predictions for this model and task.")
        return
    if data.attrs.get("primary") != family:
        st.caption(f"Primary output unavailable; spotlight uses {data.attrs.get('primary')}.")
    if task == "race" and spotlight:
        ui.podium_spotlight(data)
    ui.prediction_field(data, key=f"lab_predictions_{task}_{family}")
    known = int(data.actual_rank.notna().sum())
    st.caption(
        f"{len(data)} rostered drivers predicted; {known} have an official numeric rank "
        "for scoring. Unranked NC / DSQ entries stay in the forecast and do not receive "
        "an invented finish position."
    )
    st.download_button(
        "Download predictions (CSV)",
        ui.filter_grid(data).to_csv(index=False).encode(),
        file_name=f"{task}-{family}-predictions.csv",
        mime="text/csv",
        key=f"lab_download_{task}_{family}",
        icon=":material/download:",
        on_click="ignore",
    )


def _rank_chart(predictions: pd.DataFrame, task: str, family: str):
    data = ui.filter_grid(
        fc.comparison_frame(predictions, task=task, models=ui.selected_models(), primary=family)
    )
    if not data.empty:
        st.altair_chart(charts.rank_deltas(data), width="stretch")


def _headline_values(run: dict, task: str) -> pd.Series:
    """Saved headline score per family for one task (empty for unscored forecasts)."""
    metrics = _frame(run, "metrics")
    if not {"family", "task", "metric", "value"}.issubset(metrics):
        return pd.Series(dtype=float)
    key = HEADLINE_METRIC[task][0]
    names = {key, "position_mae"} if key == "mae" else {key}
    rows = metrics.loc[metrics.task.eq(task) & metrics.metric.isin(names)]
    rows = rows.assign(family=rows.family.replace(ui.MODEL_ALIASES))
    return rows.dropna(subset=["value"]).groupby("family", sort=False).value.first()


def _scoreboard(run: dict, task: str, *, key: str):
    """Headline comparison: each active model's saved score against the simple baseline.

    Uses stored target-event metrics only; the winner badge is text, not colour alone.
    """
    values = _headline_values(run, task)
    models = [m for m in ui.selected_models() if m in values.index]
    if not models:
        return False
    metric = HEADLINE_METRIC[task][0]
    baseline = values.get("baseline")
    best = min(models, key=lambda m: values[m])
    unit = "places" if metric == "mae" else ""
    best_text = f"{values[best]:.2f} {unit}".strip()
    name = MODEL_LABELS.get(best, best)
    if task == "podium":
        lead = f"**{name}** gave the sharpest podium chances (Brier {values[best]:.3f})"
    else:
        lead = f"**{name}** was closest: on average **{best_text}** off per driver"
    if baseline is not None and pd.notna(baseline):
        fmt = ".3f" if task == "podium" else ".2f"
        if values[best] < baseline:
            lead += f", beating the simple baseline ({baseline:{fmt}})."
        elif values[best] == baseline:
            lead += f", tied with the simple baseline ({baseline:{fmt}})."
        else:
            lead = (
                f"The simple baseline ({baseline:{fmt}}) beat every selected model this "
                f"weekend. Best model: **{name}** ({values[best]:{fmt}})."
            )
    st.markdown(lead)
    cards = [*models, *(["baseline"] if baseline is not None and pd.notna(baseline) else [])]
    help_text = (
        "Average number of places between the predicted and official position, over drivers "
        "with an official numeric rank. Lower is better."
        if metric == "mae"
        else "Mean squared gap between podium probability and the 0/1 outcome. Lower is better."
    )
    for col, family in zip(st.columns(len(cards)), cards, strict=True):
        value = values[family]
        is_best = family == best
        title = "Simple baseline" if family == "baseline" else MODEL_LABELS.get(family, family)
        delta = None
        if family != "baseline" and baseline is not None and pd.notna(baseline):
            fmt = "+.3f" if task == "podium" else "+.2f"
            delta = f"{value - baseline:{fmt}} vs baseline"
        if is_best:
            # Text marker (not colour alone) for the best selected model.
            delta = f"{delta} · best model" if delta else "Best model"
        state = "best" if is_best else "base" if family == "baseline" else "other"
        with col, st.container(key=f"f1_score_{state}_{key}_{family}"):
            st.metric(
                title,
                f"{value:.3f}" if task == "podium" else f"{value:.2f}",
                delta=delta,
                delta_color="inverse",
                delta_arrow="auto" if delta and delta[0] in "+-" else "off",
                help=(
                    f"The baseline is {BASELINE_COPY[task]}. " + help_text
                    if family == "baseline"
                    else help_text
                ),
            )
    return True


def _results_summary(run: dict, task: str, *, key: str):
    """Scoreboards for the task (and podium chances after qualifying)."""
    shown = False
    if not _headline_values(run, task).empty:
        st.markdown(f"#### How close each model got · {TASKS[task]} order")
        shown = _scoreboard(run, task, key=f"{key}_{task}")
    if task == "race" and not _headline_values(run, "podium").empty:
        st.markdown("#### Podium chances")
        shown = _scoreboard(run, "podium", key=f"{key}_podium") or shown
    return shown


def _task_results(run: dict, task: str, family: str, *, dataset=None):
    predictions = _frame(run, "predictions")
    if predictions.empty:
        st.info("No saved predictions are available.")
        return
    _results_summary(run, task, key="results")
    st.caption(
        "Historical replay: target outcomes were consulted during source review. "
        "These are exploratory results, not an untouched final holdout."
    )
    if not ui.full_telemetry():
        _executive_results(run, task, family)
        return
    data = fc.comparison_frame(predictions, task=task, models=ui.selected_models(), primary=family)
    if task == "race" and not data.empty:
        ui.podium_spotlight(data)
    labels = ["Full grid", "All scores & baselines"]
    if dataset is not None:
        labels.append("Audit trail")
    tabs = st.tabs(labels)
    with tabs[0]:
        st.caption(
            f"Ordered by {FAMILIES[family]} (spotlight) · "
            f"{len(ui.selected_models())} model(s) compared"
        )
        _prediction_table(predictions, task, family, spotlight=False)
    with tabs[1]:
        _metrics_table(_frame(run, "metrics"), task, f"lab_metrics_{task}")
        if task == "race":
            st.markdown("**Podium classification**")
            _metrics_table(_frame(run, "metrics"), "podium", "lab_metrics_podium")
        metrics = _frame(run, "metrics")
        if "family" in metrics and metrics.family.eq("baseline").any():
            with st.expander("Fixed-rule baseline reference"):
                st.caption("Simple rules provide a reference alongside the active learned models.")
                baselines = metrics.loc[metrics.family.eq("baseline")]
                _metrics_table(baselines, task, f"lab_baseline_{task}", active_only=False)
                if task == "race":
                    _metrics_table(baselines, "podium", "lab_baseline_podium", active_only=False)
    if dataset is not None:
        with tabs[2]:
            _run_details(run, dataset)


def _executive_results(run: dict, task: str, family: str):
    """Small visual summary from saved predictions; no audit tables or recalculated scores."""
    predictions = _frame(run, "predictions")
    field = fc.comparison_frame(predictions, task=task, models=ui.selected_models(), primary=family)
    data = ui.filter_grid(field)
    if data.empty:
        st.info("No saved predictions are available for this model and task.")
        return
    if task == "race":
        ui.podium_spotlight(field)
    st.markdown(f"**{FAMILIES[family]} · predicted front three**")
    for row in data.sort_values("predicted_rank").head(3).itertuples():
        st.html(
            ui.team_row(
                str(row.driver),
                str(row.team_id),
                f"P{int(row.predicted_rank):02d}",
                rank=int(row.predicted_rank),
            )
        )
    if task == "race":
        podium = ui.filter_grid(
            predictions.loc[predictions.task.eq("podium") & predictions.family.eq(family)]
        )
        if not podium.empty:
            st.markdown("**Highest independent podium probabilities**")
            for row in (
                podium.sort_values("podium_probability", ascending=False).head(3).itertuples()
            ):
                st.html(
                    ui.team_row(str(row.driver), str(row.team_id), ui.pct(row.podium_probability))
                )
            st.caption("Podium chances come from the separate classifier, not the race-order rank.")
    st.caption(
        "Want the whole grid, every score and the audit trail? Switch **Detail level** to "
        "**Full detail** in the sidebar."
    )


def _run_details(run: dict, dataset):
    ui.calibration_audit(
        _frame(run, "calibration_predictions"),
        family=st.session_state.get("_lab_family", "linear"),
        models=ui.selected_models(),
        key="lab_calibration_probabilities",
    )
    with st.expander("Development findings and saved configuration"):
        st.write(
            "Model settings are selected using earlier chronological development events. "
            "Target-event results are reported separately and are not used to choose settings."
        )
        development = _frame(run, "development_metrics")
        if development.empty:
            st.info(
                "This run has no separate development metrics. Check its saved protocol "
                "before interpreting comparisons."
            )
        else:
            selection = development.loc[development.stage.eq("development_selection")]
            for task in TASKS:
                if "task" in selection and selection.task.eq(task).any():
                    st.markdown(f"**{TASKS[task]} development results**")
                    _metrics_table(selection, task, f"lab_dev_{task}")
            calibration = development.loc[
                development.stage.eq("development_calibration_fit_diagnostic")
            ]
            if not calibration.empty:
                st.markdown("**Calibration-fit diagnostics**")
                st.caption(
                    "These events fit probability calibration and its decision threshold. "
                    "Their metrics are fit diagnostics, not independent validation results."
                )
                _metrics_table(calibration, "podium", "lab_calibration_diagnostics")
        st.json(run.get("config", {}), expanded=False)
    with st.expander("Training events, cutoffs and provenance"):
        training = _frame(run, "training_events")
        if not training.empty:
            _table(training, key="lab_training_events")
        st.write(
            "All history must be available before the target weekend. Qualifying forecasts use "
            "information available before qualifying; race forecasts additionally use "
            "the published qualifying classification. Target race results are labels only."
        )
        st.json(run.get("summary", {}), expanded=False)
        st.json(run.get("receipt", {}), expanded=False)
        st.caption(f"Dataset snapshot: {dataset.receipt.get('id', 'unknown')}")
        st.caption(f"Saved run: {Path(run['_path']).name}")


def _short_cutoff(value) -> str:
    if value is None or pd.isna(value):
        return "Unavailable"
    return pd.Timestamp(value).strftime("%d %b · %H:%M UTC")


def _empty_steps(season: int):
    """First-visit guidance: what to do, in order, and what will appear."""
    st.markdown("#### Get a result in three steps")
    steps = st.columns(3)
    copy = (
        ("1 · Pick a race", f"Choose any finished {season} Grand Prix in Race setup above."),
        ("2 · Choose the history", "Train on the last 6, last 12 or all earlier races."),
        (
            "3 · Run the test",
            "Saved results open instantly; a new combination trains on demand.",
        ),
    )
    for col, (title, body) in zip(steps, copy, strict=True):
        with col:
            ui.info_card(title, body)


def historical_test(*, _slots=None):
    title = "Test a race weekend"
    detail = (
        "Pick a finished Grand Prix, train on the races before it, and see how three model "
        "families and a simple baseline would have called it."
    )
    header, ribbon = _title(title, detail, _slots)
    model_controls = _sidebar_sections()
    with model_controls:
        family, task = _model_filters()
    box, (season_col, event_col, window_col) = _setup_panel()
    with season_col:
        season = _season_selector()
    dataset = _load_dataset(active=True, season=season)
    if dataset is None:
        with box:
            _sidebar_refresh()
        _render_context(ribbon, season=season, family=family, task=task)
        return
    fallback_ids = dataset.receipt.get("api_fallback_events", [])
    if fallback_ids:
        st.warning(
            f"{len(fallback_ids)} {season} races use captured FastF1/Jolpica API results. "
            "API publication timing and some entrant fields are historical reconstructions.",
            icon=":material/info:",
        )
        if ui.full_telemetry():
            with st.expander("API fallback sources and timing assumptions"):
                st.write(dataset.receipt["fallback_semantics"])
                _table(pd.read_parquet(dataset.root / "fallbacks.parquet"), key="lab_api_fallbacks")
    exclusions_path = dataset.root / "exclusions.parquet"
    if season != 2026 and exclusions_path.exists():
        exclusions = pd.read_parquet(exclusions_path)
        if not exclusions.empty:
            st.warning(
                f"{len(exclusions)} {season} races are unavailable because required source "
                "documents could not be verified. Training uses the eligible races shown.",
                icon=":material/info:",
            )
            if ui.full_telemetry():
                with st.expander("Unavailable races and source details"):
                    _table(exclusions, key="lab_historical_exclusions")
    completed = (
        dataset.events.qualifying_available_at.notna() & dataset.events.race_available_at.notna()
    )
    events = dataset.events.loc[completed & dataset.events.year.eq(season)].sort_values("round")
    pending = dataset.events.loc[
        ~completed
        & (pd.to_datetime(dataset.events.weekend_start, utc=True) <= pd.Timestamp.now(tz="UTC"))
    ]
    if not pending.empty:
        st.info(
            "Awaiting final qualifying and race classifications: "
            + ", ".join(pending.name)
            + ". Refresh again when the results are published.",
            icon=":material/hourglass_top:",
        )
    event_ids = events.event_id.tolist()
    as_of = _short_cutoff(dataset.receipt.get("as_of")) if dataset.receipt.get("as_of") else None
    if not event_ids:
        st.info("No reviewed completed events are eligible yet.")
        with box:
            _sidebar_refresh(as_of=as_of)
        _render_context(ribbon, dataset=dataset, season=season, family=family, task=task)
        return
    default = "2026-19" if "2026-19" in event_ids else event_ids[-1]
    if st.session_state.get("lab_target_event") not in event_ids:
        st.session_state["lab_target_event"] = default
    if st.session_state.get("lab_active_season") != season:
        active = st.session_state.get("lab_active_run")
        saved = _default_saved(st.session_state["lab_target_event"], dataset.receipt["id"])
        st.session_state["lab_active_run"] = active or (saved["_path"] if saved else None)
        st.session_state["lab_replay_artifact"] = st.session_state["lab_active_run"]
        st.session_state["lab_active_season"] = season
    with event_col:
        target = st.selectbox(
            "Race",
            event_ids,
            key="lab_target_event",
            format_func=_event_option(events),
            help="Only finished weekends with official qualifying and race results are listed.",
        )
    with window_col:
        window = st.selectbox(
            "Train on",
            list(WINDOWS),
            key="lab_history_window",
            format_func=_window_label,
            help="How many earlier races from the same season the models learn from.",
        )
    earlier = event_ids.index(target)
    available = earlier if WINDOWS[window] is None else min(earlier, WINDOWS[window])
    with box:
        info_col, run_col = st.columns([3, 1.15], vertical_alignment="center")
        with info_col:
            summary = st.empty()
            draft_note = st.empty()
        with run_col:
            with st.form("lab_historical_form", border=False):
                submitted = st.form_submit_button(
                    "Run historical test",
                    type="primary",
                    key="lab_run",
                    width="stretch",
                    icon=":material/play_arrow:",
                )
    with summary:
        if earlier == 0:
            st.caption(
                ":material/warning: The season opener has no earlier races to learn from. "
                "Pick a later race."
            )
        else:
            st.caption(
                f"Learns from **{available}** earlier {season} race"
                f"{'s' if available != 1 else ''}. This race and everything after it stay "
                "hidden from training. Practice pace and weather are not used."
            )
    if submitted:
        from f1ml_real.lab import cached_run

        _, run_root = paths()
        try:
            with st.spinner(
                f"Training nine models on {available} earlier races and sealing predictions…"
            ):
                result = cached_run(dataset.root, target, run_root, last_n=WINDOWS[window])
            st.session_state["lab_active_run"] = str(result)
            st.success(
                "Historical test saved. Reopening this configuration reuses the verified run.",
                icon=":material/check_circle:",
            )
        except (ValueError, OSError, KeyError, RuntimeError) as exc:
            st.error(f"This test could not run: {exc}")
            st.caption(
                "Choose a later completed event or a longer history window when there "
                "is too little development history."
            )
    with box:
        saved_col, refresh_col = st.columns(2)
        with saved_col:
            with st.expander("Saved runs", icon=":material/history:"):
                run = _replay_selector(dataset, season, submitted=submitted)
        with refresh_col:
            _sidebar_refresh(as_of=as_of)
    if run is not None and run.get("config", {}).get("source_snapshot_id") != dataset.receipt["id"]:
        st.error(
            "This saved run uses a different dataset snapshot. "
            "Run a test against the active reviewed dataset."
        )
        run = None
    with header:
        ui.brand_header(title, detail, dataset_version=dataset.root.name, verified_run=run)
    inspected_event = (run or {}).get("config", {}).get("event_id")
    inspected_event = inspected_event or (run or {}).get("config", {}).get("target_event_id")
    pending = None
    if run is not None and (
        inspected_event != target or run.get("config", {}).get("last_n") != WINDOWS[window]
    ):
        ready = _has_saved(target, WINDOWS[window], dataset.receipt["id"])
        pending = (
            f"{_event_label(events, target)} · {window}. Not shown yet; press Run historical "
            + ("test to open its saved results." if ready else "test to train it.")
        )
        with draft_note:
            st.caption(
                ":material/pending: Results below are still for the previous selection. "
                + (
                    "Saved results exist for this choice and will open instantly."
                    if ready
                    else "This choice hasn't been run yet."
                )
            )
    _render_context(
        ribbon,
        run=run,
        dataset=dataset,
        season=season,
        event=_event_label(events, target),
        family=family,
        task=task,
        pending=pending,
        unverified_artifact=st.session_state.get("lab_active_run") if run is None else None,
    )
    if run is None:
        st.info(
            "Choose a completed event and run a historical test to see predictions "
            "and baseline comparisons.",
            icon=":material/flag:",
        )
        _empty_steps(season)
        st.markdown("#### The three model families")
        _model_strip()
        return
    config = run.get("config", {})
    event_id = config.get("event_id", config.get("target_event_id", default))
    target_row = events.loc[events.event_id.eq(event_id)]
    event = target_row.iloc[0] if len(target_row) else None
    training = _frame(run, "training_events")
    name = str(event["name"]) if event is not None else _event_label(events, event_id)
    st.markdown(f"## {name}")
    window_text = _window_label(
        next((k for k, v in WINDOWS.items() if v == config.get("last_n")), "")
    )
    st.caption(
        f"{season}"
        + (f" · Round {int(event['round'])}" if event is not None else "")
        + (f" · trained on {window_text.lower()}" if window_text else "")
        + " · saved historical replay"
        + (
            " · target source: captured FastF1/Jolpica API fallback"
            if event_id in fallback_ids
            else ""
        )
    )
    count = training.event_id.nunique() if "event_id" in training else len(training)
    with st.container(key="f1_cutoffs"):
        cols = st.columns(3)
    with cols[0]:
        ui.metric_card(
            "Training races",
            count,
            tooltip="Eligible races before the saved target weekend in the selected season.",
        )
    with cols[1]:
        st.metric(
            "Qualifying cutoff",
            _short_cutoff(event.qualifying_cutoff if event is not None else None),
            help="Qualifying forecasts only use information published before this time.",
        )
    with cols[2]:
        st.metric(
            "Race cutoff",
            _short_cutoff(event.race_cutoff if event is not None else None),
            help="Race and podium forecasts also use the qualifying result, published "
            "before this time.",
        )
    if config.get("insufficient_development_history"):
        st.warning(
            "This training window has too few earlier events for separate development selection. "
            "Treat these settings as a fixed exploratory comparison."
        )
    task = _task_switch()
    _task_results(run, task, family, dataset=dataset)


def live_data_path() -> Path:
    path = Path(os.environ.get("F1ML_LIVE_DATA", "data/live"))
    return path if path.is_absolute() else project_root() / path


def _refresh_button(*, key: str, base: Path | None = None, state: Path | None = None):
    from f1ml_real.refresh import refresh

    if st.button(
        "Refresh FastF1 data",
        type="primary",
        key=key,
        help="Reload event options from the current API calendar and published results.",
        width="stretch",
        icon=":material/sync:",
    ):
        try:
            with st.spinner("Loading the latest event data…"):
                refresh(
                    base if base is not None else paths()[0],
                    state if state is not None else live_data_path(),
                )
            st.session_state["_lab_refresh_notice"] = (
                "Data refreshed. Event options and Data & methodology are up to date."
            )
            st.rerun()
        except Exception as exc:
            st.error(f"Refresh failed; the previous snapshot is still available: {exc}")


def _sidebar_refresh(*, live_page: bool = False, as_of: str | None = None):
    """Explicit acquisition action on the main canvas; ordinary filters stay read-only.

    On the live page refreshing is the main task, so it gets a full card. Elsewhere it is
    a secondary action tucked into a labelled expander.
    """
    season = 2026 if live_page else _selected_season()
    base, state = historical_paths(season)
    if live_page:
        with st.container(border=True, key="f1_refresh_card"):
            left, right = st.columns([3, 1.15], vertical_alignment="center")
            with left:
                st.markdown("**Step 1 · Get the latest race weekends**")
                st.caption(
                    "Downloads the current FastF1 calendar and results, plus FIA entry lists "
                    "and classifications. Nothing is trained until you press Run forecast."
                )
                st.caption(f"Refresh season: {season}")
            with right:
                _refresh_button(key="lab_refresh", base=base, state=state)
    else:
        label = "Update race data" + (f" · last updated {as_of}" if as_of else "")
        with st.expander(label, icon=":material/sync:"):
            st.caption(f"Refresh season: {season}")
            st.caption(
                "Pulls newly published weekends from FastF1 and the FIA. Nothing is trained."
            )
            _refresh_button(key="lab_refresh", base=base, state=state)
    if notice := st.session_state.pop("_lab_refresh_notice", None):
        st.success(notice, icon=":material/check_circle:")


def live_forecast(*, _with_refresh: bool = False, _slots=None):
    from f1ml_real.lab import cached_forecast
    from f1ml_real.refresh import current_dataset, forecast_tasks, upcoming_events

    title = "Refresh & predict"
    detail = (
        "Pull the latest calendar and results, then predict sessions that haven't happened yet."
    )
    header, ribbon = _title(title, detail, _slots)
    model_controls = _sidebar_sections()
    with model_controls:
        family, task_filter = _model_filters(live=True)
    if _with_refresh:
        _sidebar_refresh(live_page=True)
    base, runs = paths()
    state = live_data_path()
    try:
        path = current_dataset(base, state)
        dataset = _dataset(str(path), (path / "receipt.json").stat().st_mtime_ns)
    except (ValueError, OSError, KeyError) as exc:
        st.error(f"The refreshed dataset could not be verified: {exc}")
        return
    with header:
        ui.brand_header(title, detail, dataset_version=dataset.root.name)
    _render_context(
        ribbon,
        dataset=dataset,
        season=2026,
        event="Live discovery",
        family=family,
        task=task_filter,
    )
    if path == base:
        st.info(
            "Press Refresh FastF1 data to discover the latest race weekends.",
            icon=":material/travel_explore:",
        )
        st.caption(
            "Live discovery uses the 2026 season. After a refresh, upcoming weekends with a "
            "published entry list or the latest known field can be forecast here."
        )
        return
    st.caption(f"Last refreshed: {_cutoff(dataset.receipt.get('as_of'))}")
    events = upcoming_events(dataset)
    ids = events.event_id.tolist()
    if not ids:
        st.info("There are no upcoming sessions in this calendar. Refresh to check for updates.")
        return
    if st.session_state.get("lab_live_event") not in ids:
        st.session_state["lab_live_event"] = next(
            (eid for eid in ids if forecast_tasks(dataset, eid)), ids[0]
        )
    box, (event_col, window_col, run_col) = _setup_panel((2.4, 1.7, 1.25))
    with event_col:
        target = st.selectbox(
            "Weekend",
            ids,
            key="lab_live_event",
            format_func=_event_option(events),
            help=(
                "Weekends with an unstarted qualifying or race session. "
                "Finished weekends are in Historical test."
            ),
        )
    _render_context(
        ribbon,
        dataset=dataset,
        season=2026,
        event=_event_label(events, target),
        family=family,
        task=task_filter,
    )
    event = dataset.event(target)
    tasks = forecast_tasks(dataset, target)
    with window_col:
        window = st.selectbox(
            "Train on", list(WINDOWS), key="lab_live_window", format_func=_window_label
        )
    with run_col:
        submitted = st.button(
            "Run forecast",
            key="lab_live_run",
            disabled=not tasks,
            type="primary",
            width="stretch",
            icon=":material/play_arrow:",
        )
    with box:
        if tasks:
            st.success(
                "Ready to predict: " + ", ".join(TASKS[task] for task in tasks) + ".",
                icon=":material/check_circle:",
            )
        else:
            st.info(
                "No forecast is ready. Refresh for the latest inputs. After qualifying starts, "
                "race forecasts wait for its final results.",
                icon=":material/hourglass_top:",
            )
        if tasks:
            early = "qualifying" in tasks
            cutoff = event.qualifying_cutoff if early else event.race_cutoff
            field = dataset.forecast_field(target, "qualifying" if early else "race", cutoff)
            if field.field_source.iloc[0] == "latest_known_field_assumption":
                source = dataset.event(field.field_source_event_id.iloc[0])
                st.info(
                    f"Provisional field: using the latest known entrants from {source['name']}. "
                    "Refresh when the event entry list is published to include any changes."
                )
            if early:
                st.caption(
                    "Race and podium forecasts use each model's own predicted qualifying "
                    "positions. These are conditional forecasts; qualifying uncertainty "
                    "and grid penalties are not modeled."
                )
            else:
                st.caption("Race and podium forecasts use the published qualifying results.")
        with st.expander("Schedule and refresh details", icon=":material/event:"):
            st.write(f"Qualifying: {_cutoff(event.get('qualifying_start'))}")
            st.write(f"Race: {_cutoff(event.get('race_start'))}")
            for status in json.loads((path / "refresh_status.json").read_text()):
                if status["event_id"] == target:
                    for issue in status["issues"]:
                        st.warning(issue)
            st.caption(f"Verified snapshot: {dataset.receipt['id']}")
    if submitted:
        try:
            with st.spinner("Training on earlier events and saving predictions…"):
                result = cached_forecast(path, target, runs, last_n=WINDOWS[window])
            st.session_state["lab_live_run_path"] = str(result)
        except (ValueError, OSError, KeyError, RuntimeError) as exc:
            st.error(f"The prediction could not run: {exc}")
    active = st.session_state.get("lab_live_run_path")
    run = _read_saved(active) if active else None
    if run is None or run["summary"]["target_event_id"] != target:
        run = _default_saved(target, dataset.receipt["id"])
    if run is None or run["summary"]["target_event_id"] != target:
        return
    if run["summary"].get("last_n") != WINDOWS[window]:
        st.info("Run forecast to generate predictions for the selected training window.")
        return
    st.markdown(f"## {event.get('name', _event_label(events, target))}")
    st.caption(
        f"Saved predictions · {run['summary']['training_event_count']} earlier training events · "
        f"{run['summary']['evaluation_status']}"
    )
    if run["summary"]["source_snapshot_id"] != dataset.receipt["id"]:
        st.warning("These saved predictions use an earlier refresh snapshot.")
    if run["summary"].get("qualifying_source") == "predicted":
        st.info("Saved race and podium predictions use predicted qualifying positions.")
    elif "race" in set(_frame(run, "predictions").task):
        st.caption("Saved race and podium predictions use official qualifying results.")
    with header:
        ui.brand_header(title, detail, dataset_version=dataset.root.name, verified_run=run)
    predictions = _frame(run, "predictions")
    task_filter = _task_switch(live=True, available=set(predictions.task))
    _render_context(ribbon, run=run, dataset=dataset, family=family, task=task_filter)
    for task in ("qualifying", "race"):
        if task in set(predictions.task) and task_filter in ("all", task):
            st.markdown(f"### {TASKS[task]}")
            if run["summary"]["mode"] == "live_forecast":
                if ui.full_telemetry():
                    _prediction_table(predictions, task, family)
                else:
                    _executive_results(run, task, family)
            else:
                _task_results(run, task, family)


def _model_strip():
    families = [family for family in ui.selected_models() if family in MODEL_COPY]
    for col, family in zip(st.columns(len(families)), families, strict=True):
        with col:
            with st.container(border=True):
                st.markdown(f"**{FAMILIES[family]}**")
                st.caption(MODEL_COPY[family][0])


def three_models(*, _with_refresh: bool = False, _slots=None):
    title = "Three models, one fair comparison"
    detail = (
        "Every family gets the same data, the same cutoffs and the same scoring. The methods "
        "match the regression, trees and ensembles covered in class."
    )
    header, ribbon = _title(title, detail, _slots)
    model_controls = _sidebar_sections()
    with model_controls:
        selected_family, task = _model_filters()
    box, (season_col, event_col, window_col) = _setup_panel()
    with season_col:
        season = _season_selector()
    dataset = _load_dataset(active=True, season=season)
    with box:
        saved_col, refresh_col = st.columns(2)
    with saved_col:
        saved_slot = st.expander("Saved runs", icon=":material/history:")
    run = _shared_replay_controls(
        dataset, season, cols=(event_col, window_col), saved_slot=saved_slot
    )
    if _with_refresh:
        with refresh_col:
            _sidebar_refresh()
    active = st.session_state.get("lab_active_run")
    with header:
        ui.brand_header(title, detail, verified_run=run)
    _render_context(
        ribbon,
        run=run,
        dataset=dataset,
        season=season,
        family=selected_family,
        task=task,
        unverified_artifact=active if run is None else None,
    )
    families = [family for family in MODEL_COPY if family in ui.selected_models()]
    for col, family in zip(st.columns(len(families)), families, strict=True):
        estimator, course, explanation = MODEL_COPY[family]
        with col, st.container(border=True, key=f"f1_model_card_{family}"):
            st.subheader(FAMILIES[family])
            st.markdown(f"**{estimator}**")
            st.write(explanation)
            st.caption(f"Course connection: {course}.")
    if run:
        config = run.get("config", {})
        event_id = config.get("event_id", config.get("target_event_id"))
        where = _event_label(dataset.events, event_id) if dataset and event_id else "this run"
        st.markdown(f"### How they compared · {where}")
        task = _task_switch()
        if not _results_summary(run, task, key="models"):
            st.info("This saved run has no target-event scores for the selected models.")
        if ui.full_telemetry():
            with st.expander("All scores and baselines"):
                _metrics_table(_frame(run, "metrics"), task, f"lab_metrics_{task}")
                if task == "race":
                    st.markdown("**Podium classification**")
                    _metrics_table(_frame(run, "metrics"), "podium", "lab_metrics_podium")
        st.caption("The driver-by-driver grid for this run is on the **Historical test** tab.")
    if not ui.full_telemetry():
        st.caption("Regressors predict order; a separate classifier estimates podium probability.")
        return
    st.subheader("What each family predicts")
    _table(
        pd.DataFrame(
            [
                {
                    "Task": "Qualifying order",
                    "Estimator": "Regressor",
                    "Available information": (
                        "Earlier driver and team form; independent current entry list"
                    ),
                    "Evaluation": "Known-rank MAE, RMSE and rank correlation",
                },
                {
                    "Task": "Race order",
                    "Estimator": "Regressor",
                    "Available information": (
                        "Earlier form plus published current qualifying result"
                    ),
                    "Evaluation": "Known-rank MAE, RMSE and rank correlation",
                },
                {
                    "Task": "Race podium",
                    "Estimator": "Classifier",
                    "Available information": "Same post-qualifying cutoff as the race regressor",
                    "Evaluation": "Probability quality and classification metrics",
                },
            ]
        ),
        key="lab_model_tasks",
    )
    st.write(
        "Every family uses the same eligible evaluation drivers and events. Simple recent-form "
        "and qualifying-order baselines make it clear when a learned model adds value."
    )
    st.caption(
        "Predicted positions are field rankings of regression scores, not predicted lap times. "
        "Podium probabilities come from separate classifiers, "
        "not from a softmax of the predicted order."
    )


def data_methodology(*, _with_refresh: bool = False, _slots=None):
    title = "Real data and the rules"
    detail = "Public source records, explicit cutoffs and an honest account of what is missing."
    header, ribbon = _title(title, detail, _slots)
    model_controls = _sidebar_sections()
    with model_controls:
        family, _ = _model_filters()
    box, (season_col, note_col) = _setup_panel((1, 4.1))
    with season_col:
        season = _season_selector()
    with note_col:
        st.caption(
            "Coverage below is for the whole season. Model and grid filters in the sidebar "
            "are kept for the other tabs; they don't change these source counts."
        )
    dataset = _load_dataset(active=True, season=season)
    if _with_refresh:
        as_of = (
            _short_cutoff(dataset.receipt.get("as_of"))
            if dataset is not None and dataset.receipt.get("as_of")
            else None
        )
        with box:
            _sidebar_refresh(as_of=as_of)
    if dataset is None:
        return
    with header:
        ui.brand_header(title, detail, dataset_version=dataset.root.name)
    _render_context(
        ribbon,
        dataset=dataset,
        season=season,
        event="Source coverage",
        family=family,
        task="Coverage (filters retained)",
    )
    events, rosters, labels, receipt = (
        dataset.events,
        dataset.rosters,
        dataset.labels,
        dataset.receipt,
    )
    st.write(
        "Historical datasets use FIA final classifications and independent entry lists where "
        "verified, with the FastF1 calendar and stable public driver identities. Older seasons "
        "also contain explicitly labeled captured FastF1/Jolpica API fallback results. "
        "No synthetic observations or forecasts are used in these tests."
    )
    if receipt.get("api_fallback_events"):
        st.warning(receipt["fallback_semantics"], icon=":material/info:")
    st.caption(f"Last refreshed: {_cutoff(receipt.get('as_of'))}")
    completed = events.qualifying_available_at.notna() & events.race_available_at.notna()
    counts = st.columns(3)
    counts[0].metric("Completed events", events.loc[completed].event_id.nunique())
    counts[1].metric("Season", str(receipt.get("year", 2026)))
    counts[2].metric(
        "Source documents",
        len(dataset.documents),
        help="Official entry lists and classifications retained with URLs and hashes.",
    )
    if not ui.full_telemetry():
        left, right = st.columns(2)
        with left:
            ui.info_card(
                "Earlier races only", "History excludes the target weekend and later events."
            )
        with right:
            ui.info_card(
                "Source gaps remain visible", "Missing practice pace and weather are excluded."
            )
        st.info(
            "These are exploratory historical tests. Outcomes were consulted during development "
            "and source review; none are presented as an untouched final holdout."
        )
        return
    coverage_tab, features_tab, method_tab, sources_tab = st.tabs(
        ["Coverage", "Features & gaps", "How a test works", "Sources"]
    )
    with coverage_tab:
        coverage = []
        for _, event in events.sort_values(["year", "round"]).iterrows():
            for task in ("qualifying", "race"):
                rs = rosters.loc[rosters.event_id.eq(event.event_id) & rosters.task.eq(task)]
                ls = labels.loc[labels.event_id.eq(event.event_id) & labels.task.eq(task)]
                coverage.append(
                    {
                        "Event": _event_label(events, event.event_id),
                        "Session": TASKS[task],
                        "Status": "Final classification available"
                        if pd.notna(event[f"{task}_available_at"])
                        else "Awaiting final classification",
                        "Rostered drivers": rs.driver_id.nunique(),
                        "Known numeric ranks": int(ls.official_position.notna().sum()),
                        "Known race-podium labels": int(ls.podium.notna().sum())
                        if task == "race"
                        else None,
                        "Cutoff (UTC)": _cutoff(event[f"{task}_cutoff"]),
                    }
                )
        st.caption("One row per session: who was entered and how many official ranks exist.")
        _table(pd.DataFrame(coverage), key="lab_source_coverage", height=450)
    with features_tab:
        _table(
            pd.DataFrame(
                [
                    {
                        "Feature group": "Previous official results",
                        "Status": "Observed",
                        "Treatment": "Shifted driver and team form from earlier eligible events",
                    },
                    {
                        "Feature group": "Current entry list",
                        "Status": "Observed / API reconstruction"
                        if receipt.get("api_fallback_events")
                        else "Observed",
                        "Treatment": "Independent FIA entry lists, or labeled API entrant fields",
                    },
                    {
                        "Feature group": "Current qualifying result",
                        "Status": "Observed for race only",
                        "Treatment": "FIA publication cutoff, or the disclosed API timing proxy",
                    },
                    {
                        "Feature group": "Practice and qualifying lap pace",
                        "Status": "Unavailable in this core snapshot",
                        "Treatment": "Excluded; no invented laps or median-filled pace columns",
                    },
                    {
                        "Feature group": "Weather",
                        "Status": "Unavailable in this core snapshot",
                        "Treatment": "Excluded",
                    },
                ]
            ),
            key="lab_feature_coverage",
        )
        st.caption(
            "The FastF1 timing archive returned HTTP 401 during acquisition. Missing pace "
            "information also affects the real session-data path; it is not confined to the "
            "archived fixture. Official result documents allow the core historical experiment "
            "to run independently."
        )
    with method_tab:
        st.markdown(
            "1. Select completed events by source availability, before inspecting model "
            "performance.\n"
            "2. Build history from earlier weekends only, using source publication times "
            "or disclosed API timing proxies.\n"
            "3. Select settings on earlier chronological development events. "
            "Fit preprocessing on each training fold.\n"
            "4. Save the selected configuration and target predictions before scoring target "
            "labels.\n"
            "5. Report development and target-event findings separately, including baseline "
            "losses."
        )
        st.info(
            "These are exploratory historical tests. Outcomes including Mexico were already "
            "consulted during development and source review, so none are presented as an "
            "untouched final holdout."
        )
        st.write(
            "Historical tests support 2024, 2025 and 2026. Training uses the previous 6, "
            "previous 12 or all eligible earlier races from the target season. Each season has "
            "its own FIA evidence and labeled API reconstructions."
        )
        st.write(
            "An NC or disqualified driver can lack an official numeric rank. The driver remains "
            "in the prediction field; rank metrics use known ranks only and report their counts. "
            "A missing source label stays unknown. Publication times are conservative bounds, "
            "not measured session-end times."
        )
    with sources_tab:
        docs = dataset.documents.copy()
        columns = [
            c for c in ("event_id", "task", "purpose", "title", "published_at", "url") if c in docs
        ]
        _table(
            docs[columns],
            key="lab_source_documents",
            column_config={"url": st.column_config.LinkColumn("Official source")},
        )
        excluded_path = dataset.root / "exclusions.parquet"
        if excluded_path.exists() and len(excluded := pd.read_parquet(excluded_path)):
            st.markdown("**Source availability details**")
            _table(excluded, key="lab_source_exclusions")
        with st.expander("Dataset receipt (JSON)"):
            st.json(receipt, expanded=False)
    st.caption(
        "Downloads and source preparation run outside normal interface reruns. Only pressing "
        "Run historical test can start model fitting. Saved runs are verified and reused."
    )


def main():
    st.set_page_config(
        page_title="F1 Model Lab · Pit Wall",
        page_icon="🏁",
        layout="wide",
        initial_sidebar_state="auto",
    )
    theme.inject()
    st.logo(str(Path(__file__).parent / "assets" / "logo.svg"), size="medium")
    ui.persist_filters()
    pages = {
        "Historical test": historical_test,
        "Refresh & predict": live_forecast,
        "Three models": three_models,
        "Data & methodology": data_methodology,
    }
    # Navigation leads the page; the page title and the sticky context ribbon follow it.
    page = ui.top_navigation(list(pages))
    header = st.empty()
    with st.container(key="f1_context_ribbon"):
        ribbon = st.empty()
    kwargs = {"_slots": (header, ribbon)}
    if page != "Historical test":
        kwargs["_with_refresh"] = True
    pages[page](**kwargs)


if __name__ == "__main__":
    main()
