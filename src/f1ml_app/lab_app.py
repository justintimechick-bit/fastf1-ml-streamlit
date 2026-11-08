"""Real historical tests: reading a page never fits models or downloads data."""

from __future__ import annotations

import html
import os
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from f1ml_app import theme
from f1ml_app.settings import project_root

FAMILIES = {
    "linear": "Linear models",
    "random_forest": "Random forest",
    "gradient_boosting": "Gradient boosting",
    "baseline": "Simple baseline",
}
WINDOWS = {"All earlier 2026 events": None, "Most recent 12 events": 12, "Most recent 6 events": 6}
TASKS = {"qualifying": "Qualifying", "race": "Race", "podium": "Podium"}
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

    return load_run(Path(path))


def _load_dataset():
    path, _ = paths()
    receipt = path / "receipt.json"
    if not receipt.exists():
        st.warning("The reviewed real dataset is not installed in this app yet.")
        st.caption(
            "Prepare and verify the source snapshot outside Streamlit, "
            "then add it to the deployment."
        )
        return None
    try:
        return _dataset(str(path), receipt.stat().st_mtime_ns)
    except (ImportError, ValueError, OSError, KeyError) as exc:
        st.error(f"The real dataset could not be verified: {exc}")
        return None


def _read_saved(path: Path | str) -> dict | None:
    path = Path(path)
    markers = [path / name for name in ("receipt.json", "summary.json", "config.json")]
    stamp = max((p.stat().st_mtime_ns for p in markers if p.exists()), default=0)
    try:
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
    return f"R{int(event['round']):02d} · {event['name']}"


def _frame(run: dict, key: str) -> pd.DataFrame:
    value = run.get(key)
    return value.copy() if isinstance(value, pd.DataFrame) else pd.DataFrame(value or [])


def _table(frame: pd.DataFrame, *, key: str, **kwargs):
    st.dataframe(frame, hide_index=True, width="stretch", key=key, **kwargs)


def _title(title: str, detail: str):
    st.html(
        '<div class="lab-heading"><h1>'
        + html.escape(title)
        + "</h1><p>"
        + html.escape(detail)
        + "</p></div>"
    )


def _cutoff(value) -> str:
    return (
        "Unavailable"
        if value is None or pd.isna(value)
        else pd.Timestamp(value).strftime("%d %b %Y, %H:%M UTC")
    )


def _metrics_table(metrics: pd.DataFrame, task: str, key: str):
    view = (
        metrics.loc[metrics.task.eq(task)].copy()
        if not metrics.empty and "task" in metrics
        else pd.DataFrame()
    )
    if view.empty:
        st.info("No saved metrics are available for this task.")
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
        c: st.column_config.NumberColumn(format="%.3f")
        for c in view.select_dtypes(include="number")
        if c not in ("Events", "Scored drivers", "Rostered drivers")
    }
    if "Label coverage" in view:
        configs["Label coverage"] = st.column_config.NumberColumn(format="percent")
    _table(view, key=key, column_config=configs)


def _prediction_table(predictions: pd.DataFrame, task: str, family: str):
    data = predictions.loc[predictions.task.eq(task) & predictions.family.eq(family)].copy()
    if data.empty:
        st.info("This saved run has no predictions for this model and task.")
        return
    data = data.sort_values("predicted_rank")
    if task == "race":
        # Race order and podium probability come from separate fitted estimators.
        join_keys = [c for c in ("event_id", "driver_id", "family") if c in data]
        podium = predictions.loc[predictions.task.eq("podium") & predictions.family.eq(family)]
        if not podium.empty:
            columns = [
                c for c in ("podium_probability", "podium_prediction", "threshold") if c in podium
            ]
            data = data.drop(columns=columns, errors="ignore").merge(
                podium[join_keys + columns], on=join_keys, how="left", validate="one_to_one"
            )
    columns = {
        "predicted_rank": "Predicted rank",
        "driver": "Driver",
        "team_id": "Team",
        "actual_rank": "Official rank",
        "outcome": "Official status",
    }
    if task == "race":
        columns.update({"podium_probability": "Podium chance", "actual_podium": "Actual podium"})
    if {"actual_rank", "predicted_rank"}.issubset(data):
        data["Rank error"] = (data.predicted_rank - data.actual_rank).abs()
    view = data[
        [c for c in columns if c in data] + (["Rank error"] if "Rank error" in data else [])
    ].rename(columns=columns)
    if "Official status" in view:
        statuses = {
            "ordered": "Classified",
            "dsq": "DSQ",
            "non_classified": "NC",
            "unknown": "Unknown",
        }
        view["Official status"] = (
            view["Official status"].map(statuses).fillna(view["Official status"])
        )
    if "Actual podium" in view:
        view["Actual podium"] = view["Actual podium"].map({0: False, 1: True}).astype("boolean")
    config = {
        "Predicted rank": st.column_config.NumberColumn(format="%d"),
        "Official rank": st.column_config.NumberColumn(
            format="%d", help="Blank means no official numeric position was printed."
        ),
        "Rank error": st.column_config.NumberColumn(format="%.1f"),
        "Podium chance": st.column_config.NumberColumn(
            format="percent", help="Independent probabilities; they need not sum to three."
        ),
        "Actual podium": st.column_config.CheckboxColumn(),
    }
    _table(view, key=f"lab_predictions_{task}_{family}", column_config=config, height="content")
    known = int(data.actual_rank.notna().sum()) if "actual_rank" in data else 0
    st.caption(
        f"{len(data)} rostered drivers predicted; {known} have an official numeric rank "
        "for scoring. Unranked NC / DSQ entries stay in the forecast and do not receive "
        "an invented finish position."
    )
    st.download_button(
        "Download predictions",
        data.to_csv(index=False).encode(),
        file_name=f"{task}-{family}-predictions.csv",
        mime="text/csv",
        key=f"lab_download_{task}_{family}",
    )


def _rank_chart(predictions: pd.DataFrame, task: str, family: str):
    data = predictions.loc[predictions.task.eq(task) & predictions.family.eq(family)].copy()
    if not {"actual_rank", "predicted_rank", "driver"}.issubset(data):
        return
    data = data.dropna(subset=["actual_rank", "predicted_rank"])
    if data.empty:
        return
    limit = max(float(data.actual_rank.max()), float(data.predicted_rank.max())) + 1
    line = (
        alt.Chart(pd.DataFrame({"x": [1, limit], "y": [1, limit]}))
        .mark_line(color=theme.tokens()["muted"], strokeDash=[4, 4])
        .encode(x=alt.X("x:Q", title="Official rank"), y=alt.Y("y:Q", title="Predicted rank"))
    )
    dots = (
        alt.Chart(data)
        .mark_circle(size=80, color=theme.tokens()["series"][0])
        .encode(
            x=alt.X("actual_rank:Q", title="Official rank", scale=alt.Scale(domain=[0, limit])),
            y=alt.Y("predicted_rank:Q", title="Predicted rank", scale=alt.Scale(domain=[0, limit])),
            tooltip=[
                alt.Tooltip("driver:N", title="Driver"),
                alt.Tooltip("actual_rank:Q", title="Official"),
                alt.Tooltip("predicted_rank:Q", title="Predicted"),
            ],
        )
    )
    st.altair_chart((line + dots).properties(height=250), width="stretch")
    st.caption(
        "Points on the dashed line match the official rank. Only known numeric ranks are plotted."
    )


def _task_results(run: dict, task: str, family: str):
    predictions = _frame(run, "predictions")
    if predictions.empty:
        st.info("No saved predictions are available.")
        return
    st.subheader("Target-event results")
    st.caption(
        "Historical replay: target outcomes were consulted during source review. "
        "These are exploratory results, not an untouched final holdout."
    )
    _metrics_table(_frame(run, "metrics"), task, f"lab_metrics_{task}")
    if task == "race":
        st.markdown("**Podium classification**")
        _metrics_table(_frame(run, "metrics"), "podium", "lab_metrics_podium")
    st.subheader(f"{FAMILIES[family]} predictions")
    _prediction_table(predictions, task, family)
    with st.expander("Compare predicted and official ranks"):
        _rank_chart(predictions, task, family)


def _run_details(run: dict, dataset):
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


def historical_test():
    _title(
        "Test a race weekend",
        "Train on earlier real events. Compare three model families with simple baselines.",
    )
    dataset = _load_dataset()
    if dataset is None:
        return
    events = dataset.events.sort_values(["year", "round"])
    event_ids = events.event_id.tolist()
    if not event_ids:
        st.info("No reviewed completed events are eligible yet.")
        return
    default = "2026-19" if "2026-19" in event_ids else event_ids[-1]
    if st.session_state.get("lab_target_event") not in event_ids:
        st.session_state["lab_target_event"] = default
    if "lab_active_run" not in st.session_state:
        saved = _default_saved(default, dataset.receipt["id"])
        st.session_state["lab_active_run"] = saved["_path"] if saved else None
    with st.form("lab_historical_form", border=True):
        target_col, history_col = st.columns([3, 2])
        with target_col:
            target = st.selectbox(
                "Target event",
                event_ids,
                key="lab_target_event",
                format_func=lambda event: _event_label(events, event),
            )
        with history_col:
            window = st.selectbox("Training history", list(WINDOWS), key="lab_history_window")
        st.caption(
            "2026 only. The target and every later event are excluded from training. "
            "Practice pace and weather are unavailable and excluded from these core models."
        )
        submitted = st.form_submit_button("Run historical test", type="primary", key="lab_run")
    if submitted:
        from f1ml_real.lab import cached_run

        data_path, run_root = paths()
        try:
            with st.spinner("Training and checking the real-data test…"):
                result = cached_run(data_path, target, run_root, last_n=WINDOWS[window])
            st.session_state["lab_active_run"] = str(result)
            st.success(
                "Historical test saved. Reopening this configuration reuses the verified run."
            )
        except (ValueError, OSError, KeyError, RuntimeError) as exc:
            st.error(f"This test could not run: {exc}")
            st.caption(
                "Choose a later completed event or a longer history window when there "
                "is too little development history."
            )
    active = st.session_state.get("lab_active_run")
    run = _read_saved(active) if active else None
    if run is not None and run.get("config", {}).get("source_snapshot_id") != dataset.receipt["id"]:
        st.error(
            "This saved run uses a different dataset snapshot. "
            "Run a test against the active reviewed dataset."
        )
        run = None
    if run is None:
        st.info(
            "Choose a completed event and run a historical test to see predictions "
            "and baseline comparisons."
        )
        _model_strip()
        return
    config = run.get("config", {})
    event_id = config.get("event_id", config.get("target_event_id", default))
    target_row = events.loc[events.event_id.eq(event_id)]
    event = target_row.iloc[0] if len(target_row) else None
    training = _frame(run, "training_events")
    st.subheader(_event_label(events, event_id))
    st.caption(
        "Saved 2026 historical replay. Results below belong to this saved run; "
        "submit the form to change the event or training window."
    )
    count = training.event_id.nunique() if "event_id" in training else len(training)
    cols = st.columns([1, 2, 2])
    cols[0].metric("Earlier training events", count)
    cols[1].markdown("**Qualifying cutoff**")
    cols[1].write(_cutoff(event.qualifying_cutoff if event is not None else None))
    cols[2].markdown("**Race cutoff**")
    cols[2].write(_cutoff(event.race_cutoff if event is not None else None))
    if config.get("insufficient_development_history"):
        st.warning(
            "This training window has too few earlier events for separate development selection. "
            "Treat these settings as a fixed exploratory comparison."
        )
    family = st.segmented_control(
        "Prediction model",
        [k for k in FAMILIES if k != "baseline"],
        format_func=FAMILIES.get,
        default="linear",
        required=True,
        key="lab_prediction_family",
    )
    st.caption(MODEL_COPY[family][0])
    tabs = st.tabs(
        ["Before qualifying", "After qualifying · race"], key="lab_forecast_tab", on_change="rerun"
    )
    for tab, task in zip(tabs, ("qualifying", "race"), strict=True):
        if tab.open:
            with tab:
                _task_results(run, task, family)
    _run_details(run, dataset)


def live_data_path() -> Path:
    path = Path(os.environ.get("F1ML_LIVE_DATA", "data/live"))
    return path if path.is_absolute() else project_root() / path


def live_forecast():
    from f1ml_real.lab import RealDataset, cached_forecast, cached_run
    from f1ml_real.refresh import current_dataset, forecast_tasks, refresh

    _title("Refresh & predict", "Load the latest weekend and predict sessions before they start.")
    base, runs = paths()
    state = live_data_path()
    st.caption(
        "Refresh downloads the current FastF1 calendar and results, plus FIA entry lists and "
        "final classifications. Training starts only when you press Run forecast."
    )
    if st.button("Refresh FastF1 data", type="primary", key="lab_refresh"):
        try:
            with st.spinner("Loading the latest event data…"):
                refresh(base, state)
            st.success("New data snapshot saved. Select an event below.")
        except Exception as exc:
            st.error(f"Refresh failed; the previous snapshot is still available: {exc}")
    try:
        path = current_dataset(base, state)
        dataset = _dataset(str(path), (path / "receipt.json").stat().st_mtime_ns)
    except (ValueError, OSError, KeyError) as exc:
        st.error(f"The refreshed dataset could not be verified: {exc}")
        return
    if path == base:
        st.info("Press Refresh FastF1 data to discover the latest race weekends.")
        return
    st.caption(f"Last refreshed: {_cutoff(dataset.receipt.get('as_of'))}")
    original = RealDataset(base)
    events = dataset.events.loc[~dataset.events.event_id.isin(original.events.event_id)]
    ids = events.event_id.tolist()
    if not ids:
        st.info("The calendar has no additional events yet.")
        return
    if st.session_state.get("lab_live_event") not in ids:
        st.session_state["lab_live_event"] = next(
            (eid for eid in ids if forecast_tasks(dataset, eid)), ids[0]
        )
    target = st.selectbox(
        "Event", ids, key="lab_live_event", format_func=lambda eid: _event_label(events, eid)
    )
    event = dataset.event(target)
    tasks = forecast_tasks(dataset, target)
    completed = pd.notna(event.race_available_at) and pd.notna(event.qualifying_available_at)
    if tasks:
        st.info("Ready to predict: " + ", ".join(TASKS[task] for task in tasks) + ".")
    elif completed:
        st.info("This weekend is complete. You can run a historical test using its new data.")
    else:
        st.info(
            "No forecast is ready. A session must still be in the future, with a published "
            "entry list. Race and podium forecasts also need final qualifying results."
        )
    with st.expander("Schedule and refresh details"):
        st.write(f"Qualifying: {_cutoff(event.get('qualifying_start'))}")
        st.write(f"Race: {_cutoff(event.get('race_start'))}")
        import json

        for status in json.loads((path / "refresh_status.json").read_text()):
            if status["event_id"] == target:
                for issue in status["issues"]:
                    st.warning(issue)
        st.caption(f"Verified snapshot: {dataset.receipt['id']}")
    window = st.selectbox("Training history", list(WINDOWS), key="lab_live_window")
    action = "Run historical test" if completed else "Run forecast"
    if st.button(action, key="lab_live_run", disabled=not tasks and not completed):
        try:
            with st.spinner("Training on earlier events and saving predictions…"):
                runner = cached_run if completed else cached_forecast
                result = runner(path, target, runs, last_n=WINDOWS[window])
            st.session_state["lab_live_run_path"] = str(result)
        except (ValueError, OSError, KeyError, RuntimeError) as exc:
            st.error(f"The prediction could not run: {exc}")
    active = st.session_state.get("lab_live_run_path")
    run = _read_saved(active) if active else None
    if run is None or run["summary"]["target_event_id"] != target:
        run = _default_saved(target, dataset.receipt["id"])
    if run is None or run["summary"]["target_event_id"] != target:
        return
    st.subheader("Saved predictions")
    st.caption(
        f"{run['summary']['training_event_count']} earlier training events · "
        f"{run['summary']['evaluation_status']}"
    )
    if run["summary"]["source_snapshot_id"] != dataset.receipt["id"]:
        st.caption("These saved predictions use an earlier refresh snapshot.")
    family = st.selectbox(
        "Prediction model", list(FAMILIES), format_func=FAMILIES.get, key="lab_live_family"
    )
    predictions = _frame(run, "predictions")
    for task in ("qualifying", "race"):
        if task in set(predictions.task):
            st.markdown(f"**{TASKS[task]}**")
            if run["summary"]["mode"] == "live_forecast":
                _prediction_table(predictions, task, family)
            else:
                _task_results(run, task, family)


def _model_strip():
    for col, family in zip(st.columns(3), MODEL_COPY, strict=True):
        with col:
            st.markdown(f"**{FAMILIES[family]}**")
            st.caption(MODEL_COPY[family][0])


def three_models():
    _title(
        "Three models, one fair comparison",
        "The methods match the regression, trees and ensembles covered in class.",
    )
    for family, (estimator, course, explanation) in MODEL_COPY.items():
        with st.container(border=True):
            st.subheader(FAMILIES[family])
            st.markdown(f"**{estimator}**")
            st.write(explanation)
            st.caption(f"Course connection: {course}.")
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


def data_methodology():
    _title(
        "Real data and the rules",
        "Public source records, explicit cutoffs and an honest account of what is missing.",
    )
    dataset = _load_dataset()
    if dataset is None:
        return
    events, rosters, labels, receipt = (
        dataset.events,
        dataset.rosters,
        dataset.labels,
        dataset.receipt,
    )
    st.write(
        "The active experiment uses official FIA final classifications and independently published "
        "entry lists, with the FastF1 event schedule and stable driver identities "
        "from public results. "
        "No synthetic observations or forecasts are used in these tests."
    )
    counts = st.columns(3)
    counts[0].metric("Reviewed completed events", events.event_id.nunique())
    counts[1].metric("Season", str(receipt.get("year", 2026)))
    counts[2].metric("Source snapshot date", str(receipt.get("as_of", ""))[:10])
    st.subheader("Coverage by event and session")
    coverage = []
    for _, event in events.sort_values(["year", "round"]).iterrows():
        for task in ("qualifying", "race"):
            rs = rosters.loc[rosters.event_id.eq(event.event_id) & rosters.task.eq(task)]
            ls = labels.loc[labels.event_id.eq(event.event_id) & labels.task.eq(task)]
            coverage.append(
                {
                    "Event": _event_label(events, event.event_id),
                    "Session": TASKS[task],
                    "Rostered drivers": rs.driver_id.nunique(),
                    "Known numeric ranks": int(ls.official_position.notna().sum()),
                    "Known race-podium labels": int(ls.podium.notna().sum())
                    if task == "race"
                    else None,
                    "Cutoff (UTC)": _cutoff(event[f"{task}_cutoff"]),
                }
            )
    _table(pd.DataFrame(coverage), key="lab_source_coverage", height=450)
    st.subheader("Core features and missing session data")
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
                    "Status": "Observed",
                    "Treatment": "Roster and team membership known by each cutoff",
                },
                {
                    "Feature group": "Current qualifying result",
                    "Status": "Observed for race only",
                    "Treatment": "Published result available before the race prediction cutoff",
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
        "The FastF1 timing archive returned HTTP 401 during acquisition. Missing pace information "
        "also affects the real session-data path; it is not confined to the archived fixture. "
        "Official result documents allow the core historical experiment to run independently."
    )
    st.subheader("A chronological historical replay")
    st.markdown(
        "1. Select completed events by source availability, before inspecting model performance.\n"
        "2. Build history from earlier weekends only, using conservative publication times.\n"
        "3. Select settings on earlier chronological development events. "
        "Fit preprocessing on each training fold.\n"
        "4. Save the selected configuration and target predictions before scoring target labels.\n"
        "5. Report development and target-event findings separately, including baseline losses."
    )
    st.info(
        "These are exploratory historical tests. Outcomes including Mexico were already consulted "
        "during development and source review, so none are presented as an untouched final holdout."
    )
    st.write(
        "The active model lab currently compares 2026-only training windows. Pooled and weighted "
        "2024–2026 comparisons require the same independent source "
        "and cutoff review for older seasons."
    )
    st.write(
        "An NC or disqualified driver can lack an official numeric rank. The driver remains in the "
        "prediction field; rank metrics use known ranks only and report their counts. "
        "A missing source label stays unknown. Publication times are conservative bounds, "
        "not measured session-end times."
    )
    with st.expander("Source documents and excluded events"):
        docs = dataset.documents.copy()
        columns = [
            c for c in ("event_id", "task", "purpose", "title", "published_at", "url") if c in docs
        ]
        _table(
            docs[columns],
            key="lab_source_documents",
            column_config={"url": st.column_config.LinkColumn("Official source")},
        )
        excluded_path = paths()[0] / "exclusions.parquet"
        if excluded_path.exists() and len(excluded := pd.read_parquet(excluded_path)):
            st.markdown("**Excluded source events**")
            _table(excluded, key="lab_source_exclusions")
        st.json(receipt, expanded=False)
    st.caption(
        "Downloads and source preparation run outside normal interface reruns. Only pressing "
        "Run historical test can start model fitting. Saved runs are verified and reused."
    )


def main():
    st.set_page_config(
        page_title="Pit Wall · Real F1 model tests",
        page_icon="🏁",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    theme.inject()
    st.html("""<style>
    .lab-heading{border-left:5px solid #d7263d;padding:.15rem 0 .15rem 1.1rem;margin:0 0 1.3rem}
    .lab-heading h1{font-size:2.1rem;line-height:1.15;letter-spacing:-.025em;
      padding:0;margin:0 0 .45rem}
    .lab-heading p{max-width:72ch;margin:0;line-height:1.5;opacity:.78}
    </style>""")
    st.logo(str(Path(__file__).parent / "assets" / "logo.svg"), size="large")
    navigation = st.navigation(
        [
            st.Page(
                historical_test,
                title="Historical test",
                icon=":material/flag:",
                default=True,
                url_path="historical-test",
            ),
            st.Page(
                live_forecast,
                title="Refresh & predict",
                icon=":material/sync:",
                url_path="live",
            ),
            st.Page(
                three_models,
                title="Three models",
                icon=":material/account_tree:",
                url_path="models",
            ),
            st.Page(
                data_methodology,
                title="Data & methodology",
                icon=":material/fact_check:",
                url_path="data",
            ),
        ]
    )
    with st.sidebar:
        st.markdown("**Real F1 model lab**")
        st.caption("2026 official records\n\nQualifying, race order and podium probabilities.")
        st.caption("Historical replay · exploratory evaluation")
    navigation.run()


if __name__ == "__main__":
    main()
