"""Shared Streamlit plumbing: cached loaders, the global selection, headers and states."""

import base64
import html
import re
import tomllib
import traceback
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st

from f1ml_app import forecasts as fc
from f1ml_app import theme
from f1ml_app.catalog import RealCatalog, RealExperiment
from f1ml_app.experiment import ROLE_LABELS, ArtifactUnavailable, Experiment
from f1ml_app.settings import (
    DEFAULT_VIEW_MODE,
    MODEL_ALIASES,
    MODEL_LABELS,
    MODEL_SHORT,
    PERSISTENT_FILTER_KEYS,
    VIEW_MODES,
    load_paths,
    project_root,
)

BRAND_LOGO = Path(__file__).parent / "assets" / "logo-ml.png"

SOURCES = {
    "experiment": "Synthetic demo",
    "real": "Real experiment",
    "catalog": "FastF1 catalog",
}
ROLE_SHORT = {
    "final_test": "test",
    "development_validation": "dev",
    "development_training": "train",
    "excluded": "excl.",
    "outside_manifest": "other",
}
OUTCOME_LABELS = {
    "ordered": "Classified",
    "non_classified": "Not classified (NC)",
    "dsq": "Disqualified (DSQ)",
    "nan": "Unknown",
    "None": "Unknown",
    "<NA>": "Unknown",
}
MOMENT_CAPTIONS = {
    "pre_qualifying": "Qualifying cutoff · no results known",
    "post_qualifying": "Race cutoff · qualifying result known",
    "post_race": "All official results revealed",
}


# ----- cached, read-only loaders (no downloads, no fitting) -----------------------------


@st.cache_resource(show_spinner="Verifying the frozen experiment's receipts…")
def experiment() -> Experiment:
    return Experiment(load_paths())


@st.cache_resource(show_spinner="Reading the FastF1 catalog snapshot…")
def _catalog() -> RealCatalog:
    return RealCatalog(load_paths().catalog)


@st.cache_resource(show_spinner="Verifying the saved real-data experiment…")
def real_experiment() -> RealExperiment:
    return RealExperiment(load_paths().real_experiment)


def catalog() -> RealCatalog:
    return real_experiment() if st.session_state.get("source") == "real" else _catalog()


@st.cache_resource(show_spinner="Loading a saved model bundle…")
def _bundle(path: str, task: str, model_id: str):
    from f1ml.models.qualifying import load_model as load_qualifying
    from f1ml.models.race import load_model as load_race

    return load_qualifying(Path(path)) if task == "qualifying" else load_race(Path(path), task)


def loader(card):
    return _bundle(str(card.path), card.task, card.model_id)


@st.cache_data(show_spinner="Replaying the qualifying forecast…")
def qualifying_forecast(event_id: str, family: str | None = None, strategy: str | None = None):
    choice = (family, strategy) if family else None
    return fc.qualifying_forecast(experiment(), event_id, choice, loader)


@st.cache_data(show_spinner="Replaying the race forecast…")
def race_forecast(event_id: str, moment: str, race=(None, None), podium=(None, None)):
    return fc.race_forecast(
        experiment(),
        event_id,
        moment,
        race if race[0] else None,
        podium if podium[0] else None,
        loader,
    )


def try_experiment() -> tuple[Experiment | None, str | None]:
    try:
        return experiment(), None
    except (ArtifactUnavailable, FileNotFoundError, ValueError, KeyError) as exc:
        return None, str(exc)


def selected_forecasts(task: str, event_id: str, moment: str = "post_race") -> dict:
    """Use only eligible bundles of active families; frozen tests have one locked family."""
    exp = experiment()
    forecasts = {}
    for model in selected_models():
        if exp.role(event_id) == "final_test":
            card = exp.locked_card(task)
            if card is None or MODEL_ALIASES.get(card.family, card.family) != model:
                continue
            forecast = (
                qualifying_forecast(event_id)
                if task == "qualifying"
                else race_forecast(event_id, moment)
            )
        else:
            choice = exp.family_choice(task, event_id, model)
            if choice is None:
                continue
            if task == "qualifying":
                forecast = qualifying_forecast(event_id, *choice)
            else:
                podium = exp.family_choice("podium", event_id, model)
                if podium is None:
                    continue
                forecast = race_forecast(event_id, moment, choice, podium)
        if forecast.available:
            forecasts[model] = forecast
    return forecasts


def forecast_fields(forecasts: dict, outcome) -> pd.DataFrame:
    fields = {}
    for model in selected_models():
        forecast = forecasts.get(model)
        if forecast is None:
            fields[model] = fc.presentation_frame(pd.DataFrame())
            continue
        data = forecast.predictions.copy()
        if "team_id" not in data:
            data = data.merge(
                forecast.inputs[["driver_id", "team_id"]],
                on="driver_id",
                how="left",
                validate="one_to_one",
            )
        fields[model] = fc.presentation_frame(data, outcome=outcome)
        podium = forecast.models.get("podium")
        if podium and MODEL_ALIASES.get(podium.family, podium.family) != model:
            # A frozen lock can pair different regressor/classifier families. Never
            # attribute that classifier's probabilities to the race-order family.
            fields[model]["podium_probability"] = float("nan")
    return fc.compare_fields(fields, primary_model())


def forecast_metrics(exp, forecasts: dict, task: str):
    """Saved scores and winner among active, available forecasts, never refitted scores."""
    from f1ml_app.text import metric_help, metric_label

    rows = []
    for model, forecast in forecasts.items():
        card = forecast.models.get(task)
        if card is None or MODEL_ALIASES.get(card.family, card.family) != model:
            continue
        metrics = fc.saved_event_metrics(exp, forecast, task)
        if metrics:
            rows.append({"Model": MODEL_LABELS[model], **metrics})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return
    primary = "position_mae" if task != "podium" else "log_loss"
    if primary in frame and frame[primary].notna().any():
        winner = frame.loc[frame[primary].idxmin()]
        st.markdown(
            f":green-badge[Active selection winner] {winner['Model']} · "
            f"{metric_label(primary)}: {winner[primary]:.3f}"
        )
    metrics = [c for c in ("position_mae", "spearman", "brier", "log_loss", "f1") if c in frame]
    table(
        frame[["Model", *metrics]],
        {
            c: st.column_config.NumberColumn(metric_label(c), format="%.3f", help=metric_help(c))
            for c in metrics
        },
    )


def try_catalog() -> tuple[RealCatalog | None, str | None]:
    try:
        return catalog(), None
    except (FileNotFoundError, ValueError, KeyError) as exc:
        return None, str(exc)


# ----- global selection ------------------------------------------------------------------


def persist_filters():
    """Keep known filter widgets when Streamlit cleans up widgets absent on a page.

    Call before rendering widgets, including in independently mounted page entrypoints.
    Buttons are deliberately excluded: preserving a submit button can repeat an action.
    """
    keys = set(PERSISTENT_FILTER_KEYS) | {
        key for key in st.session_state if key.startswith(("season_", "event_", "field_"))
    }
    for key in keys:
        if key in st.session_state:
            st.session_state[key] = st.session_state[key]


def full_telemetry() -> bool:
    return st.session_state.get("view_mode", DEFAULT_VIEW_MODE) == VIEW_MODES[1]


def selected_models() -> list[str]:
    """Canonical family selection, also safe for independently mounted legacy views."""
    values = st.session_state.get("lab_selected_models", list(MODEL_LABELS))
    selected = [model for model in dict.fromkeys(values) if model in MODEL_LABELS]
    return selected or [
        st.session_state.get("_lab_family", "linear")
        if st.session_state.get("_lab_family") in MODEL_LABELS
        else "linear"
    ]


def primary_model() -> str:
    models = selected_models()
    primary = st.session_state.get(
        "_lab_family", st.session_state.get("lab_prediction_family", models[0])
    )
    return primary if primary in models else models[0]


def filter_models(frame: pd.DataFrame, *, column: str = "family") -> pd.DataFrame:
    if column not in frame:
        return frame.copy()
    return frame.loc[frame[column].replace(MODEL_ALIASES).isin(selected_models())].copy()


def filter_grid(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply global filters without recomputing ranks or evaluation metrics."""
    data = frame.copy()
    teams = st.session_state.get("grid_teams", [])
    if teams and "team_id" in data:
        data = data.loc[data.team_id.isin(teams)]
    search = st.session_state.get("grid_driver_search", "").strip()
    if search:
        cols = [c for c in ("driver", "driver_id") if c in data]
        if cols:
            mask = pd.Series(False, index=data.index)
            for column in cols:
                mask |= data[column].astype(str).str.contains(search, case=False, regex=False)
            data = data.loc[mask]
    return data.copy()


def _models_changed():
    models = selected_models()
    st.session_state["lab_selected_models"] = models
    primary = primary_model()
    st.session_state["_lab_family"] = primary
    st.session_state["lab_prediction_family"] = primary
    st.session_state["lab_live_family"] = primary


def _select_all_models():
    st.session_state["lab_selected_models"] = list(MODEL_LABELS)
    _models_changed()


def _primary_changed(key: str):
    st.session_state["_lab_family"] = st.session_state[key]


def model_preferences(*, live: bool = False) -> str:
    models = selected_models()
    st.session_state["lab_selected_models"] = models
    st.multiselect(
        "Models to compare",
        list(MODEL_LABELS),
        key="lab_selected_models",
        format_func=MODEL_LABELS.get,
        on_change=_models_changed,
        help="Compare any combination. Clearing the selection keeps the spotlight model active.",
    )
    st.button(
        "Compare all three models",
        key="select_all_models",
        on_click=_select_all_models,
        width="stretch",
        disabled=len(models) == len(MODEL_LABELS),
    )
    key = "lab_live_family" if live else "lab_prediction_family"
    st.session_state[key] = primary_model()
    primary = st.selectbox(
        "Spotlight model",
        models,
        key=key,
        format_func=MODEL_LABELS.get,
        on_change=_primary_changed,
        args=(key,),
        help="Sets the grid order and the P1–P4 podium cards. Other selected models appear "
        "alongside it for comparison.",
    )
    st.session_state["_lab_family"] = primary
    return primary


def reset_filters():
    """Button callback: reset before any widgets are instantiated on the next rerun."""
    keys = (
        set(PERSISTENT_FILTER_KEYS)
        | {
            key
            for key in st.session_state
            if key.startswith(("field_", "season_", "event_", "_lab_"))
        }
        | {"lab_active_run", "lab_active_season", "lab_selected_season"}
    )
    for key in keys:
        st.session_state.pop(key, None)
    st.session_state["lab_selected_models"] = list(MODEL_LABELS)
    st.session_state["_lab_family"] = next(iter(MODEL_LABELS))


def grid_preferences():
    teams = theme.constructor_ids()
    st.session_state["grid_teams"] = [
        team for team in st.session_state.get("grid_teams", []) if team in teams
    ]
    st.multiselect(
        "Teams",
        teams,
        key="grid_teams",
        format_func=theme.team_name,
        placeholder="All teams",
        help="Filter driver tables and charts across pages. Saved scores use the full field.",
    )
    st.text_input(
        "Driver search",
        key="grid_driver_search",
        placeholder="e.g. Leclerc",
        help="Matches driver names or IDs. Clear the box to show everyone.",
    )
    st.markdown("### Detail level")
    display_preferences()
    st.button(
        "Reset all filters",
        key="reset_filters",
        on_click=reset_filters,
        width="stretch",
        icon=":material/restart_alt:",
    )


def top_navigation(options: list[str]) -> str:
    return st.radio(
        "Main views",
        options,
        key="lab_page",
        horizontal=True,
        label_visibility="collapsed",
        width="stretch",
    )


def display_preferences():
    """Render in the sidebar's display section; the default preserves audit views."""
    if st.session_state.get("view_mode") not in VIEW_MODES:
        st.session_state["view_mode"] = DEFAULT_VIEW_MODE
    return st.radio(
        "View Mode",
        VIEW_MODES,
        key="view_mode",
        label_visibility="collapsed",
        format_func={VIEW_MODES[0]: "Summary", VIEW_MODES[1]: "Full detail"}.get,
        captions=[
            "Headline results and the podium only",
            "Adds the full grid, every score and the audit trail",
        ],
    )


def context_ribbon(
    *,
    artifact_id: str | None = None,
    season: int | None = None,
    event: str | None = None,
    family: str | None = None,
    task: str | None = None,
    lock_status: str = "No prediction lock",
    receipt_verified: bool = False,
    pending: str | None = None,
    models: list[str] | None = None,
):
    """Render escaped context inside a top-of-page ``f1_context_ribbon`` container.

    Status arguments must come from the artifact loader used by the displayed view.
    The inspected artifact and an unsubmitted request are intentionally explicit.
    """
    model_text = family or "Unavailable"
    if models:
        model_text = ", ".join(MODEL_LABELS.get(m, m) for m in models)
        model_text += f" ({len(models)}/{len(MODEL_LABELS)} active)"
    teams = st.session_state.get("grid_teams", [])
    grid_text = ", ".join(theme.team_name(team) for team in teams) or "All constructors"
    if search := st.session_state.get("grid_driver_search", "").strip():
        grid_text += f" · Driver: {search}"
    weekend = (
        str(event)
        if event and str(event).startswith(str(season))
        else " · ".join(str(x) for x in (season, event) if x is not None)
    )
    fields = (
        ("Showing", weekend or "No weekend selected"),
        ("Prediction", task or "Not selected"),
        ("Models", model_text),
        ("Grid", grid_text),
    )
    body = "".join(
        f'<div class="f1-context-field"><span class="f1-micro">{html.escape(label)}</span>'
        f'<span class="f1-context-value">{html.escape(str(value))}</span></div>'
        for label, value in fields
    )
    body += (
        f'<div class="f1-context-artifact" title="Saved run or dataset shown below">'
        f'<span class="f1-micro">Source</span>'
        f"<code>{html.escape(artifact_id or 'No saved artifact selected')}</code></div>"
    )
    locked = lock_status == "Predictions locked"
    badges = _tags(
        [
            (lock_status, "locked" if locked else "warn"),
            (
                "Receipt verified" if receipt_verified else "Receipt not verified",
                "verified" if receipt_verified else "warn",
            ),
        ]
    )
    note = (
        f'<div class="f1-context-pending"><strong>Next request:</strong> '
        f"{html.escape(pending)}</div>"
        if pending
        else ""
    )
    st.html(
        f'<section class="f1-context" aria-label="Current telemetry context">{body}'
        f'<div class="f1-context-badges" title="Prediction Lock: saved predictions were sealed '
        f'before target scoring. It does not guarantee forecast accuracy.">'
        f"{badges}</div>{note}</section>"
    )


@dataclass(frozen=True)
class Selection:
    source: str
    season: int | None
    event_id: str | None
    moment: str


def _event_label(frame: pd.DataFrame, source: str):
    rows = frame.set_index("event_id")

    def label(event_id):
        row = rows.loc[event_id]
        text = f"{event_id.split('-')[0]} · R{int(row['round']):02d} · {row['name']}"
        if source == "experiment":
            text += f" · {ROLE_SHORT[row['role']]}"
        return text

    return label


def _default_event(source: str, events: pd.DataFrame) -> str:
    if source == "experiment":
        usable = events.loc[events.role.isin(["final_test", "development_validation"])]
    else:
        usable = events.loc[events.result_sessions.fillna(0).gt(0)]
    return (usable if len(usable) else events).event_id.iloc[-1]


def _apply_url_once():
    """Deep links (?source=&event=&moment=) initialize the selection once per session."""
    if st.session_state.get("_url_applied"):
        return
    st.session_state["_url_applied"] = True
    params = st.query_params
    if params.get("source") in SOURCES:
        st.session_state["source"] = params["source"]
    if params.get("moment") in fc.MOMENTS:
        st.session_state["moment"] = params["moment"]
    if params.get("event"):
        st.session_state["_pending_event"] = params["event"]


def sidebar(*, fixed_source: str | None = None) -> Selection:
    """Data source → season → Grand Prix → replay moment. Rendered on every page."""
    persist_filters()
    _apply_url_once()
    state = st.session_state
    if fixed_source is not None:
        state["source"] = fixed_source
    with st.sidebar:
        st.markdown("### Race & Replay Context")
        source = st.segmented_control(
            "Data source",
            list(SOURCES),
            format_func=SOURCES.get,
            default=None if "source" in state else "experiment",
            required=True,
            key="source",
            disabled=fixed_source is not None,
            help="Synthetic demo models, the separate real experiment review, or source coverage. "
            "Missing real forecasts are unavailable.",
        )
        loaded, problem = try_experiment() if source == "experiment" else try_catalog()
        if loaded is None:
            st.error(f"Could not load this data source: {problem}")
            return Selection(source, None, None, "post_race")
        events = loaded.events
        pending = state.pop("_pending_event", None)
        if pending in set(events.event_id):
            year = int(events.set_index("event_id").loc[pending, "year"])
            state[f"season_{source}"] = year
            state[f"event_{source}_{year}"] = pending
        seasons = sorted(events.year.unique().tolist(), reverse=True)
        season_key = f"season_{source}"
        if state.get(season_key) not in seasons:
            state[season_key] = seasons[0]
        season = st.segmented_control("Season", seasons, required=True, key=season_key)
        frame = events.loc[events.year.eq(season)]
        key = f"event_{source}_{season}"
        if state.get(key) not in frame.event_id.tolist():
            state[key] = _default_event(source, frame)
        event_id = st.selectbox(
            "Grand Prix",
            frame.event_id.tolist(),
            key=key,
            format_func=_event_label(frame, source),
            help="Experiment roles: test = frozen final test, dev = development validation "
            "fold, train = training-only weekend, excl. = excluded at freeze."
            if source == "experiment"
            else None,
        )
        moment = "post_race"
        st.markdown("### Model Selection")
        if source == "experiment":
            model_preferences()
            if state.get("moment") not in fc.MOMENTS:
                state["moment"] = "post_race"
            moment = st.radio(
                "Replay moment",
                list(fc.MOMENTS),
                format_func=fc.MOMENTS.get,
                captions=list(MOMENT_CAPTIONS.values()),
                key="moment",
                help="Step through the weekend. Outcomes stay hidden until the replay moment "
                "reaches the session that produced them.",
            )
            st.divider()
            if loaded.is_synthetic:
                st.warning(
                    "**Synthetic fixtures.** Scores validate software and protocol, not real "
                    "Formula 1 accuracy.",
                    icon=":material/science:",
                )
            st.caption(
                f"Manifest `{loaded.manifest_id[:12]}` · dataset and feature receipts verified"
            )
        else:
            if source == "real":
                st.selectbox(
                    "Prediction task",
                    ["qualifying", "race", "podium"],
                    format_func={
                        "qualifying": "Qualifying order",
                        "race": "Race order",
                        "podium": "Podium probabilities",
                    }.get,
                    key="real_evaluation_task",
                )
                st.caption("No trained model is available for this saved review.")
            else:
                st.caption("Coverage only; no model or prediction task is available.")
            st.divider()
            if source == "real":
                st.warning(f"{loaded.status['experiment_id']} · {loaded.status['state']}")
                st.caption("No trained models · development 0 events · final test 0 events")
            else:
                st.info(
                    "Real source data. Coverage only: no model is trained on it.",
                    icon=":material/database:",
                )
            st.caption(
                f"Snapshot `{loaded.receipt['id'][:12]}` · as of "
                f"{loaded.as_of[:16].replace('T', ' ')} UTC"
            )
        st.markdown("### Grid Filter & Display Density")
        grid_preferences()
    selection = Selection(source, int(season), event_id, moment)
    state["_selection"] = selection
    st.query_params.from_dict(
        {
            "source": source,
            "event": event_id,
            **({"moment": moment} if source == "experiment" else {}),
        }
    )
    return selection


def selection() -> Selection:
    return st.session_state.get("_selection") or Selection("experiment", None, None, "post_race")


def real_review():
    """Display saved real artifacts only. No training/download/evaluation imports or calls."""
    exp = real_experiment()
    status = exp.status
    st.subheader(f"Real-data experiment · {status['experiment_id']}")
    st.warning(f"{status['state']} · no real performance result is available")
    cols = st.columns(3)
    cols[0].metric("Eligible modeling events", status["eligible_events"])
    cols[1].metric("Development evaluated events", status["development_evaluated_events"])
    cols[2].metric("Final-test evaluated events", status["final_evaluated_events"])
    st.caption(
        f"Source snapshot {status['source_snapshot_id']} · as of {when(status['as_of'])}. "
        "Training cutoffs: none; no model fitted. Final manifest and lock: not created. "
        "Evaluation ledger: unclaimed."
    )
    for blocker in status["blockers"]:
        st.markdown(f"- {blocker}")
    for limitation in status.get("limitations", []):
        st.caption(limitation)
    st.info(status["prior_consultation"])
    tabs = st.tabs(["Coverage", "Exclusions", "Development plan", "Evaluation status"])
    with tabs[0]:
        table(exp.coverage["seasons"])
        table(exp.group_coverage)
    with tabs[1]:
        table(exp.exclusions)
    with tabs[2]:
        st.write(status["protocol"])
        table(exp.strategy_eligibility)
    with tabs[3]:
        st.caption("Planned methods only; blank metric values are not zero loss.")
        table(exp.evaluation_records)
    st.download_button(
        "Download real-data review",
        exp.report.encode(),
        file_name=f"{status['experiment_id']}-review.md",
        mime="text/markdown",
    )


# ----- presentation helpers ---------------------------------------------------------------


def when(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    stamp = pd.Timestamp(value)
    stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    return stamp.strftime("%a %d %b %Y · %H:%M UTC")


def _tags(items: list[tuple[str, str]]) -> str:
    return "".join(
        f'<span class="f1-tag {html.escape(cls, quote=True)}">{html.escape(str(text))}</span>'
        for text, cls in items
    )


def _brand_image() -> str:
    data = base64.b64encode(BRAND_LOGO.read_bytes()).decode("ascii")
    return (
        f'<img class="f1-logo" src="data:image/png;base64,{data}" '
        'alt="FastF1 plus machine learning" width="336" height="112">'
    )


def _header_html(
    title: str, subtitle: str, tags: list[tuple[str, str]], *, logo: bool = False
) -> str:
    return (
        f'<header class="f1-hero">{_brand_image() if logo else ""}<div class="f1-heading">'
        f'<h1 class="f1-title">{html.escape(title)}</h1>'
        f'<div class="f1-meta">{html.escape(subtitle)}</div></div>'
        f'<div class="f1-status" aria-label="Dataset and replay status">{_tags(tags)}</div>'
        "</header>"
    )


def brand_header(
    title: str,
    subtitle: str,
    *,
    dataset_version: str | None = None,
    verified_run: dict | None = None,
):
    """Compact brand bar. Pass only a run returned by the verified saved-run loader.

    The config badge identifies the configured review, not the active dataset.
    A lock refers to sealed predictions, never to an untouched evaluation holdout.
    """
    # Lock and receipt status live in the context ribbon directly below; the header
    # only names the page and the data it reads, so the two never repeat each other.
    config = project_root() / "configs" / "real-20261105.toml"
    try:
        version = tomllib.loads(config.read_text()).get("experiment_id")
    except (OSError, ValueError):
        version = None
    badges = []
    if dataset_version:
        # Refresh directories use opaque snapshot IDs; keep the bar compact.
        display = (
            f"snapshot {dataset_version[:12]}"
            if re.fullmatch(r"[0-9a-fA-F]{32,64}", dataset_version)
            else dataset_version
        )
        badges.append((f"Data: {display}", ""))
    elif version:
        badges.append((f"Config {version}", ""))
    lock = (verified_run or {}).get("prediction_lock", {})
    if verified_run is not None and not (
        lock.get("sealed_at") and lock.get("sealed_before_target_scoring") is True
    ):
        badges.append(("Replay lock unconfirmed", "warn"))
    if verified_run is not None:
        badges.append(("Exploratory evaluation", "warn"))
    st.html(_header_html(title, subtitle, badges))


def hero(kicker: list[tuple[str, str]], title: str, meta: str):
    """Legacy header entry point; preserve caller-provided tags and escaped text."""
    st.html(_header_html(title, meta, kicker))


def team_pill(team: str | None, label: str | None = None) -> str:
    """Escaped HTML for st.html/st.markdown: constructor text plus an accent bar."""
    text = label if label is not None else theme.team_name(team)
    return (
        f'<span class="f1-team-pill" style="--team-accent:{theme.team_color(team)}">'
        f"{html.escape(text)}</span>"
    )


def team_row(
    driver: str,
    team: str | None,
    value: str = "",
    *,
    rank: int | None = None,
    caption: str | None = None,
    tooltip: str | None = None,
) -> str:
    """Escaped timing row HTML with a constructor stripe and aligned numeric value."""
    position = (
        f'<span class="f1-team-rank f1-num">{html.escape(str(rank))}</span>'
        if rank is not None
        else ""
    )
    note = f'<span class="f1-team-caption">{html.escape(caption)}</span>' if caption else ""
    tip = f' title="{html.escape(tooltip, quote=True)}" tabindex="0"' if tooltip else ""
    return (
        f'<div class="f1-team-row" style="--team-accent:{theme.team_color(team)}"{tip}>'
        f'{position}<span class="f1-team-driver">{html.escape(driver)}{note}</span>'
        f'{team_pill(team)}<span class="f1-team-value">{html.escape(value)}</span></div>'
    )


def metric_card(
    label: str,
    value,
    delta=None,
    *,
    tooltip: str | None = None,
    caption: str | None = None,
):
    """Native, accessible Streamlit metric with telemetry styling and optional context."""
    from f1ml_app.text import metric_help, metric_label

    st.metric(metric_label(label), value, delta=delta, help=tooltip or metric_help(label))
    if caption:
        st.caption(caption)


def _podium_result(row) -> tuple[str, str]:
    actual = row.get("actual_rank")
    podium = row.get("actual_podium")
    if pd.isna(podium) and pd.notna(actual):
        podium = actual <= 3
    if pd.isna(podium):
        return "Actual result unavailable", ""
    status = (
        str(row.get("outcome", "Unknown")).replace("non_classified", "NC").replace("dsq", "DSQ")
    )
    finish = (
        f"P{int(actual)}"
        if pd.notna(actual)
        else (status if status not in ("Unknown", "nan", "None", "<NA>") else "rank unavailable")
    )
    return (
        f"Podium {'hit' if bool(podium) else 'miss'} · Actual {finish}",
        "verified" if podium else "warn",
    )


def _probability_bar(value, color: str) -> str:
    if pd.isna(value):
        return '<div class="f1-probability-label">Podium probability unavailable</div>'
    width = float(value) * 100
    return (
        f'<div class="f1-probability-label"><span>Podium probability</span>'
        f'<strong class="f1-num">{pct(value)}</strong></div>'
        f'<div class="f1-probability-track" role="meter" aria-label="Podium probability" '
        f'aria-valuemin="0" aria-valuemax="100" aria-valuenow="{width:.2f}">'
        f'<div class="f1-probability-fill" style="width:{width:.2f}%;'
        f'background:{color}"></div></div>'
    )


def _model_badges(row) -> str:
    badges = []
    for model, short in MODEL_SHORT.items():
        rank = row.get(f"rank__{model}")
        if f"rank__{model}" not in row:
            continue
        probability = row.get(f"podium__{model}")
        rank_text = f"P{int(rank)}" if pd.notna(rank) else "rank unavailable"
        chance = f"{pct(probability, 0)} " if pd.notna(probability) else ""
        badges.append(f"{short}: {chance}({rank_text})")
    return (
        f'<div class="f1-model-comparison">{html.escape(" | ".join(badges))}</div>'
        if badges
        else ""
    )


def podium_spotlight(field: pd.DataFrame):
    """Race-order top three with independent classifier probabilities and observed hits."""
    cards = filter_grid(fc.spotlight_candidates(field))
    if cards.empty:
        return
    st.subheader("Predicted Podium Spotlight")
    st.caption(
        "P1–P3 follow the race-order model. Bars show the separate classifier's independent "
        "podium probability. A hit means an actual podium finish, not an exact-position match."
    )
    for col, (_, row) in zip(st.columns(3), cards.head(3).iterrows(), strict=False):
        color = theme.team_color(row.team_id)
        result, cls = _podium_result(row)
        with col:
            st.html(
                f'<article class="f1-spotlight" style="--team-accent:{color}">'
                f'<div class="f1-micro">Predicted P{int(row.predicted_rank)}</div>'
                f'<div class="f1-spotlight-driver">{html.escape(str(row.driver))}</div>'
                f"{team_pill(row.team_id)}{_probability_bar(row.podium_probability, color)}"
                f"{_model_badges(row)}{_tags([(result, cls)])}</article>"
            )
    if len(cards) > 3:
        row = cards.iloc[3]
        color = theme.team_color(row.team_id)
        result, cls = _podium_result(row)
        st.html(
            f'<aside class="f1-challenger" style="--team-accent:{color}">'
            f'<div><div class="f1-micro">P4 Challenger</div><strong>'
            f"{html.escape(str(row.driver))}</strong> · Predicted P{int(row.predicted_rank)}</div>"
            f'{team_pill(row.team_id)}<div class="f1-challenger-probability">'
            f"{_probability_bar(row.podium_probability, color)}</div>"
            f"{_model_badges(row)}{_tags([(result, cls)])}</aside>"
        )


def field_filters(field: pd.DataFrame, *, key: str) -> tuple[pd.DataFrame, bool]:
    """A constructor filter and a full-field surprise highlight; predictions stay unchanged."""
    field = filter_grid(field)
    teams = sorted(field.team_id.dropna().astype(str).unique(), key=theme.team_name)
    team_key = f"field_{key}_team"
    if st.session_state.get(team_key) not in [None, *teams]:
        st.session_state[team_key] = None
    left, right = st.columns([2, 1])
    with left:
        team = st.selectbox(
            "Team / Constructor",
            [None, *teams],
            key=team_key,
            format_func=lambda value: theme.team_name(value) if value else "All constructors",
        )
    with right:
        highlight = st.checkbox(
            "Highlight largest rank surprises",
            key=f"field_{key}_surprises",
            disabled=not field.rank_error.notna().any(),
        )
    selected = field.loc[field.team_id.eq(team)].copy() if team else field.copy()
    known = selected.loc[selected.rank_error.notna() & selected.rank_error.gt(0)]
    largest = (
        known.sort_values(
            ["rank_error", "predicted_rank", "driver_id"], ascending=[False, True, True]
        )
        .head(3)
        .index
    )
    selected["is_surprise"] = selected.index.isin(largest) & highlight
    if highlight and not known.empty:
        surprise = known.loc[largest[0]]
        st.caption(
            f"Largest gap in this selection: {surprise.driver}, "
            f"{int(surprise.rank_error)} places ({surprise.rank_delta_label} vs forecast). "
            "Up to three non-zero errors are highlighted; unknown ranks are excluded."
        )
    return selected, highlight


def prediction_field(field: pd.DataFrame, *, key: str):
    """Shared full-field chart/table. Human labels lead; raw IDs and scores stay hidden."""
    from f1ml_app import charts

    selected, highlight = field_filters(field, key=key)
    if selected.empty:
        st.info("No drivers match this constructor filter.")
        return
    st.altair_chart(
        charts.rank_deltas(selected, color_by="error" if highlight else "constructor"),
        width="stretch",
    )
    multi = any(column.startswith("rank__") for column in selected)
    st.caption(
        "Each dot is one model's predicted rank; the grey band shows how far the models "
        "disagree and the hollow diamond is the actual rank. Hover for details. Missing "
        "model outputs and unknown actual ranks are left off, never drawn as zero."
        if multi
        else "Circle = predicted rank; diamond = actual rank. Hover for details; scroll or "
        "drag to zoom/pan the rank axis. Rank delta = Predicted − Actual: ▲ means ahead of "
        "forecast, ▼ means behind. A blank actual rank stays unknown, including unranked NC/DSQ."
    )
    with st.expander("Rank errors and podium probabilities"):
        if selected.rank_error.notna().any():
            st.altair_chart(charts.model_field_comparison(selected, "rank_error"), width="stretch")
        if selected.podium_probability.notna().any():
            st.altair_chart(
                charts.model_field_comparison(selected, "podium_probability"), width="stretch"
            )
    names = {
        "driver": "Driver",
        "team_id": "Team",
        "predicted_rank": "Predicted rank",
        "actual_rank": "Official rank",
        "rank_delta_label": "Rank delta",
        "rank_error": "Rank error",
        "podium_probability": "Podium chance",
        "actual_podium": "Actual podium",
        "outcome": "Official status",
    }
    view = selected.rename(columns=names).copy()
    view["Team"] = view.Team.map(theme.team_name)
    if "Official status" in view:
        view["Official status"] = view["Official status"].map(
            lambda value: OUTCOME_LABELS.get(str(value), value)
        )
    view["Actual podium"] = (
        pd.to_numeric(view["Actual podium"], errors="coerce")
        .map({0: False, 1: True})
        .astype("boolean")
    )
    config = {c: None for c in view.columns}
    config.update(
        {
            "Driver": st.column_config.TextColumn(width="medium"),
            "Team": st.column_config.TextColumn(),
            "Predicted rank": st.column_config.NumberColumn(format="P%d"),
            "Official rank": st.column_config.NumberColumn(
                "Actual rank",
                format="P%d",
                help="Blank means no known numeric rank. NC/DSQ do not receive invented positions.",
            ),
            "Rank delta": st.column_config.TextColumn(
                help="Predicted − Actual. Positive = finished ahead of forecast."
            ),
            "Rank error": st.column_config.NumberColumn(format="%.0f"),
            "Podium chance": st.column_config.ProgressColumn(
                "Podium probability",
                min_value=0,
                max_value=1,
                format="percent",
                help="Independent P(podium), displayed on a 0–100% scale.",
            ),
            "Official status": st.column_config.TextColumn(),
            "Actual podium": st.column_config.CheckboxColumn(),
        }
    )
    order = [c for c in names.values() if c in view]
    if multi:
        for model, label in MODEL_LABELS.items():
            rank, probability = f"rank__{model}", f"podium__{model}"
            if rank not in view:
                continue
            rank_label, probability_label = (
                f"{label} · Predicted Rank",
                f"{label} · Podium Probability",
            )
            view = view.rename(columns={rank: rank_label, probability: probability_label})
            config[rank_label] = st.column_config.NumberColumn(format="P%d")
            order.append(rank_label)
            if view[probability_label].notna().any():
                config[probability_label] = st.column_config.ProgressColumn(
                    min_value=0,
                    max_value=1,
                    format="percent",
                    help="Independent podium probability from this model's classifier.",
                )
                order.append(probability_label)
            else:
                config[probability_label] = None
        for column in (
            "Consensus / Mean Rank",
            "Model Spread (Max - Min Rank)",
            "Models with rank",
        ):
            config[column] = st.column_config.NumberColumn(
                format="%.2f" if column != "Models with rank" else "%d",
                help="Describes available model ranks; missing predictions are excluded. "
                "Mean rank is not a separately evaluated model.",
            )
            order.append(column)
    if not view["Podium chance"].notna().any():
        order = [c for c in order if c not in ("Podium chance", "Actual podium")]
        config["Podium chance"] = config["Actual podium"] = None
    table_data = view
    if highlight:
        t = theme.tokens()
        table_data = view.style.apply(
            lambda row: [
                f"background-color:{t['raised']};color:{t['ink']}" if row.is_surprise else ""
                for _ in row
            ],
            axis=1,
        )
    st.dataframe(
        table_data,
        column_config=config,
        column_order=order,
        hide_index=True,
        width="stretch",
        height="content",
        key=key,
    )
    return selected


def calibration_audit(
    frame: pd.DataFrame, *, family: str, key: str, models: list[str] | None = None
):
    """Keep calibration-fit diagnostics separate from target-event accuracy."""
    if frame.empty or not {"event_id", "family"}.issubset(frame):
        return
    models = models or [family]
    data = filter_grid(frame.loc[frame.family.isin(models)])
    if data.empty:
        return
    with st.expander("Calibration probability audit"):
        st.caption(
            "Earlier development events used to fit calibration. These are fit diagnostics, "
            "not target-event predictions or independent accuracy estimates. Saved raw and "
            "calibrated probabilities are displayed without refitting."
        )
        events = sorted(data.event_id.unique())
        event_key = f"field_{key}_calibration_event"
        if st.session_state.get(event_key) not in events:
            st.session_state[event_key] = events[-1]
        event = st.selectbox("Calibration event", events, key=event_key)
        data = data.loc[data.event_id.eq(event)]
        data = pd.concat(
            [fc.presentation_frame(data.loc[data.family.eq(model)]) for model in models],
            ignore_index=True,
        )
        if len(models) > 1:
            from f1ml_app import charts

            st.altair_chart(charts.calibration_comparison(data), width="stretch")
        view = pd.DataFrame(
            {
                "Driver": data.driver,
                "Team": data.team_id.map(theme.team_name),
                "Before calibration": data.raw_podium_probability,
                "Saved calibrated probability": data.podium_probability,
                "Actual podium": pd.to_numeric(data.actual_podium, errors="coerce")
                .map({0: False, 1: True})
                .astype("boolean"),
            }
        )
        config = {
            column: st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent")
            for column in ("Before calibration", "Saved calibrated probability")
        }
        config["Actual podium"] = st.column_config.CheckboxColumn()
        if len(models) > 1:
            view.insert(0, "Model", data.family.map(MODEL_LABELS))
        st.dataframe(view, column_config=config, hide_index=True, width="stretch", key=key)


def info_card(
    title: str,
    body: str,
    *,
    label: str | None = None,
    tooltip: str | None = None,
    caption: str | None = None,
):
    """Render a compact text card; all content is plain text and HTML-escaped."""
    micro = f'<div class="f1-micro">{html.escape(label)}</div>' if label else ""
    tip = (
        f' class="f1-tooltip" title="{html.escape(tooltip, quote=True)}" tabindex="0"'
        if tooltip
        else ""
    )
    note = f'<div class="f1-card-caption">{html.escape(caption)}</div>' if caption else ""
    st.html(
        f'<div class="f1-card">{micro}<div class="f1-card-title">'
        f"<span{tip}>{html.escape(title)}</span></div><div>{html.escape(body)}</div>{note}</div>"
    )


def event_hero(page: str, sel: Selection):
    if sel.event_id is None:
        hero([(page, "")], "No event selected", "Choose a data source, season and Grand Prix.")
        return
    if sel.source == "experiment":
        exp = experiment()
        row = exp.events.set_index("event_id").loc[sel.event_id]
        tags = [(page, "solid"), (f"Round {int(row['round'])} · {int(row['year'])}", "")]
        tags.append((ROLE_LABELS[row["role"]], ""))
        if exp.is_synthetic:
            tags.append(("Synthetic data", "warn"))
        meta = f"Weekend starts {when(row['weekend_start'])} · {fc.MOMENTS[sel.moment]}"
        hero(tags, row["name"], meta)
    else:
        cat = catalog()
        row = cat.events.set_index("event_id").loc[sel.event_id]
        tags = [(page, "solid"), (f"Round {int(row['round'])} · {int(row['year'])}", "")]
        tags.append((str(row["event_format"]).replace("_", " "), ""))
        tags.append(("Real source data", ""))
        hero(tags, row["name"], f"{row['location']}, {row['country']} · FastF1 catalog snapshot")


def table(frame: pd.DataFrame, column_config: dict | None = None, max_rows: int = 25, **kwargs):
    """Dataframe with readable headers, no index, and a height that fits up to ``max_rows``."""
    frame = filter_grid(filter_models(frame))
    config = {c: str(c).replace("_", " ").capitalize() for c in frame.columns}
    config.update(column_config or {})
    kwargs.setdefault("hide_index", True)
    kwargs.setdefault("width", "stretch")
    kwargs.setdefault("height", 35 * (min(len(frame), max_rows) + 1) + 3)
    return st.dataframe(frame, column_config=config, **kwargs)


def tower(rows: list[tuple[int, str, str]]):
    """Timing-tower list: (position, driver, value) rows."""
    body = "".join(
        f'<div class="row"><span class="pos">{html.escape(str(p))}</span>'
        f'<span class="drv">{html.escape(d)}</span><span class="val">{html.escape(v)}</span></div>'
        for p, d, v in rows
    )
    st.html(f'<div class="f1-tower">{body}</div>')


def empty_state(headline: str, body: str | None = None, icon: str = "⏸"):
    st.html(
        f'<div class="f1-empty"><div class="h">{html.escape(icon)}&nbsp; '
        f"{html.escape(headline)}</div>"
        f'<div class="b">{html.escape(body or "")}</div></div>'
    )


def guard(render):
    """Run a view body; turn any failure into a visible, specific error state."""
    try:
        render()
    except ArtifactUnavailable as exc:
        st.error(f"A required saved artifact is unavailable: {exc}", icon=":material/inventory_2:")
        st.caption("Paths are configured in configs/app.toml (see README → Interface).")
    except Exception as exc:  # noqa: BLE001 - a view must never crash the whole app
        st.error(
            f"This view could not be rendered: {type(exc).__name__}: {exc}", icon=":material/error:"
        )
        with st.expander("Technical details"):
            st.code(traceback.format_exc())


def pct(value, digits=1) -> str:
    return "—" if value is None or pd.isna(value) else f"{value * 100:.{digits}f}%"


def num(value, digits=3) -> str:
    return "—" if value is None or pd.isna(value) else f"{value:.{digits}f}"


def training_range(card) -> str:
    events = card.train_events
    return f"{len(events)} weekends · {events[0]} → {events[-1]}"


def model_summary(card) -> str:
    params = ", ".join(f"{k} {v}" for k, v in card.metadata.get("parameters", {}).items())
    return f"{card.label}" + (f" ({params})" if params else "")


def provenance(forecast, extra: str | None = None):
    """Model ID, training range, cutoff and verification notes for a forecast."""
    with st.container(border=True):
        cols = st.columns([1.3, 1, 1])
        cols[0].markdown(f"**{forecast.source_label}**")
        for task, card in forecast.models.items():
            cols[0].caption(f"{task.title()}: {model_summary(card)} · ID `{card.model_id[:12]}`")
        first = next(iter(forecast.models.values()))
        cols[1].markdown("**Training data**")
        cols[1].caption(f"{training_range(first)}\n\nends {when(first.training_end)}")
        cols[2].markdown("**Prediction cutoff**")
        cols[2].caption(f"{when(forecast.cutoff)}\n\n{str(forecast.mode).replace('_', ' ')}")
        with st.expander("Lineage and integrity checks", icon=":material/verified:"):
            for line in forecast.checks:
                st.markdown(f"- {line}")
            if extra:
                st.markdown(f"- {extra}")


def export_button(forecast, label: str):
    exp = experiment()
    frame = fc.export_frame(exp, forecast)
    name = f"{forecast.event_id}_{forecast.task}_forecast_{exp.data_kind}.csv"
    st.download_button(
        label,
        frame.to_csv(index=False).encode(),
        file_name=name,
        mime="text/csv",
        icon=":material/download:",
        help="Predictions with model ID, training range, cutoff and per-driver data coverage. "
        "Official outcomes are not included.",
        on_click="ignore",
    )
