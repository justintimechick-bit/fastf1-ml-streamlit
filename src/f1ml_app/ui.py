"""Shared Streamlit plumbing: cached loaders, the global selection, headers and states."""

import html
import traceback
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st

from f1ml_app import forecasts as fc
from f1ml_app.catalog import RealCatalog, RealExperiment
from f1ml_app.experiment import ROLE_LABELS, ArtifactUnavailable, Experiment
from f1ml_app.settings import load_paths

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


def try_catalog() -> tuple[RealCatalog | None, str | None]:
    try:
        return catalog(), None
    except (FileNotFoundError, ValueError, KeyError) as exc:
        return None, str(exc)


# ----- global selection ------------------------------------------------------------------


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
        text = f"R{int(row['round']):02d} · {row['name']}"
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
    _apply_url_once()
    state = st.session_state
    if fixed_source is not None:
        state["source"] = fixed_source
    with st.sidebar:
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
        if source == "experiment":
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


def hero(kicker: list[tuple[str, str]], title: str, meta: str):
    tags = "".join(
        f'<span class="f1-tag {html.escape(cls)}">{html.escape(text)}</span>'
        for text, cls in kicker
    )
    st.html(
        f'<div class="f1-hero"><div class="f1-kicker">{tags}</div>'
        f'<div class="f1-title">{html.escape(title)}</div>'
        f'<div class="f1-meta">{html.escape(meta)}</div></div>'
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
