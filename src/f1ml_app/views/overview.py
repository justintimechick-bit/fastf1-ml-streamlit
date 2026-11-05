"""Event overview: role, weekend timeline, cutoffs, model training range and coverage."""

import pandas as pd
import streamlit as st

from f1ml_app import forecasts as fc
from f1ml_app import ui
from f1ml_app.catalog import NO_MODEL_REASON
from f1ml_app.experiment import ROLE_LABELS
from f1ml_app.theme import tokens

ROLE_TEXT = {
    "final_test": "One of the four reserved final-test weekends. Its forecasts were sealed "
    "once by the locked models (trained only on the seven development weekends) before any "
    "result was read. This interface displays them; it never recomputes or re-scores them.",
    "development_validation": "A 2026 development validation weekend. Forecasts replay the "
    "outer-fold model saved for this weekend, trained only on earlier weekends. Its results "
    "helped choose the model family and strategy, so its scores are optimistic compared with "
    "the final test.",
    "development_training": "A development training weekend. Every saved model was trained "
    "on it (or on later weekends), so no honest replay forecast exists.",
    "excluded": "Excluded when the manifest was frozen: no completed GP qualifying and race "
    "with official results. It can never be a training, validation or test weekend.",
    "outside_manifest": "Not part of the frozen experiment manifest.",
}


def _forecast_status(forecast) -> str:
    return forecast.status_label


def _timeline(exp, event_id, forecasts) -> pd.DataFrame:
    sessions = exp.session_rows(event_id)
    rows = [
        {
            "when": r.scheduled_start,
            "item": r.session,
            "status": r.status,
            "qualifying_forecast": r.usable_at_qualifying_cutoff if r.in_model_scope else None,
            "race_forecast": r.usable_at_race_cutoff if r.in_model_scope else None,
        }
        for r in sessions.itertuples()
    ]
    for task, label in (
        ("qualifying", "Qualifying forecast cutoff"),
        ("race", "Race forecast cutoff"),
    ):
        cut = exp.cutoff(event_id, task)
        if cut:
            rows.append(
                {
                    "when": pd.Timestamp(cut[0]),
                    "item": f"⏱ {label}",
                    "status": cut[1].replace("_", " "),
                }
            )
    seen = set()
    for forecast in forecasts:
        for task, card in forecast.models.items():
            if card.model_id in seen:
                continue
            seen.add(card.model_id)
            rows.append(
                {
                    "when": pd.Timestamp(card.training_end),
                    "item": f"■ {task.title()} model: training ends",
                    "status": f"Last training weekend {card.train_events[-1]} · "
                    f"{card.family.replace('_', ' ')}",
                }
            )
    frame = pd.DataFrame(rows)
    frame["when"] = pd.to_datetime(frame["when"], utc=True)
    return frame.sort_values("when").reset_index(drop=True)


def experiment_overview(sel):
    exp = ui.experiment()
    event = exp.events.set_index("event_id").loc[sel.event_id]
    role = event["role"]
    quali = ui.qualifying_forecast(sel.event_id)
    race = ui.race_forecast(sel.event_id, sel.moment)
    st.markdown(f"**{ROLE_LABELS[role]}.** {ROLE_TEXT[role]}")
    if role == "excluded":
        st.caption(f"Manifest exclusion reason: `{exp.exclusion_reason(sel.event_id)}`")

    sessions = exp.session_rows(sel.event_id)
    gp = sessions.loc[sessions.in_model_scope]
    inputs = quali.inputs if quali.available else race.inputs if race.available else None
    tiles = st.columns(4)
    tiles[0].metric(
        "Entrant field",
        len(inputs) if inputs is not None else "—",
        border=True,
        help="Reviewed cutoff-known entrants. Not inferred from results.",
    )
    tiles[1].metric("GP sessions completed", f"{int(gp.completed.sum())} / {len(gp)}", border=True)
    tiles[2].metric(
        "Practice usable for qualifying",
        int(gp.usable_at_qualifying_cutoff.sum()),
        border=True,
        help="Completed practice sessions available strictly before the qualifying cutoff.",
    )
    tiles[3].metric(
        "Forecasts available",
        f"{int(quali.available) + int(race.available)} / 2",
        border=True,
        help="Qualifying forecast and post-qualifying race/podium forecast at this replay moment.",
    )

    st.subheader("Weekend timeline and prediction cutoffs")
    st.caption(
        "A session can feed a forecast only if it completed strictly before that forecast's "
        "cutoff (pipeline rule `features.source.available`). Sprint-format sessions are "
        "outside the model scope."
    )
    ui.table(
        _timeline(exp, sel.event_id, [quali, race]),
        hide_index=True,
        width="stretch",
        column_config={
            "when": st.column_config.DatetimeColumn(
                "When (UTC)", format="ddd D MMM · HH:mm", timezone="UTC", width="small"
            ),
            "item": st.column_config.TextColumn("Session or milestone", width="medium"),
            "status": st.column_config.TextColumn("Status", width="medium"),
            "qualifying_forecast": st.column_config.CheckboxColumn(
                "Qualifying input",
                width="small",
                help="Usable by the qualifying forecast at its cutoff.",
            ),
            "race_forecast": st.column_config.CheckboxColumn(
                "Race input",
                width="small",
                help="Usable by the race/podium forecast at its cutoff.",
            ),
        },
    )

    st.subheader("Models and training cutoffs")
    cols = st.columns(2)
    for col, forecast, title in (
        (cols[0], quali, "Qualifying forecast"),
        (cols[1], race, "Race & podium forecast"),
    ):
        with col.container(border=True):
            st.markdown(f"**{title}**")
            if not forecast.available:
                st.markdown(f":gray-badge[Unavailable] {forecast.headline}")
                st.caption(forecast.explanation)
                continue
            st.markdown(f":blue-badge[{forecast.source_label}]")
            for task, card in forecast.models.items():
                gap = pd.Timestamp(forecast.cutoff) - pd.Timestamp(card.training_end)
                st.markdown(
                    f"{task.title()}: **{ui.model_summary(card)}**  \n"
                    f"Trained on {ui.training_range(card)}; training data ends "
                    f"{ui.when(card.training_end)}, **{gap.days} days before** the cutoff "
                    f"({ui.when(forecast.cutoff)})."
                )
                st.caption(f"Model ID `{card.model_id}`")

    st.subheader("Data coverage at the cutoff")
    if inputs is None:
        ui.empty_state(
            "No feature rows for this weekend",
            "Without a reviewed entrant roster "
            "there is no field to describe. No drivers are invented.",
        )
    else:
        summary = fc.event_coverage(quali.inputs, "qualifying") if quali.available else None
        race_summary = fc.event_coverage(race.inputs, "race") if race.available else None
        c = st.columns(4)
        ref = summary or race_summary
        c[0].metric(
            "Drivers with prior history",
            f"{ref['drivers_with_history']} / {ref['drivers']}",
            border=True,
        )
        c[1].metric(
            "Clean-practice pace",
            f"{ref['drivers_with_practice_pace']} / {ref['drivers']}",
            border=True,
            help="Comparable clean-lap pace needs lap timing; missing pace stays missing.",
        )
        c[2].metric(
            "Avg. imputed model inputs",
            f"{ref['mean_imputed_inputs']:.1f} / {ref['model_inputs']}",
            border=True,
            help="Missing numeric inputs are filled with the training-fold median plus a "
            "missing indicator.",
        )
        c[3].metric(
            "Qualifying result known",
            f"{race_summary['drivers_with_qualifying_classification']} / {race_summary['drivers']}"
            if race_summary
            else "—",
            border=True,
            help="Official GP qualifying classification available at the race cutoff.",
        )
        if ref["drivers_with_practice_pace"] == 0:
            st.caption(
                "Practice sessions completed, but no comparable clean-lap timing exists in this "
                "source, so practice pace is missing for every driver and is median-imputed. "
                "History, team and circuit form carry the forecast."
            )

    st.subheader(f"{int(event['year'])} season at a glance")
    season = exp.events.loc[exp.events.year.eq(event["year"])]
    rows = []
    for item in season.itertuples():
        q = ui.qualifying_forecast(item.event_id)
        r = ui.race_forecast(item.event_id, "post_race")
        rows.append(
            {
                "event": f"R{item.round:02d} · {item.name}",
                "role": ROLE_LABELS[item.role],
                "qualifying": _forecast_status(q),
                "race": _forecast_status(r),
            }
        )
    ui.table(
        pd.DataFrame(rows),
        hide_index=True,
        width="stretch",
        column_config={
            "event": "Event",
            "role": "Experiment role",
            "qualifying": "Qualifying forecast",
            "race": "Race forecast (after the race)",
        },
    )


def catalog_overview(sel):
    cat = ui.catalog()
    if sel.source == "real":
        ui.real_review()
    event = cat.events.set_index("event_id").loc[sel.event_id]
    sessions = cat.sessions(sel.event_id)
    gp = sessions.loc[sessions.in_model_scope]
    for column in (
        "completion_state",
        "results_availability",
        "laps_availability",
        "weather_availability",
    ):
        sessions[column] = sessions[column].fillna("unknown").str.replace("_", " ")
    st.info(NO_MODEL_REASON, icon=":material/info:")
    tiles = st.columns(4)
    tiles[0].metric("Drivers observed", int(event.driver_count or 0), border=True)
    tiles[1].metric(
        "Official result sessions",
        f"{int(event.result_sessions or 0)} / 2",
        border=True,
        help="GP qualifying and race classifications retrieved from Jolpica via FastF1.",
    )
    tiles[2].metric(
        "Practice sessions with laps",
        f"{int(event.practice_with_laps or 0)} / {int(event.practice_expected or 0)}",
        border=True,
    )
    tiles[3].metric("Acquisition failures", int(event.failures or 0), border=True)

    st.subheader("Session availability")
    ui.table(
        sessions[
            [
                "kind",
                "scheduled_start",
                "completion_state",
                "results_availability",
                "laps_availability",
                "weather_availability",
                "driver_count",
                "failure_count",
            ]
        ],
        hide_index=True,
        width="stretch",
        column_config={
            "kind": "Session",
            "scheduled_start": st.column_config.DatetimeColumn(
                "Scheduled (UTC)", format="ddd D MMM YYYY · HH:mm", timezone="UTC"
            ),
            "completion_state": "Completion",
            "results_availability": "Official results",
            "laps_availability": "Lap timing",
            "weather_availability": "Weather",
            "driver_count": "Drivers",
            "failure_count": "Failures",
        },
    )
    st.caption(
        f"{len(gp)} GP sessions in scope. Completion times stay missing when timing is "
        "unavailable; no session end is invented."
    )
    for kind in ("Qualifying", "Race"):
        frame = cat.results(sel.event_id, kind)
        with st.expander(
            f"Official {kind.lower()} classification ({len(frame)} rows)", expanded=False
        ):
            if frame.empty:
                st.caption("Not retrieved in this snapshot (see failures below).")
            else:
                ui.table(
                    frame,
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "official_position": st.column_config.NumberColumn("Pos", format="%d"),
                        "abbreviation": "Code",
                        "driver": "Driver",
                        "source_constructorName": "Team",
                        "outcome": "Outcome",
                        "source_status": "Status",
                        "source_positionText": "Position text",
                    },
                )
    failures = cat.failures(sel.event_id)
    if len(failures):
        with st.expander(f"Acquisition failures ({len(failures)})"):
            ui.table(
                failures[["session_id", "component", "type", "message"]],
                hide_index=True,
                width="stretch",
            )

    st.subheader(f"{sel.season} season coverage")
    season = cat.events.loc[cat.events.year.eq(sel.season)].copy()
    season["event_format"] = season.event_format.str.replace("_", " ")
    ui.table(
        season[
            [
                "round",
                "name",
                "event_format",
                "result_sessions",
                "practice_with_laps",
                "failures",
                "future_sessions",
            ]
        ],
        hide_index=True,
        width="stretch",
        column_config={
            "round": st.column_config.NumberColumn("Round", format="%d"),
            "name": "Grand Prix",
            "event_format": "Format",
            "result_sessions": st.column_config.ProgressColumn(
                "Q/R results",
                min_value=0,
                max_value=2,
                format="%d / 2",
                color=tokens()["series"][0],
            ),
            "practice_with_laps": "Practice with laps",
            "failures": "Failures",
            "future_sessions": "Future sessions",
        },
    )


def render():
    sel = ui.selection()
    ui.event_hero("Event overview", sel)
    if sel.event_id is None:
        return
    (experiment_overview if sel.source == "experiment" else catalog_overview)(sel)


ui.guard(render)
