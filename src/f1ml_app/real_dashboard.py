"""Present saved acquisition evidence separately from measured model performance."""

import pandas as pd
import streamlit as st

from f1ml_app import ui


def coverage_summary(sessions):
    """Use past scheduled GP sessions as the acquisition denominator, never completion proof."""
    frame = sessions.loc[sessions.completion_state.ne("not_started")]
    rows = []
    for year, season in frame.groupby("year", sort=True):
        q = season.loc[season.kind.eq("Qualifying")]
        r = season.loc[season.kind.eq("Race")]
        practice = season.loc[season.kind.isin(["Practice 1", "Practice 2", "Practice 3"])]
        gp = season.loc[
            season.kind.isin(["Practice 1", "Practice 2", "Practice 3", "Qualifying", "Race"])
        ]
        row = {"Season": int(year)}
        for name, group, column in (
            ("Qualifying results", q, "results_rows"),
            ("Race results", r, "results_rows"),
            ("Practice laps", practice, "laps_rows"),
            ("Qualifying laps", q, "laps_rows"),
            ("Weather", gp, "weather_rows"),
        ):
            row[name] = f"{int(group[column].gt(0).sum())} / {len(group)}"
        rows.append(row)
    return pd.DataFrame(rows)


def event_coverage(exp, year):
    frame = exp.exclusions.loc[exp.exclusions.year.eq(year)].copy()
    q = exp.coverage["sessions"]
    q = q.loc[q.year.eq(year)]
    for kind, label in (("Qualifying", "Qualifying results"), ("Race", "Race results")):
        groups = q.loc[q.kind.eq(kind)].set_index("event_id")
        states = groups.apply(
            lambda row: (
                "Collected"
                if row.results_rows > 0
                else "Not started"
                if row.completion_state == "not_started"
                else "Not collected"
            ),
            axis=1,
        )
        frame[label] = frame.event_id.map(states)
    frame["Model eligibility"] = frame.eligible.map({True: "Eligible", False: "Excluded"})
    return frame.rename(columns={"event_id": "Event", "name": "Grand Prix", "round": "Round"})


def render():
    exp = ui.real_experiment()
    status = exp.status
    ui.hero(
        [("Real F1 observations", "solid"), ("2024–2026", "")],
        "Real-data experiment",
        "Coverage, source review and the path to a reproducible evaluation",
    )
    st.warning(
        "**Training is blocked.** Timing access and independent cutoff reviews are missing. "
        "There are no real model scores or forecasts yet.",
        icon=":material/pause_circle:",
    )
    sessions = exp.coverage["sessions"]
    cols = st.columns(4)
    cols[0].metric("Result sessions collected", int(sessions.results_rows.gt(0).sum()), border=True)
    cols[1].metric("Official result rows", int(sessions.results_rows.sum()), border=True)
    cols[2].metric("Eligible modeling events", status["eligible_events"], border=True)
    cols[3].metric("Real models trained", status["models_trained"], border=True)
    st.caption(
        f"Saved experiment {status['experiment_id']} · source horizon {ui.when(status['as_of'])}"
    )

    coverage, blockers, evaluation, provenance = st.tabs(
        ["Data coverage", "What blocks training", "Evaluation", "Provenance & downloads"],
        key="real_review_section",
        on_change="rerun",
    )
    if coverage.open:
        with coverage:
            st.subheader("Collected sessions by season")
            ui.table(coverage_summary(sessions))
            st.caption(
                "Collected / past scheduled sessions. Future and Sprint sessions are excluded from "
                "these denominators. A past schedule is not evidence of actual completion."
            )
            year = ui.selection().season or 2026
            st.subheader(f"{year} event coverage")
            frame = event_coverage(exp, year)
            missing_only = st.checkbox(
                "Show events with missing results only", key="real_missing_only"
            )
            if missing_only:
                frame = frame.loc[
                    frame["Qualifying results"].ne("Collected")
                    | frame["Race results"].ne("Collected")
                ]
            ui.table(
                frame[
                    [
                        "Round",
                        "Grand Prix",
                        "Qualifying results",
                        "Race results",
                        "qualifying_drivers",
                        "race_drivers",
                        "Model eligibility",
                    ]
                ],
                {"qualifying_drivers": "Q drivers", "race_drivers": "Race drivers"},
            )
            st.caption(
                "Use the sidebar season to inspect another year. Fields are preserved separately."
            )
            with st.expander("Session detail for the selected Grand Prix"):
                if ui.selection().event_id:
                    ui.table(
                        exp.sessions(ui.selection().event_id)[
                            [
                                "kind",
                                "scheduled_start",
                                "completion_state",
                                "results_rows",
                                "laps_rows",
                                "weather_rows",
                                "results_availability",
                                "laps_availability",
                            ]
                        ]
                    )
    if blockers.open:
        with blockers:
            st.subheader("Required before a core model can train")
            for title, text in (
                (
                    "Verified availability",
                    "There are no verified absolute completion timestamps. "
                    "Scheduled end times and retrieval times cannot stand in for completion.",
                ),
                (
                    "Independent entrant and cutoff reviews",
                    "Driver and team rosters must have evidence "
                    "that they were known at each prediction cutoff. "
                    "Eventual results cannot supply them.",
                ),
                (
                    "Enough reviewed 2026 history",
                    "The protocol requires four final-test events and "
                    "five earlier development events. "
                    "Zero events currently pass the dataset review.",
                ),
            ):
                st.markdown(f"**{title}**")
                st.write(text)
            st.info(
                "Practice, qualifying-lap and weather inputs remain unavailable. A reviewed core "
                "experiment can proceed without them once the required evidence exists."
            )
            with st.expander("Saved acquisition and exclusion details"):
                for line in status["blockers"]:
                    st.markdown(f"- {line}")
                ui.table(exp.exclusions[["event_id", "year", "reasons"]])
    if evaluation.open:
        with evaluation:
            left, right = st.columns(2)
            left.metric("Development events scored", status["development_evaluated_events"])
            right.metric("Final-test events scored", status["final_evaluated_events"])
            st.markdown(
                "**No final-test manifest or model lock has been created.** "
                "The real evaluation ledger is unclaimed."
            )
            st.caption(
                "Planned comparisons below have not run. Blank metric values are not zero loss."
            )
            task = st.selectbox(
                "Prediction task",
                ["qualifying", "race", "podium"],
                format_func={
                    "qualifying": "Qualifying order",
                    "race": "Race order",
                    "podium": "Podium probabilities",
                }.get,
                key="real_evaluation_task",
            )
            records = exp.evaluation_records.loc[exp.evaluation_records.task.eq(task)].copy()
            records["status"] = "Not run"
            ui.table(records)
            st.info(status["prior_consultation"])
            with st.expander("Predeclared evaluation plan"):
                st.json(status["protocol"])
    if provenance.open:
        with provenance:
            st.markdown("**Historical reconstruction from real source observations.**")
            st.write("Training cutoffs: none; no real model has been fitted.")
            st.caption("Synthetic demo models and their sealed evaluation remain separate.")
            with st.expander("Snapshot identity and integrity"):
                st.code(status["source_snapshot_id"], language=None)
                st.caption(f"Experiment receipt: {exp.experiment_receipt['id']}")
                st.caption(
                    "Saved receipts were verified before display. The source snapshot is fixed."
                )
            ui.table(exp.group_coverage)
            downloads = (
                ("Review report", exp.report.encode(), "review.md", "text/markdown"),
                (
                    "Season coverage",
                    exp.coverage["seasons"].to_csv(index=False).encode(),
                    "coverage.csv",
                    "text/csv",
                ),
                (
                    "Event exclusions",
                    exp.exclusions.to_csv(index=False).encode(),
                    "exclusions.csv",
                    "text/csv",
                ),
                (
                    "Evaluation records",
                    exp.evaluation_records.to_csv(index=False).encode(),
                    "evaluation.csv",
                    "text/csv",
                ),
            )
            for label, data, suffix, mime in downloads:
                st.download_button(
                    label, data, file_name=f"{status['experiment_id']}-{suffix}", mime=mime
                )
