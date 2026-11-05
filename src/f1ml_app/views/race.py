"""Race predictions: full predicted order, podium probabilities, the qualifying inputs used."""

from pathlib import Path

import streamlit as st

from f1ml_app import charts, ui
from f1ml_app import forecasts as fc
from f1ml_app.catalog import NO_MODEL_REASON
from f1ml_app.experiment import FAMILY_LABELS
from f1ml_app.text import METRICS
from f1ml_app.theme import tokens


def model_choice(exp, sel, task):
    options = fc.candidates(exp, task, sel.event_id)
    usable = [(o["family"], o["strategy"]) for o in options if o["available"]]
    labels = {(o["family"], o["strategy"]): o["label"] for o in options}
    selected = exp.selected_config(task)
    return st.selectbox(
        f"Saved {'race-order' if task == 'race' else 'podium'} bundle",
        usable,
        index=usable.index(selected) if selected in usable else 0,
        format_func=lambda key: labels[key] + ("  ★ selected" if key == selected else ""),
        key=f"{task}_model_{sel.event_id}",
    )


def render():
    sel = ui.selection()
    ui.event_hero("Race predictions", sel)
    if sel.event_id is None:
        return
    if sel.source != "experiment":
        ui.empty_state("No race forecast for real catalog events", NO_MODEL_REASON)
        return
    exp = ui.experiment()
    race_choice = podium_choice = (None, None)
    if exp.role(sel.event_id) == "development_validation" and sel.moment != "pre_qualifying":
        cols = st.columns(2)
        with cols[0]:
            race_choice = model_choice(exp, sel, "race")
        with cols[1]:
            podium_choice = model_choice(exp, sel, "podium")
    forecast = ui.race_forecast(sel.event_id, sel.moment, race_choice, podium_choice)
    if not forecast.available:
        ui.empty_state(
            forecast.headline,
            forecast.explanation,
            icon="⏳" if sel.moment == "pre_qualifying" else "⏸",
        )
        if sel.moment == "pre_qualifying" and ui.qualifying_forecast(sel.event_id).available:
            st.page_link(
                Path(__file__).with_name("qualifying.py"),
                label="Available now: the qualifying forecast",
                icon=":material/timer:",
            )
        return

    ui.provenance(forecast)
    preds = forecast.predictions.sort_values("predicted_position")
    top = forecast.top_three or {}
    by_probability = preds.sort_values("podium_probability_rank")
    tiles = st.columns(4)
    tiles[0].metric("Field", f"{len(preds)} drivers", border=True)
    tiles[1].metric(
        "Decision threshold",
        ui.pct(preds.podium_threshold.iloc[0], 0),
        border=True,
        help="Probability at or above which the classifier calls a podium; chosen on earlier "
        "development weekends.",
    )
    tiles[2].metric(
        "Top-three sets",
        "Disagree" if top.get("disagree") else "Agree",
        border=True,
        help="Whether the race-order model's P1–P3 are the same drivers as the "
        "classifier's three highest probabilities.",
    )
    tiles[3].metric(
        "Sum of podium probabilities",
        f"{preds.podium_probability.sum():.2f}",
        border=True,
        help="Each probability is an independent per-driver estimate; they are not normalized "
        "to sum to exactly three.",
    )

    st.subheader("Two different top threes")
    left, right = st.columns(2)
    with left.container(border=True):
        st.markdown("**Race-order model · predicted P1–P3**")
        st.caption("Regression of official classification, ordered by ranking score.")
        ui.tower(
            [
                (int(r.predicted_position), r.driver_id, f"score {r.raw_score:.3f}")
                for r in preds.head(3).itertuples()
            ]
        )
    with right.container(border=True):
        st.markdown("**Podium classifier · three highest probabilities**")
        st.caption("Independent P(podium) per driver, after development-fitted calibration.")
        ui.tower(
            [
                (int(r.podium_probability_rank), r.driver_id, ui.pct(r.podium_probability))
                for r in by_probability.head(3).itertuples()
            ]
        )
    if top.get("disagree"):
        st.warning(
            "The two models disagree on the podium set; both are shown as produced.",
            icon=":material/compare_arrows:",
        )

    st.subheader("Podium probability by driver")
    threshold = float(preds.podium_threshold.iloc[0])
    st.altair_chart(charts.podium_probabilities(preds, threshold), width="stretch")
    st.caption(
        f"Bars are independent marginal probabilities. The dashed line is the decision "
        f"threshold ({threshold:.0%}) selected on earlier development weekends, never on this one."
    )

    st.subheader("Predicted race order")
    order = preds.merge(
        forecast.inputs[["driver_id", "team_id", "qualifying_position"]],
        on="driver_id",
        validate="one_to_one",
    )
    ui.table(
        order[
            [
                "predicted_position",
                "driver_id",
                "team_id",
                "qualifying_position",
                "raw_score",
                "podium_probability",
                "raw_podium_probability",
                "podium_probability_rank",
                "predicted_podium",
            ]
        ],
        hide_index=True,
        width="stretch",
        column_config={
            "predicted_position": st.column_config.NumberColumn("Pred.", format="P%d", width=64),
            "driver_id": st.column_config.TextColumn("Driver", width=160),
            "team_id": st.column_config.TextColumn("Team", width=140),
            "qualifying_position": st.column_config.NumberColumn("Qualified", format="P%d"),
            "raw_score": st.column_config.NumberColumn(
                "Ranking score",
                format="%.3f",
                help="Regression output; lower = ahead. Not a probability.",
            ),
            "podium_probability": st.column_config.ProgressColumn(
                "P(podium)", min_value=0, max_value=1, format="percent", color=tokens()["series"][0]
            ),
            "raw_podium_probability": st.column_config.NumberColumn(
                "Before calibration", format="percent"
            ),
            "podium_probability_rank": st.column_config.NumberColumn("Prob. rank", format="%d"),
            "predicted_podium": st.column_config.CheckboxColumn("≥ threshold"),
        },
    )

    st.subheader("Qualifying information used")
    coverage = fc.event_coverage(forecast.inputs, "race")
    q = st.columns(3)
    q[0].metric(
        "Official qualifying classification",
        f"{coverage['drivers_with_qualifying_classification']} / {coverage['drivers']}",
        border=True,
    )
    q[1].metric(
        "Q1/Q2/Q3 pace available",
        f"{coverage['drivers_with_qualifying_pace']} / {coverage['drivers']}",
        border=True,
    )
    q[2].metric(
        "Independently evidenced grid",
        f"{coverage['drivers_with_known_grid']} / {coverage['drivers']}",
        border=True,
        help="Grid is used only from an external snapshot known before the cutoff; never "
        "from race results.",
    )
    inputs = forecast.inputs.sort_values("qualifying_position")
    ui.table(
        inputs[
            [
                "driver_id",
                "qualifying_position",
                "qualifying_rank_norm",
                "qualifying_q1_gap_pct",
                "qualifying_q2_gap_pct",
                "qualifying_q3_gap_pct",
                "known_grid_position",
                "race_rank_mean_3",
                "podium_rate_5",
            ]
        ],
        hide_index=True,
        width="stretch",
        column_config={
            "driver_id": "Driver",
            "qualifying_position": st.column_config.NumberColumn(
                "Qualifying position", format="P%d"
            ),
            "qualifying_rank_norm": st.column_config.NumberColumn("Normalized", format="%.2f"),
            "qualifying_q1_gap_pct": st.column_config.NumberColumn("Q1 gap", format="%.2f%%"),
            "qualifying_q2_gap_pct": st.column_config.NumberColumn("Q2 gap", format="%.2f%%"),
            "qualifying_q3_gap_pct": st.column_config.NumberColumn("Q3 gap", format="%.2f%%"),
            "known_grid_position": st.column_config.NumberColumn("Known grid", format="P%d"),
            "race_rank_mean_3": st.column_config.NumberColumn("Race form (3)", format="%.2f"),
            "podium_rate_5": st.column_config.NumberColumn("Podium rate (5)", format="percent"),
        },
    )
    st.caption(
        "Qualifying is the official classification, not the starting grid. Empty cells are "
        "genuinely missing (no lap timing, no evidenced grid) and are median-imputed."
    )

    st.subheader("Predicted vs official race result")
    outcome = fc.actual_outcome(exp, sel.event_id, "race", sel.moment)
    if not outcome.visible:
        ui.empty_state("Official race result not shown", outcome.message, icon="🔒")
    else:
        joined = fc.compare_to_actual(preds, outcome)
        race_m = fc.saved_event_metrics(exp, forecast, "race") or {}
        pod_m = fc.saved_event_metrics(exp, forecast, "podium") or {}
        m = st.columns(5)
        m[0].metric(
            METRICS["position_mae"]["name"], ui.num(race_m.get("position_mae"), 2), border=True
        )
        m[1].metric(
            "Winner correct", "Yes" if race_m.get("winner_accuracy") == 1 else "No", border=True
        )
        m[2].metric(
            METRICS["top3_overlap"]["name"], ui.pct(race_m.get("top3_overlap"), 0), border=True
        )
        m[3].metric(
            "Podium log loss",
            ui.num(pod_m.get("log_loss"), 4),
            border=True,
            help=METRICS["log_loss"]["what"],
        )
        m[4].metric(
            "Podium Brier",
            ui.num(pod_m.get("brier"), 4),
            border=True,
            help=METRICS["brier"]["what"],
        )
        st.caption(f"Metrics from the saved evaluation record. Outcomes: {outcome.source}.")
        left, right = st.columns([1.15, 1])
        with left:
            st.altair_chart(
                charts.predicted_vs_official(joined, "Race-order forecast"), width="stretch"
            )
        with right:
            podium = (
                preds[["driver_id", "podium_probability"]]
                .merge(
                    outcome.table[["driver_id", "official_position", "podium"]],
                    on="driver_id",
                    validate="one_to_one",
                )
                .sort_values("official_position")
            )
            ui.table(
                podium,
                hide_index=True,
                width="stretch",
                column_config={
                    "driver_id": "Driver",
                    "podium_probability": st.column_config.NumberColumn(
                        "P(podium)", format="percent"
                    ),
                    "official_position": st.column_config.NumberColumn("Official", format="P%d"),
                    "podium": st.column_config.CheckboxColumn("Podium"),
                },
            )

    if forecast.references is not None and len(forecast.references):
        with st.expander("Fixed-rule references for this weekend"):
            refs = forecast.references.assign(rule=forecast.references.method.map(FAMILY_LABELS))
            st.caption(
                "Qualifying-order rule: race order = qualifying order; podium = hard 0/1 for the "
                "qualifying top three. Recent-form rule: recent race ranks / podium rates."
            )
            ui.table(
                refs[
                    [
                        "task",
                        "rule",
                        "driver_id",
                        "predicted_position",
                        "podium_probability",
                        "score_source",
                    ]
                ].sort_values(["task", "rule", "predicted_position", "driver_id"]),
                hide_index=True,
                width="stretch",
            )
    ui.export_button(forecast, "Download race & podium forecast (CSV)")


ui.guard(render)
