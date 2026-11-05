"""Qualifying predictions: full predicted order, practice/recent form, predicted vs official."""

import streamlit as st

from f1ml_app import charts, ui
from f1ml_app import forecasts as fc
from f1ml_app.catalog import NO_MODEL_REASON
from f1ml_app.experiment import FAMILY_LABELS
from f1ml_app.text import METRICS

FORM_COLUMNS = {
    "predicted_position": st.column_config.NumberColumn("Pred.", format="P%d", width=64),
    "driver_id": st.column_config.TextColumn("Driver", width=160),
    "team_id": st.column_config.TextColumn("Team", width=140),
    "raw_score": st.column_config.NumberColumn(
        "Ranking score",
        format="%.3f",
        help="Unclipped normalized-rank estimate; lower = further ahead. Not a probability "
        "and not a lap time.",
    ),
    "practice_sessions": st.column_config.NumberColumn("Practice sessions", format="%d"),
    "practice_gap_pct": st.column_config.NumberColumn(
        "Practice pace gap",
        format="%.2f%%",
        help="Mean clean-lap deficit to the session reference in matching conditions. "
        "None = no comparable clean laps (median-imputed).",
    ),
    "previous_qualifying_rank": st.column_config.NumberColumn(
        "Last qualifying",
        format="%.2f",
        help="Normalized rank at the previous eligible weekend: 0 = pole, 1 = last.",
    ),
    "qualifying_rank_mean_3": st.column_config.NumberColumn(
        "Avg. last 3",
        format="%.2f",
        help="Mean normalized qualifying rank over the previous "
        "three eligible weekends (missing appearances do not extend the window).",
    ),
    "history_count": st.column_config.NumberColumn("History weekends", format="%d"),
    "imputed_inputs": st.column_config.NumberColumn(
        "Imputed inputs",
        format="%d",
        help="Numeric model inputs missing for this driver, filled with the training-fold median.",
    ),
}


def model_choice(exp, sel):
    """Saved outer-fold candidates for a development weekend; default = selected config."""
    options = fc.candidates(exp, "qualifying", sel.event_id)
    usable = [(o["family"], o["strategy"]) for o in options if o["available"]]
    labels = {(o["family"], o["strategy"]): o["label"] for o in options}
    selected = exp.selected_config("qualifying")
    pick = st.selectbox(
        "Saved model bundle for this weekend",
        usable,
        index=usable.index(selected) if selected in usable else 0,
        format_func=lambda key: (
            labels[key] + ("  ★ selected in development" if key == selected else "")
        ),
        key=f"qualifying_model_{sel.event_id}",
        help="Outer-fold bundles trained only on weekends before this one. Viewing a "
        "candidate does not change any saved selection.",
    )
    missing = [o for o in options if not o["available"]]
    if missing:
        st.caption(
            "Not trained for this weekend: "
            + "; ".join(f"{o['label']} ({o['reason']})" for o in missing)
        )
    return pick


def official_only(exp, sel):
    outcome = fc.actual_outcome(exp, sel.event_id, "qualifying", sel.moment)
    if outcome.visible:
        st.subheader("Official qualifying classification")
        st.caption(
            f"{outcome.source}. Shown for reference; there is no eligible forecast to compare."
        )
        ui.table(
            outcome.table[["official_position", "driver_id", "outcome", "source_status"]],
            {
                "official_position": st.column_config.NumberColumn("Pos", format="P%d", width=70),
                "driver_id": "Driver",
                "source_status": "Status",
            },
        )
    elif outcome.message:
        st.caption(outcome.message)


def render():
    sel = ui.selection()
    ui.event_hero("Qualifying predictions", sel)
    if sel.event_id is None:
        return
    if sel.source != "experiment":
        ui.empty_state("No qualifying forecast for real catalog events", NO_MODEL_REASON)
        return
    exp = ui.experiment()
    choice = (None, None)
    if exp.role(sel.event_id) == "development_validation":
        choice = model_choice(exp, sel)
    forecast = ui.qualifying_forecast(sel.event_id, *choice)
    if not forecast.available:
        ui.empty_state(forecast.headline, forecast.explanation)
        official_only(exp, sel)
        return

    ui.provenance(forecast)
    table = forecast.predictions.merge(
        forecast.inputs, on=["event_id", "driver_id", "cutoff", "mode"], validate="one_to_one"
    ).merge(
        fc.input_coverage(forecast.inputs, "qualifying")[["driver_id", "imputed_inputs"]],
        on="driver_id",
        validate="one_to_one",
    )
    coverage = fc.event_coverage(forecast.inputs, "qualifying")
    ordered = table.sort_values("predicted_position")
    top, stats = st.columns([1.35, 1])
    with top.container(border=True):
        st.markdown("**Predicted top three**")
        ui.tower(
            [
                (int(r.predicted_position), r.driver_id, f"score {r.raw_score:.3f}")
                for r in ordered.head(3).itertuples()
            ]
        )
    with stats:
        a, b = st.columns(2)
        a.metric("Field", f"{coverage['drivers']} drivers", border=True)
        b.metric(
            "Practice sessions usable",
            coverage["practice_sessions_used"],
            border=True,
            help="Completed practice sessions available strictly before the qualifying cutoff.",
        )
        a.metric(
            "Practice pace",
            f"{coverage['drivers_with_practice_pace']} / {coverage['drivers']}",
            border=True,
            help="Drivers with comparable clean-lap practice pace.",
        )
        b.metric(
            "History weekends",
            f"{coverage['max_history_weekends']} max",
            border=True,
            help="Most earlier eligible weekends available for any driver's form features.",
        )

    st.subheader("Predicted qualifying order")
    st.caption(
        "Full field, ordered by ascending ranking score; exact ties break by driver ID. Form "
        "values are normalized ranks (0 = front, 1 = back). The score is a regression output, "
        "**not a probability**."
    )
    ui.table(
        table[list(FORM_COLUMNS)].sort_values("predicted_position"),
        hide_index=True,
        width="stretch",
        column_config=FORM_COLUMNS,
    )
    if coverage["drivers_with_practice_pace"] == 0:
        st.caption(
            f"Practice: {coverage['practice_sessions_used']} session(s) completed before the "
            "cutoff, but no comparable clean-lap timing exists, so practice pace is missing "
            "for every driver (median-imputed with a missing indicator)."
        )

    st.subheader("Predicted vs official classification")
    outcome = fc.actual_outcome(exp, sel.event_id, "qualifying", sel.moment)
    if not outcome.visible:
        ui.empty_state("Official result not shown", outcome.message, icon="🔒")
    else:
        joined = fc.compare_to_actual(forecast.predictions, outcome)
        metrics = fc.saved_event_metrics(exp, forecast, "qualifying") or {}
        m = st.columns(4)
        m[0].metric(
            METRICS["position_mae"]["name"],
            ui.num(metrics.get("position_mae"), 2),
            border=True,
            help=METRICS["position_mae"]["what"],
        )
        m[1].metric(
            METRICS["spearman"]["name"],
            ui.num(metrics.get("spearman"), 3),
            border=True,
            help=METRICS["spearman"]["what"],
        )
        m[2].metric(
            "Pole correct", "Yes" if metrics.get("pole_accuracy") == 1 else "No", border=True
        )
        m[3].metric(
            METRICS["top3_overlap"]["name"],
            f"{int(metrics.get('top3_overlap_count', 0))} / "
            f"{int(metrics.get('top3_denominator', 3))}",
            border=True,
        )
        st.caption(f"Metrics from the saved evaluation record. Outcomes: {outcome.source}.")
        left, right = st.columns([1.15, 1])
        with left:
            st.altair_chart(charts.predicted_vs_official(joined, "Forecast"), width="stretch")
        with right:
            ui.table(
                joined,
                hide_index=True,
                width="stretch",
                column_config={
                    "driver_id": st.column_config.TextColumn("Driver", width=150),
                    "predicted_position": st.column_config.NumberColumn(
                        "Forecast", format="P%d", width=70
                    ),
                    "official_position": st.column_config.NumberColumn(
                        "Official", format="P%d", width=70
                    ),
                    "outcome": None,
                    "absolute_error": st.column_config.NumberColumn("Error", format="%d", width=56),
                    "places_vs_forecast": st.column_config.NumberColumn(
                        "Beat forecast",
                        format="%+d",
                        width=90,
                        help="Places gained vs the forecast; positive = qualified ahead of it.",
                    ),
                },
            )

    if forecast.references is not None and len(forecast.references):
        with st.expander("Fixed-rule references for this weekend"):
            st.caption(
                "Untuned baselines saved with the evaluation. The recent-form rule ranks by "
                "mean normalized qualifying rank (3, then 5 weekends, then previous, then a "
                "neutral 0.5); the clean-practice rule uses practice pace when at least two "
                "drivers have it, otherwise the same recent-form fallback."
            )
            refs = forecast.references.assign(rule=forecast.references.method.map(FAMILY_LABELS))
            ui.table(
                refs[
                    ["rule", "predicted_position", "driver_id", "raw_score", "score_source"]
                ].sort_values(["rule", "predicted_position"]),
                hide_index=True,
                width="stretch",
                column_config={
                    "rule": "Rule",
                    "predicted_position": st.column_config.NumberColumn("Pred.", format="P%d"),
                    "driver_id": "Driver",
                    "raw_score": st.column_config.NumberColumn("Rule score", format="%.3f"),
                    "score_source": "Score source (fallback used)",
                },
            )
    ui.export_button(forecast, "Download qualifying forecast (CSV)")


ui.guard(render)
