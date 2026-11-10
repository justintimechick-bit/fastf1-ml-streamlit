"""Race predictions: full predicted order, podium probabilities, the qualifying inputs used."""

from pathlib import Path

import streamlit as st

from f1ml_app import charts, ui
from f1ml_app import forecasts as fc
from f1ml_app.catalog import NO_MODEL_REASON
from f1ml_app.experiment import FAMILY_LABELS
from f1ml_app.text import METRICS, metric_help, metric_label


def render():
    sel = ui.selection()
    ui.event_hero("Race predictions", sel)
    if sel.event_id is None:
        return
    if sel.source != "experiment":
        ui.empty_state("No race forecast for real catalog events", NO_MODEL_REASON)
        return
    exp = ui.experiment()
    forecasts = ui.selected_forecasts("race", sel.event_id, sel.moment)
    if not forecasts:
        forecast = ui.race_forecast(sel.event_id, sel.moment)
        ui.empty_state(
            "No race forecast for the active models" if forecast.available else forecast.headline,
            "Choose a model with an eligible saved bundle."
            if forecast.available
            else forecast.explanation,
            icon="⏳" if sel.moment == "pre_qualifying" else "⏸",
        )
        if sel.moment == "pre_qualifying" and ui.qualifying_forecast(sel.event_id).available:
            st.page_link(
                Path(__file__).with_name("qualifying.py"),
                label="Available now: the qualifying forecast",
                icon=":material/timer:",
            )
        return
    forecast = forecasts.get(ui.primary_model(), next(iter(forecasts.values())))
    missing = set(ui.selected_models()) - set(forecasts)
    if missing:
        st.caption("No eligible saved race forecast for: " + ", ".join(sorted(missing)))

    ui.provenance(forecast)
    preds = forecast.predictions.sort_values("predicted_position")
    top = forecast.top_three or {}
    outcome = fc.actual_outcome(exp, sel.event_id, "race", sel.moment)
    field = ui.forecast_fields(forecasts, outcome)
    by_probability = field.sort_values("podium_probability", ascending=False)
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

    ui.podium_spotlight(field)
    with st.expander("Podium classifier · three highest probabilities"):
        st.caption("These can differ from the race-order P1–P3 shown in the spotlight.")
        ui.tower(
            [
                (i + 1, str(row.driver), ui.pct(row.podium_probability))
                for i, row in enumerate(by_probability.head(3).itertuples())
            ]
        )
    if top.get("disagree"):
        st.warning(
            "The two models disagree on the podium set; both are shown as produced.",
            icon=":material/compare_arrows:",
        )

    st.subheader("Podium probability by driver")
    threshold = float(preds.podium_threshold.iloc[0])
    filtered = ui.filter_grid(field)
    if not filtered.empty:
        chart = (
            charts.model_field_comparison(filtered, "podium_probability")
            if len(ui.selected_models()) > 1
            else charts.podium_probabilities(filtered, threshold)
        )
        st.altair_chart(chart, width="stretch")
    st.caption(
        f"Bars are independent marginal probabilities. The dashed line is the decision "
        f"threshold ({threshold:.0%}) selected on earlier development weekends, never on this one."
    )

    st.subheader("Full-field race order and rank surprises")
    ui.prediction_field(field, key="race_field")

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
    if not outcome.visible:
        ui.empty_state("Official race result not shown", outcome.message, icon="🔒")
    else:
        ui.forecast_metrics(exp, forecasts, "race")
        ui.forecast_metrics(exp, forecasts, "podium")
        race_m = fc.saved_event_metrics(exp, forecast, "race") or {}
        pod_m = fc.saved_event_metrics(exp, forecast, "podium") or {}
        m = st.columns(5)
        m[0].metric(
            metric_label(METRICS["position_mae"]["name"]),
            ui.num(race_m.get("position_mae"), 2),
            border=True,
        )
        m[1].metric(
            "Winner correct", "Yes" if race_m.get("winner_accuracy") == 1 else "No", border=True
        )
        m[2].metric(
            metric_label(METRICS["top3_overlap"]["name"]),
            ui.pct(race_m.get("top3_overlap"), 0),
            border=True,
        )
        m[3].metric(
            metric_label("Podium log loss"),
            ui.num(pod_m.get("log_loss"), 4),
            border=True,
            help=metric_help("log_loss"),
        )
        m[4].metric(
            metric_label("Podium Brier"),
            ui.num(pod_m.get("brier"), 4),
            border=True,
            help=metric_help("brier"),
        )
        st.caption(f"Metrics from the saved evaluation record. Outcomes: {outcome.source}.")
        st.caption("The full-field chart and table above include these revealed actual results.")

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
