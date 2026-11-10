"""Model comparison: development selection results, separate from frozen final-test results."""

import pandas as pd
import streamlit as st

from f1ml_app import charts, ui
from f1ml_app.experiment import FAMILY_LABELS, STRATEGY_LABELS
from f1ml_app.text import METRICS, STRATEGIES, metric_help, metric_label

TASKS = {"qualifying": "Qualifying order", "race": "Race order", "podium": "Podium probability"}
TASK_METRICS = {
    "qualifying": ["position_mae", "spearman", "pole_accuracy", "top3_overlap", "top10_overlap"],
    "race": ["position_mae", "spearman", "winner_accuracy", "top3_overlap"],
    "podium": ["log_loss", "brier", "average_precision", "ece", "precision", "recall", "f1"],
}
PRIMARY = {"qualifying": "position_mae", "race": "position_mae", "podium": "log_loss"}
LOWER_IS_BETTER = {"position_mae", "log_loss", "brier", "ece"}
METHOD_LABELS = {"model": "Locked model", **FAMILY_LABELS}


SHORT_NAMES = {
    "top10_overlap": "Top-10 overlap",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "ece": "ECE",
    "raw_log_loss": "Raw log loss",
    "raw_brier": "Raw Brier",
    "raw_ece": "Raw ECE",
}


def metric_name(key):
    return SHORT_NAMES.get(key) or METRICS.get(key, {}).get("name", key.replace("_", " ").title())


def _averages(exp, task) -> tuple[pd.DataFrame, list[str]]:
    if task == "qualifying":
        summary = exp.qualifying_comparison["summary"]
        rows, common = summary["event_averages"], summary["common_evaluation_events"]
    else:
        summary = exp.race_comparison["summary"]
        rows, common = summary["event_averages"][task], summary["common_evaluation_events"][task]
    frame = ui.filter_models(pd.DataFrame(rows))
    frame["model"] = frame.family.map(FAMILY_LABELS)
    frame["strategy_label"] = frame.strategy.map(STRATEGY_LABELS)
    return frame, common


def development(exp):
    st.info(
        "**Development results** come from expanding chronological folds over the 2026 "
        "validation weekends. They were used to choose model family, strategy and "
        "hyperparameters, so they are optimistic. The frozen final test is a separate tab.",
        icon=":material/science:",
    )
    cols = st.columns(3)
    for col, key in zip(cols, ("2026_only", "equal", "weighted"), strict=True):
        title, body = STRATEGIES[key]
        with col.container(border=True):
            st.markdown(f"**{title}**")
            st.caption(body)

    task = st.segmented_control(
        "Task",
        list(TASKS),
        format_func=TASKS.get,
        default="qualifying",
        required=True,
        key="dev_task",
    )
    frame, common = _averages(exp, task)
    if frame.empty:
        st.info("No saved development scores for the active models.")
        return
    metrics = [m for m in TASK_METRICS[task] if m in frame]
    metric = st.segmented_control(
        "Metric",
        metrics,
        format_func=metric_name,
        default=PRIMARY[task],
        required=True,
        key=f"dev_metric_{task}",
    )
    better = "lower" if metric in LOWER_IS_BETTER else "higher"
    st.markdown(
        f"Primary comparison on the **{len(common)} common event(s)** where every strategy is "
        f"eligible: `{', '.join(common)}`. Equal weight per weekend."
    )
    chart = frame.assign(value=frame[metric], strategy=frame.strategy_label)
    st.altair_chart(charts.strategy_dots(chart, metric_name(metric), better), width="stretch")

    learned = frame.loc[frame.strategy.ne("fixed_rule")]
    rules = frame.loc[frame.strategy.eq("fixed_rule")]
    primary = PRIMARY[task]
    if learned.empty or not learned[primary].notna().any():
        st.info("No scored learned candidates in the active selection.")
        return
    pick = learned.sort_values(primary).iloc[0]
    best_rule = rules.sort_values(primary).iloc[0] if len(rules) else None
    with st.container(border=True):
        st.markdown(
            f":green-badge[Active selection winner] **{FAMILY_LABELS[pick.family]}** · "
            f"{STRATEGY_LABELS[pick.strategy]}: lowest common-event {metric_name(primary)} "
            f"among {len(learned)} learned candidates ({ui.num(pick[primary], 4)}; ties broken "
            "by predeclared order)."
        )
        if best_rule is not None and best_rule[primary] < pick[primary]:
            st.markdown(
                ":orange-badge[Reference wins] The fixed "
                f"{FAMILY_LABELS[best_rule.family].lower()} "
                f"scored better ({ui.num(best_rule[primary], 4)}). Baselines are kept as "
                "references; the learned model is not claimed best overall."
            )
        st.caption(
            "Selection rule: minimize equal-event common-cohort loss, then predeclared family "
            "and strategy order. One common weekend is weak evidence."
        )
    show = ["model", "strategy_label", "events", *metrics]
    if task == "podium":
        show += [c for c in ("raw_log_loss", "raw_brier", "raw_ece") if c in frame]
    ui.table(
        frame[show],
        hide_index=True,
        width="stretch",
        column_config={
            "model": "Model",
            "strategy_label": "Strategy",
            "events": st.column_config.NumberColumn("Events", format="%d"),
            **{
                m: st.column_config.NumberColumn(
                    metric_label(metric_name(m)), format="%.4f", help=metric_help(m)
                )
                for m in show[3:]
            },
        },
    )
    comparison = exp.qualifying_comparison if task == "qualifying" else exp.race_comparison
    events = ui.filter_models(comparison["event_metrics"])
    if "task" in events:
        events = events.loc[events.task.eq(task)]
    with st.expander("Every development fold, including early folds outside the common cohort"):
        st.caption(
            "2026-01 and 2026-02 have no 2026-only result: those folds had fewer than two "
            "earlier 2026 weekends. Their pooled/weighted scores are kept but not averaged "
            "with the common cohort."
        )
        table = events.assign(
            model=events.family.map(FAMILY_LABELS),
            strategy_label=events.strategy.map(STRATEGY_LABELS),
        )
        ui.table(
            table[
                [
                    "event_id",
                    "model",
                    "strategy_label",
                    "common_comparison",
                    "tuning_status",
                    *[m for m in metrics if m in table],
                ]
            ],
            hide_index=True,
            width="stretch",
        )
    with st.expander("Strategy eligibility per fold"):
        eligibility = comparison["eligibility"]
        eligibility = eligibility.loc[eligibility.task.eq("race" if task == "podium" else task)]
        ui.table(
            eligibility.assign(training_events=eligibility.training_events.map(", ".join)),
            hide_index=True,
            width="stretch",
        )


def final_test(exp):
    ok, problem = exp.final_available()
    if not ok:
        ui.empty_state("No final-test results", problem)
        return
    final = exp.final
    claim = final["ledger_claim"] or {}
    st.success(
        f"**Frozen · evaluated once.** Ledger claim {ui.when(claim.get('claimed_at'))} · lock "
        f"`{final['lock_receipt']['id'][:12]}` · evaluation `{final['receipt']['id'][:12]}`. "
        "Nothing on this page refits, recalibrates, re-thresholds or re-selects a model.",
        icon=":material/lock:",
    )
    if final["code_matches_lock"]:
        st.caption(
            ":material/verified: Current `src/f1ml` code fingerprint matches the lock "
            f"(`{final['lock_protocol']['code_id'][:12]}`)."
        )
    else:
        st.caption(
            ":material/warning: `src/f1ml` changed after the lock. Saved results stay valid as "
            "records, but any new evaluation would be exploratory."
        )
    if exp.is_synthetic:
        st.warning(
            "Synthetic holdout: the fixture generator makes race order equal qualifying order, "
            "so race/podium scores near perfection are a fixture property, not skill.",
            icon=":material/science:",
        )
    locked = []
    for task in ("qualifying", "race", "podium"):
        card = exp.locked_card(task)
        if card is None or ui.filter_models(pd.DataFrame({"family": [card.family]})).empty:
            continue
        locked.append(
            {
                "task": TASKS[task],
                "model": ui.model_summary(card),
                "training": ui.training_range(card),
                "training_end": ui.when(card.training_end),
                "model_id": card.model_id[:16],
            }
        )
    ui.table(
        pd.DataFrame(locked),
        hide_index=True,
        width="stretch",
        column_config={
            "task": "Task",
            "model": "Locked model",
            "training": "Training weekends",
            "training_end": "Training data ends",
            "model_id": "Model ID",
        },
    )

    task = st.segmented_control(
        "Task",
        list(TASKS),
        format_func=TASKS.get,
        default="qualifying",
        required=True,
        key="final_task",
    )
    card = exp.locked_card(task)
    if card is None or ui.filter_models(pd.DataFrame({"family": [card.family]})).empty:
        st.info(
            "The frozen result's locked family is outside the active selection. "
            "Other families have no sealed final-test predictions for this task."
        )
        return
    aggregate = final["aggregate_metrics"]
    aggregate = aggregate.loc[aggregate.task.eq(task)].copy()
    metrics = [m for m in TASK_METRICS[task] if m in set(aggregate.metric)]
    metric = st.segmented_control(
        "Metric",
        metrics,
        format_func=metric_name,
        default=PRIMARY[task],
        required=True,
        key=f"final_metric_{task}",
    )
    events = final["event_metrics"]
    events = events.loc[events.task.eq(task) & events.primary_eligible]
    methods = [METHOD_LABELS[m] for m in events.method.drop_duplicates()]
    plot = events.assign(value=events[metric], method_label=events.method.map(METHOD_LABELS))
    better = "lower" if metric in LOWER_IS_BETTER else "higher"
    st.altair_chart(charts.method_dots(plot, metric_name(metric), better, methods), width="stretch")

    rows = (
        aggregate.loc[aggregate.metric.isin(metrics)]
        .sort_values(["metric", "method"])
        .assign(
            method_label=lambda f: f.method.map(METHOD_LABELS),
            metric_label=lambda f: f.metric.map(metric_name),
            interval=lambda f: f.apply(
                lambda r: (
                    f"{r.ci_low:.4f}–{r.ci_high:.4f}"
                    if pd.notna(r.ci_low)
                    else f"not computed ({int(r.events)} events; needs 8)"
                ),
                axis=1,
            ),
        )
    )
    columns = [
        "metric_label",
        "method_label",
        "events",
        "mean",
        "minimum",
        "maximum",
        "loo_mean_min",
        "loo_mean_max",
        "event_sd",
        "interval",
    ]
    aggregate_config = {
        "metric_label": "Metric",
        "method_label": "Method",
        "events": st.column_config.NumberColumn("Events", format="%d"),
        "mean": st.column_config.NumberColumn("Mean", format="%.4f"),
        "minimum": st.column_config.NumberColumn("Event min", format="%.4f"),
        "maximum": st.column_config.NumberColumn("Event max", format="%.4f"),
        "loo_mean_min": st.column_config.NumberColumn("Leave-one-out min", format="%.4f"),
        "loo_mean_max": st.column_config.NumberColumn("Leave-one-out max", format="%.4f"),
        "event_sd": st.column_config.NumberColumn("Event SD", format="%.4f"),
        "interval": "Confidence interval",
    }
    ui.table(rows.loc[rows.metric.eq(metric), columns], aggregate_config)
    with st.expander(f"All {TASKS[task].lower()} metrics"):
        ui.table(rows[columns], aggregate_config)
    st.caption(
        "Ranges and leave-one-event-out means are sensitivity summaries over weekends, not "
        "confidence intervals. Driver rows are not treated as independent replicates."
    )
    paired = final["paired"]
    paired = paired.loc[paired.task.eq(task)]
    if len(paired):
        with st.expander("Paired differences vs fixed references"):
            st.caption("Positive mean = the locked model had lower loss than that reference.")
            ui.table(
                paired.assign(baseline=paired.baseline.map(FAMILY_LABELS))[
                    [
                        "baseline",
                        "metric",
                        "events",
                        "mean",
                        "minimum",
                        "maximum",
                        "loo_mean_min",
                        "loo_mean_max",
                    ]
                ],
                hide_index=True,
                width="stretch",
            )
    if task == "podium" and final["figures"]:
        st.subheader("Calibration (saved figures)")
        cols = st.columns(len(final["figures"]))
        for col, (name, path) in zip(cols, final["figures"].items(), strict=True):
            col.image(
                str(path),
                caption=("Outer development folds" if name == "development" else "Frozen holdout")
                + " "
                "· raw vs delivered probabilities",
                width="stretch",
            )
        st.caption("Sparse bins over a few weekends cannot establish calibration quality.")


def feature_groups(exp):
    data = exp.feature_experiments
    st.caption(
        "Phase 6 added optional groups cumulatively (history, sessions, practice, telemetry, "
        "weather) on identical development weekends. A group is promoted only with measured "
        "benefit, enough observed data and acceptable runtime."
    )
    retention = ui.filter_models(data["retention"]).assign(
        reasons=lambda f: f.reasons.map(lambda r: ", ".join(r).replace("_", " "))
    )
    ui.table(
        retention[
            [
                "task",
                "group",
                "family",
                "strategy",
                "decision",
                "mean_improvement_vs_core",
                "observed_fraction",
                "paired_events",
                "runtime_ratio_vs_core",
                "reasons",
            ]
        ],
        hide_index=True,
        width="stretch",
        column_config={
            "mean_improvement_vs_core": st.column_config.NumberColumn(
                "Improvement vs core", format="%.4f"
            ),
            "observed_fraction": st.column_config.NumberColumn("Observed", format="percent"),
            "runtime_ratio_vs_core": st.column_config.NumberColumn("Runtime ratio", format="%.2f×"),
        },
    )
    promoted = retention.decision.ne("do_not_promote").sum()
    st.markdown(
        f"**{promoted}** optional group(s) promoted; the locked models use core features only."
    )


def guide():
    st.caption(
        "Definitions match the scoring code in `f1ml.evaluation`. Each weekend counts equally."
    )
    for spec in METRICS.values():
        with st.container(border=True):
            st.markdown(f"**{spec['name']}** · {spec['better']} is better")
            st.markdown(spec["what"])
            if spec["formula"]:
                st.latex(spec["formula"])
            st.caption(spec["note"])
    st.markdown(
        "**Scores vs probabilities.** Qualifying and race-order models are regressions of a "
        "normalized rank; their raw scores only order drivers. Only the podium classifier "
        "produces probabilities, and each one is an independent marginal: the field's total "
        "is not forced to three."
    )


def render():
    ui.hero(
        [("Model comparison", "solid"), ("Saved evaluation records", "")],
        "Model comparison",
        "Baselines, three model families and three historical-data strategies",
    )
    if ui.selection().source == "real":
        ui.real_review()
        return
    exp, problem = ui.try_experiment()
    if exp is None:
        ui.empty_state("Experiment artifacts unavailable", problem)
        return
    tabs = st.tabs(
        [
            "Development · model selection",
            "Final test · frozen",
            "Feature-group experiments",
            "Metric guide",
        ]
    )
    with tabs[0]:
        ui.guard(lambda: development(exp))
    with tabs[1]:
        ui.guard(lambda: final_test(exp))
    with tabs[2]:
        ui.guard(lambda: feature_groups(exp))
    with tabs[3]:
        guide()


ui.guard(render)
