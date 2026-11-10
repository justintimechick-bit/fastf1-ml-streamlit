"""Data quality and methodology: coverage, fallbacks, exclusions, cutoffs, models, limits."""

import pandas as pd
import streamlit as st

from f1ml.domain import utc
from f1ml_app import ui
from f1ml_app.experiment import FAMILY_LABELS, ROLE_LABELS, STRATEGY_LABELS
from f1ml_app.settings import MODEL_ALIASES
from f1ml_app.text import CUTOFF_RULES, FAMILIES, LIMITATIONS, PREPROCESSING, STRATEGIES
from f1ml_app.theme import tokens

SESSION_ORDER = {
    "Practice 1": "P1",
    "Practice 2": "P2",
    "Practice 3": "P3",
    "Sprint Qualifying": "Sprint Q",
    "Sprint": "Sprint",
    "Qualifying": "Qualifying",
    "Race": "Race",
}
FEATURE_GROUPS = {
    "Practice pace": ["practice_gap_pct"],
    "Previous qualifying": ["previous_qualifying_rank"],
    "Qualifying form (3)": ["qualifying_rank_mean_3"],
    "Qualifying form (5)": ["qualifying_rank_mean_5"],
    "Race form (3)": ["race_rank_mean_3"],
    "Teammate delta (3)": ["teammate_qualifying_delta_mean_3"],
    "Team current-season form": ["team_qualifying_strength_mean_3"],
    "Team prior (older seasons)": ["team_prior_qualifying_strength"],
    "Circuit prior": ["circuit_prior_qualifying_rank"],
    "Podium rate (5)": ["podium_rate_5"],
    "Qualifying Q1–Q3 pace (race)": [
        "qualifying_q1_gap_pct",
        "qualifying_q2_gap_pct",
        "qualifying_q3_gap_pct",
    ],
    "Known grid (race)": ["known_grid_rank_norm"],
}


def _cell(row) -> str:
    if not row.in_model_scope:
        return "· out of scope"
    if row.completed and row.saved_table:
        return "✓ saved"
    if row.completed:
        return "✓ no data"
    return "✗ pending"


def coverage(exp, cat):
    st.subheader("Model experiment · session availability")
    if exp is None:
        ui.empty_state("Experiment unavailable", "The frozen experiment could not be loaded.")
    else:
        rows = []
        for event in exp.events.itertuples():
            cells = {r.session: _cell(r) for r in exp.session_rows(event.event_id).itertuples()}
            rows.append(
                {
                    "event": event.event_id,
                    "role": ROLE_LABELS[event.role],
                    **{label: cells.get(kind, "—") for kind, label in SESSION_ORDER.items()},
                }
            )
        ui.table(pd.DataFrame(rows), {label: label for label in SESSION_ORDER.values()})
        st.caption(
            "✓ saved = completed with a saved table · ✗ pending = not completed (no results) · "
            "— = not part of that weekend format · out of scope = Sprint sessions. Practice is "
            "saved as participation summaries without lap timing, so comparable clean-lap pace "
            "cannot be computed for any driver."
        )
        ok, _ = exp.final_available()
        if ok:
            ui.table(
                exp.final["dataset_coverage"],
                hide_index=True,
                width="stretch",
                column_config={"year": st.column_config.NumberColumn("Season", format="%d")},
            )
    st.subheader("FastF1 catalog · real source coverage")
    if cat is None:
        ui.empty_state("Catalog unavailable", "No readable catalog snapshot was found.")
        return
    seasons = cat.coverage["seasons"]
    ui.table(
        seasons[
            [
                "year",
                "events",
                "result_sessions",
                "sessions_missing_targets",
                "practice_expected",
                "practice_with_laps",
                "qualifying_with_laps",
                "sessions_with_weather",
                "failures",
                "blocked_sessions",
                "future_sessions",
            ]
        ],
        hide_index=True,
        width="stretch",
        column_config={"year": st.column_config.NumberColumn("Season", format="%d")},
    )
    sessions = cat.coverage["sessions"]
    gp = sessions.loc[sessions.kind.isin(["Qualifying", "Race"])]
    summary = gp.groupby(["year", "kind"]).results_availability.value_counts().unstack(fill_value=0)
    st.markdown("**Official Q/R results by season**")
    ui.table(
        summary.reset_index(),
        hide_index=True,
        width="stretch",
        column_config={"year": st.column_config.NumberColumn("Season", format="%d")},
    )
    failures = cat.coverage["failures"]
    st.markdown("**Acquisition failures by type**")
    ui.table(
        failures.groupby(["component", "type"]).size().rename("rows").reset_index(),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        f"Snapshot as of {ui.when(cat.as_of)}. Timing (laps/weather) failed for the whole "
        "catalog; rate-limited result sessions remain recorded failures, never filled in."
    )


def fallbacks(exp):
    st.markdown(PREPROCESSING)
    if exp is None:
        return
    frames = [exp.qualifying_features, exp.race_features]
    ok, _ = exp.final_available()
    if ok:
        frames += [exp.final["qualifying_inputs"], exp.final["race_inputs"]]
    inputs = pd.concat(frames, ignore_index=True)
    rows = []
    for event_id, group in inputs.groupby("event_id"):
        row = {"event": event_id}
        for name, columns in FEATURE_GROUPS.items():
            # Each input is measured on the cutoff rows of the task that uses it.
            rows_for = group.loc[group.task.eq("race" if name.endswith("(race)") else "qualifying")]
            row[name] = (
                float(1 - rows_for[columns].notna().any(axis=1).mean()) if len(rows_for) else None
            )
        rows.append(row)
    st.markdown("**Share of driver rows with the input missing (median-imputed)**")
    ui.table(
        pd.DataFrame(rows),
        hide_index=True,
        width="stretch",
        column_config={
            k: st.column_config.ProgressColumn(
                k, min_value=0, max_value=1, format="percent", color=tokens()["series"][0]
            )
            for k in FEATURE_GROUPS
        },
    )
    st.caption(
        "Measured on qualifying-cutoff rows; race-only inputs on race-cutoff rows. Early "
        "weekends lack history by construction (no earlier eligible weekend)."
    )
    st.markdown(
        "**Fixed-rule fallback chain.** Recent-form rule: mean normalized qualifying rank over "
        "3 weekends → 5 weekends → previous weekend → neutral 0.5. Clean-practice rule: "
        "practice pace when at least two drivers have it, else the recent-form chain. The "
        "`score_source` column in each reference table records which fallback was used."
    )


def exclusions(exp):
    if exp is None:
        return
    st.markdown("**Weekends excluded when the manifest was frozen**")
    ui.table(pd.DataFrame(exp.manifest["excluded"]), hide_index=True, width="stretch")
    st.markdown("**Strategy exclusions in development folds**")
    tables = []
    for task, comparison in (
        ("qualifying", exp.qualifying_comparison),
        ("race/podium", exp.race_comparison),
    ):
        frame = comparison["eligibility"]
        frame = frame.loc[~frame.eligible].drop_duplicates(["event_id", "strategy", "reason"])
        tables.append(
            frame.assign(task=task, strategy=frame.strategy.map(STRATEGY_LABELS))[
                ["task", "event_id", "strategy", "reason", "training_event_count"]
            ]
        )
    ui.table(pd.concat(tables), hide_index=True, width="stretch")
    target = exp.race_comparison["target_coverage"]
    st.markdown(
        f"**Training-label exclusions:** {int((~target.race_training_eligible).sum())} race and "
        f"{int((~target.podium_training_eligible).sum())} podium label rows excluded of "
        f"{len(target)}."
    )
    ok, _ = exp.final_available()
    if ok:
        final = exp.final
        st.markdown("**Final-test field and target coverage**")
        cov = final["coverage"].assign(outcomes=final["coverage"].outcomes.map(str))
        ui.table(cov, hide_index=True, width="stretch")
        groups = final["subgroups"].status.value_counts().rename("rows").reset_index()
        st.markdown("**Subgroup summaries** (need ≥ 3 weekends; smaller groups are suppressed)")
        ui.table(groups, hide_index=True, width="stretch")
        conditions = final["conditions"].condition.value_counts().to_dict()
        st.markdown(
            f"**Weather conditions after prediction:** {conditions}. Missing weather is "
            "*unknown*, not dry, so no wet/dry error conclusion is possible."
        )


def cutoffs(exp):
    ui.table(pd.DataFrame(CUTOFF_RULES), hide_index=True, width="stretch")
    st.markdown(
        "**Model lineage guard.** Before any replay forecast, `predict_checked` refuses a model "
        "if the weekend was used for its training, tuning or selection; if its training "
        "features, labels or weekends reach the cutoff; if it depends on a weekend that had not "
        "finished before this one started; or if the weekend is in the sealed final block."
    )
    if exp is None:
        return
    rows = []
    for item in exp.cutoffs.itertuples():
        spec = exp.gp_session(item.event_id, "Qualifying" if item.task == "qualifying" else "Race")
        lead = utc(spec["scheduled_start"]) - utc(item.cutoff) if spec else None
        rows.append(
            {
                "event": item.event_id,
                "task": item.task,
                "cutoff": pd.Timestamp(item.cutoff),
                "target_start": pd.Timestamp(spec["scheduled_start"]) if spec else None,
                "lead_minutes": lead.total_seconds() / 60 if lead is not None else None,
                "mode": item.mode.replace("_", " "),
                "origin": item.origin.replace("_", " "),
            }
        )
    frame = pd.DataFrame(rows)
    for column in ("cutoff", "target_start"):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    ui.table(
        frame.sort_values(["cutoff"]),
        hide_index=True,
        width="stretch",
        column_config={
            "cutoff": st.column_config.DatetimeColumn(
                "Cutoff (UTC)", format="D MMM YYYY HH:mm:ss", timezone="UTC"
            ),
            "target_start": st.column_config.DatetimeColumn(
                "Target session start", format="D MMM YYYY HH:mm", timezone="UTC"
            ),
            "lead_minutes": st.column_config.NumberColumn("Minutes before start", format="%.2f"),
        },
    )
    st.caption(
        "Historical reconstruction: source completion is an availability proxy, not "
        "proof of what was public at the original cutoff."
    )


def models(exp):
    cols = st.columns(2)
    active = [
        text
        for family, text in FAMILIES.items()
        if MODEL_ALIASES.get(family, family) in ui.selected_models()
    ]
    for i, (title, body) in enumerate(active):
        with cols[i % 2].container(border=True):
            st.markdown(f"**{title}**")
            st.caption(body)
    st.markdown("**Historical-data strategies**")
    for title, body in STRATEGIES.values():
        st.markdown(f"- **{title}** — {body}")
    st.markdown(
        "**Podium post-processing.** Calibration is a sigmoid fitted on the raw logit using "
        "later out-of-fold development weekends only; the decision threshold maximizes "
        "equal-event F1 on those weekends (ties: closest to 0.5, then higher). With too few "
        "earlier out-of-fold weekends, calibration stays identity and the threshold stays 0.5."
    )
    if exp is None:
        return
    rows = []
    for task in ("qualifying", "race", "podium"):
        card = exp.locked_card(task)
        if card is None:
            continue
        if MODEL_ALIASES.get(card.family, card.family) not in ui.selected_models():
            continue
        meta = card.metadata
        rows.append(
            {
                "task": task,
                "model": FAMILY_LABELS[card.family],
                "strategy": STRATEGY_LABELS[card.strategy],
                "parameters": ", ".join(f"{k}={v}" for k, v in meta.get("parameters", {}).items()),
                "weights": ", ".join(f"{k}={v}" for k, v in meta.get("weight_profile", {}).items())
                or "equal",
                "calibration": (meta.get("calibration") or {}).get("status", "n/a"),
                "threshold": (meta.get("threshold") or {}).get("value"),
                "selected_on": ", ".join(meta.get("selection_events", [])),
                "inputs": len(meta["model_feature_columns"]),
            }
        )
    st.markdown("**Locked models (Phase 7)**")
    ui.table(pd.DataFrame(rows), hide_index=True, width="stretch")


def limitations(exp, cat):
    for item in LIMITATIONS:
        st.markdown(f"- {item}")
    if cat is not None:
        seasons = cat.coverage["seasons"].set_index("year")
        counts = ", ".join(f"{y}: {int(r.result_sessions)}" for y, r in seasons.iterrows())
        st.caption(f"Real catalog official Q/R result sessions by season — {counts}.")


def render():
    ui.hero(
        [("Data quality", "solid"), ("Methodology", "")],
        "Data quality & methodology",
        "Missing sessions, fallbacks, exclusions, cutoffs and the model families used",
    )
    if ui.selection().source == "real":
        ui.real_review()
        return
    exp, _ = ui.try_experiment()
    cat, _ = ui.try_catalog()
    tabs = st.tabs(
        [
            "Coverage",
            "Feature fallbacks",
            "Exclusions",
            "Prediction cutoffs",
            "Models",
            "Limitations",
        ]
    )
    with tabs[0]:
        ui.guard(lambda: coverage(exp, cat))
    with tabs[1]:
        ui.guard(lambda: fallbacks(exp))
    with tabs[2]:
        ui.guard(lambda: exclusions(exp))
    with tabs[3]:
        ui.guard(lambda: cutoffs(exp))
    with tabs[4]:
        ui.guard(lambda: models(exp))
    with tabs[5]:
        ui.guard(lambda: limitations(exp, cat))


ui.guard(render)
