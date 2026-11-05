"""The feature registry is the matrix contract and the generated dictionary."""

from pathlib import Path

from f1ml.storage import digest, file_digest, write_json

FEATURE_VERSION = "3.0.0"
WINDOWS = (3, 5)
CATEGORIES = ("driver_id", "team_history_id", "circuit_id")
METADATA = (
    "event_id",
    "driver_id",
    "team_id",
    "task",
    "cutoff",
    "mode",
    "feature_version",
    "feature_code_id",
)
RACE_METADATA = ("qualifying_position", "known_grid_position")


def implementation_id():
    root = Path(__file__).parent
    return digest(
        {
            name: file_digest(root / name)
            for name in (
                "build.py",
                "history.py",
                "pace.py",
                "source.py",
                "contract.py",
                "dictionary.py",
            )
        }
    )


def registry() -> dict:
    features = {}

    def add(
        name,
        source,
        formula,
        *,
        race=False,
        missing="NaN; fold-only median imputation",
        era="Driver history may span 2024–2026; normalized ranks are not equal-car estimates.",
        kind="numeric",
        cutoff="Only earlier completed eligible weekends; target excluded before aggregation.",
    ):
        features[name] = {
            "name": name,
            "dtype": kind,
            "tasks": ["race"] if race else ["qualifying", "race"],
            "source": source,
            "formula": formula,
            "cutoff": cutoff,
            "missing_value_policy": missing,
            "regulation_era": era,
        }

    def statistic(name, source, formula, **kwargs):
        add(name, source, formula, **kwargs)
        add(
            f"{name}_missing",
            source,
            f"1 if {name} is missing, else 0",
            missing="Always observed: 0/1",
            **kwargs,
        )

    add(
        "driver_id",
        "Reviewed cutoff entrant list",
        "Stable source driver identity",
        kind="categorical",
        missing="Required; unknown training categories use OneHotEncoder(handle_unknown='ignore')",
        cutoff="Reviewed entrant identity known by the task cutoff.",
    )
    add(
        "team_history_id",
        "Reviewed entrant team era; otherwise explicit year-isolated fallback",
        "Performance-history ID, never an implicit predecessor-team mapping",
        kind="categorical",
        missing="Fallback unverified:<season>:<team_id>; never pooled across seasons",
        cutoff="Reviewed current team membership/history ID known by the task cutoff.",
        era=(
            "Renamed/new eras stay separate; older priors require the same "
            "explicitly reviewed history ID."
        ),
    )
    add(
        "circuit_id",
        "Verified event circuit/layout ID or reviewed event-context CSV",
        "Stable circuit/layout identity",
        kind="categorical",
        missing="unknown; circuit history then unavailable",
        cutoff="Circuit/layout context reviewed as known at the task cutoff.",
        era=(
            "Layout identity is required; no circuit identity inferred from raw lap"
            " times or event names."
        ),
    )
    for name, formula in (
        ("season", "Calendar season"),
        ("round", "Discovered GP round number"),
        ("regulation_2026", "1 if season == 2026, else 0"),
        ("field_size", "Number of reviewed cutoff entrants, including missing-data entrants"),
        ("circuit_identity_verified", "1 if circuit/layout ID has source evidence"),
    ):
        add(
            name,
            "Schedule or reviewed cutoff context",
            formula,
            missing="Required, or explicit 0 flag",
            era=(
                "2026 indicator identifies the regulation boundary; no implicit "
                "old-team form transfer."
            ),
            cutoff="Context known at the task cutoff; no results-derived roster.",
        )
    for kind in ("qualifying", "race"):
        statistic(
            f"previous_{kind}_rank",
            f"Official GP {kind} classification",
            (
                "(position-1)/(actual source field size-1) at the immediately previous "
                "eligible weekend; missing if absent"
            ),
        )
        for window in WINDOWS:
            for metric, source, formula in (
                (f"{kind}_rank", f"Official GP {kind} classification", "mean normalized rank"),
                (
                    f"teammate_{kind}_delta",
                    f"Official GP {kind} entries/classification",
                    (
                        "mean of driver normalized rank minus mean of other valid same-team "
                        "entrants, current season only"
                    ),
                ),
                (
                    f"team_{kind}_strength",
                    f"Official GP {kind} entries/classification",
                    (
                        "mean of per-weekend team mean normalized ranks, current season/current"
                        " history ID only"
                    ),
                ),
            ):
                statistic(
                    f"{metric}_mean_{window}",
                    source,
                    (
                        f"{formula}, within the previous {window} eligible calendar weekends; "
                        "no row-count window"
                    ),
                    era="Current-season team/teammate form only; older priors are separate."
                    if metric.startswith(("team_", "teammate_"))
                    else "Normalized driver history may span 2024–2026.",
                )
                add(
                    f"{metric}_count_{window}",
                    source,
                    "Number of usable weekend observations in that window",
                    missing="0 when no observations",
                )
        statistic(
            f"team_prior_{kind}_strength",
            f"Older-season official GP {kind} results",
            (
                "Mean of at most five older-season team weekend means for the exact "
                "reviewed team_history_id"
            ),
            era=(
                "Older-season prior is separate from current-season/2026 team form; "
                "unreviewed identities have no prior."
            ),
        )
        add(
            f"team_prior_{kind}_count",
            "Older-season team results",
            "Number of prior weekend means used",
            missing="0",
        )
        statistic(
            f"circuit_prior_{kind}_rank",
            f"Earlier driver GP {kind} results at verified circuit/layout",
            "Mean normalized rank at this circuit/layout over available prior eligible weekends",
        )
        add(
            f"circuit_prior_{kind}_count",
            "Same-circuit prior results",
            "Number of usable driver appearances",
            missing="0",
        )
    for metric, source in (
        ("podium", "Official prior race podium labels"),
        ("retirement", "Explicit prior race is_retired flag; classified retirement can be true"),
    ):
        for window in WINDOWS:
            statistic(
                f"{metric}_rate_{window}",
                source,
                (
                    f"Mean known binary outcome in previous {window} eligible calendar weekends; "
                    "DNS/DSQ are not automatically retirements"
                ),
            )
            add(f"{metric}_count_{window}", source, "Number of known outcomes used", missing="0")
    for name, formula in (
        ("history_count", "Usable driver race rows in the previous five eligible weekends"),
        ("history_quali_mean", "Alias of qualifying_rank_mean_5, normalized units"),
        ("history_race_mean", "Alias of race_rank_mean_5, normalized units"),
        ("history_podium_rate", "Alias of podium_rate_5"),
    ):
        add(
            name,
            "Shifted historical features",
            formula,
            missing="0 for history_count; NaN for absent means",
        )
    for name, formula in (
        (
            "practice_sessions",
            "Completed cutoff-available practice sessions with a driver observation",
        ),
        (
            "practice_clean_laps",
            (
                "Clean retained practice laps; excludes pits, "
                "deleted/inaccurate/generated laps and nonclear/unknown track status"
            ),
        ),
        ("practice_comparable_laps", "Clean laps with condition-matched peer comparisons"),
        ("practice_pace_sessions", "Practice sessions with a comparable driver pace estimate"),
    ):
        add(
            name,
            "Earlier current-weekend GP practice laps",
            formula,
            missing="0; keep the entrant row",
            cutoff="Only completed practices available before the chosen task cutoff.",
            era="Current-weekend observations only; fuel loads/run plans are unknown.",
        )
    statistic(
        "practice_gap_pct",
        "Earlier current-weekend clean practice laps and prior weather samples",
        (
            "Mean across sessions of median driver deficits: 100*(lap/best "
            "peer-cell lap-1); cells separate session, segment, compound, time bin,"
            " rainfall, temperature bin"
        ),
        cutoff=(
            "Only completed practices available before cutoff; weather matched "
            "backward to lap start."
        ),
        era="Current weekend only. No raw cross-circuit times. Fuel/run-plan differences remain.",
    )
    statistic(
        "qualifying_rank_norm",
        "Available actual target GP qualifying classification",
        (
            "(official qualifying position-1)/(actual qualifying field size-1); "
            "ambiguous/out-of-range/missing ranks masked"
        ),
        race=True,
        cutoff=(
            "Race cutoff only, after GP qualifying classification is available; never Q predictor."
        ),
    )
    add(
        "qualifying_classification_count",
        "Target GP qualifying classification",
        "1 if valid normalized Q rank is available, else 0",
        race=True,
        missing="0",
        cutoff="Race cutoff only, after target GP qualifying classification is available.",
    )
    for segment in ("q1", "q2", "q3"):
        statistic(
            f"qualifying_{segment}_gap_pct",
            "Explicitly segmented GP qualifying laps and GP qualifying weather",
            (
                f"Median within-cell deficit in {segment.upper()} only; "
                "same compound/time/rainfall/temperature cell as practice method"
            ),
            race=True,
            cutoff="Race cutoff only; GP qualifying session must be completed and available.",
            era="No Q1/Q3 raw-time comparison; unknown segment/conditions leave pace missing.",
        )
        add(
            f"qualifying_{segment}_lap_count",
            "Comparable segmented GP Q laps",
            "Number of driver laps used in that segment",
            race=True,
            missing="0",
            cutoff="Race cutoff only, after completed GP qualifying is available.",
        )
    statistic(
        "known_grid_rank_norm",
        "Independently evidenced grid snapshot in entrant CSV",
        "(grid-1)/(cutoff field size-1); pit-lane grid=0 has missing rank and its own flag",
        race=True,
        cutoff="Only grid_known_at <= race cutoff with source evidence; never target race results.",
    )
    add(
        "known_pit_lane_start",
        "Reviewed grid snapshot",
        "1 if independently known grid slot is 0",
        race=True,
        missing="0; grid-rank missing flag distinguishes unknown",
        cutoff="Only an independently evidenced grid snapshot known by the race cutoff.",
    )
    return features


def columns(task, dtype="numeric"):
    task = "race" if task == "podium" else task
    if task not in ("qualifying", "race"):
        raise ValueError(f"Unsupported task: {task}")
    return tuple(
        name
        for name, spec in registry().items()
        if task in spec["tasks"] and spec["dtype"] == dtype
    )


def write_dictionary(output: Path):
    document = {
        "feature_version": FEATURE_VERSION,
        "features": registry(),
        "metadata": {"common": METADATA, "race_only": RACE_METADATA},
        ("rank_policy"): (
            "No invented positions; invalid/ambiguous ranks masked; N is actual "
            "complete official field size."
        ),
    }
    write_json(output / "feature_dictionary.json", document)
    lines = [
        "# Feature dictionary",
        "",
        f"Schema {FEATURE_VERSION}. Lower normalized ranks/deficits are better.",
        "",
        "| Feature | Tasks | Source | Formula | Cutoff | Missing-value policy | Regulation era |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, spec in registry().items():
        values = [name, ", ".join(spec["tasks"])] + [
            spec[k]
            for k in ("source", "formula", "cutoff", "missing_value_policy", "regulation_era")
        ]
        lines.append("| " + " | ".join(v.replace("|", "\\|") for v in values) + " |")
    (output / "feature_dictionary.md").write_text("\n".join(lines) + "\n")
    return digest(document)
