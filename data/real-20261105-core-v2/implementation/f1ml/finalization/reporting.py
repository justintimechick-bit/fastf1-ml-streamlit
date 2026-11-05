"""Event-level uncertainty, subgroup coverage and standalone calibration figures."""

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from f1ml.evaluation.race_metrics import PODIUM_METRICS, RACE_METRICS, reliability
from f1ml.features.pace import boolean, values
from f1ml.ingestion.tables import json_value
from f1ml.storage import save_table, write_json

QUALIFYING_METRICS = ("position_mae", "spearman", "pole_accuracy", "top3_overlap", "top10_overlap")


def table(root, name, frame):
    save_table(root / f"{name}.parquet", frame)
    frame.to_csv(root / f"{name}.csv", index=False, mode="x")


def uncertainty(values_, protocol):
    x = np.asarray(values_, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    result = {
        "events": n,
        "mean": float(x.mean()) if n else None,
        "event_sd": float(x.std(ddof=1)) if n > 1 else None,
        "minimum": float(x.min()) if n else None,
        "maximum": float(x.max()) if n else None,
        "loo_mean_min": None,
        "loo_mean_max": None,
        "ci_low": None,
        "ci_high": None,
        "interval_status": "insufficient_events_for_confidence_interval",
    }
    if n > 1:
        means = (x.sum() - x) / (n - 1)
        result.update(loo_mean_min=float(means.min()), loo_mean_max=float(means.max()))
    if n >= protocol["bootstrap_min_events"]:
        rng = np.random.default_rng(protocol["random_seed"])
        draws = rng.choice(x, size=(protocol["bootstrap_replicates"], n), replace=True).mean(axis=1)
        low, high = np.quantile(draws, [0.025, 0.975])
        result.update(
            ci_low=float(low),
            ci_high=float(high),
            interval_status="descriptive_event_bootstrap_serial_dependence_not_modeled",
        )
    return result


def aggregate(metrics, protocol):
    rows = []
    for (task, method), group in metrics.groupby(["task", "method"], sort=False):
        names = (
            QUALIFYING_METRICS
            if task == "qualifying"
            else RACE_METRICS
            if task == "race"
            else PODIUM_METRICS
        )
        for name in names:
            rows.append(
                {
                    "task": task,
                    "method": method,
                    "metric": name,
                    **uncertainty(group[name].dropna(), protocol),
                }
            )
    return pd.DataFrame(rows)


def paired_differences(metrics, protocol):
    rows = []
    for task, group in metrics.groupby("task"):
        loss = "log_loss" if task == "podium" else "position_mae"
        model = group.loc[group.method.eq("model"), ["event_id", loss]]
        for method, baseline in group.loc[~group.method.eq("model")].groupby("method"):
            pair = model.merge(
                baseline[["event_id", loss]],
                on="event_id",
                suffixes=("_model", "_baseline"),
                validate="one_to_one",
            )
            differences = pair[f"{loss}_baseline"] - pair[f"{loss}_model"]
            rows.append(
                {
                    "task": task,
                    "baseline": method,
                    "metric": loss,
                    "meaning": "positive means model has lower loss",
                    **uncertainty(differences, protocol),
                }
            )
    return pd.DataFrame(rows)


def session_condition(source, event_id, task):
    spec = next(
        s
        for s in source.event(event_id)["sessions"]
        if s["kind"] == ("Qualifying" if task == "qualifying" else "Race")
    )
    try:
        weather = source.component(spec, "weather")
        rain = boolean(values(weather, "rainfall", "Rainfall", "source_Rainfall"))
        known = rain.dropna()
        condition = "unknown"
        if len(known) >= 3:
            if known.any():
                condition = "wet_observed"
            elif len(known) == len(weather):
                condition = "dry_observed"
        return {
            "event_id": event_id,
            "task": task,
            "condition": condition,
            "weather_rows": len(weather),
            "known_rain_rows": len(known),
            "interpretation": "observed session samples only; not complete weather reconstruction",
        }
    except (ValueError, KeyError, OSError) as exc:
        return {
            "event_id": event_id,
            "task": task,
            "condition": "unknown",
            "weather_rows": 0,
            "known_rain_rows": 0,
            "error": str(exc),
        }


def subgroup_errors(tables, protocol):
    rows = []
    for task, frame in tables.items():
        for dimension in ("driver_id", "team_at_cutoff", "condition"):
            for (method, value), group in frame.groupby(["method", dimension], dropna=False):
                valid = group.loc[group.primary_eligible & group._merge.eq("both")]
                events = valid.event_id.nunique()
                enough = events >= protocol["subgroup_min_events"] and str(value) != "unknown"
                error = (
                    (valid.podium_probability - valid.podium.astype(float)) ** 2
                    if task == "podium"
                    else (valid.predicted_position - valid.official_position.astype(float)).abs()
                )
                rows.append(
                    {
                        "task": task,
                        "method": method,
                        "dimension": dimension,
                        "value": str(value),
                        "events": events,
                        "driver_rows": len(valid),
                        "metric": "brier" if task == "podium" else "position_mae",
                        "mean_error": float(error.mean()) if enough and len(error) else None,
                        "status": "descriptive_only"
                        if enough
                        else "insufficient_events_or_unknown_condition",
                    }
                )
    return pd.DataFrame(rows)


def calibration_figure(frame, path: Path, title, bins):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "f1ml-matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), constrained_layout=True)
    axes[0].plot([0, 1], [0, 1], color="0.6", linestyle="--", label="reference")
    cells = []
    if not frame.empty:
        for method, group in frame.groupby("method", sort=False):
            group = group.loc[group.podium.notna()]
            if group.empty:
                continue
            weights = 1 / group.groupby("event_id").driver_id.transform("size")
            for column in (
                ("raw_podium_probability", "podium_probability")
                if method == "model"
                else ("podium_probability",)
            ):
                _, records = reliability(group[column], group.podium.astype(float), bins, weights)
                valid = [r for r in records if r["weight"] > 0]
                label = method + (" raw" if column.startswith("raw_") else " delivered")
                axes[0].plot(
                    [r["mean_probability"] for r in valid],
                    [r["observed_rate"] for r in valid],
                    marker="o",
                    linestyle=":" if column.startswith("raw_") else "-",
                    label=label,
                )
                cells.extend({"method": method, "probability_source": column, **r} for r in records)
            if method == "model":
                axes[1].hist(
                    group.podium_probability,
                    bins=np.linspace(0, 1, bins + 1),
                    weights=weights,
                    color="#336c9f",
                    edgecolor="white",
                )
    else:
        axes[0].text(0.5, 0.5, "No eligible probability/label pairs", ha="center")
    axes[0].set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Mean predicted podium probability",
        ylabel="Observed podium frequency",
        title="Reliability; no fit on these outcomes",
    )
    axes[0].legend(fontsize=7, loc="best")
    axes[1].set(
        xlim=(0, 1),
        xlabel="Delivered model probability",
        ylabel="Event-balanced mass",
        title="Probability distribution",
    )
    fig.suptitle(
        title + f"\n{frame.event_id.nunique() if not frame.empty else 0} weekends; descriptive bins"
    )
    fig.savefig(path.with_suffix(".png"), dpi=160)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)
    return pd.DataFrame(cells)


def report(output, protocol, metrics, summaries, coverage, subgroups):
    label = (
        "SYNTHETIC HOLDOUT SOFTWARE VALIDATION"
        if protocol["data_kind"] == "synthetic_validation"
        else "OBSERVED-SOURCE HOLDOUT EVALUATION"
    )
    exploratory = protocol["exploratory"]
    lines = [
        f"# {label}",
        "",
        "**EXPLORATORY — fresh future holdout required.**"
        if exploratory
        else "One recorded primary evaluation; no fitting inside the test block.",
        "",
        "## Locked choices",
        "",
    ]
    for task, choice in protocol["models"].items():
        meta = choice["metadata"]
        lines.append(
            f"- {task}: {choice['family']}, {choice['strategy']}; "
            f"parameters `{meta['parameters']}`; "
            f"{len(meta['train_events'])} pre-block training weekends; core inputs."
        )
    lines += [
        "",
        "History is fixed to pre-block weekends. Qualifying results enter race/podium inputs only.",
        "Preprocessing, weights, calibration and thresholds were copied from development refits.",
        (
            "All qualifying forecasts were sealed before race inputs; all "
            "forecasts were sealed before scoring."
        ),
        "",
        "## Observed results in the supplied source",
        "",
        "| Task | Method | Metric | Events | Mean | Event range | Leave-one-event-out mean range |",
        "| --- | --- | --- | ---: | ---: | --- | --- |",
    ]
    for row in summaries.to_dict("records"):
        if row["metric"] not in (
            "position_mae",
            "spearman",
            "pole_accuracy",
            "winner_accuracy",
            "top3_overlap",
            "log_loss",
            "brier",
            "average_precision",
            "ece",
            "precision",
            "recall",
        ):
            continue

        def show(v):
            return "—" if pd.isna(v) else f"{v:.4f}"

        lines.append(
            f"| {row['task']} | {row['method']} | {row['metric']} | {row['events']} | "
            f"{show(row['mean'])} | {show(row['minimum'])}–{show(row['maximum'])} | "
            f"{show(row['loo_mean_min'])}–{show(row['loo_mean_max'])} |"
        )
    lines += [
        "",
        "Per-event metrics and complete predicted-versus-actual tables are CSV and Parquet.",
        "Baselines are fixed references; the locked learned models are not claimed to beat them.",
        "Podium probabilities are independent and are not normalized to sum to three.",
        "",
        "## Coverage and uncertainty",
        "",
        f"Event/task coverage records: {len(coverage)}; full details are in coverage.csv.",
        (
            "Missing or mismatched official fields are excluded transparently per "
            "task/metric; no tail is invented."
        ),
        (
            "Driver/team/condition summaries need at least three weekends; smaller "
            "groups are suppressed."
        ),
        (
            "Wet/dry refers to observed target-session samples read only after "
            "prediction seals; absent weather is unknown."
        ),
        "Blocks below eight weekends get no confidence interval. Event ranges, SD and",
        (
            "leave-one-event-out mean ranges are sensitivity summaries, not "
            "confidence intervals or accuracy guarantees."
        ),
        "",
        "## Development comparisons and interpretation",
        "",
        "development/ preserves all A (2026-only), B (pooled equal) and C (2026-favored) results,",
        (
            "eligibility, searches and feature-group comparisons. Primary strategy "
            "means use identical common events."
        ),
        "Early A folds below the configured earlier-2026 minimum are ineligible, not pooled.",
        "Common development event counts: "
        + ", ".join(
            f"{task}={len(choice['common_events'])}" for task, choice in protocol["models"].items()
        )
        + ". Small cohorts make family/weight selection weak evidence.",
        (
            "Optional practice/telemetry/weather groups were not promoted under "
            "their development evidence/complexity gates."
        ),
        (
            "Calibration plots show raw versus delivered predictions; a lower "
            "Brier score alone does not establish calibration."
        ),
        (
            "Historical reconstructions may contain later source revisions and "
            "cannot certify original publication-time knowledge."
        ),
    ]
    if protocol["data_kind"] == "synthetic_validation":
        lines += [
            "",
            (
                "These are generated fixture outcomes, not observed Formula 1 results. "
                "The generator deliberately"
            ),
            (
                "makes race order equal qualifying order; excellent race/reference "
                "scores are a fixture property."
            ),
            (
                "No real 2026 accuracy, calibration or feature benefit follows. "
                "Real-data finalization remains"
            ),
            (
                "unavailable until a genuine reviewed dataset, cutoffs, rosters and "
                "frozen manifest exist."
            ),
        ]
    lines += [
        "",
        "## Hypotheses and next confirmation",
        "",
        (
            "Whether 2026 weighting, extra session inputs or calibration help on "
            "real events remains a hypothesis."
        ),
        (
            "No choices are changed in response to these test results. Any later "
            "model/feature changes or test reuse"
        ),
        (
            "are exploratory and require a fresh future holdout for confirmation. "
            "The original manifest is unchanged."
        ),
        "",
    ]
    with (output / "report.md").open("x") as stream:
        stream.write("\n".join(lines))
    write_json(
        output / "summary.json",
        json_value(
            {
                "data_kind": protocol["data_kind"],
                "exploratory": exploratory,
                "fresh_future_holdout_required": protocol["fresh_future_holdout_required"],
                "metrics": summaries.to_dict("records"),
                "retuned_after_test": False,
                "real_2026_performance_claim": False
                if protocol["data_kind"] == "synthetic_validation"
                else None,
            }
        ),
    )
