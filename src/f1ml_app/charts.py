"""Altair charts. Every chart has a table view on the same page (palette relief rule)."""

import json

import altair as alt
import pandas as pd

from f1ml_app import forecasts as fc
from f1ml_app.settings import MODEL_LABELS
from f1ml_app.theme import team_color, team_name, tokens

ROW = 30  # px per categorical row


def _sized(chart, height: int):
    """Fit width to the column; keep ``height`` as the plot height (Streamlit defaults to
    ``autosize: fit``, which squeezes discrete rows to make room for legends and axes)."""
    return chart.properties(
        height=height, autosize=alt.AutoSizeParams(type="fit-x", contains="padding")
    )


def _legend():
    return alt.Legend(orient="top", title=None, symbolSize=110, labelFontSize=12)


def podium_probabilities(frame: pd.DataFrame, threshold: float) -> alt.Chart:
    """Ranked bars of independent podium probabilities; classifier top three emphasized."""
    t = tokens()
    data = frame.assign(
        group=frame.classifier_top_three.map({True: "Classifier top 3", False: "Other drivers"}),
        inside=frame.podium_probability >= 0.78,
    ).sort_values(["podium_probability", "driver_id"], ascending=[False, True])
    if "driver" not in data:
        data["driver"] = data.driver_id.str.replace("_", " ").str.title()
    order = data.driver.tolist()
    y = alt.Y("driver:N", sort=order, title=None, axis=alt.Axis(labelLimit=220))
    x = alt.X(
        "podium_probability:Q",
        scale=alt.Scale(domain=[0, 1]),
        axis=alt.Axis(format="%", values=[0, 0.25, 0.5, 0.75, 1], title="Podium probability"),
    )
    tooltip = [
        alt.Tooltip("driver:N", title="Driver"),
        alt.Tooltip("podium_probability:Q", title="Podium probability", format=".1%"),
        alt.Tooltip("raw_podium_probability:Q", title="Before calibration", format=".1%"),
        alt.Tooltip("podium_probability_rank:Q", title="Probability rank"),
        alt.Tooltip("predicted_position:Q", title="Race-order model position"),
    ]
    base = alt.Chart(data).encode(y=y)
    bars = base.mark_bar(size=18, cornerRadiusEnd=4).encode(
        x=x,
        color=alt.Color(
            "group:N",
            scale=alt.Scale(
                domain=["Classifier top 3", "Other drivers"], range=[t["series"][0], t["neutral"]]
            ),
            legend=_legend(),
        ),
        tooltip=tooltip,
    )
    outside = (
        base.transform_filter("!datum.inside")
        .mark_text(align="left", dx=6, fontSize=12, color=t["ink2"])
        .encode(x=x, text=alt.Text("podium_probability:Q", format=".1%"))
    )
    inside = (
        base.transform_filter("datum.inside")
        .mark_text(align="right", dx=-8, fontSize=12, fontWeight=600)
        .encode(
            x=x,
            text=alt.Text("podium_probability:Q", format=".1%"),
            # Literal colors: a second color *scale* would merge with the bars' shared scale.
            color=alt.condition(
                "datum.group == 'Classifier top 3'",
                alt.value("#ffffff"),
                alt.value(t["on_neutral"]),
            ),
        )
    )
    line = pd.DataFrame(
        {"threshold": [threshold], "label": [f"Decision threshold {threshold:.0%}"]}
    )
    rule = (
        alt.Chart(line)
        .mark_rule(color=t["muted"], strokeDash=[4, 3], strokeWidth=1.5)
        .encode(x="threshold:Q", tooltip=[alt.Tooltip("label:N", title="Threshold")])
    )
    return _sized(bars + outside + inside + rule, ROW * len(data))


def predicted_vs_official(frame: pd.DataFrame, forecast_label: str) -> alt.Chart:
    """Dumbbell: forecast vs official position per driver (P1 at the left)."""
    t = tokens()
    data = frame.dropna(subset=["official_position"]).copy()
    order = data.sort_values("predicted_position").driver_id.tolist()
    n = int(max(data.predicted_position.max(), data.official_position.max()))
    long = data.melt(
        id_vars=["driver_id", "absolute_error"],
        value_vars=["predicted_position", "official_position"],
        var_name="kind",
        value_name="position",
    )
    long["kind"] = long.kind.map(
        {"predicted_position": forecast_label, "official_position": "Official classification"}
    )
    x = alt.X(
        "position:Q",
        scale=alt.Scale(domain=[0.5, n + 0.5], nice=False),
        axis=alt.Axis(values=list(range(1, n + 1)), format="d", title="Position (P1 = front)"),
    )
    y = alt.Y("driver_id:N", sort=order, title=None, axis=alt.Axis(labelLimit=220))
    links = (
        alt.Chart(data)
        .mark_rule(color=t["neutral"], strokeWidth=2)
        .encode(y=y, x=alt.X("predicted_position:Q"), x2="official_position:Q")
    )
    domain = [forecast_label, "Official classification"]
    points = (
        alt.Chart(long)
        .mark_point(filled=True, size=120, opacity=1, stroke=t["surface"], strokeWidth=2)
        .encode(
            y=y,
            x=x,
            color=alt.Color(
                "kind:N",
                scale=alt.Scale(domain=domain, range=list(t["series"][:2])),
                legend=_legend(),
            ),
            shape=alt.Shape(
                "kind:N", scale=alt.Scale(domain=domain, range=["circle", "diamond"]), legend=None
            ),
            tooltip=[
                alt.Tooltip("driver_id:N", title="Driver"),
                alt.Tooltip("kind:N", title="Series"),
                alt.Tooltip("position:Q", title="Position", format="d"),
                alt.Tooltip("absolute_error:Q", title="Absolute error (places)", format="d"),
            ],
        )
    )
    return _sized(links + points, ROW * len(data))


def rank_deltas(field: pd.DataFrame, *, color_by: str = "constructor") -> alt.Chart:
    """Full-field rank dumbbells: circle = forecast, diamond = actual, P1 at left.

    Unknown actual positions retain their forecast marker and row. Hover links both
    endpoints; the numeric axis can be zoomed/panned without reordering the field.
    """
    if any(column.startswith("rank__") for column in field):
        return model_field_comparison(
            field, "rank_error" if color_by == "error" else "predicted_rank"
        )
    t = tokens()
    data = field.sort_values(["predicted_rank", "driver_id"], kind="stable").copy()
    data["_row"] = range(len(data))
    data["Team"] = data.team_id.map(team_name)
    data["constructor_color"] = data.team_id.map(team_color)
    labels = {str(i): str(name) for i, name in zip(data._row, data.driver, strict=True)}
    order = data._row.tolist()
    maximum = max([len(data), *data.predicted_rank.dropna(), *data.actual_rank.dropna(), 1])
    y = alt.Y(
        "_row:O",
        title=None,
        sort=order,
        scale=alt.Scale(domain=order),
        axis=alt.Axis(labelExpr=f"{json.dumps(labels)}[datum.label]", labelLimit=220),
    )
    x = alt.X(
        "position:Q",
        title="Rank · P1 at left",
        scale=alt.Scale(domain=[0.5, maximum + 0.5]),
        axis=alt.Axis(tickMinStep=1, format="d"),
    )
    hover = alt.selection_point(fields=["driver_id"], on="pointerover", clear="pointerout")
    tooltip = [
        alt.Tooltip("driver:N", title="Driver"),
        alt.Tooltip("Team:N"),
        alt.Tooltip("predicted_rank:Q", title="Predicted rank", format="d"),
        alt.Tooltip("actual_rank:Q", title="Actual rank", format="d"),
        alt.Tooltip("rank_delta_label:N", title="Beat forecast"),
        alt.Tooltip("rank_error:Q", title="Absolute rank error", format="d"),
    ]
    color = (
        alt.condition(
            "isValid(datum.rank_error)",
            alt.Color(
                "rank_error:Q",
                title="Rank error",
                scale=alt.Scale(
                    domain=[
                        0,
                        max(1, data.rank_error.max() if data.rank_error.notna().any() else 1),
                    ],
                    range=[t["verified"], t["accent"]],
                ),
            ),
            alt.value(t["muted"]),
        )
        if color_by == "error"
        else alt.Color("constructor_color:N", scale=None, legend=None)
    )
    links = (
        alt.Chart(data)
        .transform_filter("isValid(datum.actual_rank)")
        .mark_rule(strokeWidth=2)
        .encode(
            y=y,
            x=alt.X("predicted_rank:Q"),
            x2="actual_rank:Q",
            color=color,
            opacity=alt.condition(hover, alt.value(0.85), alt.value(0.15)),
            tooltip=tooltip,
        )
    )
    long = data.melt(
        id_vars=[c for c in data if c not in ("predicted_rank", "actual_rank")],
        value_vars=["predicted_rank", "actual_rank"],
        var_name="kind",
        value_name="position",
    )
    # Keep both rank fields available in every endpoint's tooltip.
    long = long.merge(
        data[["_row", "predicted_rank", "actual_rank"]], on="_row", validate="many_to_one"
    )
    long["kind"] = long.kind.map({"predicted_rank": "Predicted", "actual_rank": "Actual"})
    points = (
        alt.Chart(long)
        .transform_filter("isValid(datum.position)")
        .mark_point(filled=True, size=95, stroke=t["surface"], strokeWidth=1.5)
        .encode(
            y=y,
            x=x,
            color=color,
            shape=alt.Shape(
                "kind:N",
                scale=alt.Scale(domain=["Predicted", "Actual"], range=["circle", "diamond"]),
                legend=_legend(),
            ),
            opacity=alt.condition(hover, alt.value(1), alt.value(0.25)),
            tooltip=tooltip,
        )
    )
    return _sized(
        (links + points).add_params(hover).interactive(bind_y=False), max(150, ROW * len(data))
    )


def model_field_comparison(field: pd.DataFrame, value: str = "predicted_rank") -> alt.Chart:
    """Grouped series for all selected models; missing outputs never become zeroes."""
    data = fc.comparison_long(field)
    if "family" not in data:
        data["family"] = field.attrs.get("primary", "Selected model")
    data["Model"] = data.family.map(MODEL_LABELS).fillna(data.family)
    models = data.Model.drop_duplicates().tolist()
    order = field.driver.tolist()
    if value == "predicted_rank":
        return _model_rank_dots(data, order, models)
    titles = {
        "predicted_rank": "Predicted rank · P1 at left",
        "rank_error": "Absolute rank error · lower is better",
        "podium_probability": "Podium probability",
    }
    chart = (
        alt.Chart(data)
        .mark_bar()
        .encode(
            y=alt.Y("driver:N", sort=order, title=None),
            yOffset=alt.YOffset("Model:N", sort=models),
            x=alt.X(
                f"{value}:Q",
                title=titles[value],
                scale=alt.Scale(domain=[0, 1])
                if value == "podium_probability"
                else alt.Scale(zero=True),
            ),
            color=alt.Color(
                "Model:N",
                scale=alt.Scale(domain=models, range=list(tokens()["series"][: len(models)])),
                legend=_legend(),
            ),
            tooltip=[
                "driver:N",
                "Model:N",
                "predicted_rank:Q",
                "actual_rank:Q",
                "rank_error:Q",
                alt.Tooltip("podium_probability:Q", format=".1%"),
            ],
        )
    )
    return _sized(chart, max(150, len(field) * max(28, 10 * len(models) + 8)))


def _model_rank_dots(data: pd.DataFrame, order: list[str], models: list[str]) -> alt.Chart:
    """One row per driver: a dot per model on a P1-left rank axis, a band for model
    disagreement, and a hollow diamond for the official rank when it is known.

    Rank is a position, not a quantity, so bars (where P1 is the shortest bar) mislead.
    All layers share one dataset; unknown ranks are filtered out, never drawn as zero.
    """
    t = tokens()
    ranks = [*data.predicted_rank.dropna(), *data.actual_rank.dropna(), len(order), 1]
    maximum = int(max(ranks))
    hover = alt.selection_point(fields=["driver_id"], on="pointerover", clear="pointerout")
    y = alt.Y("driver:N", sort=order, title=None, axis=alt.Axis(labelLimit=220))
    x_scale = alt.Scale(domain=[0.5, maximum + 0.5], nice=False)
    axis = alt.Axis(tickMinStep=1, format="d", values=list(range(1, maximum + 1)))
    base = alt.Chart(data).encode(y=y)
    spread = (
        base.transform_filter("isValid(datum.predicted_rank)")
        .transform_aggregate(
            low="min(predicted_rank)", high="max(predicted_rank)", groupby=["driver"]
        )
        .mark_rule(strokeWidth=7, strokeCap="round", color=t["grid"], opacity=0.7)
        .encode(x=alt.X("low:Q", scale=x_scale, title=None), x2="high:Q")
    )
    tooltip = [
        alt.Tooltip("driver:N", title="Driver"),
        alt.Tooltip("Model:N"),
        alt.Tooltip("predicted_rank:Q", title="Predicted rank", format="d"),
        alt.Tooltip("actual_rank:Q", title="Actual rank", format="d"),
        alt.Tooltip("rank_error:Q", title="Rank error (places)", format="d"),
        alt.Tooltip("podium_probability:Q", title="Podium probability", format=".1%"),
    ]
    points = (
        base.transform_filter("isValid(datum.predicted_rank)")
        .mark_point(filled=True, size=120, opacity=1, stroke=t["surface"], strokeWidth=1.2)
        .encode(
            x=alt.X(
                "predicted_rank:Q", scale=x_scale, axis=axis, title="Predicted rank · P1 at left"
            ),
            color=alt.Color(
                "Model:N",
                scale=alt.Scale(domain=models, range=list(tokens()["series"][: len(models)])),
                legend=_legend(),
            ),
            opacity=alt.condition(hover, alt.value(1), alt.value(0.85)),
            tooltip=tooltip,
        )
    )
    first = json.dumps(str(data.family.iloc[0])) if len(data) else '""'
    actual = (
        base.transform_filter(f"datum.family == {first} && isValid(datum.actual_rank)")
        .mark_point(shape="diamond", filled=False, size=150, strokeWidth=2, color=t["ink"])
        .encode(
            x=alt.X("actual_rank:Q", scale=x_scale),
            tooltip=[
                alt.Tooltip("driver:N", title="Driver"),
                alt.Tooltip("actual_rank:Q", title="Actual rank", format="d"),
            ],
        )
    )
    return _sized((points + spread + actual).add_params(hover), max(150, ROW * len(order)))


def calibration_comparison(frame: pd.DataFrame) -> alt.Chart:
    """Reliability curves from saved fit diagnostics, with a reference diagonal."""
    data = frame.dropna(subset=["podium_probability", "actual_podium"]).copy()
    data["bin"] = pd.cut(
        data.podium_probability, bins=[0, 0.2, 0.4, 0.6, 0.8, 1], labels=False, include_lowest=True
    )
    data = (
        data.groupby(["family", "bin"], observed=True)
        .agg(
            probability=("podium_probability", "mean"),
            observed=("actual_podium", "mean"),
            drivers=("driver_id", "count"),
        )
        .reset_index()
    )
    data["Model"] = data.family.map(MODEL_LABELS).fillna(data.family)
    curves = (
        alt.Chart(data)
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "probability:Q", title="Mean predicted probability", scale=alt.Scale(domain=[0, 1])
            ),
            y=alt.Y("observed:Q", title="Observed podium rate", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Model:N", legend=_legend()),
            tooltip=["Model:N", "probability:Q", "observed:Q", "drivers:Q"],
        )
    )
    diagonal = (
        alt.Chart(pd.DataFrame({"x": [0, 1]}))
        .mark_line(strokeDash=[4, 3], color=tokens()["muted"])
        .encode(x="x:Q", y="x:Q")
    )
    return _sized(curves + diagonal, 250)


def strategy_dots(frame: pd.DataFrame, metric_title: str, better: str) -> alt.Chart:
    """Family rows, one dot per historical-data strategy; fixed rules as neutral references."""
    t = tokens()
    domain = [
        "2026-only",
        "Pooled 2024–26 (equal)",
        "Weighted history (2026-favored)",
        "Fixed rule",
    ]
    colors = [*t["series"], t["neutral"]]
    shapes = ["circle", "square", "triangle-up", "diamond"]
    rows = frame.model.drop_duplicates().tolist()
    chart = (
        alt.Chart(frame)
        .mark_point(filled=True, size=120, opacity=1, stroke=t["surface"], strokeWidth=2)
        .encode(
            y=alt.Y("model:N", sort=rows, title=None, axis=alt.Axis(labelLimit=240)),
            yOffset=alt.YOffset("strategy:N", sort=domain, scale=alt.Scale(domain=domain)),
            x=alt.X(
                "value:Q",
                title=f"{metric_title} ({better} is better)",
                scale=alt.Scale(zero=False, padding=12),
                axis=alt.Axis(tickCount=6),
            ),
            color=alt.Color(
                "strategy:N", scale=alt.Scale(domain=domain, range=colors), legend=_legend()
            ),
            shape=alt.Shape(
                "strategy:N", scale=alt.Scale(domain=domain, range=shapes), legend=None
            ),
            tooltip=[
                alt.Tooltip("model:N", title="Model"),
                alt.Tooltip("strategy:N", title="Strategy"),
                alt.Tooltip("value:Q", title=metric_title, format=".4f"),
                alt.Tooltip("events:Q", title="Events"),
            ],
        )
    )
    return _sized(chart, 54 * len(rows))


def method_dots(
    frame: pd.DataFrame, metric_title: str, better: str, methods: list[str]
) -> alt.Chart:
    """Per-event final-test scores for the locked model and its fixed references."""
    t = tokens()
    shapes = ["circle", "square", "triangle-up"]
    events = sorted(frame.event_id.unique())
    chart = (
        alt.Chart(frame)
        .mark_point(filled=True, size=120, opacity=1, stroke=t["surface"], strokeWidth=2)
        .encode(
            y=alt.Y("event_id:N", sort=events, title=None),
            yOffset=alt.YOffset("method_label:N", sort=methods, scale=alt.Scale(domain=methods)),
            x=alt.X(
                "value:Q",
                title=f"{metric_title} ({better} is better)",
                scale=alt.Scale(zero=True),
                axis=alt.Axis(tickCount=6),
            ),
            color=alt.Color(
                "method_label:N",
                scale=alt.Scale(domain=methods, range=list(t["series"][: len(methods)])),
                legend=_legend(),
            ),
            shape=alt.Shape(
                "method_label:N",
                scale=alt.Scale(domain=methods, range=shapes[: len(methods)]),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("event_id:N", title="Event"),
                alt.Tooltip("method_label:N", title="Method"),
                alt.Tooltip("value:Q", title=metric_title, format=".4f"),
            ],
        )
    )
    return _sized(chart, 60 * len(events))
