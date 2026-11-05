"""Altair charts. Every chart has a table view on the same page (palette relief rule)."""

import altair as alt
import pandas as pd

from f1ml_app.theme import tokens

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
    order = data.driver_id.tolist()
    y = alt.Y("driver_id:N", sort=order, title=None, axis=alt.Axis(labelLimit=220))
    x = alt.X(
        "podium_probability:Q",
        scale=alt.Scale(domain=[0, 1]),
        axis=alt.Axis(format="%", values=[0, 0.25, 0.5, 0.75, 1], title="Podium probability"),
    )
    tooltip = [
        alt.Tooltip("driver_id:N", title="Driver"),
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
