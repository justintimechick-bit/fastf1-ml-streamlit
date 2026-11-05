"""Cutoff-bound optional feature supplements; never rebuild or overwrite core features."""

import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from f1ml.config import ExperimentConfig, FeatureConfig
from f1ml.domain import PRACTICES, utc
from f1ml.evaluation.splits import assert_development
from f1ml.experiments.acquisition import TelemetryCatalog
from f1ml.experiments.aggregates import (
    cells,
    pace_summary,
    teammate_deltas,
    telemetry_summaries,
    weather_summary,
)
from f1ml.experiments.contract import EXTRA, KEYS, VERSION, implementation_id
from f1ml.features.build import validate_cutoff
from f1ml.features.contract import model_matrix
from f1ml.features.pace import clean_laps
from f1ml.storage import new_directory, save_table, seal_directory, verify_directory, write_json


def mean(values):
    return float(np.mean(values)) if values else np.nan


def bounded_weather(raw, spec, capture, cutoff):
    """A relative weather clock without a verified origin cannot establish sample availability."""
    if raw.empty:
        return raw
    column = next((c for c in ("observed_at", "Date", "source_Date") if c in raw), None)
    if column is None:
        return pd.DataFrame()
    observed = pd.to_datetime(raw[column], utc=True, errors="coerce")
    upper = min(
        [cutoff] + [utc(t) for t in (spec.get("ended_at"), capture.get("completed_at")) if t]
    )
    lower = utc(spec["scheduled_start"])
    return raw.loc[observed.ge(lower) & observed.le(upper) & observed.lt(cutoff)].copy()


def event_supplement(source, manifest, core, catalog, options):
    event_id, task = core.event_id.iloc[0], core.task.iloc[0]
    assert_development(manifest, [event_id])
    if core.cutoff.nunique() != 1 or core["mode"].nunique() != 1:
        raise ValueError("Optional features require one event/task cutoff")
    cutoff, mode = utc(core.cutoff.iloc[0]), core["mode"].iloc[0]
    event = source.event(event_id)
    validate_cutoff(event, task, cutoff, mode)
    feature_options = FeatureConfig(**manifest["config"].get("features", {}))
    allowed = PRACTICES | ({"Qualifying"} if task == "race" else set())
    sources, diagnostics = [], []
    for spec in sorted(event["sessions"], key=lambda s: utc(s["scheduled_start"])):
        if spec["kind"] not in allowed:
            continue
        captured, audit = catalog.read(
            source.receipt["id"], event_id, spec["kind"], cutoff, mode, options
        )
        from f1ml.features.source import available

        raw, weather = pd.DataFrame(), pd.DataFrame()
        known = available(spec, cutoff, mode, timing=True)
        if known:
            try:
                raw = source.component(spec, "laps")
                weather = source.component(spec, "weather")
            except (ValueError, KeyError, OSError) as exc:
                diagnostics.append(
                    {"kind": spec["kind"], "state": "parse_failed", "message": str(exc)}
                )
        if not known and not captured:
            diagnostics.append(
                {"kind": spec["kind"], "state": "unavailable_at_cutoff", "capture": audit}
            )
            continue
        if raw.empty:
            raw = captured.get("session_laps", captured.get("laps", pd.DataFrame()))
        if weather.empty:
            weather = captured.get("weather", pd.DataFrame())
        observed_weather = bounded_weather(weather, spec, audit, cutoff)
        if observed_weather.empty and "weather" in captured:
            observed_weather = bounded_weather(captured["weather"], spec, audit, cutoff)
        session = {
            "kind": spec["kind"],
            "comparable": pd.DataFrame(),
            "clean": pd.DataFrame(),
            "telemetry": pd.DataFrame(),
            "weather": {},
        }
        for name, operation in (
            ("clean", lambda raw=raw: clean_laps(raw)),
            (
                "comparable",
                lambda raw=raw, weather=weather, spec=spec: cells(
                    raw, weather, spec["kind"], feature_options
                ),
            ),
            (
                "telemetry",
                lambda captured=captured, session=session: telemetry_summaries(
                    captured, session["comparable"], options
                ),
            ),
            (
                "weather",
                lambda observed_weather=observed_weather: weather_summary(observed_weather),
            ),
        ):
            try:
                session[name] = operation()
            except (ValueError, KeyError, TypeError) as exc:
                diagnostics.append(
                    {
                        "kind": spec["kind"],
                        "component": name,
                        "state": "aggregation_failed",
                        "message": str(exc),
                    }
                )
        sources.append(session)
        diagnostics.append(
            {
                "kind": spec["kind"],
                "state": "available",
                "capture": audit,
                "lap_rows": len(raw),
                "weather_rows": len(weather),
                "telemetry_laps": len(session["telemetry"]),
            }
        )
    rows = []
    for driver in core.driver_id:
        row = core.loc[core.driver_id.eq(driver), list(KEYS)].iloc[0].to_dict()
        row.update({name: np.nan for name in EXTRA})
        ranks, lap_cvs, sector_cvs, gaps, telemetry, weather = [], [], [], [], [], []
        consistency_laps, detailed_laps, comparable_laps = 0, 0, 0
        deltas = {
            key: [] for key in ("speed_mean_kph", "full_throttle_fraction", "braking_fraction")
        }
        comparisons = 0
        for session in sources:
            if session["kind"] in PRACTICES:
                pace = pace_summary(session["comparable"], driver)
                ranks += pace["ranks"]
                lap_cvs += pace["lap_cvs"]
                sector_cvs += pace["sector_cvs"]
                consistency_laps += pace["laps"]
                detailed_laps += (
                    int(session["clean"].driver_id.eq(driver).sum())
                    if not session["clean"].empty
                    else 0
                )
                if not session["comparable"].empty:
                    own = session["comparable"].loc[session["comparable"].driver_id.eq(driver)]
                    gaps += own.gap_pct.dropna().tolist()
                    comparable_laps += len(own)
            if not session["telemetry"].empty:
                own = session["telemetry"].loc[session["telemetry"].driver_id.eq(driver)]
                telemetry.extend(own.to_dict("records"))
                difference, count = teammate_deltas(session["telemetry"], core, driver)
                comparisons += count
                for key in deltas:
                    deltas[key] += difference[key]
            if session["weather"]:
                weather.append(session["weather"])
        row.update(
            practice_result_rank_norm=mean(ranks),
            practice_result_cells=len(ranks),
            practice_lap_cv=mean(lap_cvs),
            practice_sector_cv=mean(sector_cvs),
            practice_consistency_laps=consistency_laps,
            practice_detail_gap_pct=mean(gaps),
            practice_detail_laps=detailed_laps,
            practice_detail_comparable_laps=comparable_laps,
            telemetry_teammate_comparisons=comparisons,
            telemetry_teammate_speed_delta_kph=mean(deltas["speed_mean_kph"]),
            telemetry_teammate_throttle_delta=mean(deltas["full_throttle_fraction"]),
            telemetry_teammate_braking_delta=mean(deltas["braking_fraction"]),
        )
        for target, key in (
            ("telemetry_speed_mean_kph", "speed_mean_kph"),
            ("telemetry_speed_p95_kph", "speed_p95_kph"),
            ("telemetry_full_throttle_fraction", "full_throttle_fraction"),
            ("telemetry_braking_fraction", "braking_fraction"),
        ):
            row[target] = mean([v[key] for v in telemetry if pd.notna(v[key])])
        row["telemetry_laps"] = sum(
            any(
                pd.notna(v[k])
                for k in ("speed_mean_kph", "full_throttle_fraction", "braking_fraction")
            )
            for v in telemetry
        )
        row["telemetry_channel_coverage"] = mean(
            [
                v[k]
                for v in telemetry
                for k in ("speed_coverage", "throttle_coverage", "brake_coverage")
            ]
        )
        row["observed_weather_sessions"] = len(weather)
        row["observed_weather_samples"] = sum(w["observed_weather_samples"] for w in weather)
        for name in (
            "observed_air_temp_c",
            "observed_track_temp_c",
            "observed_humidity_pct",
            "observed_wind_speed_mps",
            "observed_rain_fraction",
        ):
            row[name] = mean([w[name] for w in weather if pd.notna(w[name])])
        rows.append(row)
    return pd.DataFrame(rows), diagnostics


def build_supplements(
    source, manifest, feature_root: Path, catalog_root: Path, output: Path, options=None
):
    options = options or ExperimentConfig()
    receipt = verify_directory(feature_root)
    if receipt["manifest_id"] != manifest["id"] or receipt["source_id"] != source.receipt["id"]:
        raise ValueError("Optional features must match the frozen core source and manifest")
    catalog = TelemetryCatalog(catalog_root)
    new_directory(output)
    audits, timings = [], []
    for task in ("qualifying", "race"):
        frame = pd.read_parquet(feature_root / f"{task}_features.parquet")
        assert_development(manifest, frame.event_id)
        model_matrix(frame, task)
        supplements = []
        for event_id, core in frame.groupby("event_id", sort=False):
            start = time.perf_counter()
            extra, audit = event_supplement(source, manifest, core, catalog, options)
            supplements.append(extra)
            audits.append({"task": task, "event_id": event_id, "sources": audit})
            timings.append(
                {
                    "task": task,
                    "event_id": event_id,
                    "runtime_seconds": time.perf_counter() - start,
                    "driver_rows": len(extra),
                    "telemetry_drivers": int(extra.telemetry_laps.gt(0).sum()),
                    "weather_drivers": int(extra.observed_weather_sessions.gt(0).sum()),
                    "practice_drivers": int(extra.practice_detail_comparable_laps.gt(0).sum()),
                }
            )
        save_table(output / f"{task}_optional.parquet", pd.concat(supplements, ignore_index=True))
    save_table(output / "coverage.parquet", pd.DataFrame(timings))
    write_json(output / "audit.json", audits)
    from f1ml.experiments.dictionary import dictionary

    write_json(output / "feature_dictionary.json", dictionary())
    return seal_directory(
        output,
        {
            "version": VERSION,
            "implementation_id": implementation_id(),
            "manifest_id": manifest["id"],
            "core_feature_snapshot_id": receipt["id"],
            "source_id": source.receipt["id"],
            "options": asdict(options),
            "data_kind": "synthetic"
            if all("SYNTHETIC" in str(e.get("name", "")) for e in source.events)
            else "observed_source_snapshot",
            "forecast_policy": "none; no verified historical forecasts obtained",
        },
    )
