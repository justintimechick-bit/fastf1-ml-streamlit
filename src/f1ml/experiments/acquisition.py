"""Small explicit session captures through public FastF1 interfaces, with persistent reuse."""

import logging
import time
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from f1ml.config import ExperimentConfig
from f1ml.domain import PRACTICES, utc
from f1ml.evaluation.splits import assert_development, gp_session
from f1ml.features.pace import clean_laps
from f1ml.ingestion.provider import DiagnosticHandler, timestamp
from f1ml.storage import digest, read_json, save_table, seal_directory, verify_directory, write_json

ALLOWED = PRACTICES | {"Qualifying"}


class TelemetryProvider:
    def __init__(self, cache: Path, options: ExperimentConfig):
        import fastf1

        self.fastf1, self.options = fastf1, options
        cache.mkdir(parents=True, exist_ok=True)
        fastf1.Cache.enable_cache(str(cache.resolve()))

    def capture(self, event, spec):
        if spec["kind"] not in ALLOWED:
            raise ValueError("Only GP practice/qualifying telemetry is eligible")
        handler, logger = DiagnosticHandler(), logging.getLogger("fastf1")
        logger.addHandler(handler)
        try:
            session = self.fastf1.get_session(event["year"], event["round"], spec["kind"])
            session.load(laps=True, telemetry=True, weather=True, messages=True)
            original = session.laps
            raw = pd.DataFrame(original).copy()
            identities = pd.DataFrame(session.results)[["DriverNumber", "DriverId"]]
            if identities.DriverNumber.duplicated().any():
                raise ValueError("Ambiguous driver-number identity")
            lookup = identities.set_index("DriverNumber").DriverId
            raw["driver_id"] = raw.DriverNumber.astype(str).map(lookup)
            raw["lap_number"] = raw.LapNumber
            raw["source_lap_index"] = raw.index
            if spec["kind"] == "Qualifying":
                raw["qualifying_segment"] = pd.NA
                for index, part in enumerate(original.split_qualifying_sessions(), start=1):
                    if part is not None:
                        raw.loc[part.index, "qualifying_segment"] = f"Q{index}"
            selected = clean_laps(raw).sort_values(["driver_id", "start", "lap_number"])
            selected = selected.groupby("driver_id", sort=True).head(
                self.options.max_laps_per_driver
            )
            if selected.empty:
                raise ValueError("No clean eligible laps; no telemetry samples retained")
            telemetry, lap_errors = [], []
            for row in selected.itertuples():
                try:
                    data = pd.DataFrame(
                        original.loc[row.source_lap_index].get_car_data(interpolate_edges=False)
                    ).copy()
                    data["driver_id"], data["lap_number"] = row.driver_id, row.lap_number
                    telemetry.append(data)
                except Exception as exc:
                    lap_errors.append(
                        {
                            "driver_id": row.driver_id,
                            "lap_number": float(row.lap_number),
                            "type": type(exc).__name__,
                            "message": str(exc)[:1000],
                        }
                    )
            if not telemetry or not any(len(t) for t in telemetry):
                raise ValueError(f"No raw car telemetry for selected laps: {lap_errors[:3]}")
            status = pd.DataFrame(session.session_status)
            values = status.Status.tolist()
            last = max((i for i, v in enumerate(values) if v == "Finalised"), default=-1)
            completed = (
                last >= 0
                and values[-1] in ("Finalised", "Ends")
                and "Started" not in values[last + 1 :]
            )
            ended = None
            if completed:
                origin = pd.Timestamp(session.t0_date)
                origin = origin.tz_localize("UTC") if origin.tzinfo is None else utc(origin)
                ended = (origin + status.iloc[-1].Time).isoformat()
            try:
                weather = pd.DataFrame(session.weather_data).copy()
                if completed and "Time" in weather:
                    weather["observed_at"] = origin + pd.to_timedelta(weather.Time)
                    weather = weather.loc[weather.observed_at.le(utc(ended))]
            except Exception as exc:
                weather = pd.DataFrame()
                lap_errors.append(
                    {"component": "weather", "type": type(exc).__name__, "message": str(exc)[:1000]}
                )
            frames = {
                "telemetry": pd.concat(telemetry, ignore_index=True),
                "laps": selected.reset_index(drop=True),
                "session_laps": raw.reset_index(drop=True),
                "weather": weather,
            }
            return frames, {
                "source": "fastf1_public_car_data",
                "fastf1_version": version("fastf1"),
                "retrieved_at": timestamp(),
                "completed_at": ended,
                "completion_state": "completed" if completed else "unknown",
                "errors": lap_errors,
                "warnings": handler.messages,
                "load_scope": "selected sessions; FastF1 loads all cars/car+position upstream",
                "retained_scope": "earliest clean bounded laps per driver; car data only",
                "interpolation": "none; get_car_data(interpolate_edges=False)",
                "drs_policy": "schema audit only; no DRS/aero/energy model feature",
            }
        except Exception as exc:
            exc.telemetry_diagnostics = handler.messages
            raise
        finally:
            logger.removeHandler(handler)


def schema_report(frames):
    report = {}
    for name, frame in frames.items():
        report[name] = {
            "rows": len(frame),
            "columns": {
                c: {"dtype": str(frame[c].dtype), "missing": int(frame[c].isna().sum())}
                for c in frame
            },
        }
    car = frames.get("telemetry", pd.DataFrame())
    report["drs"] = {
        "column_present": "DRS" in car,
        "used_in_features": False,
        "values": sorted(car.DRS.dropna().astype(str).unique().tolist())[:30]
        if "DRS" in car
        else [],
        "caution": "A parser-created column/default is not proof of an observed wire channel",
    }
    return report


class TelemetryCatalog:
    def __init__(self, root: Path):
        self.root = root
        self.index = read_json(root / "index.json") if (root / "index.json").exists() else {}

    @staticmethod
    def key(source_id, event_id, kind, options):
        return digest(
            {
                "source_id": source_id,
                "event_id": event_id,
                "kind": kind,
                "max_laps_per_driver": options.max_laps_per_driver,
                "selection_version": 1,
            }
        )

    def _save_index(self):
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.root / "index.pending.json"
        if temporary.exists():
            temporary.unlink()
        write_json(temporary, self.index)
        temporary.replace(self.root / "index.json")

    def run(self, source, manifest, requests, provider, options, retry_failed=False):
        required = {"event_id", "task", "kind", "cutoff", "mode"}
        if not required.issubset(requests) or requests.empty:
            raise ValueError(f"Telemetry requests require {sorted(required)}")
        assert_development(manifest, requests.event_id)
        if requests[["event_id", "kind"]].drop_duplicates().shape[0] > options.max_sessions:
            raise ValueError(
                "Explicit telemetry request exceeds max_sessions; select a smaller pilot"
            )
        report = []
        for row in requests.to_dict("records"):
            event = source.event(row["event_id"])
            kind, task, cutoff = row["kind"], row["task"], utc(row["cutoff"])
            if (
                kind not in ALLOWED
                or task not in ("qualifying", "race")
                or (task == "qualifying" and kind == "Qualifying")
            ):
                raise ValueError(
                    "Target qualifying, target race and Sprint telemetry are forbidden"
                )
            if row["mode"] not in ("historical_reconstruction", "publication_snapshot"):
                raise ValueError("Unknown provenance mode")
            target = gp_session(event, "Qualifying" if task == "qualifying" else "Race")
            if cutoff >= min(
                utc(t) for t in (target["scheduled_start"], target.get("started_at")) if t
            ):
                raise ValueError("Telemetry cutoff must precede target session start")
            if task == "race" and cutoff <= utc(gp_session(event, "Qualifying")["scheduled_start"]):
                raise ValueError("Race telemetry cutoff must follow qualifying")
            spec = next(s for s in event["sessions"] if s["kind"] == kind)
            if utc(spec["scheduled_start"]) >= cutoff:
                raise ValueError("Telemetry session is not earlier than cutoff")
            key = self.key(source.receipt["id"], event["event_id"], kind, options)
            existing = self.index.get(key)
            if existing and existing["status"] in ("complete", "partial"):
                saved = verify_directory(self.root / existing["path"])
                if utc(saved["completed_at"]) >= cutoff or (
                    row["mode"] == "publication_snapshot" and utc(saved["retrieved_at"]) > cutoff
                ):
                    report.append(
                        {
                            **row,
                            "key": key,
                            **existing,
                            "action": "ineligible_at_cutoff",
                            "status": "ineligible_at_cutoff",
                        }
                    )
                    continue
                report.append({**row, "key": key, "action": "reused", **existing})
                continue
            if (
                existing
                and (utc(timestamp()) - utc(existing["last_attempt"])).total_seconds()
                < options.failure_cooldown_hours * 3600
                and not retry_failed
            ):
                report.append({**row, "key": key, "action": "cooldown", **existing})
                continue
            attempt_errors, frames, metadata = [], {}, {}
            start = time.perf_counter()
            for attempt in range(options.max_attempts):
                try:
                    frames, metadata = provider.capture(event, spec)
                    if metadata.get("completion_state") != "completed" or not metadata.get(
                        "completed_at"
                    ):
                        raise ValueError("Observed absolute session completion is unavailable")
                    if utc(metadata["completed_at"]) >= cutoff:
                        raise ValueError("Session did not complete before the requested cutoff")
                    if (
                        row["mode"] == "publication_snapshot"
                        and utc(metadata["retrieved_at"]) > cutoff
                    ):
                        raise ValueError(
                            "Telemetry revision was retrieved after publication cutoff"
                        )
                    break
                except Exception as exc:
                    attempt_errors.append(
                        {
                            "type": type(exc).__name__,
                            "message": str(exc)[:2000],
                            "diagnostics": getattr(exc, "telemetry_diagnostics", []),
                        }
                    )
                    if isinstance(exc, ValueError) or attempt + 1 == options.max_attempts:
                        frames = {}
                        break
                    time.sleep(min(1.0 * 2**attempt, 4.0))
            status = ("partial" if metadata.get("errors") else "complete") if frames else "failed"
            revision = (
                self.root
                / "captures"
                / key
                / str(len(list((self.root / "captures" / key).glob("*"))))
            )
            revision.mkdir(parents=True)
            for name, frame in frames.items():
                save_table(revision / f"{name}.parquet", frame)
            schema = schema_report(frames)
            write_json(revision / "schema.json", schema)
            payload = {
                "retrieved_at": timestamp(),
                "fastf1_version": version("fastf1"),
                "source_dataset_id": source.receipt["id"],
                "guard_manifest_id": manifest["id"],
                "schema_audit_only": source.receipt["id"] != manifest["dataset_id"],
                "event_id": event["event_id"],
                "kind": kind,
                "year": event["year"],
                "status": status,
                "attempt_errors": attempt_errors,
                **metadata,
            }
            receipt = seal_directory(revision, payload)
            entry = {
                "path": str(revision.relative_to(self.root)),
                "id": receipt["id"],
                "status": status,
                "last_attempt": timestamp(),
                "runtime_seconds": time.perf_counter() - start,
                "errors": attempt_errors,
                "schema": schema,
            }
            self.index[key] = entry
            self._save_index()
            report.append({**row, "key": key, "action": "downloaded", **entry})
        return report

    def read(self, source_id, event_id, kind, cutoff, mode, options):
        key = self.key(source_id, event_id, kind, options)
        entry = self.index.get(key)
        if not entry or entry["status"] not in ("complete", "partial"):
            return {}, {"state": entry["status"] if entry else "not_requested", "key": key}
        root = self.root / entry["path"]
        receipt = verify_directory(root)
        if receipt["source_dataset_id"] != source_id or receipt["schema_audit_only"]:
            raise ValueError("Telemetry source is not bound to this frozen experiment")
        if utc(receipt["completed_at"]) >= cutoff or (
            mode == "publication_snapshot" and utc(receipt["retrieved_at"]) > cutoff
        ):
            return {}, {"state": "unavailable_at_cutoff", "key": key}
        frames = {
            name: pd.read_parquet(root / f"{name}.parquet")
            for name in ("telemetry", "laps", "session_laps", "weather")
            if (root / f"{name}.parquet").exists()
        }
        return frames, {
            "state": entry["status"],
            "key": key,
            "receipt_id": receipt["id"],
            "completed_at": receipt["completed_at"],
            "retrieved_at": receipt["retrieved_at"],
        }
