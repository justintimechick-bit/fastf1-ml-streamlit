"""Public FastF1 boundary. Results and timing are independent resumable components."""

import logging
import re
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import pandas as pd
from fastf1.exceptions import DataNotLoadedError

from f1ml.domain import GP_SESSIONS, PRACTICES, event_key, session_kind, utc
from f1ml.ingestion.normalize import normalize_results, summarize_practice
from f1ml.ingestion.types import SourceBatch


def timestamp() -> str:
    return pd.Timestamp.now(tz="UTC").isoformat()


class DiagnosticHandler(logging.Handler):
    def __init__(self):
        super().__init__(logging.WARNING)
        self.messages = []

    def emit(self, record):
        if len(self.messages) < 50:
            self.messages.append(
                {
                    "logger": record.name,
                    "level": record.levelname,
                    "message": record.getMessage()[:2000],
                }
            )


class FastF1Provider:
    def __init__(self, cache_dir: Path, refresh: bool = False, max_api_pages: int = 20):
        import fastf1
        from fastf1.ergast import Ergast

        self.fastf1 = fastf1
        self.cache_dir = cache_dir.resolve()
        self.max_api_pages = max_api_pages
        self._active_cache = None
        self._enable_cache(self.cache_dir / "schedules")
        self.results_api = Ergast(result_type="pandas", auto_cast=False)
        self.legacy_refresh = refresh
        self.legacy_cache_generation = uuid4().hex if refresh else "0"

    def _enable_cache(self, path: Path):
        if self._active_cache != path:
            path.mkdir(parents=True, exist_ok=True)
            self.fastf1.Cache.enable_cache(str(path))
            self._active_cache = path

    @contextmanager
    def event_cache(self, event_id: str, generation: int = 0):
        # Refresh uses a new persistent event cache, leaving all other events intact.
        self._enable_cache(self.cache_dir / "events" / event_id / str(generation))
        yield

    def schedule(self, year: int) -> list[dict]:
        if hasattr(self, "cache_dir"):
            self._enable_cache(self.cache_dir / "schedules")
        schedule = self.fastf1.get_event_schedule(year, include_testing=False)
        retrieved = timestamp()
        events = []
        for _, event in schedule.iterrows():
            if int(event.RoundNumber) <= 0 or str(event.EventFormat) == "testing":
                continue
            sessions = []
            slots = sorted(
                int(re.fullmatch(r"Session(\d+)", str(c))[1])
                for c in event.index
                if re.fullmatch(r"Session(\d+)", str(c))
            )
            for index in slots:
                name = event.get(f"Session{index}")
                if not isinstance(name, str) or not name:
                    continue
                kind = session_kind(name)
                date = event.get(f"Session{index}DateUtc")
                if pd.isna(date):
                    raise ValueError(f"No UTC session start for {year} {event.EventName} {name}")
                start = pd.Timestamp(date)
                start = start.tz_localize("UTC") if start.tzinfo is None else utc(start)
                sessions.append(
                    {
                        "kind": kind.value,
                        "source_kind": name,
                        "slot": index,
                        "scheduled_start": start.isoformat(),
                    }
                )
            events.append(
                {
                    "event_id": event_key(year, int(event.RoundNumber)),
                    "year": year,
                    "round": int(event.RoundNumber),
                    "name": str(event.EventName),
                    "location": str(event.Location),
                    "country": event.get("Country"),
                    "event_format": str(event.EventFormat),
                    "sessions": sessions,
                    "schedule_retrieved_at": retrieved,
                    "schedule_source_columns": list(schedule.columns),
                    "fastf1_version": version("fastf1"),
                }
            )
        return events

    def fetch_results(self, event: dict, spec: dict) -> SourceBatch:
        kind = spec["kind"]
        if kind not in ("Qualifying", "Race"):
            raise ValueError("Official results are requested only for GP Q/R")
        method = (
            self.results_api.get_qualifying_results
            if kind == "Qualifying"
            else self.results_api.get_race_results
        )
        response = method(season=event["year"], round=event["round"])
        frames, descriptions, diagnostics = [], [], []
        complete, page_count = False, 0
        while True:
            page_count += 1
            description = getattr(response, "description", pd.DataFrame())
            descriptions.append(pd.DataFrame(description))
            for _, row in description.iterrows():
                if int(row["season"]) != event["year"] or int(row["round"]) != event["round"]:
                    raise ValueError("Official result response has the wrong event identity")
            frames.extend(pd.DataFrame(frame) for frame in response.content)
            total = getattr(response, "total_results", None)
            count = sum(len(frame) for frame in frames)
            if response.is_complete or (total is not None and count >= total):
                complete = True
                break
            if page_count >= getattr(self, "max_api_pages", 20):
                diagnostics.append(
                    {"type": "PaginationLimit", "message": "Result paging bound reached"}
                )
                break
            try:
                response = response.get_next_result_page()
            except Exception as exc:
                diagnostics.append({"type": type(exc).__name__, "message": str(exc)})
                break
        raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        return SourceBatch(
            {"results": raw, "description": pd.concat(descriptions, ignore_index=True)},
            {
                "retrieved_at": timestamp(),
                "fastf1_version": version("fastf1"),
                "source": "jolpica_via_fastf1_ergast",
                "source_version": "ergast-compatible-v1",
                "response_complete": complete,
                "page_count": page_count,
                "errors": diagnostics,
                "warnings": [],
                "completion_state": "results_available" if not raw.empty else "unknown",
                "completion_evidence": "official_results_endpoint" if not raw.empty else "none",
                "completed_at": None,
                "absolute_time_available": False,
            },
        )

    def fetch_timing(self, event: dict, spec: dict) -> SourceBatch:
        kind = session_kind(spec["kind"])
        if kind not in GP_SESSIONS:
            raise ValueError("Sprint observations are outside this phase")
        session = self.fastf1.get_session(event["year"], event["round"], kind.value)
        diagnostics, frames = [], {}
        source_info = {}
        handler = DiagnosticHandler()
        logger = logging.getLogger("fastf1")
        logger.addHandler(handler)
        try:
            try:
                session.load(
                    laps=kind in PRACTICES | {"Qualifying"},
                    telemetry=False,
                    weather=True,
                    messages=kind in PRACTICES | {"Qualifying"},
                )
            except Exception as exc:
                diagnostics.append(
                    {"component": "load", "type": type(exc).__name__, "message": str(exc)}
                )
            properties = [("entries", "results"), ("weather", "weather_data")]
            if kind in PRACTICES | {"Qualifying"}:
                properties += [("laps", "laps"), ("status", "session_status")]
            for name, attribute in properties:
                try:
                    frames[name] = pd.DataFrame(getattr(session, attribute)).copy()
                except Exception as exc:
                    diagnostics.append(
                        {"component": name, "type": type(exc).__name__, "message": str(exc)}
                    )
            try:
                source_info = dict(session.session_info)
            except (AttributeError, DataNotLoadedError):
                pass
            if kind == "Qualifying" and "laps" in frames and not frames["laps"].empty:
                try:
                    original = session.laps
                    if not original.index.is_unique:
                        raise ValueError("Qualifying lap indices are not unique")
                    labels = pd.Series(pd.NA, index=original.index, dtype="string")
                    for part, laps in enumerate(original.split_qualifying_sessions(), start=1):
                        if laps is not None:
                            labels.loc[laps.index] = f"Q{part}"
                    frames["laps"]["QualifyingSegment"] = labels.to_numpy()
                    source_info["QualifyingSegmentationMethod"] = (
                        "FastF1.Laps.split_qualifying_sessions"
                    )
                except (AttributeError, ValueError, DataNotLoadedError) as exc:
                    diagnostics.append(
                        {
                            "component": "qualifying_segment",
                            "type": type(exc).__name__,
                            "message": str(exc),
                        }
                    )
        finally:
            logger.removeHandler(handler)
        status = frames.get("status", pd.DataFrame())
        values = status["Status"].tolist() if "Status" in status else []
        last_finalised = max(
            (i for i, value in enumerate(values) if value == "Finalised"), default=-1
        )
        finalised = (
            bool(values)
            and values[-1] in ("Finalised", "Ends")
            and last_finalised >= 0
            and "Started" not in values[last_finalised + 1 :]
        )
        state = "completed" if finalised else "in_progress" if values else "unknown"
        observed_start, observed_end = None, None
        # Accessing an already available property never triggers telemetry loading.
        # In a fresh telemetry=False Session this is normally not loaded.
        try:
            origin = pd.Timestamp(session.t0_date)
            if pd.notna(origin) and session.session_start_time is not None:
                origin = origin.tz_localize("UTC") if origin.tzinfo is None else utc(origin)
                observed_start = (origin + session.session_start_time).isoformat()
                if finalised:
                    observed_end = (origin + status.iloc[-1].Time).isoformat()
        except (AttributeError, ValueError, DataNotLoadedError):
            pass
        return SourceBatch(
            frames,
            {
                "retrieved_at": timestamp(),
                "fastf1_version": version("fastf1"),
                "source": "fastf1_public_session",
                "source_version": version("fastf1"),
                "completion_state": state,
                "completion_evidence": "terminal_finalised_status"
                if finalised
                else "session_status"
                if values
                else "none",
                "completion_status_values": values,
                "source_session_info": source_info,
                "started_at": observed_start,
                "completed_at": observed_end,
                "absolute_time_available": observed_end is not None,
                "errors": diagnostics,
                "warnings": handler.messages,
                "load_options": {
                    "laps": kind != "Race",
                    "telemetry": False,
                    "weather": True,
                    "messages": kind != "Race",
                },
            },
        )

    def session(self, event: dict, spec: dict, as_of: pd.Timestamp) -> tuple[dict, pd.DataFrame]:
        """Phase 1 compatibility export: require absolute times, never invent them.

        Phase 2 catalog commands retain results/partial timing independently. This
        legacy export makes no raw telemetry requests and stays strict about cutoffs.
        """
        kind = session_kind(spec["kind"])
        if hasattr(self, "cache_dir"):
            self._enable_cache(
                self.cache_dir / "legacy" / event["event_id"] / self.legacy_cache_generation
            )
        timing = self.fetch_timing(event, spec)
        info = timing.metadata
        if info["completion_state"] == "completed" and not info["completed_at"]:
            raise ValueError("No absolute session timing; cannot safely export cutoff inputs")
        completed = info["completion_state"] == "completed" and utc(info["completed_at"]) <= as_of
        meta = {
            **spec,
            "started_at": info["started_at"],
            "ended_at": info["completed_at"],
            "retrieved_at": info["retrieved_at"],
            "completed": bool(completed),
            "completion_evidence": info["completion_evidence"],
            "official_results": False,
        }
        if not completed:
            return meta, pd.DataFrame()
        if kind in PRACTICES:
            from types import SimpleNamespace

            return meta, summarize_practice(
                SimpleNamespace(laps=timing.frames["laps"], results=timing.frames["entries"])
            )
        batch = self.fetch_results(event, spec)
        if not batch.metadata["response_complete"]:
            raise ValueError("Paginated/incomplete official results; source review required")
        table = normalize_results(batch.frames["results"], kind.value)
        return {
            **meta,
            "retrieved_at": batch.metadata["retrieved_at"],
            "official_results": not table.empty,
            "results_source": batch.metadata["source"],
        }, table
