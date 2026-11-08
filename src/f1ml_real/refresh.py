"""Explicit FastF1 refresh with independently published FIA fields and final labels."""

from __future__ import annotations

import html
import re
import threading
from pathlib import Path
from urllib.parse import urljoin
from uuid import uuid4

import pandas as pd
import requests

from f1ml.ingestion.catalog import atomic_json, writer_lock
from f1ml.ingestion.pipeline import validate_events
from f1ml.ingestion.provider import FastF1Provider
from f1ml.storage import (
    file_digest,
    read_json,
    save_table,
    seal_directory,
    verify_directory,
    write_json,
)
from f1ml_real.fia import (
    REVIEWED_EVENT_URLS,
    PublicCache,
    document_index,
    normalized,
    parse_classification,
    parse_roster,
    pdf_text,
)
from f1ml_real.lab import RealDataset

# FastF1's cache configuration is process-global, including Cache.disabled().
_REFRESH_LOCK = threading.Lock()


def _now():
    return pd.Timestamp.now(tz="UTC")


def current_dataset(base: Path, state: Path) -> Path:
    """Only activate a completely written, verified snapshot for this base dataset."""
    marker = Path(state) / "CURRENT.json"
    if not marker.exists():
        return Path(base)
    pointer = read_json(marker)
    if pointer["base_snapshot_id"] != verify_directory(Path(base))["id"]:
        return Path(base)
    path = (Path(state) / pointer["path"]).resolve()
    if not path.is_relative_to((Path(state) / "snapshots").resolve()):
        raise ValueError("Refresh pointer is outside its snapshot directory")
    if verify_directory(path)["id"] != pointer["snapshot_id"]:
        raise ValueError("Refresh pointer does not match its snapshot")
    return path


def _roster(text, identities, raw_results):
    """Match new identities by both printed car number and name, never a previous field."""
    identities = dict(identities)
    for line in text.split("In addition to the list")[0].splitlines():
        match = re.match(r"^(\d{1,2})\s+([A-Z]{3})\s+(.+?)\s+([A-Z]{3})\s+(.+)$", normalized(line))
        if not match:
            continue
        number, code, name, _, _ = match.groups()
        if code in identities:
            continue
        for raw in raw_results:
            if "number" not in raw:
                continue
            for row in raw.loc[raw.number.astype(str).eq(number)].to_dict("records"):
                api_name = normalized(f"{row['givenName']} {row['familyName']}").casefold()
                if api_name == normalized(name).casefold():
                    identities[code] = row["driverId"]
    return parse_roster(text, identities)


def refresh(base: Path, state: Path) -> Path:
    """Download on explicit action, seal a revision, then atomically activate it.

    FastF1 supplies the current calendar and Q/R endpoint captures. FIA final PDFs
    retain the lab's numeric-rank and publication semantics. A new acquisition
    cache directory ensures mutable indexes and empty API responses are refreshed.
    """
    with _REFRESH_LOCK, writer_lock(Path(state)):
        return _refresh(Path(base), Path(state))


def _refresh(base, state):
    original = RealDataset(base)
    previous = RealDataset(current_dataset(base, state))
    generation = uuid4().hex
    cache = PublicCache(state / "cache" / generation / "fia")
    provider = FastF1Provider(state / "cache" / generation / "fastf1")
    year = int(original.events.year.max())
    with provider.fastf1.Cache.disabled():
        schedule = provider.schedule(year)
        validate_events(schedule)
        if not schedule:
            raise ValueError("FastF1 returned an empty calendar; previous snapshot remains active")
        # Only completed original facts are immutable. Partial/new events are retried.
        fixed = set(original.events.event_id)
        events = original.events.to_dict("records")
        rosters, labels, documents = [original.rosters], [original.labels], [original.documents]
        statuses, captures = [], []
        identities = read_json(previous.root / "identity_mapping.json")
        index = cache.get(f"https://www.fia.com/documents/championships/110/{year}").read_text()
        options = {
            normalized(html.unescape(name)).casefold(): urljoin("https://www.fia.com", url)
            for url, name in re.findall(r'<option value="([^\"]+)"[^>]*>([^<]+)', index)
            if f"/{year}/event/" in url
        }
        for event in schedule:
            eid = event["event_id"]
            if eid in fixed:
                continue
            starts = {s["kind"]: pd.Timestamp(s["scheduled_start"]) for s in event["sessions"]}
            qstart, rstart = starts["Qualifying"], starts["Race"]
            row = {k: v for k, v in event.items() if k != "sessions"}
            row.update(
                weekend_start=min(starts.values()).isoformat(),
                qualifying_start=qstart.isoformat(),
                race_start=rstart.isoformat(),
                qualifying_cutoff=(qstart - pd.Timedelta(seconds=1)).isoformat(),
                race_cutoff=None,
                qualifying_available_at=None,
                race_available_at=None,
            )
            problems, raw_results = [], []
            horizon = _now()
            if horizon < qstart:
                row["qualifying_cutoff"] = horizon.isoformat()
            # Do not ask result endpoints about sessions which have not started.
            for kind in ("Qualifying", "Race"):
                if starts[kind] >= horizon:
                    continue
                try:
                    batch = provider.fetch_results(event, {"kind": kind})
                    captures.append((eid, kind, batch))
                    if not batch.metadata["response_complete"]:
                        problems.append(f"{kind}: incomplete FastF1 result response")
                    else:
                        raw_results.append(batch.frames["results"])
                except Exception as exc:
                    problems.append(f"{kind} FastF1: {type(exc).__name__}: {exc}")
            event_rosters, event_labels, event_docs = [], [], []
            try:
                url = REVIEWED_EVENT_URLS.get(event["name"]) or options.get(
                    normalized(event["name"]).casefold()
                )
                if not url:
                    raise ValueError("FIA event documents not published yet")
                records = document_index(cache.get(url).read_text())
                records = [
                    d
                    for d in records
                    if pd.Timestamp(d["published_at"]) <= horizon and f"/{year}_" in d["url"]
                ]
                finals = {}
                for kind in ("Qualifying", "Race"):
                    choices = [
                        d
                        for d in records
                        if re.search(rf"Final {kind} Classification\s*$", d["title"], re.I)
                        and starts[kind] < horizon
                    ]
                    if choices:
                        finals[kind] = max(choices, key=lambda d: d["published_at"])
                qfinal = finals.get("Qualifying")
                row["race_cutoff"] = (
                    (
                        horizon
                        if horizon < rstart
                        else pd.Timestamp(qfinal["published_at"]) + pd.Timedelta(minutes=1)
                    ).isoformat()
                    if qfinal
                    else None
                )
                for kind, task in (("Qualifying", "qualifying"), ("Race", "race")):
                    cutoff = row[f"{task}_cutoff"]
                    if cutoff is None or (task == "race" and pd.Timestamp(cutoff) >= rstart):
                        continue
                    entries = [
                        d
                        for d in records
                        if "entry list" in d["title"].casefold()
                        and "practice" not in d["title"].casefold()
                        and pd.Timestamp(d["published_at"]) <= pd.Timestamp(cutoff)
                    ]
                    if not entries:
                        problems.append(f"{kind}: no independent pre-cutoff entry list")
                        continue
                    entry = max(entries, key=lambda d: d["published_at"])
                    entry_path = cache.get(entry["url"])
                    text = pdf_text(entry_path, roster=True)
                    if str(year) not in text[:1000]:
                        raise ValueError("Entry list has the wrong season")
                    roster = _roster(text, identities, raw_results)
                    event_rosters.append(
                        roster.assign(
                            event_id=eid,
                            task=task,
                            known_at=entry["published_at"],
                            source=entry["url"],
                            cutoff=cutoff,
                        )
                    )
                    identities.update(roster.set_index("abbreviation").driver_id.to_dict())

                    event_docs.append(
                        {
                            **entry,
                            "event_id": eid,
                            "task": task,
                            "purpose": "roster",
                            "sha256": file_digest(entry_path),
                        }
                    )
                    if kind not in finals:
                        continue
                    final = finals[kind]
                    final_path = cache.get(final["url"])
                    text = pdf_text(final_path)
                    if str(year) not in normalized(text)[:600]:
                        raise ValueError("Classification has the wrong season")
                    parsed = parse_classification(text, roster, kind)
                    event_labels.append(
                        parsed.assign(
                            event_id=eid,
                            task=task,
                            available_at=final["published_at"],
                            source=final["url"],
                        )
                    )
                    row[f"{task}_available_at"] = final["published_at"]
                    event_docs.append(
                        {
                            **final,
                            "event_id": eid,
                            "task": task,
                            "purpose": "label",
                            "sha256": file_digest(final_path),
                        }
                    )
                old = previous.events.loc[previous.events.event_id.eq(eid)]
                if not old.empty:
                    for task in ("qualifying", "race"):
                        if pd.notna(old.iloc[0][f"{task}_available_at"]) and pd.isna(
                            row[f"{task}_available_at"]
                        ):
                            raise ValueError(f"Previously collected {task} classification missing")
            except (ValueError, KeyError, OSError, requests.RequestException) as exc:
                problems.append(f"{type(exc).__name__}: {exc}")
                # Never replace previously collected evidence with a transient failure.
                old = previous.events.loc[previous.events.event_id.eq(eid)]
                if not old.empty:
                    events.extend(old.to_dict("records"))
                    for name, destination in (
                        ("rosters", rosters),
                        ("labels", labels),
                        ("documents", documents),
                    ):
                        frame = getattr(previous, name)
                        destination.append(frame.loc[frame.event_id.eq(eid)])
                    statuses.append(
                        {
                            "event_id": eid,
                            "name": event["name"],
                            "issues": problems,
                            "retained_previous": True,
                        }
                    )
                    continue
            events.append(row)
            rosters.extend(event_rosters)
            labels.extend(event_labels)
            if event_docs:
                documents.append(pd.DataFrame(event_docs))
            statuses.append({"event_id": eid, "name": event["name"], "issues": problems})
    output = state / "snapshots" / generation
    output.mkdir(parents=True)
    for name, frame in {
        "events": pd.DataFrame(events),
        "rosters": pd.concat(rosters, ignore_index=True),
        "labels": pd.concat(labels, ignore_index=True),
        "documents": pd.concat(documents, ignore_index=True),
        "exclusions": pd.DataFrame(statuses),
    }.items():
        save_table(output / f"{name}.parquet", frame)
    for eid, kind, batch in captures:
        prefix = output / "fastf1" / eid / kind
        for name, frame in batch.frames.items():
            save_table(prefix / f"{name}.parquet", frame)
        write_json(prefix / "metadata.json", batch.metadata)
    write_json(output / "identity_mapping.json", identities)
    write_json(output / "schedule.json", schedule)
    write_json(output / "refresh_status.json", statuses)
    receipt = seal_directory(
        output,
        {
            "data_kind": "real",
            "kind": "refreshed_live_dataset",
            "year": year,
            "base_snapshot_id": original.receipt["id"],
            "source": (
                "FastF1 calendar/results and FIA final classifications/independent entry lists"
            ),
            "as_of": _now().isoformat(),
            "missing_groups": ["practice_laps", "qualifying_laps", "weather"],
        },
    )
    RealDataset(output)
    atomic_json(
        state / "CURRENT.json",
        {
            "path": str(output.relative_to(state)),
            "snapshot_id": receipt["id"],
            "base_snapshot_id": original.receipt["id"],
        },
    )
    return output


def forecast_tasks(dataset, event_id, now=None):
    """Enable only unstarted targets with the required independently sourced inputs."""
    now = _now() if now is None else pd.Timestamp(now).tz_convert("UTC")
    event = dataset.event(event_id)
    tasks = []
    for task in ("qualifying", "race"):
        start = event.get(f"{task}_start")
        if pd.isna(start) or now >= pd.Timestamp(start):
            continue
        roster = dataset.rosters.loc[
            dataset.rosters.event_id.eq(event_id) & dataset.rosters.task.eq(task)
        ]
        if roster.empty or dataset.history_events(event_id).empty:
            continue
        if task == "race" and (
            pd.isna(event.qualifying_available_at)
            or pd.Timestamp(event.qualifying_available_at) >= now
        ):
            continue
        dataset.feature_frame(event_id, task)  # Enforce roster and qualifying cutoffs too.
        tasks.extend(("race", "podium") if task == "race" else ("qualifying",))
    return tuple(tasks)
