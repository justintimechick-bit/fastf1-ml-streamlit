"""Explicitly labeled historical reconstructions from captured FastF1/Jolpica results.

API captures do not establish historical publication times or independent fields.
Use available FIA entry lists/publication records where possible; mark every
calendar proxy and reconstructed field rather than claiming verified availability.
"""

import hashlib
import re

import numpy as np
import pandas as pd
import requests

from f1ml_real.fia import parse_roster, pdf_text


def catalog_event(event, sessions, results, records, cache, identities, driver_names):
    year, number, eid = int(event["year"]), int(event["round"]), event["event_id"]
    if year not in (2024, 2025):
        raise ValueError("API fallback is restricted to archived 2024/2025 seasons")
    starts = sessions.set_index("kind").scheduled_start
    qstart, rstart = pd.Timestamp(starts["Qualifying"]), pd.Timestamp(starts["Race"])
    qcut = qstart - pd.Timedelta(seconds=1)
    publications = {}
    for kind in ("Qualifying", "Race"):
        choices = [
            d
            for d in records
            if re.search(rf"Final {kind} Classification\s*$", d["title"], re.I)
            and str(year) in d["url"]
        ]
        if choices:
            publications[kind] = max(choices, key=lambda d: d["published_at"])
    # Race inputs are reconstructed immediately before the race. When the FIA
    # publication is absent/late, this is explicitly a proxy, not an observed time.
    qdoc, rdoc = publications.get("Qualifying"), publications.get("Race")
    qtime = pd.Timestamp(qdoc["published_at"]) if qdoc else rstart
    qverified = qtime < rstart - pd.Timedelta(seconds=1)
    qavailable = qtime if qverified else rstart - pd.Timedelta(seconds=2)
    ravailable = pd.Timestamp(rdoc["published_at"]) if rdoc else rstart + pd.Timedelta(days=1)
    rcut = min(qavailable + pd.Timedelta(minutes=1), rstart - pd.Timedelta(seconds=1))
    row = {
        **event,
        "weekend_start": sessions.scheduled_start.min(),
        "qualifying_cutoff": qcut.isoformat(),
        "race_cutoff": rcut.isoformat(),
        "qualifying_available_at": qavailable.isoformat(),
        "race_available_at": ravailable.isoformat(),
        "evidence_mode": "api_fallback",
        "qualifying_availability_kind": "fia_publication_proxy" if qverified else "calendar_proxy",
        "race_availability_kind": "fia_publication_proxy" if rdoc else "calendar_proxy",
    }
    rosters, labels, docs, captures = [], [], [], {}
    for kind, task, cutoff, available in (
        ("Qualifying", "qualifying", qcut, qavailable),
        ("Race", "race", rcut, ravailable),
    ):
        raw = results.loc[results.event_id.eq(eid) & results.kind.eq(kind)].copy()
        if raw.empty or raw.driver_id.duplicated().any() or not raw.official_field_complete.all():
            raise ValueError(f"Incomplete captured API {kind} field for {eid}")
        endpoint = "qualifying" if task == "qualifying" else "results"
        url = f"https://api.jolpi.ca/ergast/f1/{year}/{number}/{endpoint}/"
        relative = f"api_captures/{eid}-{task}.json"
        capture = raw.to_json(orient="records", indent=2) + "\n"
        captures[relative] = capture
        sha = hashlib.sha256(capture.encode()).hexdigest()
        roster = None
        entries = [
            d
            for d in records
            if "entry list" in d["title"].casefold()
            and "practice" not in d["title"].casefold()
            and str(year) in d["url"]
            and pd.Timestamp(d["published_at"]) <= cutoff
        ]
        if entries:
            entry = max(entries, key=lambda d: d["published_at"])
            try:
                path = cache.get(entry["url"])
                roster = parse_roster(
                    pdf_text(path, roster=True), identities, driver_names=driver_names
                )
            except (ValueError, OSError, requests.RequestException):
                pass
            else:
                from f1ml.storage import file_digest

                roster = roster.assign(
                    source=entry["url"],
                    known_at=entry["published_at"],
                    field_evidence="independent_fia_entry_list",
                )
                docs.append(
                    {
                        **entry,
                        "event_id": eid,
                        "task": task,
                        "purpose": "roster",
                        "sha256": file_digest(path),
                    }
                )
        if roster is None:
            roster = pd.DataFrame(
                {
                    "driver_number": pd.to_numeric(raw.driver_number).astype(int),
                    "driver_id": raw.driver_id,
                    "abbreviation": raw.source_driverCode,
                    "driver": raw.source_givenName + " " + raw.source_familyName,
                    "team_id": raw.team_id,
                    "team_name": raw.source_constructorName,
                    "nationality": raw.source_driverNationality,
                    "source": url,
                    "known_at": cutoff.isoformat(),
                    "field_evidence": "api_field_reconstruction",
                }
            )
            docs.append(
                {
                    "title": "API fallback · reconstructed entrant field",
                    "url": url,
                    "published_at": cutoff.isoformat(),
                    "event_id": eid,
                    "task": task,
                    "purpose": "roster",
                    "sha256": sha,
                    "source_file": relative,
                    "availability_kind": "calendar_proxy",
                }
            )
        roster = roster.assign(event_id=eid, task=task, cutoff=cutoff.isoformat())
        rosters.append(roster)
        frame = roster[["driver_id"]].merge(raw, on="driver_id", how="left", validate="one_to_one")
        observed = frame.outcome.notna()
        ordered = frame.outcome.eq("ordered")
        rank = frame.official_position.where(ordered)
        podium = (rank <= 3).astype(float).where(observed)
        if task == "race" and podium.sum() != 3:
            raise ValueError(f"Captured API podium is incomplete for {eid}")
        labels.append(
            pd.DataFrame(
                {
                    "driver_id": frame.driver_id,
                    "official_position": rank,
                    "outcome": frame.outcome.where(observed, "unknown"),
                    "podium": podium if task == "race" else np.nan,
                    "printed_driver": frame.source_givenName + " " + frame.source_familyName,
                    "label_observed": observed,
                    "event_id": eid,
                    "task": task,
                    "available_at": available.isoformat(),
                    "source": url,
                    "evidence_mode": "api_fallback",
                }
            )
        )
        docs.append(
            {
                "title": f"API fallback · captured {kind} classification",
                "url": url,
                "published_at": available.isoformat(),
                "event_id": eid,
                "task": task,
                "purpose": "label",
                "sha256": sha,
                "source_file": relative,
                "availability_kind": row[f"{task}_availability_kind"],
                "fia_publication_reference": ((qdoc if task == "qualifying" else rdoc) or {}).get(
                    "url"
                ),
            }
        )
    return row, rosters, labels, docs, captures
