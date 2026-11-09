"""Public FIA document evidence, cached independently of the restricted timing archive.

Publication is a conservative availability/completion upper bound, never an
invented observed session-end time. Raw PDFs stay in the local acquisition cache.
"""

import html
import re
import time
import unicodedata
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse

import pandas as pd
import requests

from f1ml.ingestion.catalog import current_snapshot
from f1ml.storage import digest, file_digest, read_json, seal_directory, write_json

INDEX = "https://www.fia.com/documents/championships/110/2026"
REVIEWED_EVENT_URLS = {
    # The short Barcelona facets point to F3, not the F1 weekend.
    "Barcelona Grand Prix": "https://www.fia.com/documents/championships/"
    "fia-formula-one-world-championship-14/season/season-2026-2072/"
    "event/Barcelona-Catalunya%20Grand%20Prix",
}
TEAM_NAMES = {
    "racing bulls": "racing_bulls",
    "red bull racing": "red_bull",
    "mclaren": "mclaren",
    "williams": "williams",
    "aston martin": "aston_martin",
    "haas": "haas",
    "audi": "audi",
    "alpine": "alpine",
    "cadillac": "cadillac",
    "mercedes": "mercedes",
    "sauber": "sauber",
    "ferrari": "ferrari",
    " rb ": "racing_bulls",
}


def normalized(value):
    text = unicodedata.normalize("NFKD", str(value))
    return " ".join("".join(c for c in text if not unicodedata.combining(c)).split())


class PublicCache:
    def __init__(self, root, offline=False):
        self.root, self.offline = Path(root), offline
        self.root.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.last_request = 0.0

    def get(self, url):
        basename = unquote(Path(urlparse(url).path).name) or "index.html"
        path = self.root / f"{digest(url)[:16]}-{basename}"
        metadata = path.with_suffix(path.suffix + ".json")
        if path.exists() and metadata.exists():
            if file_digest(path) != read_json(metadata)["sha256"]:
                raise ValueError(f"Cached source changed: {path}")
            return path
        if self.offline:
            raise ValueError(f"Source absent from offline cache: {url}")
        time.sleep(max(0, 0.5 - (time.monotonic() - self.last_request)))
        response = self.session.get(url, timeout=40)
        self.last_request = time.monotonic()
        response.raise_for_status()
        if urlparse(url).path.endswith(".pdf") and not response.content.startswith(b"%PDF"):
            raise ValueError(f"Expected a PDF, received another content type: {url}")
        path.write_bytes(response.content)
        write_json(
            metadata,
            {
                "url": url,
                "retrieved_at": pd.Timestamp.now(tz="UTC").isoformat(),
                "sha256": file_digest(path),
                "status_code": response.status_code,
            },
        )
        return path


def document_index(text):
    records = []
    for link, body in re.findall(r'<a href="([^\"]+\.pdf)"[^>]*>(.*?)</a>', text, re.S):
        plain = html.unescape(re.sub("<[^>]+>", " ", body))
        plain = " ".join(plain.split())
        match = re.search(r"Published on (\d{2}\.\d{2}\.\d{2} \d{2}:\d{2}) (CEST|CET)", plain)
        if not match:
            continue
        dt = datetime.strptime(match[1], "%d.%m.%y %H:%M")
        offset = 2 if match[2] == "CEST" else 1
        published = pd.Timestamp(dt.replace(tzinfo=timezone(timedelta(hours=offset)))).tz_convert(
            "UTC"
        )
        records.append(
            {
                "title": plain.split("Published on")[0].strip(),
                "url": urljoin("https://www.fia.com", link),
                "published_at": published.isoformat(),
                "printed_timezone": match[2],
            }
        )
    return records


def pdf_text(path, *, roster=False):
    import pdfplumber

    # Tiny superscript footnotes otherwise merge into car numbers (e.g. car 1 -> 13).
    with pdfplumber.open(path) as doc:
        return "\n".join(
            (
                page.filter(lambda o: o["object_type"] != "char" or o["size"] >= 8)
                if roster
                else page
            ).extract_text()
            or ""
            for page in doc.pages
        )


def parse_roster(text, identities, driver_names=None):
    text = normalized(text).split("In addition to the list")[0] if "\n" not in text else text
    text = text.split("In addition to the list")[0]
    rows = []
    for line in text.splitlines():
        line = normalized(line)
        match = re.match(r"^(\d{1,2})\s+([A-Z]{3})\s+(.+?)\s+([A-Z]{3})\s+(.+)$", line)
        if not match:
            # 2024/2025 FIA lists print names without three-letter driver codes.
            match = re.match(r"^(\d{1,2})\s+(.+?)\s+([A-Z]{3})\s+(.+)$", line)
            if not match or not driver_names:
                continue
            number, name, nation, team_text = match.groups()
            identity = driver_names.get(normalized(name).casefold())
            if identity is None:
                raise ValueError(f"Unreviewed driver identity: {name}")
            tla = identity
        else:
            number, tla, name, nation, team_text = match.groups()
        if tla not in identities:
            raise ValueError(f"Unreviewed driver identity {tla}: {name}")
        team = next(
            (canonical for token, canonical in TEAM_NAMES.items() if token in team_text.casefold()),
            None,
        )
        if team is None:
            raise ValueError(f"Unreviewed team identity: {team_text}")
        rows.append(
            {
                "driver_number": int(number),
                "driver_id": identities[tla],
                "abbreviation": tla,
                "driver": name,
                "team_id": team,
                "team_name": team_text,
                "nationality": nation,
            }
        )
    frame = pd.DataFrame(rows)
    if (
        len(frame) < 18
        or frame.driver_id.duplicated().any()
        or frame.driver_number.duplicated().any()
    ):
        raise ValueError(f"Independent entry list is incomplete or ambiguous ({len(frame)} rows)")
    return frame


def parse_classification(text, roster, kind):
    """Numeric positions only when explicitly printed; preserve NC/DNS/DSQ without a tail rank."""
    rows, section, started = [], "ordered", False
    known = roster.set_index("driver_number")
    for raw in text.splitlines():
        line = normalized(raw)
        if re.match(r"^NO\s+DRIVER\s+NAT", line):
            started = True
            continue
        if not started:
            continue
        if line.startswith(
            (
                "FASTEST LAP",
                "OVERALL FASTEST LAP",
                "POLE POSITION LAP",
                "* PENALTIES",
                "Timekeeper:",
            )
        ):
            break
        if re.match(r"^\d", line) and "KM/H" in line:
            # Some PDFs omit the pole-position heading in their text layer.
            break
        if line in ("NOT CLASSIFIED", "DISQUALIFIED", "DID NOT START", "WITHDRAWN"):
            section = {
                "NOT CLASSIFIED": "non_classified",
                "DISQUALIFIED": "dsq",
                "DID NOT START": "dns",
                "WITHDRAWN": "withdrawn",
            }[line]
            continue
        match = re.match(r"^(?:(\d{1,2})\s+)?(\d{1,2})\s+([A-Za-z].*)$", line)
        if not match:
            continue
        position, number, name = match.groups()
        number = int(number)
        if number not in known.index:
            raise ValueError(f"Classification driver {number} absent from independent entry list")
        # Never match a timing/team line merely because it begins with a number.
        expected = normalized(known.loc[number, "driver"]).casefold().replace(" ", "")
        actual = normalized(name.replace("*", "")).casefold().replace(" ", "")
        aliases = [expected]
        if known.loc[number, "driver_id"] == "antonelli":
            aliases.extend(("andreakimiantonelli", "kimiantonelli"))
        if not any(alias in actual or actual in alias for alias in aliases):
            raise ValueError(f"Classification/roster identity mismatch: {number} {name}")
        rank = float(position) if position and section == "ordered" else float("nan")
        rows.append(
            {
                "driver_id": known.loc[number, "driver_id"],
                "official_position": rank,
                "outcome": section,
                "podium": float(rank <= 3) if pd.notna(rank) else 0.0,
                "printed_driver": name,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty or frame.driver_id.duplicated().any():
        raise ValueError("Empty or repeated classification identities")
    if set(frame.driver_id) - set(roster.driver_id):
        raise ValueError("Classification includes an unreviewed entrant")
    frame["label_observed"] = True
    missing = set(roster.driver_id) - set(frame.driver_id)
    if missing:
        # These are genuine roster rows with absent outcomes, not fabricated DNS labels.
        frame = pd.concat(
            [
                frame,
                pd.DataFrame(
                    [
                        {
                            "driver_id": driver,
                            "official_position": float("nan"),
                            "outcome": "unknown",
                            "podium": float("nan"),
                            "printed_driver": None,
                            "label_observed": False,
                        }
                        for driver in sorted(missing)
                    ]
                ),
            ],
            ignore_index=True,
        )
    if frame.official_position.dropna().duplicated().any():
        raise ValueError("Repeated official numeric classifications")
    if kind == "Race" and frame.podium.sum() != 3:
        raise ValueError("Official podium is incomplete")
    if kind == "Qualifying":
        frame["podium"] = float("nan")
    return frame


def prepare_dataset(
    catalog, cache_root, output, *, as_of, year=2026, offline=False, allow_result_fallback=False
):
    if Path(output).exists():
        raise FileExistsError(output)
    if allow_result_fallback and year not in (2024, 2025):
        raise ValueError("API result fallback is restricted to archived 2024/2025 seasons")
    snapshot, receipt = current_snapshot(Path(catalog))
    events = pd.read_parquet(snapshot / "tables/events.parquet")
    sessions = pd.read_parquet(snapshot / "tables/sessions.parquet")
    source_results = pd.read_parquet(snapshot / "tables/results.parquet")
    pairs = source_results[["source_driverCode", "driver_id"]].dropna().drop_duplicates()
    if pairs.source_driverCode.duplicated().any():
        raise ValueError("Driver abbreviations need an explicit identity review")
    identities = pairs.set_index("source_driverCode").driver_id.to_dict()
    cache = PublicCache(cache_root, offline)
    index_path = cache.get(f"https://www.fia.com/documents/championships/110/{year}")
    if year != 2026:
        # The bare year URL does not select a season on FIA's faceted site.
        season = re.search(
            rf'<option value="([^\"]*/season/season-{year}-\d+)"', index_path.read_text()
        )
        if not season:
            raise ValueError(f"FIA season index not found for {year}")
        index_path = cache.get(
            "https://www.fia.com/documents/championships/"
            "fia-formula-one-world-championship-14/season/" + season[1].rsplit("/", 1)[-1]
        )
    options = {
        html.unescape(name).strip(): urljoin("https://www.fia.com", url)
        for url, name in re.findall(
            r'<option value="([^\"]*/event/[^\"]+)"[^>]*>([^<]+)', index_path.read_text()
        )
    }
    driver_names = {}
    for row in source_results.to_dict("records"):
        for name in (
            f"{row['source_givenName']} {row['source_familyName']}",
            f"{row['source_familyName']} {row['source_givenName']}",
        ):
            driver_names[normalized(name).casefold()] = row["source_driverCode"]
    # The FIA prints Antonelli's full given name; the results API abbreviates it.
    if "ANT" in identities:
        driver_names["andrea kimi antonelli"] = "ANT"
        driver_names["kimi antonelli"] = "ANT"
    horizon = pd.Timestamp(as_of)
    rows, rosters, result_rows, docs, exclusions = [], [], [], [], []
    fallbacks, captures = [], {}
    for event in events.loc[events.year.eq(year)].sort_values("round").to_dict("records"):
        eid = event["event_id"]
        ss = sessions.loc[sessions.event_id.eq(eid)]
        q = ss.loc[ss.kind.eq("Qualifying")].iloc[0]
        race = ss.loc[ss.kind.eq("Race")].iloc[0]
        if pd.Timestamp(race.scheduled_start) >= horizon:
            exclusions.append({"event_id": eid, "reason": "race_not_started_as_of"})
            continue
        name = "Barcelona-Catalunya" if event["name"] == "Barcelona Grand Prix" else event["name"]
        print(f"Reviewing {eid} {name}", flush=True)
        records = []
        try:
            reviewed_url = REVIEWED_EVENT_URLS.get(event["name"]) if year == 2026 else None
            if name not in options and not reviewed_url:
                raise ValueError("FIA event identity not found in the observed index")
            page = cache.get(reviewed_url or options[name])
            records = [
                d
                for d in document_index(page.read_text())
                if pd.Timestamp(d["published_at"]) <= horizon
            ]
            finals = {}
            for kind in ("Qualifying", "Race"):
                selected = [
                    d
                    for d in records
                    if re.search(rf"Final {kind} Classification\s*$", d["title"], re.I)
                    and pd.Timestamp(d["published_at"]) <= horizon
                    and re.search(rf"\b{year}\b", unquote(d["url"]).replace("_", " "))
                ]
                if not selected:
                    raise ValueError(f"Missing official final {kind} classification")
                finals[kind] = max(selected, key=lambda d: d["published_at"])
            qcut = pd.Timestamp(q.scheduled_start) - pd.Timedelta(seconds=1)
            rcut = pd.Timestamp(finals["Qualifying"]["published_at"]) + pd.Timedelta(minutes=1)
            if rcut >= pd.Timestamp(race.scheduled_start):
                raise ValueError("Published final qualifying was unavailable before the race")
            event_rosters, event_labels, event_docs = [], [], []
            for kind, cutoff in (("Qualifying", qcut), ("Race", rcut)):
                eligible = [
                    d
                    for d in records
                    if "entry list" in d["title"].casefold()
                    and "practice" not in d["title"].casefold()
                    and pd.Timestamp(d["published_at"]) <= cutoff
                ]
                if not eligible:
                    raise ValueError(f"No independent entrant list before {kind} cutoff")
                entry = max(eligible, key=lambda d: d["published_at"])
                entry_path = cache.get(entry["url"])
                roster = parse_roster(
                    pdf_text(entry_path, roster=True), identities, driver_names=driver_names
                )
                final = finals[kind]
                final_path = cache.get(final["url"])
                if not re.search(rf"\b{year}\b", normalized(pdf_text(final_path))[:600]):
                    raise ValueError("Classification source has the wrong season identity")
                labels = parse_classification(pdf_text(final_path), roster, kind)
                task = "qualifying" if kind == "Qualifying" else "race"
                roster = roster.assign(
                    event_id=eid,
                    task=task,
                    known_at=entry["published_at"],
                    source=entry["url"],
                    cutoff=cutoff.isoformat(),
                )
                labels = labels.assign(
                    event_id=eid, task=task, available_at=final["published_at"], source=final["url"]
                )
                event_rosters.append(roster)
                event_labels.append(labels)
                event_docs.extend(
                    [
                        {
                            **entry,
                            "event_id": eid,
                            "task": task,
                            "purpose": "roster",
                            "sha256": file_digest(entry_path),
                        },
                        {
                            **final,
                            "event_id": eid,
                            "task": task,
                            "purpose": "label",
                            "sha256": file_digest(final_path),
                        },
                    ]
                )
            rosters.extend(event_rosters)
            result_rows.extend(event_labels)
            docs.extend(event_docs)
            rows.append(
                {
                    **event,
                    "weekend_start": ss.scheduled_start.min(),
                    "qualifying_cutoff": qcut.isoformat(),
                    "race_cutoff": rcut.isoformat(),
                    "qualifying_available_at": finals["Qualifying"]["published_at"],
                    "race_available_at": finals["Race"]["published_at"],
                }
            )
        except (ValueError, KeyError, requests.RequestException) as exc:
            if allow_result_fallback:
                from f1ml_real.historical import catalog_event

                try:
                    row, rs, ls, ds, captured = catalog_event(
                        event, ss, source_results, records, cache, identities, driver_names
                    )
                except (ValueError, KeyError, OSError) as fallback_error:
                    exc = ValueError(f"{exc}; API fallback unavailable: {fallback_error}")
                else:
                    rows.append(row)
                    rosters.extend(rs)
                    result_rows.extend(ls)
                    docs.extend(ds)
                    captures.update(captured)
                    fallbacks.append({"event_id": eid, "name": name, "reason": str(exc)})
                    print(f"API fallback {eid}: {exc}", flush=True)
                    continue
            exclusions.append({"event_id": eid, "reason": f"{type(exc).__name__}: {exc}"})
            print(f"Excluded {eid}: {exc}", flush=True)
    output = Path(output)
    if not rows:
        raise ValueError(f"No eligible {year} events: {exclusions}")
    output.mkdir(parents=True)
    for name, frame in {
        "events": pd.DataFrame(rows),
        "rosters": pd.concat(rosters, ignore_index=True) if rosters else pd.DataFrame(),
        "labels": pd.concat(result_rows, ignore_index=True) if result_rows else pd.DataFrame(),
        "documents": pd.DataFrame(docs),
        "exclusions": pd.DataFrame(exclusions),
    }.items():
        frame.to_parquet(output / f"{name}.parquet", index=False)
        frame.to_csv(output / f"{name}.csv", index=False)
    if allow_result_fallback:
        pd.DataFrame(fallbacks, columns=["event_id", "name", "reason"]).to_parquet(
            output / "fallbacks.parquet", index=False
        )
    for relative, capture in captures.items():
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(capture)
    write_json(output / "identity_mapping.json", identities)
    return seal_directory(
        output,
        {
            "data_kind": "real",
            "source": (
                "FIA evidence with labeled captured FastF1/Jolpica API fallback"
                if fallbacks
                else "FIA final classifications and independent entry lists"
            ),
            "api_fallback_events": [row["event_id"] for row in fallbacks],
            "fallback_semantics": (
                "API captures reconstruct historical results and sometimes entrant fields. "
                "FIA publication records are timing proxies for API labels; absent records use "
                "qualifying just before race start and race start plus 24 hours as calendar "
                "proxies. Historical API availability/revision timing is not verified."
            )
            if fallbacks
            else None,
            "source_catalog_id": receipt["id"],
            "year": year,
            "as_of": as_of,
            "eligible_events": len(rows),
            "excluded_events": len(exclusions),
            "mode": "historical_reconstruction_exploratory",
            "availability_semantics": (
                "FIA-only rows use official document publication as a conservative upper bound. "
                "API fallback rows use explicitly recorded FIA publication/calendar proxies; "
                "historical API availability and revisions are not verified."
                if fallbacks
                else "Official document publication is a conservative availability/completion "
                "upper bound, not an observed session-end timestamp"
            ),
            "exposure": (
                "Historical outcomes were previously accessible and consulted; "
                "no untouched holdout claim"
            ),
            "missing_groups": ["practice_laps", "qualifying_laps", "weather"],
            "implementation_id": file_digest(Path(__file__)),
        },
    )
