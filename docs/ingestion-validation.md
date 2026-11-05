# Ingestion validation — November 4, 2026

Environment: Python 3.13, FastF1 **3.8.1**, pandas 2.3.3, PyArrow 23.0.1.
Read the Phase 1 methodology/configuration before implementation. No `AGENTS.md`
was present in the workspace or checked parent paths. Existing synthetic data,
frozen experiment snapshots and model artifacts were preserved.

## Representative sample

```sh
.venv/bin/f1ml --config configs/default.toml download --output data/ingestion-v2 \
  --as-of 2026-11-04T23:00:00Z \
  --event 2024-01 --event 2025-02 --event 2026-01
```

| Event | Discovered format | Official Q rows | Official R rows |
| --- | --- | ---: | ---: |
| 2024 Bahrain | Conventional | 20 | 20 |
| 2025 China | Sprint qualifying/Sprint weekend | 20 | 20 |
| 2026 Australia | Conventional | 22 | 22 |

Sprint Qualifying and Sprint stayed separate schedule rows and were not loaded.
The China schedule exposed one practice session; conventional schedules exposed
three. Driver counts came from source rows.

FastF1 session loads returned partial identity data but no usable timing/status
or weather. Property reads raised `DataNotLoadedError`; underlying FastF1 warnings
are retained with each attempt. The timing circuit opened after three distinct
failed events. No lap/weather observations or absolute completion times were
fabricated. A sample resume reused 15 saved job records with **zero new attempts**
and deferred four additional sample timing jobs.

## Configured scope

```sh
.venv/bin/f1ml --config configs/default.toml resume --output data/ingestion-v2
.venv/bin/f1ml --config configs/default.toml reparse --output data/ingestion-v2
.venv/bin/f1ml coverage --catalog data/ingestion-v2 --level seasons
```

| Season | Events discovered | Result sessions retained | Started Q/R sessions missing results | Future sessions |
| --- | ---: | ---: | ---: | ---: |
| 2024 | 24 | 48 | 0 | 0 |
| 2025 | 24 | 33 | 15 | 0 |
| 2026 | 23 | 10 | 28 | 20 |

The normalized snapshot contains **71 events, 355 sessions, 29 observed source
driver identities, 2,014 entry observations and 1,832 result rows**. Driver entries
from different source interfaces remain distinct observations. There are **zero**
normalized practice/qualifying laps or weather samples, with explicit empty
schemas and coverage states. The timing circuit deferred 290 further started
session loads rather than repeatedly requesting inaccessible data.

The other 43 started GP result sessions returned FastF1
`ErgastInvalidRequestError` with the server reason **`Too Many Requests`**.
All remain recorded, including event/session and URL, for resume. Source rate
limiting now stops further requests behind a persisted cooldown; focused tests
verify this for the first throttled job. The observed cooldown ends at
**2026-11-05 00:12:52 UTC (November 4, 7:12:52 p.m. EST)**. Default resume respects
that deadline. It does not override shared upstream rate limits. The separate
timing-access cooldown ends on November 5 at 23:08:14 UTC; use `--retry-failed`
only if access has changed.

Actual source data also exposed repeated qualifying positions in four result
fields (10 driver rows), and the `Lapped` status. Repeated positions are now
preserved with ambiguity flags; `Lapped` is a finish, independent of retirement.
Reviewed source team names included `Cadillac F1 Team`, now an explicit alias.
Offline reparse reused 143 retained source records, changed no retrieval times,
and made **zero** download attempts. No unmapped team entries remain in the
current collected data; new aliases remain visible as unmapped values.

Verified full-scope snapshot (later resumes can update `CURRENT.json`):

```text
data/ingestion-v2/snapshots/5f218e4b8b8549e69c3f1f6c5cc4e44d
receipt: 9451b9b6821aa7789d3ed5b7a201c4557da16fdee4c584b929e33129a12162f8
```

`CURRENT.json` identifies any later snapshot. All original source revisions and
previous snapshots remain on disk. The latest successful source data is retained
if an event refresh fails, with explicit staleness flags. The catalog is **not** a
cutoff-verified Phase 1 experiment dataset; no real-data models or final-test
metrics were produced.

## Software verification

```sh
.venv/bin/pytest -q
.venv/bin/ruff check src tests
.venv/bin/ruff format --check src tests
.venv/bin/python -m pip check
.venv/bin/python -m build --no-isolation
```

**55 tests passed**; lint, formatting, dependency consistency and wheel/source
builds passed. Tests cover duplicate keys, session separation, classified
retirements, missing/ambiguous positions, literal source preservation, unknown
identities, team eras, unfinalised practice, public load flags/pagination,
interrupted resume, retry bounds, rate-limit cooldown, scoped refresh and offline
mapping reparse. The Phase 1 cutoff/evaluation tests continue to pass.

Hosted FastF1 documentation returned HTTP 403 here. Public signatures and official
docstrings from the installed wheel were checked and hashed in
`docs/fastf1-interface-check.json`, alongside the official source/time-reference
links in `docs/ingestion.md`. Timing ingestion has offline contract coverage but
could not be validated against accessible live timing data in this environment.
