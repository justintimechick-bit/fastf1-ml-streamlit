# Phase 2 ingestion

The catalog is a resumable source-acquisition layer for 2024–2026. It preserves
every discovered weekend/session, including future sessions, unavailable data,
and Sprint identities. It does not change frozen experiment manifests or train
models. Accounts, databases and raw telemetry are not required by this layer.

## Public API contract

Verified against installed **FastF1 3.8.1**, its shipped public API docstrings and
official source documentation. Signatures/docstring hashes are recorded in
[fastf1-interface-check.json](fastf1-interface-check.json). The hosted documentation
returned HTTP 403 here, so the installed official docstrings were used to check
load flags, timing semantics and caching; the relevant official references are
[Session](https://docs.fastf1.dev/core.html),
[events](https://docs.fastf1.dev/events.html),
[cache API source](https://github.com/theOehrly/Fast-F1/blob/v3.8.1/fastf1/req.py),
[Ergast interface](https://github.com/theOehrly/Fast-F1/blob/v3.8.1/fastf1/ergast/interface.py),
and [time handling](https://github.com/theOehrly/Fast-F1/blob/main/docs/data_reference/time_explanation.rst).

| Interface | Use and observed behavior |
| --- | --- |
| `get_event_schedule(year, include_testing=False)` | Discover all three seasons; retain source session names, format and UTC starts. Exclude round zero/testing. |
| `Ergast.get_qualifying_results/get_race_results` | Official classifications redistributed by Jolpica; independent of timing access. `auto_cast=False` preserves literal flattened source fields. Check response event identity and paginate to the reported total with a configured page bound. |
| `Session.load` | GP practice/Q: `laps=True, telemetry=False, weather=True, messages=True`; Race: `laps=False, telemetry=False, weather=True, messages=False`. No car/position telemetry or race laps are requested. |
| `Session.results` | Only observed identity/team participation; never a substitute for official result labels because FastF1 can calculate fallback classifications. |
| `Session.laps/weather_data/session_status` | Read separately after loading, because FastF1 can catch failures internally and return partial data. Preserve warnings, property-access failures and source columns. |
| `Cache.enable_cache` | Persistent raw HTTP and parsed caches, version checking enabled. Separate cache namespaces for schedules and each event/generation. A refresh does not clear other events. |

All public session loads explicitly disable telemetry. A `Finalised` status
confirms completion observed at retrieval; without an absolute UTC origin,
`completed_at` remains null. Retain relative status/lap/weather times. Do not
substitute the scheduled end or claim the data was available at a historical
prediction cutoff. Practice/Q laps from an unfinalised session remain only in
its source revision and are withheld from normalized lap tables.

## Download, resume and refresh

From the repository root, after activating `.venv`:

```sh
f1ml --config configs/default.toml download --output data/ingestion-v2 \
  --as-of 2026-11-04T23:00:00Z \
  --event 2024-01 --event 2025-02 --event 2026-01
f1ml --config configs/default.toml resume --output data/ingestion-v2
f1ml coverage --catalog data/ingestion-v2 --level seasons
f1ml coverage --catalog data/ingestion-v2 --level events --format csv
f1ml coverage --catalog data/ingestion-v2 --level sessions --event 2025-02 --format json
f1ml coverage --catalog data/ingestion-v2 --level failures --format csv
f1ml --config configs/default.toml refresh-event --output data/ingestion-v2 --event 2026-01
```

Omit `--event` for the whole configured scope. The sample and full scope can use
the same catalog. Successful components are skipped on resume; failed, partial
and empty components have a persisted retry time (24 hours by default).
`resume --retry-failed` explicitly bypasses that ordinary cooldown and reopens
the timing circuit. `resume --as-of <timezone-bearing timestamp>` advances the
horizon. `--as-of` limits which scheduled sessions are requested; it is **not** a
claim that a newly downloaded revision existed at that time. Discovery failures
are retained by season and retried on resume. Schedules are frozen within a
catalog; create a new catalog to rediscover revised schedules, reusing caches.

Transient connection/timeouts, server errors and malformed JSON get bounded
exponential retries: default three attempts, waits capped at eight seconds.
Authentication/missing-data/normalization failures are not repeatedly downloaded
inside a job. Missing timing across three distinct events opens a persisted
24-hour timing circuit; results continue independently.

HTTP 429, FastF1's rate-limit exception, and the installed Ergast interface's
`Too Many Requests` error pause further acquisition behind a persisted cooldown.
Jolpica currently documents anonymous burst/sustained limits of four requests per
second and 500/hour; shared IP usage also counts. [Official rate-limit guide](https://github.com/jolpica/jolpica-f1/blob/main/docs/rate_limits.md).
FastF1 does not expose `Retry-After` through `ErgastInvalidRequestError`; this
layer uses a configurable conservative one-hour cooldown. `--retry-failed` and
event refresh cannot override that active source cooldown. Resume after it
expires; the pipeline does not sleep for an hour or hammer the remaining events.

Exit 0 means requested components completed; 1 means coverage is partial or
failures remain, with work saved; 2 means an invalid command/configuration or
validation condition. Normalization ambiguity is flagged for review rather than
converted into a fictitious order. The POSIX single-writer lock supports this
macOS/Linux workflow and is released by the OS on process exit.

## Storage and provenance

`state.json` is atomically checkpointed after each component. An interrupted run
resumes pending work; incomplete orphan revisions are not published. Each
`records/<event>/<session>/<component>/<revision>/` contains typed source Parquet,
normalized Parquet, metadata and a content-hashed receipt. Refresh retains prior
revisions; if the latest attempt fails, prior usable tables remain visible with
`stale_after_latest_attempt=True`.

Each completed run publishes an immutable `snapshots/<id>/` and atomically
updates `CURRENT.json`. Its receipt binds the component revisions, transformation
version, mapping hash, horizon and circuit state. Every table is local Parquet.
Coverage is also CSV. Verification detects mutated source/snapshot contents.

Metadata records retrieval time, FastF1 version, source/version, session/event
identity, load flags, completion evidence, available source columns, null counts,
attempts, warnings and download/parsing errors. Normalized rows retain their
source revision and retrieval/transformation versions. `source_*` columns use
consistent string representations; `source_record_json` preserves the flattened
row, while the component's `source/` Parquet retains FastF1-provided dtypes. These
are FastF1 observations, not verbatim HTTP bodies or publication-time archives.

## Tables and identity

| Table | Key and meaning |
| --- | --- |
| `events` | `event_id`: season/round from the discovered GP schedule; all events stay visible. |
| `sessions` | `session_id`: event plus FP1/FP2/FP3/SQ/S/Q/R; schedule, completion and availability remain distinct. |
| `drivers` | Stable source `driver_id`; observed names only, without a team column. |
| `driver_entries` | Session, entry source, driver number; observed participation/result membership, with event-specific team identity and history ID. Different interfaces remain separate observations. |
| `results` | Session, stable driver ID; official position, classification code, retirement/classification flags, podium target, missing/ambiguity flags and source fields. |
| `practice_laps`, `qualifying_laps` | Session, driver number, lap key; all completed-session timing rows and source quality/deletion flags. No elimination counts are assumed. Missing lap numbers get explicitly marked source-row keys, not invented lap numbers. |
| `session_weather` | Session, sample key; relative timestamp, measured values, null indicators and source fields. |
| `session_status` | Session, source status index; relative state transitions. |
| `team_aliases` | Season/source ID/name mapped explicitly to team identity and a separate performance-history era. |

No driver list, field size or fixed team roster is embedded. Driver numbers are
session-local observation keys, not cross-event identities. Missing stable IDs
remain null/unresolved; later race numbers are not used to identify reserve
practice drivers. Ingestion entries must not become cutoff-time prediction
rosters without a separate dated entry-list review.

`configs/team_mapping.json` defines reviewed aliases and **separate history IDs**.
RB/Racing Bulls aliases can identify the same organization while using different
performance eras. Sauber and Audi are not automatically pooled; new teams keep
their own histories. Unknown ID/name/year combinations stay explicitly unmapped
with isolated IDs. No fuzzy matching or implicit predecessor-performance pooling
is applied. Future feature code should choose `team_history_id` deliberately.

After reviewing an alias or changing a transformation, apply it without network:

```sh
f1ml --config configs/default.toml reparse --output data/ingestion-v2
```

Reparse checkpoints its work, preserves original retrieval timestamps, and
publishes new normalized revisions without changing original source records or
frozen snapshots. Download resume refuses a half-finished reparse or an
unapplied mapping change.

## Results and coverage

Positive integral source positions are preserved. Missing, zero or invalid
positions stay null. Numeric `positionText` and `Finished`/`Lapped` establish
classification; retirement reason is independent. A numerically classified
retirement can retain its rank and podium target. DNS, withdrawn, DSQ, excluded,
DNQ and non-classified entries remain distinguishable. Unknown outcomes remain
unknown. Repeated official positions are retained with `position_ambiguous=True`
and require review before full-order evaluation; no tail order is invented.

Coverage includes every discovered season/event/session: driver and result
counts, lap/weather counts, missing and ambiguous targets, unusual outcomes,
unresolved identities, unmapped teams, source columns, warnings and failures.
Practice counts use the actual discovered format (for example one versus three
scheduled sessions); unavailable practice is visible. Future sessions are marked
`not_started` and excluded from started-session missing-target totals. Empty
normalized tables retain canonical columns and never contain synthetic rows.

See [live validation](ingestion-validation.md). The Phase 1 offline demonstration
remains explicitly synthetic and is not an access-failure fallback.
