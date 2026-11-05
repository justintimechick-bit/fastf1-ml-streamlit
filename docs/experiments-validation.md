# Phase 6 validation — November 4, 2026 (America/New_York)

The subsequent [resume diagnosis](experiments-resume.md) established HTTP 401
from the official archive and HTTP 404 from the documented mirror for one pilot
session. Upstream confirms the archive now requires authentication. The original
failure records below remain unchanged.

No genuine optional feature benefit is claimed. The live pilot failed; synthetic
fixtures validate software behavior only. No telemetry observations were invented
or substituted into the live source, and no added group was promoted.

## Live selective capture

Prepared `data/phase6-live-requests.csv` from the existing immutable real schedule
for FP1 at `2024-01`, Sprint-format `2025-02`, and `2026-01`. Qualifying cutoffs
are one second before each source schedule's GP qualifying start. Only these
three practice sessions were requested; no Sprint/target/final-test session was
loaded. The existing manifest is an event guard for this schema audit, not a
claim that the real source is compatible with its synthetic training data.
Captured revisions explicitly mark the source mismatch as schema-audit-only.

Exact command, initially attempted in the network-restricted sandbox:

```sh
.venv/bin/f1ml telemetry-download \
  --dataset data/ingestion-v2/snapshots/999ec6cdd75547f8acc7bc79a67a833d \
  --manifest data/phase3-final/manifest.json \
  --requests data/phase6-live-requests.csv --output data/phase6-live-telemetry
```

The sandbox failed DNS resolution while loading schedules. Retried the same
three-session pilot with authorized network access and `--retry-failed`, sharing
the existing persistent schedule cache. Each session then made at most two
capture attempts and failed with `DataNotLoadedError`: “The data you are trying
to access has not been loaded yet. See Session.load”. FastF1 diagnostics include
`SessionNotAvailableError: No data for this session`, failed session info, driver,
status/timing/weather/message loads, and inability to establish `t0_date`.
No unsupported explanation such as an authentication requirement is inferred.

| Session | Result | Network-run attempts | Capture wall seconds |
| --- | --- | ---: | ---: |
| 2024-01 FP1 | failed | 2 | 12.57 |
| 2025-02 FP1 | failed | 2 | 12.24 |
| 2026-01 FP1 | failed | 2 | 12.55 |

The command returned status 1. `data/phase6-live-telemetry/index.json` and immutable
capture receipts retain errors, warnings, timing and empty observed-schema
reports. No valid live telemetry samples were returned, so no 2026 car-channel
schema is asserted observed. Repeating the command observes the failure cooldown;
explicit retries remain possible. The [interface audit](telemetry-interface-check.json)
records installed FastF1 3.8.1 signatures, parser hash, units, DRS defaults and
documentation access. Hosted docs returned HTTP 403; official repository source
and installed documentation were inspected instead. Downloads use public APIs.

## Verification

```sh
.venv/bin/ruff check src tests
.venv/bin/ruff format --check src tests
.venv/bin/pytest -q tests/test_experiments.py
.venv/bin/pytest -q
.venv/bin/python -m build --no-isolation
```

**22 new tests, 145 total tests passed.** Lint/formatting passed; wheel and source
distribution built. Checks include unequal-interval integration (0.1 versus
0.9 seconds), units, invalid throttle sentinel, missing channels/long gaps,
duplicate timestamps, no interpolated samples, no Q1/Q3 consistency pooling,
time-weighted rain, absolute weather timestamps and post-session/cutoff exclusion,
cache reuse/cooldown/retry, and source/implementation mismatch rejection.

A labelled synthetic channel fixture exercises native public Lap/Laps-shaped
loader interfaces through persistent capture, comparable practice/sector features,
teammate deltas, observed weather, and unavailable publication revisions. It verifies
that the loader requests `interpolate_edges=False`. Separate tests cover all nine
task/family combinations, full fields with missing optional data, identical paired
events, strict allowlists, fixed core matrix/prediction equivalence, and rejection
of retention based on synthetic data or a harmful incremental feature group.

One nonfatal joblib physical-core warning appeared in pytest. Standalone Parquet
reads also logged sandbox CPU-cache `sysctl` probe messages; checks succeeded.

## Paired missing-data experiment

Exact commands using the existing synthetic core snapshot, without synthetic raw
telemetry or fabricated live weather:

```sh
.venv/bin/f1ml build-experiments --dataset data/phase3-final/dataset \
  --manifest data/phase3-final/manifest.json --features data/phase3-final/features \
  --telemetry data/phase6-synthetic-telemetry --output data/phase6-verified-features
.venv/bin/f1ml --config configs/default.toml experiment-compare \
  --manifest data/phase3-final/manifest.json --features data/phase3-final/features \
  --optional data/phase6-verified-features --output data/phase6-verified-experiments
```

| Output | Receipt ID |
| --- | --- |
| Optional supplement | `1b3d5565b0094572e44779ee6135c68312f95bdf9917810389a41e217f52401d` |
| Paired comparison | `b3e8dd73098168f72a98adb3ede011cd0e1ccea1584b1cd2db25fa10bdfa88ae` |

Saved 162 event-metric rows/model bundles: three tasks × three families × six
groups (core plus five stages) × three identical 2026 development events per
cell. Saved 135 predecessor/core paired differences, feature-level missingness,
runtime/coverage, predictions, dictionaries and model/source/cutoff lineage.
Both task supplements retain all 46 development driver rows. All detailed-practice,
telemetry and timestamp-verified weather coverage is zero in this minimal fixture.
Race/podium session groups do retain the actual eligible qualifying inputs.

All 45 retention decisions are `do_not_promote`. There is no live optional-data
evidence and no observed additional signal to justify retaining telemetry/weather.
Synthetic metric changes, including forest changes caused by altered feature
sets/random feature ordering, are not treated as evidence of a real signal.
Local runtimes were measured while other verification work ran and should be
repeated for a genuine cost comparison. See the saved `report.md`,
`event_metrics.parquet`, `paired_differences.parquet`, `missingness.parquet`,
supplement `coverage.parquet`, and `retention.json`.

## Preserved core and final test

Verified unchanged receipts:

- Phase 3 features: `ccc37825281ab0102fb44f15dc516e04211ad3896fec45bfbf78b71699c3a64d`.
- Phase 4 models: `911ef333cd8d600a851515974d9717a63813aaf980b5870c944c98e2a8bc8839`.
- Phase 5 models: `891b065891bb56ae9dc6b116ac4089a5d50f999ab38c6c5d9928bd9dc6d6093a`.
- Manifest: `3cc7b3bf005e808e360595275f3e24c35bbd8a26c962bafdef8e6ee766f968dd`.

The sealed IDs remain `2026-04` through `2026-07`. No core feature implementation
file was modified. Model integration is an explicit opt-in matrix specification;
absent that specification, existing core training/prediction contracts remain.
All prior tests pass. No core artifact or manifest was overwritten.

Remaining limits: unavailable live timing/telemetry, real cutoff/identity reviews,
unverified absolute weather timing in older sources, unknown fuel/run plans,
small development cohorts, and no verified archived pre-cutoff weather forecasts.
No race-weather forecasts are included. Use fresh output directories for reruns;
a genuine observed-source experiment and frozen retention protocol are needed
before considering model promotion or a separately authorized final evaluation.
