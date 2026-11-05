# Phase 9 real-data experiment — November 5, 2026

**Blocked before training. Phase 9 is not complete.** The separate experiment is
`real-20261105-core-v2`. No real models, forecasts, development scores or final
scores were produced. No real final manifest/model lock was created and no
evaluation ledger was claimed. Planned-method records have **zero evaluated
events and blank metric values**, not zero loss.

Select **Real experiment** in the interface for this review. **Synthetic demo**
continues to load the existing Phase 3–7 models and sealed forecasts. The original
FastF1 catalog remains separately selectable. Interface reruns only read saved
artifacts; they do not download, fit, calibrate or score.

## Coverage actually observed

Source horizon: **November 5, 2026, 09:00 EST / 14:00 UTC**. This is an acquisition
horizon, not a training cutoff. The schedule contains 71 events and 355 sessions,
including Sprint sessions outside modeling scope and 20 future sessions. A past
schedule timestamp does not establish actual completion.

| Season | Scheduled events | Official Q/R result sessions | Practice sessions with laps | Q sessions with laps | Weather sessions |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2024 | 24 | 48 | 0 | 0 | 0 |
| 2025 | 24 | 48 | 0 | 0 | 0 |
| 2026 | 23 | 16 | 0 | 0 | 0 |

The snapshot holds **2,263 official result rows**. The isolated resume recovered
21 sessions beyond the original catalog's 91: 15 in 2025 and six in 2026.
It attempted 22 jobs, reused 100 and deferred 311. The last attempt returned
Jolpica's `Too Many Requests` for **2026 round 5 qualifying**, setting a results
cooldown until **10:13:39 EST / 15:13:39 UTC**. The original timing cooldown
remains until **18:08:14 EST / 23:08:14 UTC**. Neither cooldown was overridden.
Continue with `resume` after the relevant deadline.

Portable reports:

- [Season coverage](real-data-seasons.csv).
- [Feature-group coverage by season](real-data-feature-coverage.csv).
- [Every event and its exclusion reasons](real-data-exclusions.csv).

Complete local artifacts: `data/real-20261105-core-v2/`. They include session and
event feature coverage, classified source failures, component diagnostics,
target/identity/status audits, strategy eligibility, planned evaluation records,
protocol, full search configuration, environment versions, copied implementation,
immutable source snapshot and **451 component revisions** with raw frames and
receipts. The source snapshot ID is
`cfa596bf8ed4dcd35d670f78215567a1949acda2583d809fe7812b9ed9a6b57a`;
the experiment receipt ID is
`7a80e53e48247527c965a9220df278f7342cc91813982e1d3c8c8b2608ed0c9d`.

`data/real-20261105-core-v1/` retains the preliminary audit revision. It has no
models/evaluated predictions and is not selected by the interface. Neither audit
is an evaluation lock. New audits use new output directories.

## Access diagnosis and ingestion validation

At 09:14 EST on November 5, a new unauthenticated request for official **2024
Bahrain FP1 SessionInfo.json** returned **HTTP 401, zero bytes**. Its documented
mirror path returned **HTTP 404**. Exact URLs/timestamps are saved in
`data/phase9-endpoint-check.json` and the experiment's `endpoint_checks.json`.
These checks are diagnostic requests, not feature observations. The FastF1
maintainer confirms that the archive now requires F1 TV Access or higher;
Jolpica results remain separately accessible.
[Upstream maintainer discussion](https://github.com/theOehrly/Fast-F1/issues/978).

The current snapshot's 45 failure rows comprise 22 rate-limit errors and 23
`DataNotLoadedError` diagnostics. The latter alone does not prove that a session
never existed. The sampled endpoint check establishes an access problem,
consistent with the upstream restriction. Unrequested/circuit-deferred sessions
are distinguished from failed requests in `dataset/coverage/sessions.csv`.
Missing stored timing is not proof that the underlying observations do not exist.

Four historical normalization failures rejected duplicate official positions:
**2024-08, 2024-15, 2024-17 and 2025-07 qualifying**. The existing reparsing code
already retains these source positions with ambiguity flags. Phase 9 preserves
the earlier failures and normalized rows; no unique order is invented. These
events are excluded from the strict complete-order cohort.

Offline replay through the current normalizer checked **112 sessions / 2,263
rows with zero mismatches** for driver numbers, canonical team/team-era identity,
official position, classification code, retirement/classification flags, podium,
ambiguity and response completeness. Separate sealed evidence is in
`data/real-20261105-source-validation/`, bound to the experiment receipt.
This validates source-to-table transformations, not independent FIA-document
verification of Jolpica classifications.

Q/R driver sets differ for **2025-09, 2025-21 and 2026-01**. For example,
2026-01 has 19 qualifying rows and 22 race rows. Missing entrants are not inferred
or inner-joined away. Independently evidenced task rosters may explain a real
substitution or absent qualifying entry; these remain review exclusions without
that evidence. Complete label pairs before timing/roster checks number **21 in
2024, 23 in 2025 and six in 2026**. Those counts do not establish roster agreement.

## What works and what blocks modeling

Already working with real data: dynamic schedules, distinct GP/Sprint sessions,
cached requests, resumable component jobs, hashed immutable revisions, independent
official Q/R results, provider driver IDs, explicit team aliases/eras and
administrative classification/status preservation. Timing-derived fallback
results never become official targets.

Catalog review overlays, cutoff-aware features, shifted history, chronological
comparisons and final locking already exist. Their software tests pass, but
they have not completed a real-data modeling run.

The real source has **zero verified absolute completion timestamps**. There are
**no independently evidenced real cutoff rosters, explicit real cutoff CSVs or
availability-review CSVs** in the repository. Therefore **zero events meet the
modeling contract**, including for core features. Eventual labels do not establish
completion, first publication or entrants known before qualifying. Scheduled end
times, retrieval times and the race-result grid cannot fill those gaps.

Missing optional timing alone does **not** block a reviewed core experiment.
Driver/team history and post-qualifying official classification remain the planned
fallback. They still need evidenced completion, availability and roster/cutoff
inputs. Authenticated timing access or a compatible provenanced cache would
address timing access; independent review evidence is also needed. This run did
not sign into an account or change FastF1 dependencies.

Missing pace affects **both** sources: the synthetic fixture has participation
summaries and zero observed pace gaps across its 46 development rows per task;
the real snapshot has zero laps/weather samples. Existing pace code matches a
single session, Q1/Q2/Q3 segment, compound, time, rainfall and temperature cell
against a field reference. Formulas and segmentation have software tests, but
could not be validated against downloaded real laps in this run.

The tested `f1ml_real.preprocessing.ObservedCorePreprocessor` removes entirely
missing/constant training columns and uninformative missing flags. Median
imputation, scaling and unknown-safe encoding fit within the training fold;
later observations cannot promote excluded features. This preparatory component
has **not been integrated into or used by a fitted real comparison**. Legacy
comparators retain their original preprocessing; running them directly would
not finish Phase 9.

## Planned evaluation and prior exposure

`configs/real-20261105.toml` saves the real ID, evidence paths, protocol and
reference to the full search grid. Eligibility uses source labels, identity/team
resolution and independent timing/roster review, never prediction performance.
Optional timing does not select the core comparison cohort.

After sufficient eligible events exist, freeze the latest **four eligible
completed 2026 weekends** before selection. Require at least **five earlier 2026
development events**, so the two-event 2026-only training minimum can leave
three common development comparisons: at least nine eligible 2026 events overall.
Neither minimum shrinks automatically. **No final membership is frozen now.**

Planned families are Ridge, random forest and histogram gradient boosting for
qualifying/race, and logistic regression, random forest and histogram gradient
boosting for podiums. Compare 2026-only, pooled and weighted history on identical
eligible events with fixed recent-form/practice/qualifying reference rules as
appropriate. Tune features, parameters and weights only on chronological
development data; use later development OOF events for calibration/thresholds.
Report insufficient 2026 history. No strategy currently has an eligible cell.
Even the six source-only complete 2026 label pairs fall short of the nine-event
minimum; completion of the pending downloads must precede cohort review.

Real outcomes were accessible during ingestion/source QA, including this audit.
Saved trained model metadata identifies synthetic experiments; no prior real
model comparison was found. Unlogged consultation cannot be ruled out. Do not
call a retrospective block of these outcomes demonstrably untouched. Disclose
exposure and use fresh prospective events for an untouched confirmation claim.

After review and selection, freeze real bundles/configuration with distinct
source/manifest IDs and a separate persistent evaluation ledger. Seal qualifying
forecasts before target Q enters race features, then seal race/podium forecasts
before scoring once. Keep pre-block history and parameters fixed. The synthetic
ledger remains consumed and unchanged.

## Reproduction and remaining work

Use the pinned environment from the project root. The saved output paths already
exist: fork/audit/validation refuse to overwrite them. Use a new path when
repeating, and update configuration paths for a new acquisition catalog.

```sh
.venv/bin/python -m f1ml_real fork-catalog \
  --source data/ingestion-v2 --output data/real-20261105-catalog
.venv/bin/f1ml --config configs/default.toml resume \
  --output data/real-20261105-catalog --as-of 2026-11-05T14:00:00Z

# Exit 1: audit saved successfully but dataset remains blocked.
.venv/bin/python -m f1ml_real audit --config configs/real-20261105.toml \
  --output data/real-20261105-core-replay
.venv/bin/python -m f1ml_real validate-source \
  --experiment data/real-20261105-core-v2 \
  --output data/real-source-validation-replay

# Expected exit 2; this is a preflight check, not a training runner.
.venv/bin/python -m f1ml_real check-ready \
  --experiment data/real-20261105-core-v2

.venv/bin/pytest -q
.venv/bin/ruff check src tests
.venv/bin/ruff format --check src tests
.venv/bin/streamlit run src/f1ml_app/streamlit_app.py
```

For a later horizon, use a new ID/configuration/output path; resume acquisition
without changing a frozen source snapshot. Reviews require:

- Availability: `session_id,completed_at,available_at,evidence`.
- Entrants: `event_id,task,driver_id,team_id,known_at,source`.
- Cutoffs: `event_id,task,cutoff,mode`.

Evidence must establish the claimed timestamp; do not copy eventual target
rosters or scheduled end times into review rows.

Actual training commands, saved real models, baseline-versus-model performance,
calibration selection and final evaluation **remain undelivered** because the
dataset gate fails. There is no successful real training command to reproduce.
After data review, remaining implementation includes connecting observed-only
preprocessing to chronological comparison and extending locking for the real
bundles while preserving the synthetic lock.

The original `src/f1ml` fingerprint remains
`3dacc77e40fc73192127b0ad3b0416eb04e92306d8d773d165545acdff67901f`.
All **1,037** preserved source/synthetic-artifact/ledger files match pre-work
SHA-256 values. New code lives in `src/f1ml_real`; interface changes stay in the
separate `src/f1ml_app` package.

## Verification completed

- Full suite: **191 passed** (`data/phase9-tests-final.log`).
- Real-artifact smoke check traversed overview, qualifying, race, comparison and
  methodology with downloads and estimator fitting patched to raise. All views
  rendered successfully and real forecasts stayed unavailable.
- Source normalization replay: **112 sessions, 2,263 rows, zero mismatches**.
- `ruff check src tests` and `ruff format --check src tests` passed.
- Wheel and source distribution builds passed.
- Original synthetic lock/final receipts verified; **1,037 unchanged files** and
  the unchanged locked code fingerprint were checked again after the full suite.
- Real audit and source-replay receipts verified; readiness exits **2** with
  explicit missing-evidence errors. No real training or final scoring occurred.
