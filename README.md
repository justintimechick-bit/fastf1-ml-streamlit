# F1 ML foundation

A local Python package for full Grand Prix qualifying order, race classification,
and driver podium probabilities. Only **2024, 2025 and 2026** are supported. A local,
read-only Streamlit interface is included. Phase 9 adds a separate real-data audit
and interface source; **real training is blocked**, not completed. The app can be
packaged for Streamlit Community Cloud; training and downloads remain offline.

The workspace initially had no `AGENTS.md`. Phase 1 supplies cutoff/evaluation
contracts, fixed baselines and an offline example. Phase 2 adds resumable source
acquisition and coverage reports. Phase 3 adds reproducible, cutoff-aware feature
datasets and a generated feature dictionary. Phase 4 adds chronological qualifying
model comparison and full-field prediction. Phase 5 adds race classification and
independent podium classifiers with chronological calibration and threshold
selection. Phases 6–7 add optional-data experiments and one locked final evaluation;
Phase 8 adds the [interface](#interface-phase-8). No real 2026 performance claim is made.

## Setup and verification

Python **3.11+** is required (verified with 3.13). From this directory:

```sh
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock.txt
python -m pip install -r requirements-app.lock.txt   # interface only (Streamlit)
python -m pip install --no-build-isolation --no-deps -e '.[dev,app]'
pytest -q
ruff check src tests
ruff format --check src tests
python -m build --no-isolation
```

`pyproject.toml` holds direct dependencies. `requirements.lock.txt` pins the tested
environment, including development tools; it was resolved on macOS ARM/Python
3.13, has no wheel hashes, and is not a universal multi-platform lock. On other
platforms, use `python -m pip install -e '.[dev]'` to resolve compatible versions.
FastF1 is constrained to 3.8.x to avoid the automatic account sign-in introduced
in 3.9. This preserves the tested environment; it does not restore anonymous
archive access. On the November 4 resume, the selected official timing archive
returned HTTP 401 and the documented mirror returned 404. FastF1's maintainer
[confirms the archive authentication requirement](https://github.com/theOehrly/Fast-F1/issues/978).
New timing/telemetry downloads therefore require authenticated access or an
already populated compatible local cache. Account support is outside the current
account-free implementation. Core/offline workflows and separately available
official results remain usable. See the [resume diagnosis](docs/experiments-resume.md).

## Run the offline example

```sh
f1ml demo --output data/features-demo-v3
cat data/features-demo-v3/artifacts/summary.json
```

Use a new output directory for each run. All data is visibly **synthetic**, with
variable field sizes, a Sprint weekend, seven completed 2026 fixtures and one
future fixture. Three expanding 2026 validation folds run all three tasks. The
latest four completed 2026 fixtures remain sealed. Synthetic metrics demonstrate
software behavior only.

The demo saves a source snapshot, frozen manifest, cutoff/entrant CSVs, separate
feature and label Parquet files, source audits, model artifacts, predictions and
weekend-level metrics. Every snapshot has a content-hashed `receipt.json`.

## Real-data acquisition (Phase 2)

Read [ingestion commands and table contracts](docs/ingestion.md) and the
[methodology](docs/methodology.md) before using real data. From the project root:

```sh
# First validate a normal weekend, a Sprint weekend and available 2026 data.
f1ml --config configs/default.toml download --output data/ingestion-v2 \
  --as-of 2026-11-04T23:00:00Z \
  --event 2024-01 --event 2025-02 --event 2026-01

# Then process the configured scope, reusing successful components.
f1ml --config configs/default.toml resume --output data/ingestion-v2
f1ml coverage --catalog data/ingestion-v2 --level seasons
f1ml coverage --catalog data/ingestion-v2 --level sessions --event 2026-01
f1ml coverage --catalog data/ingestion-v2 --level failures --format csv

# Fetch a new revision of just one event, respecting upstream cooldowns.
f1ml --config configs/default.toml refresh-event --output data/ingestion-v2 \
  --event 2026-01

# Reapply reviewed mappings/normalization to retained source frames offline.
f1ml --config configs/default.toml reparse --output data/ingestion-v2
```

`download` can also process the whole scope directly. `resume --as-of <UTC time>`
advances the download horizon. `--retry-failed` explicitly bypasses ordinary
failure cooldown and reopens the timing circuit; it cannot override an active
upstream rate-limit cooldown. After that cooldown expires, `resume` retries the
rate-limited jobs automatically. Commands return **1** for partial availability
and **2** for configuration/validation errors; saved work remains resumable.

The current snapshot is identified by `data/ingestion-v2/CURRENT.json`. Its
`tables/` directory contains events, sessions, drivers, driver entries, results,
practice laps, qualifying laps, weather, session status and team aliases as
Parquet. `coverage/` contains season/event/session/failure CSV and Parquet reports.
Source frames and provenance live in immutable, hashed component revisions.

**Observed on November 4, 2026:** 71 events were discovered. The representative
sample retained Q/R fields of 20, 20 and 22 drivers, respectively. The configured
scope retained 91 Q/R result sessions (48 in 2024, 33 in 2025 and 10 in 2026).
Another 43 started Q/R sessions hit Jolpica's `Too Many Requests` response and
remain in coverage as failures for a later resume. Practice/qualifying lap and
weather access failed in the live sample; the timing circuit deferred the
remaining timing requests. No fabricated observations were substituted.
See the [validation record](docs/ingestion-validation.md) for exact results.

Only GP practice and qualifying lap timing is requested. `telemetry=False` is
explicit on every session load; race laps are not requested. Weather is stored
as a source observation, including race weather, and is **never automatically a
model input**. Sprint sessions are listed separately and not downloaded.

Driver entries describe observed participation/result membership, **not** entry
lists known at a prediction cutoff. Team aliases and distinct performance eras
are explicit in `configs/team_mapping.json`; unknown aliases remain visible.
Repeated source positions are preserved and flagged, never arbitrarily ordered.

The Phase 2 catalog is a source layer, separate from Phase 1's strict immutable
experiment `Dataset`. It does not change frozen manifests or open final tests.
Absolute completion times are often unavailable without telemetry; they remain
missing. Cutoff-verified export and prospective publication snapshots still
need review before real-data feature/model work. The legacy `ingest` export
fails closed when absolute timing cannot be established. The offline demo and
existing `freeze`, `build-features`, `backtest`, and `predict` commands remain.

## Feature generation (Phase 3)

[Feature methodology and commands](docs/features.md) cover normalized ranks,
shifted 3/5-weekend histories, podium/retirement rates, teammate/team form,
reviewed circuit priors and comparable practice/Q pace. See the generated
[feature dictionary](docs/feature_dictionary.md) for every formula and cutoff.

```sh
f1ml build-features --dataset data/features-demo-v3/dataset \
  --manifest data/features-demo-v3/manifest.json \
  --entrants data/features-demo-v3/entrants.csv \
  --cutoffs data/features-demo-v3/cutoffs.csv --output data/rebuilt-features-v3
f1ml feature-dictionary --output data/dictionary-v3
```

Outputs retain the complete reviewed entrant field even when optional practice,
weather or qualifying pace is missing. Race features alone include actual GP
qualifying classification and separate Q1/Q2/Q3 pace. Statistical preprocessing
is fitted only inside each training fold; unseen categories are supported.

Immutable Phase 2 snapshots are also supported by `freeze`, `build-features` and
`predict`, with `--availability` and `--event-context` review CSVs where needed.
Missing verified completion timestamps remain a real-data limitation; no session
end is invented. Team form uses current-season history, with explicit older-era
priors. Final-test events remain sealed. Existing legacy artifacts are preserved;
v3 matrices require models trained with the v3 schema and implementation.

## Qualifying-order prediction (Phase 4)

Compare Ridge, random forest and histogram gradient boosting against recent-form
and clean-practice ranking rules. Each regressor uses earlier 2026-only, equal
2024–2026, or recency/season-weighted training. Nested chronological development
folds select parameters and weights; primary averages share the same eligible
events. The frozen final block stays sealed.

```sh
f1ml --config configs/default.toml qualifying-compare \
  --manifest data/features-demo-v3/manifest.json \
  --features data/features-demo-v3/features --output data/qualifying-models
cat data/qualifying-models/report.md
```

Outputs include per-event metrics, common-event averages, eligibility reasons,
raw ranking scores, positions 1 through the actual field size, fold model bundles
and a selected learned-model refit. Exact score ties sort by stable driver ID.
Scores are neither probabilities nor lap-time forecasts. The tiny synthetic demo
has only one common event with default minima and establishes no real accuracy.

See [qualifying methodology and runnable prediction commands](docs/qualifying.md)
for `predict-qualifying`, baselines/fallbacks, weight formulas, metrics, cutoff
guards and artifact contracts. The [validation record](docs/qualifying-validation.md)
records the completed checks. Existing `backtest` and `predict` remain available.

## Race classification and podium prediction (Phase 5)

Predict after available GP qualifying using the Phase 3 race-cutoff inputs.
Ridge, random forest and gradient boosting predict official classification;
logistic regression, random forest and gradient boosting predict independent
podium probabilities. All three historical-data strategies use chronological
development folds. Later inner folds fit calibration and select thresholds;
the final test remains sealed. Qualifying classification and independently
evidenced grid updates stay separate. No target-race observation enters inputs.

```sh
f1ml --config configs/default.toml race-compare \
  --manifest data/features-demo-v3/manifest.json \
  --features data/features-demo-v3/features --output data/race-models
cat data/race-models/report.md
```

Outputs include event-level order/probability metrics, raw versus delivered
calibration diagnostics, transparent target exclusions/status, complete-field
orders and traced model bundles. Podium probabilities are never normalized to
sum to three. `predict-race` reports both the race-order top three and the
classifier's three highest probabilities, including disagreements.

See [race/podium methodology and prediction commands](docs/race-podium.md) and the
[validation record](docs/race-validation.md). Missing historical publication
snapshots and real-data availability/roster reviews remain limitations.

## Controlled optional-data experiments (Phase 6)

Core models remain available. A separate path selectively caches practice/GP
qualifying telemetry and builds observed pace, telemetry and weather supplements.
Five cumulative groups use identical complete development events, with paired
loss, missingness, coverage and runtime reports. No final test is opened.

```sh
f1ml build-experiments --dataset data/features-demo-v3/dataset \
  --manifest data/features-demo-v3/manifest.json --features data/features-demo-v3/features \
  --telemetry data/optional-telemetry --output data/optional-features
f1ml experiment-compare --manifest data/features-demo-v3/manifest.json \
  --features data/features-demo-v3/features --optional data/optional-features \
  --output data/feature-experiments
```

This example exercises missing optional data, not fabricated telemetry. See
[selective downloads, formulas and retention rules](docs/experiments.md) and the
[live pilot/verification record](docs/experiments-validation.md). No DRS equivalence,
inferred energy/fuel/aero channels or unverified weather forecasts are included.
Live failures remain visible; synthetic evidence cannot promote feature groups.

## Locked final evaluation (Phase 7)

The authorized finalization path audits development lineage, freezes all three
model bundles/configurations and claims a single-use ledger before opening the
reserved block. Qualifying forecasts are sealed first, then race/podium forecasts,
before scoring. History and fitted models remain fixed to pre-block weekends.

```sh
f1ml synthetic-final-inputs --dataset data/phase3-final/dataset \
  --manifest data/phase3-final/manifest.json --output data/phase7-inputs
f1ml final-lock --dataset data/phase3-final/dataset \
  --manifest data/phase3-final/manifest.json --features data/phase3-final/features \
  --qualifying data/phase4-final --race data/phase5-final \
  --experiments data/phase6-verified-experiments \
  --entrants data/phase7-inputs/entrants.csv --cutoffs data/phase7-inputs/cutoffs.csv \
  --output data/phase7-lock
f1ml final-evaluate --lock data/phase7-lock --dataset data/phase3-final/dataset \
  --output data/phase7-final
```

The existing manifest is **synthetic**. Results validate software/protocol only,
not real 2026 accuracy. Outputs include baseline comparisons, event/aggregate
metrics, predicted-versus-actual tables, calibration plots, subgroup errors,
coverage and event sensitivity summaries. The ledger rejects another primary
run. Reuse/model changes are exploratory and need a fresh future holdout. Read
the [protocol](docs/final-evaluation.md) and [results](docs/final-evaluation-validation.md).

## Interface (Phase 8)

A local Streamlit app for classroom demonstration. It loads the prepared Phase 3 dataset
and features, saved Phase 4/5 model bundles and the Phase 7 lock/evaluation, and the
real catalog snapshot. Nothing is downloaded, fitted, recalibrated or re-scored on page
reruns, and the final-test block is shown exactly as sealed. From the project root, with
the environment from [Setup](#setup-and-verification) active:

```sh
streamlit run src/f1ml_app/streamlit_app.py
# equivalent shortcut installed by the editable install above:
f1ml-app
# different port:
f1ml-app --server.port 8600
```

Then open <http://localhost:8501> (`.streamlit/config.toml` runs headless, so no browser
opens automatically). Views: **Event overview**, **Qualifying predictions**, **Race
predictions**, **Model comparison** and **Data quality & methodology**. Pick the data
source, season, Grand Prix and replay moment in the sidebar. The URL keeps the selection,
e.g. <http://localhost:8501/race?source=experiment&event=2026-04&moment=post_qualifying>.

* **Synthetic demo** (default) is the frozen experiment above. Its data is
  **synthetic** and every page says so. Final-test weekends (2026-04…07) show the sealed
  Phase 7 forecasts. Development validation weekends (2026-01…03) replay the outer-fold
  model saved for that weekend through `predict_checked`, and the replay matches the saved
  predictions exactly. Training-only and future weekends say that no eligible model or
  field exists.
* **FastF1 catalog** shows real 2024–2026 session coverage and retrieved classifications.
  It has no forecasts because no reviewed real-data experiment exists.
* **Real experiment** shows the separate Phase 9 review and its pinned snapshot.
  It is blocked before training and shows no real forecasts or measured model scores.
* **Before qualifying** shows the post-qualifying race forecast as unavailable. Official
  outcomes stay hidden until the replay moment reaches their session.
* Each forecast exports to CSV with model ID, training range, cutoff and per-driver data
  coverage (no outcome columns).

The interface lives in `src/f1ml_app` rather than `src/f1ml/app` so the Phase 7 lock's
fingerprint of `src/f1ml` still matches current code. Artifact paths can be overridden
in `configs/app.toml` (`[paths]`). Integration checks: `pytest -q tests/test_app.py`
(skipped when `data/` is absent). See [interface details and a classroom
walkthrough](docs/interface.md).

## Streamlit Community Cloud

Hosted app: [Pit Wall F1 dashboard](https://fastf1-ml-pit-wall.streamlit.app/).
Deployment source: [fastf1-ml-streamlit](https://github.com/justintimechick-bit/fastf1-ml-streamlit).
Only the prepared deployment repository was made public, with the user's approval.

The root entrypoint opens the dedicated **Real-data experiment** dashboard:

```sh
.venv/bin/streamlit run streamlit_app.py
```

It includes season/event coverage filters, training blockers, evaluation status
and report/CSV downloads. The synthetic demo remains separately selectable in
the weekend views. No real forecasts are substituted with demo predictions.

For hosting, create an isolated deployment copy that includes the saved artifacts
(the workspace's `data/` is otherwise ignored by Git):

```sh
.venv/bin/python -m f1ml_app.cloud --output deploy/streamlit
```

Deploy that copy from GitHub using `streamlit_app.py` on branch `main`, with
**Python 3.13** in Advanced settings. See the [cloud deployment instructions](docs/streamlit-cloud.md).
Preparing or publishing the repository does not itself create a live Cloud app.

## Real-data experiment (Phase 9 · blocked before training)

The separate experiment **`real-20261105-core-v2`** preserves the synthetic models,
sealed forecasts, consumed ledger and locked implementation unchanged. Its
isolated resumable catalog now retains **112 official Q/R result sessions**:
48 from 2024, 48 from 2025 and 16 from 2026. All **2,263 result rows** match an
offline replay of their retained raw responses. Practice laps, qualifying laps
and weather remain unavailable; missing pace affects real data as well as the demo.

The timing endpoint still returns **HTTP 401** and its documented mirror **404**.
The results resume recovered 21 sessions before hitting a new rate-limit cooldown.
There are **zero cutoff-eligible modeling events**: completion timestamps,
independently evidenced real cutoff rosters and cutoff reviews are absent. No real
model was fitted, no final manifest/lock created and no real ledger claimed.
Model/baseline records explicitly show **0 development and 0 final-test events**
with blank metrics. This is an ingestion/review artifact, not a finished experiment.

Select **Real experiment** in the interface to inspect provenance, coverage,
exclusions and evaluation status. **Synthetic demo** remains separately labeled.
Real forecast pages show unavailable; no demo forecast substitutes for missing data.
Artifact location: `[paths].real_experiment` in `configs/app.toml`, defaulting to
`data/real-20261105-core-v2`. Read the [coverage, blockers, protocol and reproducible
commands](docs/real-data-experiment.md) before continuing. The new observed-only
preprocessor is tested but has not yet been connected to real model comparison.

## Package boundaries

| Module | Responsibility |
| --- | --- |
| `ingestion` | FastF1 schedules/timing, official results, normalization, source revisions |
| `features` | Cutoffs, source selection, rosters, allowlisted matrices and audits |
| `models` | Order regressors, podium classifiers/calibration, fold preprocessing, saved bundles |
| `evaluation` | Frozen manifest, complete-weekend folds, coverage and event metrics |
| `prediction` | Artifact lineage, chronological guards, orders and probabilities |
| `experiments` | Selective raw capture, optional supplements, paired development ablations |
| `finalization` | Development audit, immutable lock, one-time holdout evaluation and reports |
| `app` | Reserved boundary inside the locked code fingerprint; intentionally left empty |
| `f1ml_app` (separate package) | Phase 8 Streamlit interface; read-only over saved artifacts |
| `f1ml_real` (separate package) | Phase 9 real-source audit, normalization replay and preparatory observed-only preprocessing |

Remaining work includes completing unavailable downloads, cutoff/identity review,
publication-time snapshot collection and genuine real-data model validation.
The synthetic holdout is consumed once under Phase 7 and cannot be reused as
fresh confirmation for later changes.
