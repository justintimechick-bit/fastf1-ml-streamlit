# Real F1 model lab

The active Streamlit interface uses real observations. Choose a completed event,
select 2024, 2025 or 2026 and how much earlier history from that year to use,
and press **Run historical test**.
The interface compares three learned model families on the same eligible drivers
and events. It also displays simple baselines, so a trained model's failure to
improve on a baseline stays visible.

| Family shown in the interface | Qualifying and race order | Podium probability | Course connection |
| --- | --- | --- | --- |
| Linear / logistic | Ridge regression | Logistic regression | Sessions 1–2: regression and regularization |
| Random forest | Random forest regression | Random forest classification | Session 3: trees and ensembles |
| Gradient boosting | Histogram gradient boosting regression | Histogram gradient boosting classification | Session 3: boosting |

These are **three families applied to three prediction tasks**. Qualifying and
race models estimate normalized rank scores and sort the full entrant field.
Podium models estimate each driver's marginal probability of finishing in the
top three; these probabilities are not a joint distribution over three winners.
The simple reference rules are baselines, not a fourth learned family.

## Use the interface

1. Open **Historical test**, select the season and completed event to reconstruct.
2. Select **Previous 6 races that year**, **Previous 12 races that year**, or
   **All earlier races that year**. Only eligible earlier races from the selected
   season can train the models. Short windows use all available earlier races
   when fewer exist; a season opener cannot train on another year's results.
3. Press **Run historical test**. Training occurs only on this explicit action.
   An identical request loads its saved, verified run.
4. Compare qualifying, race and podium outputs. Review the displayed training
   events, development-event counts, eligible labels and source coverage before
   interpreting a score.
5. Read **Three models** for the syllabus mapping and **Data & methodology** for
   provenance and limitations.

Opening the application or changing an ordinary display control does not start
a download or fit. Missing real forecasts are shown as missing. The application
never fills them with a demo forecast. Source acquisition stays outside ordinary
Streamlit reruns.

The 2024 and 2025 snapshots each contain all 24 races. Where FIA documents could
not be verified, the app uses labeled cached FastF1/Jolpica classifications
(8 races in 2024; 15 in 2025). Independent FIA entrant fields are retained where
available; other fields are labeled API reconstructions. Publication timing is
an explicit proxy for API records, using FIA publication records where present,
otherwise qualifying just before race start and race labels 24 hours after race
start. This does not verify historical API publication/revision timing. The
interface, dataset receipt and saved run summary retain these limitations.

### Refresh and predict a new weekend

The **Refresh FastF1 data** button at the bottom of the sidebar is available on
every page. It refreshes the selected season; **Refresh & predict** always refreshes
2026. It explicitly downloads the season calendar and started Q/R result endpoints, together with FIA entry lists
and final classification evidence. Each acquisition bypasses stale FastF1 caches
and uses fresh FIA indexes. The original reviewed dataset stays fixed; a verified
revision is activated atomically under `data/live`. A failed acquisition leaves
the previous revision available, with event-level source problems shown in the UI.
**Data & methodology** reads this active revision too, so refreshing updates its
coverage, source details, completed-event count and last-refreshed timestamp.

**Run forecast** fits only unstarted tasks whose required inputs are present.
Qualifying needs an independent entry list. Race and podium need an independent
race field and final qualifying published before their forecast cutoff. Live
cutoffs use the acquisition horizon; new completed-event reconstructions retain
the historical cutoffs described below. Current-season earlier published events
supply training, development selection and calibration. The target's labels are
never joined or scored for a live forecast. Training checks the session clock
again before sealing; if the target starts during fitting, it does not seal a
live forecast. Saved forecasts remain viewable after the target starts.

```sh
python -m f1ml_real refresh
python -m f1ml_real forecast --event 2026-20
```

Both commands accept `--dataset` and `--state`; forecast also accepts `--runs` and
`--last-n`. `F1ML_LIVE_DATA` configures the app's state directory. The active year
comes from the reviewed base dataset. Unknown entrants require a matching name
and car number in FastF1's captured results before their new identity is accepted;
a new driver without that evidence remains unavailable. Raw API captures and
publication/source hashes are retained independently of saved model runs.

## Source data and availability

The original FastF1/Jolpica catalog and the new FIA modeling snapshot have
separate roles. The catalog audits 2024, 2025 and completed 2026 schedules and
result acquisition. FastF1 timing access remains incomplete: practice laps,
qualifying laps and weather are unavailable in the retained source. An HTTP
access failure is not proof that the underlying session never existed. The
session-level audit distinguishes source failures, deferred downloads and
parsing failures. Cached responses and resumable jobs preserve prior work.

The original 2026 core snapshot uses **FIA final classifications and independently
published entry lists**. Each factual table records public source URLs; the
source manifest records publication timestamps and SHA-256 hashes. Parsed facts
and links can be deployed; raw PDFs stay in the local source cache. A revised
FIA final classification can differ from an earlier Jolpica response after an
administrative decision. The source is chosen before model selection; it is not
selected according to a model's error.

For FIA-only evidence, document publication is a **conservative availability/completion upper
bound**, not a measured session-end timestamp. The qualifying cutoff is one
second before the scheduled qualifying start. The race reconstruction cutoff is
one minute after publication of the chosen final qualifying classification and
must precede the scheduled race. The independently published entrant list must
already be available at the task's cutoff. Different task rosters can represent
a real substitution; target results do not supply the roster or team membership.

A latest revised final published after an early historical cutoff is not known
at that cutoff. That revised result cannot provide labels or history there;
independently available task information can still appear in its calendar window. Eligibility depends
on source availability and label quality, never model performance. Only events
whose race labels were available before the target weekend can supply its
fitted training rows. The selected target itself and later events are excluded.

## What the core models can observe

Inputs are shifted, field-relative driver and team classification history over
prior event windows, counts of available history and independently known driver
and team identity. Race and podium models can additionally observe the target
weekend's official qualifying classification after its publication. Qualifying
models cannot observe target qualifying or race outcomes. Historical order is
normalized by the event's entrant field size before averaging across events.
A driver's absence or unknown rank remains missing; it is not converted to a
last-place result. The recent-6/recent-12 choices limit fitted training events.
Each event's features still use its earlier publicly available three/five-event
calendar windows, including observations outside that fitted-row window.
Unavailable components are masked without extending the window to older appearances.

There are **no observed lap-pace or weather features** in this core run. Entirely
missing or constant training columns are removed within each fit. Median
imputation, missingness indicators, scaling and category encoding are fitted on
that fold's training rows only. A later event cannot supply an imputation median,
a category vocabulary, or promote an excluded feature into the trained schema.

This is a useful result-history experiment while richer session data remains
incomplete. It does not validate lap-time segmentation, tyre comparability or
weather effects against observations that were never downloaded. Comparisons
using 2024–2025 pooled or weighted history still require reviewed equivalent
roster/availability evidence; the functional lab currently exposes 2026 history
rather than implying those broader Phase 9 comparisons were completed.

## Validation and honest metric interpretation

The target event is selected before its models are fitted. Earlier chronological
development events select small predefined hyperparameter grids. Calibration
uses separate later development predictions where enough observations exist.
Development scores are displayed separately from the selected historical-target
scores. Configuration and forecasts are sealed before joining target labels and
computing target metrics. A separate run directory holds this evidence; the
original final-evaluation ledger and sealed experiment are unchanged.

All these historical outcomes were available and consulted during source QA.
The result is an **exploratory historical replay**, not an untouched final test.
Repeatedly changing the event/history after looking at outcomes remains
exploration. A genuinely untouched confirmation needs a newly specified
prospective cohort and its own locked protocol.

Rank metrics use only **explicitly printed numeric ranks**. FIA `NC`, `DSQ`,
`DNS`, withdrawal, or a roster entrant absent from a classification is not given
an invented tail position. Known-rank MAE/RMSE/correlation therefore measure a
partial-label problem; inspect the known-rank and event counts. They are not
complete-field finishing-order scores. Explicitly nonclassified/disqualified
race drivers are observed non-podium outcomes. An absent, unobserved label stays
unknown and is excluded from the corresponding label metric.

The Mexico example includes a pre-qualifying replacement (Ryo Hirakawa) and
Lawson's qualifying disqualification. Its final race classification prints 17
numeric ranks and five nonclassified drivers. The interface still predicts the
whole independently evidenced 22-driver field, while rank metrics use the 17
observed numeric labels. Podium scoring can include all 22 observed race outcomes.

## Reproduce and test

Use the pinned environment from the project root. Acquisition, model training and
report export are explicit commands; no command below regenerates an old lock.
New immutable outputs need a new path. See `configs/` and the source receipts for
the exact acquisition configuration rather than treating a local retrieval time
as a historical availability timestamp.

```sh
# Explicit real-data training, or verified reuse of the identical saved run.
.venv/bin/python -m f1ml_real backtest \
  --dataset data/real-core-20261105-v4 --event 2026-19 \
  --runs artifacts/real-lab

# An exploratory recent-history comparison uses a different saved run.
.venv/bin/python -m f1ml_real backtest \
  --dataset data/real-core-20261105-v4 --event 2026-18 \
  --last-n 12 --runs artifacts/real-lab

.venv/bin/python -m pytest
.venv/bin/streamlit run streamlit_app.py
```

The default test suite is `tests/real/` and reads the sealed real FIA snapshot.
It checks source hashes, independent rosters, task cutoffs, official special
statuses, shifted features, training-only preprocessing and the real interface.
Guard tests may deliberately alter a copy of actual rows or a copied artifact to
verify a rejection; they do not create synthetic race observations. Legacy demo
software tests and sealed synthetic artifacts remain archived and are outside
the default real-only workflow.

Once a saved run exists, an executed notebook can be exported for the project
submission. The exporter actually runs its analysis cells and records their
outputs; it does not claim to have rerun training or invent notebook results.

```sh
.venv/bin/python tools/export_real_notebook.py \
  --dataset data/real-core-20261105-v4 \
  --run artifacts/real-lab/replay-2026-19-all-bdc1c440786e \
  --output notebooks/real-mexico-2026-reproduced.ipynb
```

Read [rubric alignment](project-rubric-alignment.md) for the report and notebook
requirements. The actual saved-run commands, counts and results are recorded
with the run artifacts and should be reported with their dataset/run IDs.

## Mexico run actually executed

The saved example is
`artifacts/real-lab/replay-2026-19-all-bdc1c440786e/`.
Its dataset is `data/real-core-20261105-v4/`, ID
`08bd7449688fda332cab42379d89cdeb295c3b200c45d2d9119e3aefc6befb2a`.
It contains **19 completed 2026 events**, with all 22 independent entrants per
qualifying/race roster. Four future scheduled events remain excluded.
**410/418** qualifying numeric ranks and **339/418** race numeric ranks are
observed in the snapshot. There are five entrants absent from their final
qualifying documents (three in Australia,
two in Spain); their numeric labels remain unknown. Special statuses also remain
unranked unless the document explicitly prints a numeric position.

All **18 earlier 2026 events (rounds 1–18)** fitted the final Mexico estimators.
Parameter selection used **three development events**, rounds **14–16**;
rounds **17–18** supplied separate chronological podium-calibration predictions.
Calibration-fit diagnostics are saved separately and are not independent
validation estimates. Nine final fitted estimators and the three baseline rules
produced forecasts for the whole 22-driver Mexico field.

| Method | Development Q MAE, 3 events | Development race MAE, 3 events | Development podium Brier, 3 events |
| --- | ---: | ---: | ---: |
| Simple baseline | 2.788 | **3.276** | 0.0734 |
| Linear / logistic | 2.812 | 3.617 | 0.0764 |
| Random forest | 2.812 | 3.433 | **0.0671** |
| Gradient boosting | **2.721** | 3.746 | 0.0853 |

Development rank metrics use **64/66 qualifying** and **54/66 race** numeric
labels; podium metrics use **66/66**. Podium selection scores above are from
uncalibrated development predictions; calibration uses the later two events.

| Method | Mexico Q MAE, 1 event (21/22 ranks) | Mexico race MAE, 1 event (17/22 ranks) | Mexico podium Brier, 1 event (22/22 labels) |
| --- | ---: | ---: | ---: |
| Simple baseline | 2.286 | **3.412** | 0.1275 |
| Linear / logistic | 1.952 | 3.824 | **0.0312** |
| Random forest | **1.857** | 3.588 | 0.0589 |
| Gradient boosting | 2.048 | 3.471 | 0.0738 |

**All three learned race regressors lost to the qualifying-order baseline on
Mexico's known-rank MAE.** Every learned qualifying model improved on the
qualifying-form baseline for this event. The podium probability models improved
on their fixed probability baseline, but one consulted event cannot establish
general superiority. No settings were retuned against these target results.
The table's best cells describe measurements; they do not select a new model.
Full RMSE, rank correlation, podium log loss/precision/recall/F1, driver-level
predictions, parameter choices, source/cutoff audits and hashes are saved with
the run. The executed [analysis notebook](../notebooks/real-mexico-2026.ipynb)
shows the actual output and both development stages.

To reconstruct the factual snapshot from the retained verified cache, use a new
output path:

```sh
.venv/bin/python -m f1ml_real prepare-real \
  --catalog data/real-20261105-catalog --cache data/fia-2026-source-cache \
  --output data/real-core-reproduction --as-of 2026-11-05T16:00:00Z --offline
```

Omit `--offline` only for an explicit acquisition run that may fetch missing
public FIA documents. This does not unlock or overwrite any previous snapshot.
The FastF1 timing access restriction and the unreviewed older-season
availability/roster evidence still limit the broader pooled/weighted Phase 9
protocol. This functional 2026 core lab does not claim that protocol is complete.
