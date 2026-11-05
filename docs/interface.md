# Interface (Phase 8)

A local, read-only Streamlit interface over the saved Phase 2–9 artifacts. It never
downloads data, fits or recalibrates a model, selects thresholds, or opens the final-test
block. Launch commands are in the [README](../README.md#interface-phase-8).

## Data sources

| Source | What it is | Forecasts |
| --- | --- | --- |
| **Synthetic demo** (default) | The frozen experiment `data/phase3-final` with the Phase 4/5 comparisons, the Phase 7 lock and the one-time final evaluation. Its fixtures are **synthetic**; every page says so. | Yes, where an eligible model exists |
| **FastF1 catalog** | The real 2024–2026 snapshot named by `data/ingestion-v2/CURRENT.json`, read through `ingestion.coverage.inspect_coverage`. | None: no reviewed cutoff rosters or frozen real manifest exist |
| **Real experiment** | The independent, receipt-verified Phase 9 source snapshot, coverage/exclusion reports, protocol and evaluation status at `data/real-20261105-core-v2`. | None: blocked before training; never substitutes demo models |

Paths default to the README commands. Override them in `configs/app.toml` under a
`[paths]` table (keys: `dataset`, `manifest`, `features`, `development_cutoffs`,
`qualifying`, `race`, `experiments`, `lock`, `final`, `ledger`, `catalog`, `real_experiment`), or point
`F1ML_APP_CONFIG` at another TOML file. `F1ML_PROJECT_ROOT` sets the base directory.

When Real experiment is selected, overview, comparison and methodology display its
saved provenance, coverage, exclusions and explicitly unrun development/final
records. Forecast pages remain unavailable. Its immutable copied dataset is loaded
directly; moving the acquisition catalog's `CURRENT.json` cannot change this
experiment. See the [Phase 9 execution record](real-data-experiment.md).

## Which model a weekend uses

| Weekend role | Forecast shown | Model |
| --- | --- | --- |
| Final test (2026-04 … 07) | Phase 7 sealed forecasts, displayed unchanged; seal receipts verified against `prediction_seals.json` | Locked refit trained on the seven development weekends |
| Development validation (2026-01 … 03) | Replayed live from saved features through `prediction.{qualifying,race}.predict_checked`, then `combine_predictions` | The outer-fold bundle saved for that weekend (trained only on earlier weekends); default = the family/strategy selected in development; other saved candidates can be viewed |
| Development training (2024–2025) | Unavailable | None: every saved model trained on the weekend or later |
| Excluded / future (2026-08) | Unavailable | The locked model would be eligible, but no reviewed entrant roster or completed session exists |

Replays are cross-checked against the saved Phase 4/5 predictions and match exactly.
`predict_checked` remains the authority: it rejects any model that used the weekend for
training, tuning or selection, or whose data reaches the cutoff. The interface also
explains ineligibility up front (for example, 2026-only needs two earlier 2026 weekends).

## Replay moment and outcomes

The sidebar's replay moment steps through the weekend:

* **Before qualifying**: the qualifying forecast only. The post-qualifying race/podium
  forecast is shown as unavailable because official qualifying does not yet exist.
* **After qualifying**: adds the race/podium forecast and reveals the official qualifying
  classification.
* **After the race**: reveals the official race classification and podiums.

Outcomes come from label-side tables (Phase 3 labels for development, the Phase 7 scoring
record for the test). They are joined to forecasts only for display, never passed to a
predictor, and never included in CSV exports. Event metrics shown beside a comparison are
read from the saved evaluation records, not recomputed.

## Views

1. **Event overview**: weekend role, session timeline with both cutoffs and the model's
   training end, per-task model and training range, input coverage, season at a glance.
   For catalog events: session availability, retrieved classifications, failures.
2. **Qualifying predictions**: full predicted order with ranking scores (labeled as
   *not probabilities*), practice and recent-form inputs, imputed-input counts,
   predicted-vs-official dumbbell chart, fixed-rule references, CSV export.
3. **Race predictions**: race-order top three beside the podium classifier's three highest
   independent probabilities, podium probability chart with the development-selected
   threshold, full order, the qualifying information used (classification, Q1–Q3 pace,
   evidenced grid), predicted vs official, CSV export.
4. **Model comparison**: development results (A 2026-only / B pooled / C weighted; common
   cohort; selected model; all folds; eligibility) in one tab, frozen final-test results
   (aggregate with event and leave-one-out ranges, per-event chart, paired differences,
   saved calibration figures) in a separate tab, Phase 6 feature-group decisions, and a
   metric guide for presentations.
5. **Data quality & methodology**: session availability matrices, real catalog coverage
   and failures, per-feature missingness (median-imputed fallbacks), exclusions, all
   cutoffs with lead times, model families and locked parameters, limitations.

## CSV export

One row per driver. Columns: data kind, manifest ID, event, role, prediction source,
cutoff (UTC), provenance mode, seal ID (final test), predicted position, ranking score
(`*_not_probability`), podium probabilities (race), the qualifying position and evidenced
grid used as inputs (race), per-driver coverage (practice sessions, clean laps, practice
pace available, history weekends, imputed inputs and their names), and for each model:
ID, family, strategy, bundle role, training event count, first/last training event,
training event list and training data end (UTC).

## Why the package is `f1ml_app`, not `f1ml.app`

`finalization.locking.code_id()` fingerprints every `.py` file under `src/f1ml`, and the
Phase 7 lock records that fingerprint. Adding the interface inside `src/f1ml/app` would
have made current code differ from the lock even though no model or feature code changed.
The interface therefore lives in `src/f1ml_app`; `src/f1ml` is unchanged, and the
final-test tab shows that the current fingerprint still matches the lock.

## Classroom walkthrough (about 5 minutes)

1. **Event overview**, R04 · test: the timeline shows practice → qualifying cutoff →
   qualifying → race cutoff, and the locked models' training data ending weeks earlier.
2. Set the replay moment to **Before qualifying** and open **Race predictions**: the
   post-qualifying forecast is unavailable. Open **Qualifying predictions**: the forecast
   exists and the official result stays hidden.
3. Move to **After the race**: compare forecast vs official on the dumbbell chart.
4. Pick **R03 · dev** and switch the saved bundle between strategies, then pick
   **R01 · dev** to see 2026-only listed as not trained.
5. **Model comparison**: development (where the recent-form rule beat every learned model)
   versus the frozen final test, then the metric guide.

## Verification

`pytest -q tests/test_app.py` runs the integration checks against the saved artifacts
(skipped when `data/` is absent): roles, every saved outer-fold bundle replaying its saved
predictions, ineligible strategies and training weekends, guard rejection of a
later-trained model, final-test seals unchanged and unrecomputable, race forecast
unavailable before qualifying, outcomes absent from inputs/exports, export lineage
columns, catalog coverage, and an app run across views, events, moments, seasons and both
sources with model fitting, calibration, locking, evaluation and HTTP requests patched to
fail and the saved artifact directories hashed before and after.
