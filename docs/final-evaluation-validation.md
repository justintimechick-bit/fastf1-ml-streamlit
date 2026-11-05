# Phase 7 results and verification — November 5, 2026

**SYNTHETIC HOLDOUT SOFTWARE VALIDATION.** This does not establish observed Formula
1 accuracy. The only frozen experiment contains generated fixtures; a genuine
2026 study still needs reviewed real cutoffs/rosters and its own frozen manifest.
No missing live data was fabricated or substituted.

## Locked choices before outcome access

The development audit checked **228 saved model metadata files**, comparison
tables, source/feature audits and receipt integrity. No reserved event ID appeared
in recorded feature-selection, training, weighting, tuning or calibration
dependencies. Inner/outer fold chronology passed. This is a project-lineage audit,
not proof of no unlogged human access; the synthetic generator also limits
independence. Manifest eligibility had inspected result validity/availability,
which is distinct from predictive model selection.

All learned families and A/B/C strategies were compared on the same eligible
development event, `2026-03`. A's first two folds were ineligible because they
lacked two earlier 2026 training weekends. Only one common event means selection
is weak evidence. Additional early B/C scores remain in the development files.
Baseline results were retained even where better than the learned winners.

| Task | Locked learned model | Weighting | Fixed settings |
| --- | --- | --- | --- |
| Qualifying | Ridge | pooled equal | alpha 10 |
| Official race classification | Ridge | pooled equal | alpha 1 |
| Podium | histogram gradient boosting | 2026-favored | 100 iterations, rate 0.1, 15 leaves, minimum leaf 5, L2 1 |

All refits use the seven development weekends only: two 2024, two 2025, three
2026. Numeric median imputation/missing indicators/scaling and unknown-safe
categorical encoding are fitted on those training rows. Core feature lists are
frozen in the model metadata; optional groups were not promoted. Podium weights
use half-life 12 eligible events and season multipliers 0.5/1/2 for 2024/25/26.
Sigmoid calibration and threshold 0.5 were fitted/selected on earlier OOF events
`2026-02`, `2026-03`; the refit's tuning event was `2026-01`.

## One recorded evaluation

Independent fixture rosters/cutoffs were prepared without consulting target
tables. Exact commands executed, in this order:

```sh
.venv/bin/f1ml synthetic-final-inputs --dataset data/phase3-final/dataset \
  --manifest data/phase3-final/manifest.json --output data/phase7-inputs
.venv/bin/f1ml final-lock --dataset data/phase3-final/dataset \
  --manifest data/phase3-final/manifest.json --features data/phase3-final/features \
  --qualifying data/phase4-final --race data/phase5-final \
  --experiments data/phase6-verified-experiments \
  --entrants data/phase7-inputs/entrants.csv --cutoffs data/phase7-inputs/cutoffs.csv \
  --output data/phase7-lock
.venv/bin/f1ml final-evaluate --lock data/phase7-lock \
  --dataset data/phase3-final/dataset --output data/phase7-final
```

| Artifact | Receipt ID |
| --- | --- |
| Independent synthetic inputs | `3f87e2bdea321581da4ed9b6a6b56c7ce49e0dec1dc89bab384707f14f93f13f` |
| Immutable model/protocol lock | `0f22d28a3240d522ad0906f01d3e95e8c34f14e6fc6094e7048f35f627db6ec8` |
| One-time evaluation | `e0eaf8f1ec59006b98c9b02abb679ffcd358ebf32448161575f6aa91f2317a18` |

The manifest remains unchanged:
`3cc7b3bf005e808e360595275f3e24c35bbd8a26c962bafdef8e6ee766f968dd`.
The reserved `2026-04` through `2026-07` block has now been consumed once under
`data/final-evaluation-ledger/<manifest-id>/`. A different output directory or
lock cannot create another primary run. Deliberate reuse must be exploratory.

All qualifying predictions were sealed before GP qualifying could be used as
race input; race/podium predictions were sealed before scoring/diagnostic weather.
Model bytes, calibration, weights and thresholds stayed fixed. History also
stayed pre-block: no earlier block outcome entered later features. No model or
feature code changed after the primary result, and no test-driven tuning occurred.

## Results observed in the generated source

Equal-weekend means across four weekends:

| Task/method | Position MAE | Spearman | Other metrics |
| --- | ---: | ---: | --- |
| Qualifying Ridge/equal | 2.8333 | −0.4286 | pole 0; top-three overlap 0.25 |
| Qualifying recent baseline | 2.7500 | −0.4464 | pole 0; top-three overlap 0.25 |
| Qualifying practice baseline | 2.7500 | −0.4464 | full recent-form fallback |
| Race Ridge/equal | 0 | 1 | winner 1; top-three overlap 1 |
| Race qualifying-order baseline | 0 | 1 | winner 1; top-three overlap 1 |
| Race recent baseline | 2.7500 | −0.4464 | winner 0; top-three overlap 0.25 |

Podium model: log loss **0.012279**, Brier **0.000150**, AP **1.0**, ECE **0.012203**;
precision/recall are both 1 at the frozen threshold. The hard qualifying reference
has Brier 0 and effectively zero clipped log loss. Recent podium-rate baseline
log loss is 10.058242 and Brier 0.538095. Probabilities were not normalized.

**Interpretation:** qualifying Ridge did not beat recent form, and its locked
choice is unchanged. The fixture intentionally makes race order equal qualifying
order, explaining excellent race/podium/qualifying-reference scores. These are
not evidence of real F1 skill, calibration or deployment readiness. The broader
effect of 2026 weighting and calibration remains a hypothesis for real data.

The qualifying model's event MAE range is 2.5714–3.4286; leave-one-event-out mean
range is 2.6349–2.9206. These are sensitivity summaries, **not confidence intervals**.
All confidence interval fields remain null with only four event units. Driver
rows are not treated as independent replicates.

## Coverage and saved outputs

The source has 12 fixture events: seven development, four test, one unfinished
future event excluded at manifest freeze. Test fields contain 6, 7, 6, 7 drivers:
26 driver-event predictions per task, 78 rows including references per task.
There were no field mismatches or missing test positions. All 12 task/event
coverage rows are present, with outcome/status retained label-side.

Saved under `data/phase7-final/`:

- `event_metrics`, `aggregate_metrics`, and `paired_baseline_differences`: CSV/Parquet.
- `qualifying_predicted_vs_actual`, `race_predicted_vs_actual`, and
  `podium_predicted_vs_actual`: CSV/Parquet with explicit synthetic names/kind.
- `race_and_podium_forecasts` and `top_three_comparison.json` distinguish order and
  probability top threes.
- `calibration-development` and `calibration-final`: PNG/SVG plus bin CSV/Parquet;
  final PNG was visually inspected. No calibration fit uses these test outcomes.
- `subgroup_errors`: driver/team descriptive means where at least three weekends
  exist; counts and suppression reasons otherwise. The intermittent seventh
  fixture driver has only two held-out appearances and is suppressed.
- `conditions_after_prediction`: all conditions unknown because target weather
  is unavailable. **No wet/dry error conclusion is possible.** Missing is not dry.
- `dataset_sessions`, `dataset_coverage`, `coverage`: source season/role/session
  availability and final task/target exclusions.
- `development/`: original all-strategy comparisons, eligibility/searches,
  feature-group comparisons, missingness and retention decisions.
- Original lock/protocol/audit/prediction-seal receipts and `summary.json`.

## Verification and future confirmation

**155 tests passed**, including ten finalization tests. Lint and formatting checks
passed. Wheel/source distribution built with explicit Matplotlib dependency,
already pinned in the tested lock. Tests cover immutable locks, no held-out reads
before lock, no fitting in evaluation, prediction seals before scoring, fixed
history, later-fold rejection, one-primary ledger and failed-claim behavior,
exploratory labeling, source changes, uncertainty and missing weather. Reversing
held-out race outcomes changes scoring but leaves predictions unchanged.

The source snapshots, original manifest, Phase 3 features, Phase 4/5 models and
Phase 6 comparisons still match their prior receipts. Nonfatal sandbox CPU-probe
and joblib core-discovery warnings did not prevent successful checks.

No accuracy target is promised. Any later model/feature/calibration change or
reuse of this test block is **exploratory** and requires a fresh future holdout for
confirmation. A real-data final evaluation remains outstanding because the
available genuine source lacks the reviewed experiment inputs; the successful
synthetic evaluation must not be relabeled as that result.
