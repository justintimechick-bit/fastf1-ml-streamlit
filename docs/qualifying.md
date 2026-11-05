# Qualifying-order models (Phase 4)

This phase consumes the immutable Phase 3 qualifying feature/label snapshot.
It leaves feature generation, ingestion, legacy baselines and the frozen final
test manifest intact. Read [methodology](methodology.md) and [features](features.md)
first. No target-event qualifying or race observations enter qualifying inputs.
The target is official **qualifying classification**, never starting grid.

## Target, scores and complete fields

Regress `(official_position - 1) / (official_field_size - 1)` on known targets.
Missing training targets are masked, never imputed; their input rows remain in
the dataset. The model predicts every reviewed eligible entrant, including
rookies, substitutes and drivers without practice data. No qualifying
elimination counts, teams, drivers or field size are hardcoded.

`raw_score` is an unclipped normalized-rank estimate. Lower scores rank ahead.
Scores are neither probabilities nor precise lap-time forecasts. For each
event, sort by exact raw score ascending, then stable `driver_id` lexicographically
ascending. Do not round before sorting. Assign `predicted_position = 1..N` to
this order. Duplicate keys, missing drivers, nonfinite scores and incomplete
fields are rejected. Raw scores remain alongside positions in Parquet outputs.

## Models and fixed reference rules

All learned models share a training-fold-only numeric median imputer with
missing indicators, standard scaling, and dense one-hot categorical encoding
with unknown categories ignored. Empty numeric columns remain supported. The
estimator receives sample weights; preprocessing uses only selected, labeled
training rows. Scaling is retained for a consistent pipeline across families.

| Family | Default candidate search | Fixed settings |
| --- | --- | --- |
| Ridge | alpha 1, 10, 100 | regularized squared error |
| Random forest | maximum depth 4, 8 | 80 trees, minimum leaf 2, seed 42, one worker |
| Histogram gradient boosting | learning rate 0.05, 0.1 | 100 iterations, 15 leaves, minimum leaf 5, L2 1, seed 42 |

Boosting uses `early_stopping=False` so it creates no random internal validation
split. Verified against installed scikit-learn 1.9.1 and the official
[boosting](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html)
and [encoding](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)
interfaces. Compatible dependency bounds and a tested lock file already exist.

Fixed baselines are evaluated on the same common events:

- **Recent:** shifted previous-three-event mean normalized qualifying rank,
  then previous-five mean, then previous appearance rank, then fixed neutral 0.5.
- **Practice:** use Phase 3 clean comparable `practice_gap_pct`. With at least two
  observed drivers, convert gaps to `(ascending_min_rank - 1)/(observed_count - 1)`.
  Each missing driver falls back to the recent rule; with fewer than two observed
  drivers, the whole field falls back. Exact equal gaps get equal raw scores.
  Store `score_source` and the source practice deficit for inspection.

Practice gaps are not mixed directly with normalized historical ranks. The
observed-field practice rank is a simple proxy: missing participants can still
make that proxy optimistic or pessimistic. Unknown fuel/run plans and unequal
practice coverage remain limitations. Neither rule is tuned on validation data.

## Training strategies and weights

| Strategy | Training examples | Event weight |
| --- | --- | --- |
| `2026_only` (A) | Earlier eligible 2026 weekends | 1 |
| `equal` (B) | Earlier eligible 2024–2026 weekends | 1 |
| `weighted` (C) | Earlier eligible 2024–2026 weekends | season multiplier × `2^(-age/half_life)` |

Age is the number of eligible calendar weekends since the training event up to
the latest training event. Season multipliers are 0.5 for 2024, 1 for 2025,
and a development-tuned 2 or 4 for 2026; half-life candidates are 6 and 12 eligible
weekends. Within an event, each labeled row gets event weight divided by the
number of labeled drivers. Normalize all row weights together to mean one.
This gives equal weekend mass for B and preserves C's recency/season preference,
including when field sizes or label coverage differ. Record actual event weights.

A restricts training examples, not the historical inputs of those examples:
permitted Phase 3 older priors are still available. Current-season team form,
older team priors and the regulation-era indicator remain separate Phase 3
features. Extra 2026 weighting is a hypothesis to validate, not evidence that
older seasons are irrelevant or that renamed teams share performance history.

## Chronological development and selection

Use the existing frozen manifest without changing its test block or event IDs.
Outer folds are its complete-weekend expanding 2026 development folds. For each
outer fold, search model parameters and C's weight profiles only on inner
manifest folds whose **training and validation events all belong to that outer
training block**. Use at most the three latest eligible inner folds. Refit every
candidate's preprocessing separately inside each inner training fold. Minimize
the equal-event mean inner positional MAE; exact ties keep predeclared candidate
order. The outer event never selects its own model parameters or weights.

The default minimum is two usable training events for B/C and two earlier 2026
events for A. Ineligible folds are explicit in `eligibility.json`. If a strategy
has enough training data but no eligible inner fold, fit the first predeclared
candidate as a **fixed default**, labeled `fixed_default_no_inner_folds`; do not
pretend it was tuned. `[qualifying]` in `configs/default.toml` configures minima,
search values and inner-fold cap. The exact settings and seed are saved.

Primary averages and learned-family/strategy selection use only the intersection
of complete evaluation events eligible for **all three strategies**. Per-event
outputs also retain other eligible development folds, clearly flagged
`common_comparison=False`. Never compare unequal-cohort averages as the main
result. With no common event, write the feasibility report, return CLI status 1,
and save no selected bundle. Do not silently relax the minimum or final block.

Select the learned family/strategy with lowest common-event mean positional MAE;
exact ties use Ridge, forest, boosting order, then A, B, C order. Retune this cell
on chronological development folds and refit on its allowed development events.
`selected_model` means best **learned** model; a fixed rule may outperform it.
Its selection scores remain development results, not unbiased final-test scores.
The latest four eligible completed 2026 weekends (or explicitly configured size)
remain untouched. There is no final-test execution flag in this phase.

## Metrics, artifacts and prediction eligibility

Evaluate only complete official orders with exactly positions 1 through N and
matching normalized targets. Missing/ambiguous official order makes the event
ineligible for full-order metrics across all methods, with a recorded reason;
optional practice missingness does not exclude it. Never manufacture positions.

- Position MAE: mean `abs(predicted_position - official_position)` over the field.
- Spearman: correlation of the two complete rank permutations.
- Pole accuracy: one if the predicted first driver is the official qualifier first.
- Top-three/top-ten overlap: intersection size divided by `min(k, N)`; per-event
  tables also retain intersection count and denominator. Small-field top-ten
  scores can be trivially one.

Aggregate each metric with equal weight per common event, not per driver.
`report.md` presents per-event results, common averages and eligibility;
`summary.json`, `event_metrics.parquet`, `predictions.parquet` and `tuning.json`
are machine-readable. Every outer model and the selected refit get a bundle
with `model.joblib`, `model.json` and content-hashed `receipt.json`. Metadata
contains the fitted pipeline, exact feature list/version/code ID, feature
snapshot/manifest IDs, training IDs, label coverage, seed, library versions,
parameters, weights, inner folds, selection events and information dependencies.

Prediction rebuilds inputs through the Phase 3 qualifying cutoff contract and
requires a complete externally reviewed roster. The bundle must match the
manifest and feature implementation. The event must be outside all training,
tuning and selection dependencies and outside the sealed final test. All model
dependencies must precede the cutoff, and their weekends must end before the
target weekend begins. This prevents using a late selected refit to claim an
early historical prediction. Use the relevant earlier fold bundle for that.
Publication mode additionally checks exact label-revision retrieval times and
rejects reconstructed training. Historical reconstructions retain the existing
limitation that true publication-time snapshots may be unavailable.
Bundles use pickle internally: load only trusted local artifacts; integrity
hashes detect alteration but do not authenticate an outside sender.

## Commands

With the virtual environment active, create a fresh synthetic demo and compare:

```sh
f1ml demo --output data/qualifying-demo
f1ml --config configs/default.toml qualifying-compare \
  --manifest data/qualifying-demo/manifest.json \
  --features data/qualifying-demo/features --output data/qualifying-models
cat data/qualifying-models/report.md
cat data/qualifying-models/summary.json
```

Predict the third **synthetic** development event with its earlier fold model:

```sh
python - <<'PY'
import pandas as pd
rows = pd.read_csv("data/qualifying-demo/entrants.csv", dtype=str)
rows.loc[rows.event_id.eq("2026-03") & rows.task.eq("qualifying")].to_csv(
    "data/qualifying-demo/2026-03-qualifying-entrants.csv", index=False, mode="x"
)
PY
f1ml predict-qualifying \
  --model data/qualifying-models/fold_models/2026-03/ridge/equal \
  --dataset data/qualifying-demo/dataset \
  --manifest data/qualifying-demo/manifest.json --event 2026-03 \
  --entrants data/qualifying-demo/2026-03-qualifying-entrants.csv \
  --cutoff 2026-03-23T13:59:59Z --mode historical_reconstruction \
  --output data/qualifying-prediction
```

For a genuinely future discovered event use `--model <comparison>/selected_model`,
the event's reviewed cutoff/roster, and a source snapshot containing that event.
Phase 2 source snapshots support `--availability` and `--event-context` as in
Phase 3. Unknown session completion times or rosters still require review; they
are not inferred from future target results. Choose new output paths to rerun.
The existing live source limitations prevent an honest real-data accuracy claim
until those reviews and missing downloads are resolved. No dashboard is built.
