# SYNTHETIC HOLDOUT SOFTWARE VALIDATION

One recorded primary evaluation; no fitting inside the test block.

## Locked choices

- podium: gradient_boosting, weighted; parameters `{'l2_regularization': 1.0, 'learning_rate': 0.1, 'max_iter': 100, 'max_leaf_nodes': 15, 'min_samples_leaf': 5}`; 7 pre-block training weekends; core inputs.
- qualifying: ridge, equal; parameters `{'alpha': 10.0}`; 7 pre-block training weekends; core inputs.
- race: ridge, equal; parameters `{'alpha': 1.0}`; 7 pre-block training weekends; core inputs.

History is fixed to pre-block weekends. Qualifying results enter race/podium inputs only.
Preprocessing, weights, calibration and thresholds were copied from development refits.
All qualifying forecasts were sealed before race inputs; all forecasts were sealed before scoring.

## Observed results in the supplied source

| Task | Method | Metric | Events | Mean | Event range | Leave-one-event-out mean range |
| --- | --- | --- | ---: | ---: | --- | --- |
| qualifying | model | position_mae | 4 | 2.8333 | 2.5714–3.4286 | 2.6349–2.9206 |
| qualifying | model | spearman | 4 | -0.4286 | -0.6786–-0.1786 | -0.5119–-0.3452 |
| qualifying | model | pole_accuracy | 4 | 0.0000 | 0.0000–0.0000 | 0.0000–0.0000 |
| qualifying | model | top3_overlap | 4 | 0.2500 | 0.0000–0.3333 | 0.2222–0.3333 |
| qualifying | baseline_recent | position_mae | 4 | 2.7500 | 2.3333–3.4286 | 2.5238–2.8889 |
| qualifying | baseline_recent | spearman | 4 | -0.4464 | -0.7143–-0.2143 | -0.5238–-0.3571 |
| qualifying | baseline_recent | pole_accuracy | 4 | 0.0000 | 0.0000–0.0000 | 0.0000–0.0000 |
| qualifying | baseline_recent | top3_overlap | 4 | 0.2500 | 0.0000–0.3333 | 0.2222–0.3333 |
| qualifying | baseline_practice | position_mae | 4 | 2.7500 | 2.3333–3.4286 | 2.5238–2.8889 |
| qualifying | baseline_practice | spearman | 4 | -0.4464 | -0.7143–-0.2143 | -0.5238–-0.3571 |
| qualifying | baseline_practice | pole_accuracy | 4 | 0.0000 | 0.0000–0.0000 | 0.0000–0.0000 |
| qualifying | baseline_practice | top3_overlap | 4 | 0.2500 | 0.0000–0.3333 | 0.2222–0.3333 |
| race | model | position_mae | 4 | 0.0000 | 0.0000–0.0000 | 0.0000–0.0000 |
| race | model | spearman | 4 | 1.0000 | 1.0000–1.0000 | 1.0000–1.0000 |
| race | model | winner_accuracy | 4 | 1.0000 | 1.0000–1.0000 | 1.0000–1.0000 |
| race | model | top3_overlap | 4 | 1.0000 | 1.0000–1.0000 | 1.0000–1.0000 |
| race | baseline_qualifying | position_mae | 4 | 0.0000 | 0.0000–0.0000 | 0.0000–0.0000 |
| race | baseline_qualifying | spearman | 4 | 1.0000 | 1.0000–1.0000 | 1.0000–1.0000 |
| race | baseline_qualifying | winner_accuracy | 4 | 1.0000 | 1.0000–1.0000 | 1.0000–1.0000 |
| race | baseline_qualifying | top3_overlap | 4 | 1.0000 | 1.0000–1.0000 | 1.0000–1.0000 |
| race | baseline_recent | position_mae | 4 | 2.7500 | 2.3333–3.4286 | 2.5238–2.8889 |
| race | baseline_recent | spearman | 4 | -0.4464 | -0.7143–-0.2143 | -0.5238–-0.3571 |
| race | baseline_recent | winner_accuracy | 4 | 0.0000 | 0.0000–0.0000 | 0.0000–0.0000 |
| race | baseline_recent | top3_overlap | 4 | 0.2500 | 0.0000–0.3333 | 0.2222–0.3333 |
| podium | model | log_loss | 4 | 0.0123 | 0.0122–0.0123 | 0.0123–0.0123 |
| podium | model | brier | 4 | 0.0001 | 0.0001–0.0002 | 0.0001–0.0002 |
| podium | model | average_precision | 4 | 1.0000 | 1.0000–1.0000 | 1.0000–1.0000 |
| podium | model | ece | 4 | 0.0122 | 0.0121–0.0123 | 0.0122–0.0122 |
| podium | model | precision | 4 | 1.0000 | 1.0000–1.0000 | 1.0000–1.0000 |
| podium | model | recall | 4 | 1.0000 | 1.0000–1.0000 | 1.0000–1.0000 |
| podium | baseline_qualifying | log_loss | 4 | 0.0000 | 0.0000–0.0000 | 0.0000–0.0000 |
| podium | baseline_qualifying | brier | 4 | 0.0000 | 0.0000–0.0000 | 0.0000–0.0000 |
| podium | baseline_qualifying | average_precision | 4 | 1.0000 | 1.0000–1.0000 | 1.0000–1.0000 |
| podium | baseline_qualifying | ece | 4 | 0.0000 | 0.0000–0.0000 | 0.0000–0.0000 |
| podium | baseline_qualifying | precision | 4 | 1.0000 | 1.0000–1.0000 | 1.0000–1.0000 |
| podium | baseline_qualifying | recall | 4 | 1.0000 | 1.0000–1.0000 | 1.0000–1.0000 |
| podium | baseline_recent | log_loss | 4 | 10.0582 | 8.2201–12.5056 | 9.2425–10.6709 |
| podium | baseline_recent | brier | 4 | 0.5381 | 0.4000–0.6857 | 0.4889–0.5841 |
| podium | baseline_recent | average_precision | 4 | 0.4349 | 0.3524–0.5000 | 0.4132–0.4624 |
| podium | baseline_recent | ece | 4 | 0.6310 | 0.4571–0.8000 | 0.5746–0.6889 |
| podium | baseline_recent | precision | 4 | 0.2500 | 0.0000–0.3333 | 0.2222–0.3333 |
| podium | baseline_recent | recall | 4 | 0.2500 | 0.0000–0.3333 | 0.2222–0.3333 |

Per-event metrics and complete predicted-versus-actual tables are CSV and Parquet.
Baselines are fixed references; the locked learned models are not claimed to beat them.
Podium probabilities are independent and are not normalized to sum to three.

## Coverage and uncertainty

Event/task coverage records: 12; full details are in coverage.csv.
Missing or mismatched official fields are excluded transparently per task/metric; no tail is invented.
Driver/team/condition summaries need at least three weekends; smaller groups are suppressed.
Wet/dry refers to observed target-session samples read only after prediction seals; absent weather is unknown.
Blocks below eight weekends get no confidence interval. Event ranges, SD and
leave-one-event-out mean ranges are sensitivity summaries, not confidence intervals or accuracy guarantees.

## Development comparisons and interpretation

development/ preserves all A (2026-only), B (pooled equal) and C (2026-favored) results,
eligibility, searches and feature-group comparisons. Primary strategy means use identical common events.
Early A folds below the configured earlier-2026 minimum are ineligible, not pooled.
Common development event counts: podium=1, qualifying=1, race=1. Small cohorts make family/weight selection weak evidence.
Optional practice/telemetry/weather groups were not promoted under their development evidence/complexity gates.
Calibration plots show raw versus delivered predictions; a lower Brier score alone does not establish calibration.
Historical reconstructions may contain later source revisions and cannot certify original publication-time knowledge.

These are generated fixture outcomes, not observed Formula 1 results. The generator deliberately
makes race order equal qualifying order; excellent race/reference scores are a fixture property.
No real 2026 accuracy, calibration or feature benefit follows. Real-data finalization remains
unavailable until a genuine reviewed dataset, cutoffs, rosters and frozen manifest exist.

## Hypotheses and next confirmation

Whether 2026 weighting, extra session inputs or calibration help on real events remains a hypothesis.
No choices are changed in response to these test results. Any later model/feature changes or test reuse
are exploratory and require a fresh future holdout for confirmation. The original manifest is unchanged.
