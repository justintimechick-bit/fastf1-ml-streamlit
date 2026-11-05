# Qualifying development comparison

Status: `development_only_final_test_sealed`. Final test remains sealed.

Common evaluation events: 2026-03

## Event-level averages on the common cohort

| Model | Strategy | Events | Position MAE | Spearman | Pole | Top 3 | Top 10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ridge | 2026_only | 1 | 2.8571 | -0.2500 | 0.0000 | 0.3333 | 1.0000 |
| ridge | equal | 1 | 2.5714 | -0.1071 | 0.0000 | 0.3333 | 1.0000 |
| ridge | weighted | 1 | 2.8571 | -0.2857 | 0.0000 | 0.3333 | 1.0000 |
| random_forest | 2026_only | 1 | 2.8571 | -0.2857 | 0.0000 | 0.3333 | 1.0000 |
| random_forest | equal | 1 | 2.8571 | -0.2857 | 0.0000 | 0.3333 | 1.0000 |
| random_forest | weighted | 1 | 2.8571 | -0.2500 | 0.0000 | 0.3333 | 1.0000 |
| gradient_boosting | 2026_only | 1 | 2.8571 | -0.3571 | 0.0000 | 0.3333 | 1.0000 |
| gradient_boosting | equal | 1 | 2.8571 | -0.2500 | 0.0000 | 0.3333 | 1.0000 |
| gradient_boosting | weighted | 1 | 2.8571 | -0.2500 | 0.0000 | 0.3333 | 1.0000 |
| baseline_recent | fixed_rule | 1 | 2.0000 | 0.1429 | 0.0000 | 0.6667 | 1.0000 |
| baseline_practice | fixed_rule | 1 | 2.0000 | 0.1429 | 0.0000 | 0.6667 | 1.0000 |

MAE is in positions. Pole and overlaps are fractions. Top-k overlap uses min(k, N).
The selected bundle is the best learned regressor; a fixed rule can outperform it.
Development selection is not an unbiased final-test estimate.

## Per-event results (including additional eligible development folds)

| Event | Model | Strategy | Common cohort | Position MAE | Spearman | Pole | Top 3 | Top 10 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2026-01 | baseline_recent | fixed_rule | False | 1.1429 | 0.6429 | 0.0000 | 0.6667 | 1.0000 |
| 2026-01 | baseline_practice | fixed_rule | False | 1.1429 | 0.6429 | 0.0000 | 0.6667 | 1.0000 |
| 2026-01 | ridge | equal | False | 0.2857 | 0.9643 | 1.0000 | 0.6667 | 1.0000 |
| 2026-01 | ridge | weighted | False | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 2026-01 | random_forest | equal | False | 0.2857 | 0.9643 | 0.0000 | 1.0000 | 1.0000 |
| 2026-01 | random_forest | weighted | False | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 2026-01 | gradient_boosting | equal | False | 0.5714 | 0.8929 | 0.0000 | 1.0000 | 1.0000 |
| 2026-01 | gradient_boosting | weighted | False | 0.5714 | 0.8929 | 0.0000 | 1.0000 | 1.0000 |
| 2026-02 | baseline_recent | fixed_rule | False | 1.0000 | 0.6571 | 1.0000 | 0.6667 | 1.0000 |
| 2026-02 | baseline_practice | fixed_rule | False | 1.0000 | 0.6571 | 1.0000 | 0.6667 | 1.0000 |
| 2026-02 | ridge | equal | False | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 2026-02 | ridge | weighted | False | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 2026-02 | random_forest | equal | False | 0.3333 | 0.9429 | 1.0000 | 1.0000 | 1.0000 |
| 2026-02 | random_forest | weighted | False | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 2026-02 | gradient_boosting | equal | False | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 2026-02 | gradient_boosting | weighted | False | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 2026-03 | baseline_recent | fixed_rule | True | 2.0000 | 0.1429 | 0.0000 | 0.6667 | 1.0000 |
| 2026-03 | baseline_practice | fixed_rule | True | 2.0000 | 0.1429 | 0.0000 | 0.6667 | 1.0000 |
| 2026-03 | ridge | 2026_only | True | 2.8571 | -0.2500 | 0.0000 | 0.3333 | 1.0000 |
| 2026-03 | ridge | equal | True | 2.5714 | -0.1071 | 0.0000 | 0.3333 | 1.0000 |
| 2026-03 | ridge | weighted | True | 2.8571 | -0.2857 | 0.0000 | 0.3333 | 1.0000 |
| 2026-03 | random_forest | 2026_only | True | 2.8571 | -0.2857 | 0.0000 | 0.3333 | 1.0000 |
| 2026-03 | random_forest | equal | True | 2.8571 | -0.2857 | 0.0000 | 0.3333 | 1.0000 |
| 2026-03 | random_forest | weighted | True | 2.8571 | -0.2500 | 0.0000 | 0.3333 | 1.0000 |
| 2026-03 | gradient_boosting | 2026_only | True | 2.8571 | -0.3571 | 0.0000 | 0.3333 | 1.0000 |
| 2026-03 | gradient_boosting | equal | True | 2.8571 | -0.2500 | 0.0000 | 0.3333 | 1.0000 |
| 2026-03 | gradient_boosting | weighted | True | 2.8571 | -0.2500 | 0.0000 | 0.3333 | 1.0000 |

## Eligibility

| Event | Strategy | Training events | Reason |
| --- | --- | ---: | --- |
| 2026-01 | 2026_only | 0 | needs_2_usable_training_events |
| 2026-01 | equal | 4 | eligible |
| 2026-01 | weighted | 4 | eligible |
| 2026-02 | 2026_only | 1 | needs_2_usable_training_events |
| 2026-02 | equal | 5 | eligible |
| 2026-02 | weighted | 5 | eligible |
| 2026-03 | 2026_only | 2 | eligible |
| 2026-03 | equal | 6 | eligible |
| 2026-03 | weighted | 6 | eligible |

See tuning.json and each model.json for inner folds, fixed defaults, parameters,
weights and information dependencies. Raw scores remain in predictions.parquet.
Scores are ranking estimates, not probabilities or lap-time forecasts.
