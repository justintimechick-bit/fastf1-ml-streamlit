# Race classification and podium development

Final test remains sealed. Classification includes official ordered retirement/
administrative outcomes; it is not a pure-pace forecast. Probabilities are independent
and are never rescaled to sum to three.

## Race common-event averages

Events: 2026-03

| Model | Strategy | Events | position_mae | spearman | winner_accuracy | top3_overlap |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ridge | 2026_only | 1 | 2.2857 | 0.1429 | 1.0000 | 0.3333 |
| ridge | equal | 1 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| ridge | weighted | 1 | 0.5714 | 0.8929 | 1.0000 | 1.0000 |
| random_forest | 2026_only | 1 | 2.0000 | 0.2500 | 1.0000 | 0.6667 |
| random_forest | equal | 1 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| random_forest | weighted | 1 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| gradient_boosting | 2026_only | 1 | 2.8571 | -0.3571 | 0.0000 | 0.3333 |
| gradient_boosting | equal | 1 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| gradient_boosting | weighted | 1 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| baseline_qualifying | fixed_rule | 1 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| baseline_recent | fixed_rule | 1 | 2.0000 | 0.1429 | 0.0000 | 0.6667 |

### Race per-event results

| Event | Model | Strategy | Common | position_mae | spearman | winner_accuracy | top3_overlap |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| 2026-01 | baseline_qualifying | fixed_rule | False | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| 2026-01 | baseline_recent | fixed_rule | False | 1.1429 | 0.6429 | 0.0000 | 0.6667 |
| 2026-01 | ridge | equal | False | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| 2026-01 | ridge | weighted | False | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| 2026-01 | random_forest | equal | False | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| 2026-01 | random_forest | weighted | False | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| 2026-01 | gradient_boosting | equal | False | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| 2026-01 | gradient_boosting | weighted | False | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| 2026-02 | baseline_qualifying | fixed_rule | False | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| 2026-02 | baseline_recent | fixed_rule | False | 1.0000 | 0.6571 | 1.0000 | 0.6667 |
| 2026-02 | ridge | equal | False | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| 2026-02 | ridge | weighted | False | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| 2026-02 | random_forest | equal | False | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| 2026-02 | random_forest | weighted | False | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| 2026-02 | gradient_boosting | equal | False | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| 2026-02 | gradient_boosting | weighted | False | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| 2026-03 | baseline_qualifying | fixed_rule | True | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| 2026-03 | baseline_recent | fixed_rule | True | 2.0000 | 0.1429 | 0.0000 | 0.6667 |
| 2026-03 | ridge | 2026_only | True | 2.2857 | 0.1429 | 1.0000 | 0.3333 |
| 2026-03 | ridge | equal | True | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| 2026-03 | ridge | weighted | True | 0.5714 | 0.8929 | 1.0000 | 1.0000 |
| 2026-03 | random_forest | 2026_only | True | 2.0000 | 0.2500 | 1.0000 | 0.6667 |
| 2026-03 | random_forest | equal | True | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| 2026-03 | random_forest | weighted | True | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| 2026-03 | gradient_boosting | 2026_only | True | 2.8571 | -0.3571 | 0.0000 | 0.3333 |
| 2026-03 | gradient_boosting | equal | True | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| 2026-03 | gradient_boosting | weighted | True | 0.0000 | 1.0000 | 1.0000 | 1.0000 |

## Podium common-event averages

Events: 2026-03

| Model | Strategy | Events | log_loss | brier | average_precision | ece | precision | recall | f1 | raw_log_loss | raw_brier | raw_ece |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| logistic | 2026_only | 1 | 0.6762 | 0.2383 | 0.7556 | 0.2353 | 0.6667 | 0.6667 | 0.6667 | 0.6762 | 0.2383 | 0.2353 |
| logistic | equal | 1 | 0.7331 | 0.2418 | 0.7556 | 0.3421 | 0.6667 | 0.6667 | 0.6667 | 0.6275 | 0.2207 | 0.3424 |
| logistic | weighted | 1 | 0.9520 | 0.3007 | 0.7556 | 0.3253 | 0.5000 | 0.6667 | 0.5714 | 0.7844 | 0.2714 | 0.2969 |
| random_forest | 2026_only | 1 | 0.6152 | 0.2230 | 0.7556 | 0.2686 | 0.6667 | 0.6667 | 0.6667 | 0.6152 | 0.2230 | 0.2686 |
| random_forest | equal | 1 | 0.8021 | 0.2839 | 0.7556 | 0.2459 | 0.6667 | 0.6667 | 0.6667 | 0.6842 | 0.2536 | 0.2127 |
| random_forest | weighted | 1 | 0.9542 | 0.3143 | 0.7556 | 0.2842 | 0.5000 | 0.3333 | 0.4000 | 0.7942 | 0.2866 | 0.2577 |
| gradient_boosting | 2026_only | 1 | 1.6668 | 0.5099 | 0.3968 | 0.5155 | 0.3333 | 0.3333 | 0.3333 | 1.6668 | 0.5099 | 0.5155 |
| gradient_boosting | equal | 1 | 0.0196 | 0.0004 | 1.0000 | 0.0194 | 1.0000 | 1.0000 | 1.0000 | 0.0259 | 0.0007 | 0.0256 |
| gradient_boosting | weighted | 1 | 0.0196 | 0.0004 | 1.0000 | 0.0194 | 1.0000 | 1.0000 | 1.0000 | 0.0259 | 0.0007 | 0.0256 |
| baseline_qualifying | fixed_rule | 1 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline_recent | fixed_rule | 1 | 8.0405 | 0.3314 | 0.5317 | 0.4000 | 0.6667 | 0.6667 | 0.6667 | 8.0405 | 0.3314 | 0.4000 |

### Podium per-event results

| Event | Model | Strategy | Common | log_loss | brier | average_precision | ece | precision | recall | f1 | raw_log_loss | raw_brier | raw_ece |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-01 | baseline_qualifying | fixed_rule | False | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| 2026-01 | baseline_recent | fixed_rule | False | 0.1980 | 0.0714 | 0.9167 | 0.0000 | 0.7500 | 1.0000 | 0.8571 | 0.1980 | 0.0714 | 0.0000 |
| 2026-01 | logistic | equal | False | 0.2361 | 0.0566 | 1.0000 | 0.1980 | 1.0000 | 1.0000 | 1.0000 | 0.2361 | 0.0566 | 0.1980 |
| 2026-01 | logistic | weighted | False | 0.2126 | 0.0505 | 1.0000 | 0.1783 | 1.0000 | 1.0000 | 1.0000 | 0.2126 | 0.0505 | 0.1783 |
| 2026-01 | random_forest | equal | False | 0.2183 | 0.0551 | 1.0000 | 0.1808 | 1.0000 | 1.0000 | 1.0000 | 0.2183 | 0.0551 | 0.1808 |
| 2026-01 | random_forest | weighted | False | 0.1700 | 0.0398 | 1.0000 | 0.1441 | 1.0000 | 1.0000 | 1.0000 | 0.1700 | 0.0398 | 0.1441 |
| 2026-01 | gradient_boosting | equal | False | 0.0345 | 0.0012 | 1.0000 | 0.0339 | 1.0000 | 1.0000 | 1.0000 | 0.0345 | 0.0012 | 0.0339 |
| 2026-01 | gradient_boosting | weighted | False | 0.0345 | 0.0012 | 1.0000 | 0.0339 | 1.0000 | 1.0000 | 1.0000 | 0.0345 | 0.0012 | 0.0339 |
| 2026-02 | baseline_qualifying | fixed_rule | False | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| 2026-02 | baseline_recent | fixed_rule | False | 0.3054 | 0.1200 | 0.9167 | 0.2000 | 0.6667 | 0.6667 | 0.6667 | 0.3054 | 0.1200 | 0.2000 |
| 2026-02 | logistic | equal | False | 0.2115 | 0.0516 | 1.0000 | 0.1757 | 1.0000 | 1.0000 | 1.0000 | 0.2115 | 0.0516 | 0.1757 |
| 2026-02 | logistic | weighted | False | 0.2020 | 0.0496 | 1.0000 | 0.1673 | 1.0000 | 1.0000 | 1.0000 | 0.2020 | 0.0496 | 0.1673 |
| 2026-02 | random_forest | equal | False | 0.1529 | 0.0274 | 1.0000 | 0.1363 | 1.0000 | 1.0000 | 1.0000 | 0.1529 | 0.0274 | 0.1363 |
| 2026-02 | random_forest | weighted | False | 0.1541 | 0.0317 | 1.0000 | 0.1344 | 1.0000 | 1.0000 | 1.0000 | 0.1541 | 0.0317 | 0.1344 |
| 2026-02 | gradient_boosting | equal | False | 0.0295 | 0.0009 | 1.0000 | 0.0291 | 1.0000 | 1.0000 | 1.0000 | 0.0295 | 0.0009 | 0.0291 |
| 2026-02 | gradient_boosting | weighted | False | 0.0296 | 0.0009 | 1.0000 | 0.0291 | 1.0000 | 1.0000 | 1.0000 | 0.0296 | 0.0009 | 0.0291 |
| 2026-03 | baseline_qualifying | fixed_rule | True | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| 2026-03 | baseline_recent | fixed_rule | True | 8.0405 | 0.3314 | 0.5317 | 0.4000 | 0.6667 | 0.6667 | 0.6667 | 8.0405 | 0.3314 | 0.4000 |
| 2026-03 | logistic | 2026_only | True | 0.6762 | 0.2383 | 0.7556 | 0.2353 | 0.6667 | 0.6667 | 0.6667 | 0.6762 | 0.2383 | 0.2353 |
| 2026-03 | logistic | equal | True | 0.7331 | 0.2418 | 0.7556 | 0.3421 | 0.6667 | 0.6667 | 0.6667 | 0.6275 | 0.2207 | 0.3424 |
| 2026-03 | logistic | weighted | True | 0.9520 | 0.3007 | 0.7556 | 0.3253 | 0.5000 | 0.6667 | 0.5714 | 0.7844 | 0.2714 | 0.2969 |
| 2026-03 | random_forest | 2026_only | True | 0.6152 | 0.2230 | 0.7556 | 0.2686 | 0.6667 | 0.6667 | 0.6667 | 0.6152 | 0.2230 | 0.2686 |
| 2026-03 | random_forest | equal | True | 0.8021 | 0.2839 | 0.7556 | 0.2459 | 0.6667 | 0.6667 | 0.6667 | 0.6842 | 0.2536 | 0.2127 |
| 2026-03 | random_forest | weighted | True | 0.9542 | 0.3143 | 0.7556 | 0.2842 | 0.5000 | 0.3333 | 0.4000 | 0.7942 | 0.2866 | 0.2577 |
| 2026-03 | gradient_boosting | 2026_only | True | 1.6668 | 0.5099 | 0.3968 | 0.5155 | 0.3333 | 0.3333 | 0.3333 | 1.6668 | 0.5099 | 0.5155 |
| 2026-03 | gradient_boosting | equal | True | 0.0196 | 0.0004 | 1.0000 | 0.0194 | 1.0000 | 1.0000 | 1.0000 | 0.0259 | 0.0007 | 0.0256 |
| 2026-03 | gradient_boosting | weighted | True | 0.0196 | 0.0004 | 1.0000 | 0.0194 | 1.0000 | 1.0000 | 1.0000 | 0.0259 | 0.0007 | 0.0256 |

## Eligibility and exclusions

| Task | Event | Strategy | Usable training events | Reason |
| --- | --- | --- | ---: | --- |
| race | 2026-01 | 2026_only | 0 | needs_2_usable_training_events |
| race | 2026-01 | equal | 4 | eligible |
| race | 2026-01 | weighted | 4 | eligible |
| race | 2026-02 | 2026_only | 1 | needs_2_usable_training_events |
| race | 2026-02 | equal | 5 | eligible |
| race | 2026-02 | weighted | 5 | eligible |
| race | 2026-03 | 2026_only | 2 | eligible |
| race | 2026-03 | equal | 6 | eligible |
| race | 2026-03 | weighted | 6 | eligible |
| podium | 2026-01 | 2026_only | 0 | needs_2_usable_training_events |
| podium | 2026-01 | equal | 4 | eligible |
| podium | 2026-01 | weighted | 4 | eligible |
| podium | 2026-02 | 2026_only | 1 | needs_2_usable_training_events |
| podium | 2026-02 | equal | 5 | eligible |
| podium | 2026-02 | weighted | 5 | eligible |
| podium | 2026-03 | 2026_only | 2 | eligible |
| podium | 2026-03 | equal | 6 | eligible |
| podium | 2026-03 | weighted | 6 | eligible |

Actual status and target exclusions are separate in target_coverage.parquet and
evaluation_labels.parquet; these fields never enter prediction inputs. Calibration
bins (raw and delivered probabilities) are in calibration.parquet. Thresholds and
calibration eligibility are recorded per model. Selected bundles are best learned
models; fixed rules may outperform them. Development selection is not a final-test
estimate. Small samples do not establish calibration or real F1 accuracy.
