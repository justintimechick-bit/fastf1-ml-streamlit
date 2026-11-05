# Controlled feature experiments

Data kind: synthetic.
Final test remains sealed. Core models and artifacts are preserved.

Every group uses identical complete events within each task/family/strategy.
Fixed settings; calibration and thresholds use earlier development OOF only.
Runtime includes fitting, transforms and postprocessing; downloads are separate.
Optional coverage and capture receipts record aggregation and acquisition costs.

| Task | Family | Strategy | Group | Events | Position MAE | Log loss | Seconds | Missing | Observed |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| qualifying | linear | equal | core | 3 | 0.8571 | — | 0.030 | 0.077 | 1.000 |
| qualifying | linear | equal | history | 3 | 0.8571 | — | 0.025 | 0.069 | 1.000 |
| qualifying | linear | equal | sessions | 3 | 0.8571 | — | 0.035 | 0.080 | 0.000 |
| qualifying | linear | equal | practice | 3 | 0.8571 | — | 0.030 | 0.115 | 0.000 |
| qualifying | linear | equal | telemetry | 3 | 0.8571 | — | 0.038 | 0.186 | 0.000 |
| qualifying | linear | equal | weather | 3 | 0.8571 | — | 0.026 | 0.222 | 0.000 |
| qualifying | random_forest | equal | core | 3 | 1.1587 | — | 0.052 | 0.077 | 1.000 |
| qualifying | random_forest | equal | history | 3 | 1.1587 | — | 0.052 | 0.069 | 1.000 |
| qualifying | random_forest | equal | sessions | 3 | 1.1587 | — | 0.050 | 0.080 | 0.000 |
| qualifying | random_forest | equal | practice | 3 | 1.1587 | — | 0.050 | 0.115 | 0.000 |
| qualifying | random_forest | equal | telemetry | 3 | 1.1587 | — | 0.049 | 0.186 | 0.000 |
| qualifying | random_forest | equal | weather | 3 | 1.1587 | — | 0.049 | 0.222 | 0.000 |
| qualifying | gradient_boosting | equal | core | 3 | 1.1429 | — | 0.185 | 0.077 | 1.000 |
| qualifying | gradient_boosting | equal | history | 3 | 1.1429 | — | 0.178 | 0.069 | 1.000 |
| qualifying | gradient_boosting | equal | sessions | 3 | 1.1429 | — | 0.168 | 0.080 | 0.000 |
| qualifying | gradient_boosting | equal | practice | 3 | 1.1429 | — | 0.178 | 0.115 | 0.000 |
| qualifying | gradient_boosting | equal | telemetry | 3 | 1.1429 | — | 0.160 | 0.186 | 0.000 |
| qualifying | gradient_boosting | equal | weather | 3 | 1.1429 | — | 0.174 | 0.222 | 0.000 |
| race | linear | equal | core | 3 | 0.3810 | — | 0.036 | 0.107 | 1.000 |
| race | linear | equal | history | 3 | 0.8571 | — | 0.028 | 0.069 | 1.000 |
| race | linear | equal | sessions | 3 | 0.3810 | — | 0.029 | 0.086 | 1.000 |
| race | linear | equal | practice | 3 | 0.3810 | — | 0.027 | 0.138 | 0.000 |
| race | linear | equal | telemetry | 3 | 0.3810 | — | 0.027 | 0.197 | 0.000 |
| race | linear | equal | weather | 3 | 0.3810 | — | 0.026 | 0.227 | 0.000 |
| race | random_forest | equal | core | 3 | 0.0000 | — | 0.052 | 0.107 | 1.000 |
| race | random_forest | equal | history | 3 | 1.1587 | — | 0.053 | 0.069 | 1.000 |
| race | random_forest | equal | sessions | 3 | 0.0000 | — | 0.053 | 0.086 | 1.000 |
| race | random_forest | equal | practice | 3 | 0.0000 | — | 0.052 | 0.138 | 0.000 |
| race | random_forest | equal | telemetry | 3 | 0.0000 | — | 0.052 | 0.197 | 0.000 |
| race | random_forest | equal | weather | 3 | 0.0000 | — | 0.051 | 0.227 | 0.000 |
| race | gradient_boosting | equal | core | 3 | 0.0000 | — | 0.152 | 0.107 | 1.000 |
| race | gradient_boosting | equal | history | 3 | 1.1429 | — | 0.165 | 0.069 | 1.000 |
| race | gradient_boosting | equal | sessions | 3 | 0.0000 | — | 0.154 | 0.086 | 1.000 |
| race | gradient_boosting | equal | practice | 3 | 0.0000 | — | 0.150 | 0.138 | 0.000 |
| race | gradient_boosting | equal | telemetry | 3 | 0.0000 | — | 0.170 | 0.197 | 0.000 |
| race | gradient_boosting | equal | weather | 3 | 0.0000 | — | 0.160 | 0.227 | 0.000 |
| podium | linear | equal | core | 3 | — | 0.2531 | 0.058 | 0.107 | 1.000 |
| podium | linear | equal | history | 3 | — | 0.7334 | 0.059 | 0.069 | 1.000 |
| podium | linear | equal | sessions | 3 | — | 0.2531 | 0.088 | 0.086 | 1.000 |
| podium | linear | equal | practice | 3 | — | 0.2531 | 0.062 | 0.138 | 0.000 |
| podium | linear | equal | telemetry | 3 | — | 0.2531 | 0.059 | 0.197 | 0.000 |
| podium | linear | equal | weather | 3 | — | 0.2531 | 0.058 | 0.227 | 0.000 |
| podium | random_forest | equal | core | 3 | — | 0.3911 | 0.107 | 0.107 | 1.000 |
| podium | random_forest | equal | history | 3 | — | 0.5687 | 0.108 | 0.069 | 1.000 |
| podium | random_forest | equal | sessions | 3 | — | 0.3654 | 0.108 | 0.086 | 1.000 |
| podium | random_forest | equal | practice | 3 | — | 0.4088 | 0.110 | 0.138 | 0.000 |
| podium | random_forest | equal | telemetry | 3 | — | 0.4419 | 0.112 | 0.197 | 0.000 |
| podium | random_forest | equal | weather | 3 | — | 0.4128 | 0.140 | 0.227 | 0.000 |
| podium | gradient_boosting | equal | core | 3 | — | 0.0279 | 0.325 | 0.107 | 1.000 |
| podium | gradient_boosting | equal | history | 3 | — | 0.5988 | 0.425 | 0.069 | 1.000 |
| podium | gradient_boosting | equal | sessions | 3 | — | 0.0279 | 0.291 | 0.086 | 1.000 |
| podium | gradient_boosting | equal | practice | 3 | — | 0.0279 | 0.400 | 0.138 | 0.000 |
| podium | gradient_boosting | equal | telemetry | 3 | — | 0.0279 | 0.292 | 0.197 | 0.000 |
| podium | gradient_boosting | equal | weather | 3 | — | 0.0279 | 0.263 | 0.227 | 0.000 |

retention.json requires paired improvement versus both predecessor and core,
observed coverage, live evidence and runtime budget. No group is auto-promoted.
Synthetic metrics are software checks only. No pre-cutoff forecasts were obtained.
