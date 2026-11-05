# Controlled optional-data experiments (Phase 6)

This opt-in path preserves the frozen Phase 3 feature implementation and all
working core models/artifacts. It writes separate supplements and experimental
bundles. [Core methodology](methodology.md) and the [race/podium protocol](race-podium.md)
still apply. No final-test feature generation or evaluation is permitted.

## Selective capture and provenance

`telemetry-download` accepts explicit CSV rows:
`event_id,task,kind,cutoff,mode`. Every event must be in the supplied manifest's
development block. Only GP practice, plus GP qualifying for the race task, is
allowed. Sprint, target qualifying/race and late sessions are rejected. Session
completion must be observed strictly before cutoff; neither scheduled end nor
download time is relabeled as completion. Publication mode also checks retrieval.

Defaults: at most four explicitly selected sessions, two bounded attempts and
three retained clean laps per driver/session. The public FastF1 interface loads
car **and position payloads for all drivers in the selected session** upstream;
there is no per-lap network-download claim. Selectivity is by session and retained
laps. Position is cached by FastF1 but not exported/modeled. Persistent cache is
enabled and shares the existing schedule cache. Use public `Session.load` and
`Lap.get_car_data(interpolate_edges=False)` only for acquisition.

Select earliest clean laps by session time/lap number, not target performance.
The Phase 3 filter requires accurate, undeleted, non-pit, non-generated laps,
clear track status and positive times where source flags permit. Preserve full
session lap metadata/weather; retain car samples only for selected laps. Unknown
driver identity or completion is a visible failure. No field list is hardcoded.

Immutable capture revisions retain source/session identity, retrieval/version,
completion state, columns/dtypes/missingness, raw Parquet, failures/warnings and
runtime. Successful/partial captures are reused; failures observe a 24-hour
cooldown. `--retry-failed` explicitly retries. Eligibility is rechecked on cache
reuse. A capture from a source different from the guard manifest is marked
**schema-audit-only** and cannot enter an experiment; matching event IDs do not
permit joining real sources to synthetic fixtures. Supplement construction binds
to the exact frozen source and core feature snapshot.

## Channels, formulas and missingness

Verified installed FastF1 3.8.1 signatures/docstrings and official
[core](https://github.com/theOehrly/Fast-F1/blob/main/fastf1/core.py) and
[parser](https://github.com/theOehrly/Fast-F1/blob/main/fastf1/_api.py) documentation.
The hosted documentation returned HTTP 403. See the [interface audit](telemetry-interface-check.json).
The installed parser defaults absent wire channel `45` to zero and notes its
2026 absence. A DRS column/default is not proof of a comparable observed channel.
**Exclude DRS in every season.** Do not infer active aero, energy, battery, fuel
load or deployment from speed/throttle patterns.

Use native `Source=car` samples and elapsed `SessionTime`, with no resampling,
merged position telemetry, padding or edge interpolation. Reject duplicate
timestamps. Default valid intervals are `0 < dt <= 1 second`; each channel has
its own valid-time denominator and coverage relative to lap duration. Below 80%
coverage its measurement is missing. Throttle outside 0–100, including sentinel
104, is invalid; brake is boolean, not pressure.

| Feature | Formula/units |
| --- | --- |
| Mean speed | km/h; trapezoidal `sum(dt*(v_left+v_right)/2)/valid_speed_seconds` |
| Speed p95 | km/h; interval-duration-weighted left-sample 95th percentile |
| Full throttle | `sum(dt*[Throttle_left >= configured threshold])/valid_throttle_seconds`, default threshold 99% |
| Braking | `sum(dt*[Brake_left=True])/valid_brake_seconds` |
| Lap/sector consistency | population standard deviation / mean of positive seconds, within a matched cell with at least two driver laps |
| Teammate difference | own minus same-event team peers' cell mean speed/throttle/brake summary |

Trapezoidal speed assumes linear change between adjacent samples; discrete
states are held from the left. These are integration assumptions, not added
observations. Driver telemetry features are equal-lap means: a mean lap fraction
is not a pooled whole-session fraction. Counts/coverage accompany measurements.
Missing channels stay missing; counts are zero. Imputation, encoding and scaling
fit only on the corresponding training fold.

Comparisons require matching session, Q segment, compound, five-minute time cell,
rain and five-degree track-temperature cell, with at least two drivers. Never
pool Q1/Q3 or unrelated circuits' raw lap/sector times. Compute consistency within
each driver's matched cell, then average. Missing conditions, sectors or matching
teammates leave the relevant metric missing. Speed also reflects circuit geometry,
setup and run plans; unknown fuel, traffic and tyre preparation remain confounders.

## Weather and feature groups

Use observed weather only from completed eligible practice, plus GP qualifying
for race/podium. Never read eventual target qualifying/race weather. **No verified
historical forecasts issued before cutoff were obtained; race-weather forecasts
are not included.** Practice weather is not a forecast of race conditions.

Explicit weather features require verified per-sample UTC times. Capture derives
`observed_at` from public `Session.t0_date + weather.Time` and filters samples
beyond completion. Feature generation also clips to completion and cutoff. If
an older weather table has only relative time and no verified origin, explicit
weather features stay missing; it may still condition backward-matched lap cells.

Weather means and rain fractions weight adjacent observed intervals no more
than 120 seconds apart, holding left values without final-sample extrapolation.
One numeric sample can remain an observation; one rain sample has no duration
and no rain fraction. Average session summaries equally and retain sample/session
counts. Weather used to match pace conditions already affects earlier groups;
the last group tests explicit weather values, not removal of that quality logic.

| Cumulative group | Inputs |
| --- | --- |
| `history` | Shifted driver/team/circuit histories, counts, identity, field size, season/round/era |
| `sessions` | Practice participation and matched best-lap ranks; actual qualifying classification/evidenced grid for race/podium only |
| `practice` | Detailed practice gaps/counts and lap/sector CV; segment-specific qualifying pace for race/podium |
| `telemetry` | Speed/throttle/brake aggregates, counts/coverage and matched teammate differences |
| `weather` | Observed air/track temperature, humidity, wind, rain fraction/counts |

Also retain a `core` reference containing exactly the original Phase 3 matrix.
New observations never rewrite/fill core columns in place. Each supplement has
`feature_dictionary.json` with every formula and exact task/group allowlist.

## Paired evaluation and decisions

Defaults compare all three existing families on qualifying, race and podium,
using equal-event 2024–2026 training. A/C strategies can be enabled under
`[experiments].strategies`; their prior eligibility rules remain. Within each
task/family/strategy, all groups use identical complete evaluation events,
training IDs and full entrant fields. Optional availability never chooses a
rich-data-only cohort. Do not compare different strategy cohorts as a controlled
strategy result.

Freeze settings across groups: Ridge alpha 10/logistic C 1 if configured, otherwise
the first grid value; first forest depth/boosting rate, configured tree/iteration
counts. C uses its first predeclared season/half-life profile. No group-specific
search is performed. Podium calibration and thresholds use only earlier eligible
chronological OOF folds under the Phase 5 minimums. This initial experiment
isolates feature changes; it does not claim the best tuned model in each group.

Save all task metrics per event, predecessor/core paired loss differences,
feature/row missingness, observed coverage and fit/transform/postprocessing/runtime.
Supplement coverage records aggregation time; capture receipts record acquisition
time and raw availability. Local timing is noisy and warrants repeat measurement.

Advisory retention defaults require three paired events, improvement of 0.05
positions or 0.005 log loss against **both predecessor and core**, at least 25%
observed new-input coverage, runtime at most 3× core and genuine observed-source
evidence. Synthetic data cannot recommend promotion. A group that only inherits
a beneficial predecessor but worsens it fails. No artifact is auto-promoted or
overwritten. These exploratory gates do not establish statistical significance;
freeze a subsequent protocol before any final test.

## Commands

Use reviewed explicit development requests (header: `examples/telemetry_requests.csv`):

```sh
f1ml --config configs/default.toml telemetry-download \
  --dataset <immutable-source> --manifest <manifest.json> \
  --requests <selected-sessions.csv> --output data/telemetry-pilot
# Resume the same command; add --retry-failed only for an explicit retry.
```

Capture exits 1 for partial/failed/ineligible requests and 2 for invalid inputs.
`index.json` and `captures/<key>/<revision>` expose schemas, errors and receipts.

Runnable synthetic core/missing-data comparison, with no fabricated telemetry:

```sh
f1ml demo --output data/optional-demo
f1ml build-experiments --dataset data/optional-demo/dataset \
  --manifest data/optional-demo/manifest.json --features data/optional-demo/features \
  --telemetry data/optional-telemetry --output data/optional-features
f1ml --config configs/default.toml experiment-compare \
  --manifest data/optional-demo/manifest.json --features data/optional-demo/features \
  --optional data/optional-features --output data/feature-experiments
cat data/feature-experiments/report.md
cat data/feature-experiments/retention.json
```

Use matching genuine source/core paths and populated captures for real studies;
review overlays remain supported. Choose new output paths for reruns. Bundles
record group/version/code specifications, source/core/optional hashes, training
and postprocessing events, cutoffs and prediction model IDs. Experimental bundles
need matching augmented inputs; ordinary core prediction commands continue to
use the core contract. See [validation](experiments-validation.md) for actual live
failures and completed checks. No optional group has yet earned promotion.
