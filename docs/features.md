# Feature generation, schema 3.0.0

`build-features` creates separate qualifying and race Parquet datasets, one row
per reviewed driver/event/task. Podium prediction shares the race information
cutoff. Labels are separate files. Optional timing coverage never determines
which entrants survive in the core dataset.

## Run and reproduce

The existing offline example now exercises the new builder and fold-local models:

```sh
f1ml demo --output data/features-demo-v3
cat data/features-demo-v3/features/feature_dictionary.md
```

For a previously frozen Phase 1 dataset, use the normal commands with reviewed
entrant/cutoff CSVs. For a Phase 2 source, use an **immutable snapshot path** from
`CURRENT.json`, not the mutable catalog directory. Review completion/availability
and circuit context before freezing the experiment:

```sh
f1ml --config configs/default.toml freeze \
  --dataset data/ingestion-v2/snapshots/SNAPSHOT_ID \
  --availability data/session_availability.csv --event-context data/event_context.csv \
  --output data/manifest-v3.json
f1ml build-features --dataset data/ingestion-v2/snapshots/SNAPSHOT_ID \
  --availability data/session_availability.csv --event-context data/event_context.csv \
  --manifest data/manifest-v3.json --entrants data/entrants-v3.csv \
  --cutoffs data/cutoffs-v3.csv --output data/features-v3
f1ml backtest --manifest data/manifest-v3.json --features data/features-v3 \
  --output artifacts/baseline-v3
```

`SNAPSHOT_ID` is a placeholder for the actual snapshot directory. CSV headers are
in `examples/`; the demo produces complete synthetic entrant/cutoff examples.
Review files are part of the source identity: changing either requires a new
manifest. No absolute completion time is inferred from a schedule, retrieval
time, or race outcome. An optional reviewed `started_at` checks an unexpectedly
early session start. Missing optional practice timing leaves core rows intact;
unverified mandatory GP completion/availability may make an event ineligible.
The final block still requires configurable feasibility and stays sealed.

Every feature artifact includes `source_binding.json`, copied sorted inputs,
`coverage.parquet`, source-use audits, the feature dictionary, dependency versions
and a source-code fingerprint. Output creation is exclusive. Rebuilding with the
same source, reviews, manifest, inputs, code and environment produces the same
receipt ID; no generation timestamp or random imputation enters the features.

Generate the dictionary independently with:

```sh
f1ml feature-dictionary --output data/dictionary-v3
```

The generated [dictionary](feature_dictionary.md) documents every model feature's
source, formula, cutoff, missing policy and regulation considerations; a
[JSON version](feature_dictionary.json) is also available. The dictionary registry
is the model-column allowlist, rather than a separately maintained list.

## Historical features and the event shift

First select eligible development weekends **strictly before the target weekend**,
whose race has finished before both the target weekend's first session and the
chosen cutoff. Then select the last 3 or 5 of those calendar weekends. This is an
explicit event shift before aggregation; a target row never enters its own
window. If a driver missed one of those weekends, it stays a missing observation
instead of extending the window to an earlier driver appearance.

Official GP qualifying and race ranks use `(position - 1)/(N - 1)`, with `N` the
actual complete official result field for that session. Rank 1 becomes 0 and the
last rank becomes 1. Missing, repeated, nonintegral or out-of-range positions do
not become invented ranks. Incomplete fields have unavailable normalized ranks.
Administrative DNS/DSQ tail positions are retained when officially supplied;
these features describe classification, not pure on-track speed. No raw lap time
from one circuit is compared with another.

Driver features include previous-calendar-weekend rank, 3/5-weekend rank means,
known podium and retirement rates, with observation counts and missing flags.
Retirement comes from the explicit normalized `is_retired` flag: DNS and DSQ are
not automatically retirements, and a classified retirement can have both a rank
and a true retirement flag. Unknown flags do not become zeros.

Teammate deltas compare the driver's normalized rank with the mean of **all other
valid entrants in that team/history era in that session**, without assuming two
drivers. Recent teammate deltas and team strengths use current-season results
only. Team strength is the mean of per-weekend team mean ranks, so weekends get
equal weight rather than teams with more recorded drivers dominating.

Older-season team priors are separate features, using at most the five most
recent older eligible weekends and the **same explicitly reviewed history ID**.
New or renamed performance eras have no inherited prior by default. The fallback
history ID is year-isolated, leaving prior counts zero. Driver rolling history
may span 2024–2026, so `season`, `round` and `regulation_2026` remain explicit;
field normalization does not remove regulation changes or equipment effects.

Same-circuit priors average only the driver's earlier normalized results at a
verified circuit/layout ID, with counts. Circuit identity is never guessed from
the target result, raw lap times or a fuzzy event name. No verified identity
means `unknown` plus missing circuit priors. Small counts should be interpreted
as sparse historical evidence, not stable circuit-specific skill estimates.

## Current-weekend practice and qualifying pace

Session selection uses UTC timestamps and completion/availability evidence,
regardless of weekday or practice numbering. Sprint Qualifying and Sprint remain
distinct identities but contribute no outcome, pace or historical observation.

Practice retains only positive timed laps with known accuracy, explicitly
nondeleted flags, no known generated-lap flag, known pit-in/out columns with no
pit entry/exit, a resolved driver, and clear track status `1`. Other/unknown track
statuses are excluded conservatively, including yellow/red/SC/VSC disruption.
Source flags cannot establish whether the whole run was representative.

Relative pace cells are defined within **one session and qualifying part**, the
same compound, a configurable time bin (default 300 seconds), rainfall state and
track-temperature bin (default 5°C). Match weather backward to lap start with a
maximum age of 120 seconds. Require at least two distinct drivers in a cell.
Calculate `100 * (lap / fastest retained cell lap - 1)`, take the driver's median
within each session, then average practice session estimates equally. Counts
report clean laps, comparable laps, observed practice sessions and pace sessions.

Unknown weather, compound, timing or qualifying segment produces missing pace,
not a comparison across incomparable laps. Legacy best-lap-only practice summaries
can establish driver participation but cannot establish condition-matched pace.
Unknown fuel loads, engine settings, tyre preparation, traffic, evolving track
conditions and team run plans remain limitations even inside these cells.

The race dataset alone contains normalized **actual GP qualifying
classification**, plus separate Q1/Q2/Q3 pace deficits and lap counts. The ingestion
adapter records qualifying parts through the installed FastF1 public
`Laps.split_qualifying_sessions()` when available; it never assumes elimination
counts. Existing unsegmented source records remain missing for those pace
features until valid segmentation evidence is available. Q1 and Q3 raw times are
never pooled or treated as identical conditions.

Raw `qualifying_position` and independently evidenced `known_grid_position` are
race-only metadata columns, excluded from the model matrix in favor of normalized
rank features. Known pit-lane starts have a separate flag. Grid provenance/time
must meet the race cutoff; target race results are never used to recover a grid.

## Missing data, fitting and leakage checks

Rookies and substitutes retain their rows, with zero history counts and NaN
historical means when appropriate. Missing practice/pace never removes an event
or entrant. Absent target labels remain masked for fitting; an unexpected result
entrant outside the reviewed cutoff roster requires a roster review instead of
silently adding a driver using future information.

Feature generation performs no learned imputation, encoding or scaling. Each
chronological training fold fits median imputation with missing indicators,
scaling and categorical encoding on its own rows. Unseen driver/team/circuit
categories are handled by the fitted encoder. The Ridge baseline now trains on
normalized rank targets; official positions remain the evaluation targets.

Tests mutate future results, target Q results, target race results, optional
weather and disrupted laps; verify shifted windows, early cutoffs, sparse driver
appearances, team-era separation and reproducible artifacts; and check fold-only
preprocessing and unseen categories. Fold construction and manifest loading both
reject overlapping train/validation/test event IDs. The final-test block cannot
be used to build features or enter training history in this phase.

Historical reconstructions can still contain later revisions to otherwise
earlier sessions. `publication_snapshot` additionally requires the exact source
revision to have been retrieved by cutoff. A reviewed time does not turn a
retrospectively downloaded revision into a true publication-time snapshot.

Old artifacts and datasets are preserved. Legacy matrices remain readable by
legacy models; new v3 models use the new schema/fingerprint and require retraining
when the implementation changes. The original `history_window` setting is retained
for compatibility; v3 explicitly supplies both required windows, 3 and 5.
