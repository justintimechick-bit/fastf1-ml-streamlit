# Methodology and test protocol

The opt-in [Phase 6 experiment protocol](experiments.md) adds bounded raw
telemetry capture and paired practice/telemetry/observed-weather ablations.
Core features/models and the sealed final block remain independent; optional
groups require measured development benefit, coverage and acceptable complexity.

## Scope and targets

Use only 2024–2026 GP weekends, discovered dynamically. Exclude preseason testing.
Preserve Sprint Qualifying (including its former Sprint Shootout name), Sprint,
GP Qualifying and GP Race as distinct sessions. No Sprint observations enter the
initial models, including historical features.

The qualifying target is the official qualifying classification, using
`position` from FastF1's Jolpica/Ergast-compatible qualifying result endpoint.
It is **not the starting grid**. Grid penalties and pit-lane starts can change
the grid without changing qualifying position. Do not rank Q1/Q2/Q3 lap times to
reconstruct official positions, infer elimination counts, or use FastF1's
timing-derived result fallback as official labels.

The race target preserves the result provider's official `position` order.
Retirements can have a numeric classification. Retain the official ordered tail
when the provider supplies one, including an administrative position for a DNS,
DSQ or non-classified entry. Preserve `positionText` and normalized outcome
separately. If a position is missing, zero, negative or nonintegral, leave the
rank target missing; never manufacture a tail order, infer position from laps,
or drop the driver. Qualifying follows the same missing-position policy.

The podium target is one for official race positions 1–3; explicitly DNS, DSQ,
excluded and non-classified outcomes are zero. An entry with an unknown result
and no position has an unknown podium target, not an assumed negative.
Contradictory/incomplete podium data makes a weekend ineligible pending review.

The [Phase 4 qualifying protocol](qualifying.md) adds three regression families,
fixed ranking rules and nested chronological selection. It uses normalized
qualifying rank, complete-field permutations and common-event averages; events
with incomplete official qualifying order remain reported as ineligible for
full-order comparison. The frozen final test remains sealed.

The [Phase 5 race/podium protocol](race-podium.md) adds official-classification
regressors, marginal podium classifiers, chronological OOF calibration and
development-only threshold selection. It preserves the target/status policy
above and reports finishing-order top three separately from classifier top three.

## Information cutoffs

All timestamps require a timezone and are normalized to UTC. Cutoffs are explicit
CSV records, not inferred from the eventual outcome. Require the cutoff to be
strictly before the earlier of scheduled and observed target-session start. This
conservative rule also handles a revised schedule or an unexpectedly early start.

| Predictor | Permitted target-weekend inputs | Forbidden target-weekend inputs |
| --- | --- | --- |
| Qualifying | Reviewed entrants/teams known by cutoff; completed practice available strictly before cutoff | GP qualifying results, times, classification, grid from later results; all race/Sprint observations |
| Race and podium | The same prior information, completed practice and official GP qualifying available before cutoff; independently evidenced known grid | Target race laps, telemetry, weather, pit stops, results, points, retirement/DNS/DSQ status; Sprint observations |

Race cutoff must follow available official GP qualifying and precede race start.
Grid is optional and missing by default. Accept it only from an external snapshot
with `grid_known_at <= cutoff` and a provenance source. Never obtain it from
`Race.results.GridPosition`. Grid changes announced later stay unavailable.

History uses an explicit event shift: only prior eligible development weekends
whose races finished before the target weekend's first session and cutoff enter
the previous 3/5-calendar-weekend windows. Missing appearances do not extend the
window. Normalize rank by the actual official session field, retain counts and
missingness, and keep current-season team/teammate form separate from reviewed
older team priors. Circuit priors require stable circuit/layout identity.

Current-weekend practice uses only cutoff-available completed sessions and clean
laps, excluding pit, invalid/deleted/generated or disrupted/unknown-track-status
observations. Deficits are computed within matching session, qualifying segment,
compound, time, rainfall and temperature cells; raw times are never pooled across
circuits or Q1/Q3. Unknown conditions leave pace missing while core rows remain.
Only the race predictor receives actual GP qualifying classification/pace.
Fuel loads and run plans remain unknown. No target race observation enters a
predictor. See [feature methodology and dictionary](features.md) for formulas,
fallbacks and fold-only preprocessing.

FastF1 status `Finished` can describe a qualifying segment. A terminal
`Finalised`, or `Ends` following `Finalised` without a resumed session, establishes
observed completion. Phase 2 explicitly uses `telemetry=False` and records this
state independently of an absolute UTC completion timestamp. In the installed
FastF1 version, `t0_date` and `LapStartDate` normally require telemetry; without
them, the completion time remains missing. Neither schedule `EndDate` nor the
retrieval time is relabeled as the actual session end. Completed lap observations
may be retained for ingestion even when their absolute timing is unknown; a
cutoff-aware feature export must still fail closed until that time is verified.
Official results load independently of timing access and do not establish an
exact publication time. See [Phase 2 ingestion](ingestion.md).

## Missing data and information provenance

Entrant lists are explicit, sourced and timestamped for each event/task. They are
not inferred from the target result or practice participant list. Field size,
drivers and teams are dynamic. Enforce exact roster/result agreement for training
and evaluation; substitutions or unmatched DNS entries require review, not an
inner join. Inference can retain an announced entrant with no prior data.

Missing practice/history stays missing with observed counts. Numeric median
imputation with missing indicators, scaling and categorical encoding are learned
only on the training fold. Unknown drivers/teams are supported. Mask missing
rank targets during fitting, retaining their feature rows for full-field
prediction. Unknown label entries exclude a weekend at manifest freeze. Known
special outcomes with missing ranks may remain eligible; report coverage and
partial-order metrics, and suppress full-order MAE/exact-match when complete
numeric order is unavailable.

Two provenance modes are deliberately separate:

- `publication_snapshot`: source sessions must have completed before cutoff and
  the exact revision must have been downloaded by cutoff. Roster/grid knowledge
  times need genuine external evidence. Training label revisions must also have
  been retrieved before the later validation cutoff. A reconstructed training
  model cannot claim snapshot evaluation.
- `historical_reconstruction`: source completion/finalisation is an availability
  proxy. Data downloaded today may contain later lap deletions, corrected timing,
  classifications, penalties, identities or team names. True publication-time
  state often **cannot be reconstructed**. These backtests are event-time
  reconstructions, not proof of what was public at the original cutoff. Record
  the mode and source revisions in every feature/prediction audit.

Snapshots establish when this project observed data, not the source's precise
first publication time. In particular, a revised target-weekend qualifying
classification may be a valid eventual label yet unavailable at the race cutoff
in a reconstructed backtest. Collect prospective snapshots for a defensible
publication-time evaluation.

## Revisions and chronological evaluation

Store cached downloads separately from immutable normalized Parquet revisions.
Each source revision records retrieval times, failures, schedules and hashes.
Refresh into a new revision; never overwrite data under a frozen experiment.
Official results here mean the official classification redistributed by Jolpica,
not an independently verified FIA document. Later steward decisions and source
corrections can change labels. Keep the original experiment reproducible, record
the reason for any new revision, and compare revisions before starting a new
experiment. Do not switch revisions in response to test errors.

Before feature/model work, freeze an event manifest with dataset hash,
eligibility rules/configuration, exclusion reasons, chronological ordering,
development weekends and final-test IDs. Eligibility checks only completion and
usable official-label availability. Reserve the latest **four eligible completed
2026 GP weekends** as one untouched final block. The block size is configurable;
insufficient training, 2026 validation or final weekends causes a feasibility
error, with no automatic shrinking. Review exclusions before freezing: source
availability can introduce selection bias.

Use expanding 2026 development folds. Train on earlier completed weekends from
2024–2026; validate on one entire later 2026 weekend. Every driver and task from a
weekend shares that weekend's role. An earlier validation weekend can become
training in a later fold. No random driver-row split, shuffled cross-validation,
test-driven feature selection or preprocessing fit on validation/test is allowed.

The legacy fixed Ridge and logistic baselines establish the pipeline; their
hyperparameters remain fixed. Phases 4/5 add nested chronological development
selection under their linked protocols. Report per-weekend rank Spearman correlation, pairwise accuracy,
top-three recall, and full-order MAE/exact-match where valid. For podiums report
Brier score and log loss. Aggregate weekends equally and expose each metric's
usable-weekend count and label coverage. Phase 5 examines calibration/reliability
on later development weekends; marginal probabilities
have no joint constraint and need not sum to three. Results across few weekends
have high uncertainty; report weekend-level uncertainty, not driver-row confidence.

After all choices are frozen, plan one final block evaluation with a single model
trained on all prior development weekends. Keep model parameters fixed throughout
the block; earlier block observations may enter later block features only as
predeclared, cutoff-available history, never as training/tuning labels. Report
each weekend and the aggregate once. Phases 1–6 provide no final-test opening
command. The user's later explicit authorization is implemented by the
[Phase 7 lock/evaluation protocol](final-evaluation.md), which fixes pre-block
history as well as fitted models and records a single-use ledger. Further changes
or test reuse are exploratory and require a fresh future holdout, not iterative
test tuning.
