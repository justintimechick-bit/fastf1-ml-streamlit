# Input schemas

The CSVs here contain headers only; the offline demo generates complete examples
in its output directory. Real timestamps and entrants require a source review.

`cutoffs.csv`: one row per development event and task (`qualifying` or `race`).
Use timezone-bearing ISO timestamps. `mode` is `historical_reconstruction` or
`publication_snapshot`. Race and podium share the `race` cutoff.

`entrants.csv`: one row per declared driver per event/task. `driver_id` and
`team_id` use stable Jolpica/FastF1 IDs; driver numbers are not cross-event keys.
`known_at` states when the entry/team was known; `source` identifies the dated
entry list or other reviewed evidence. Include the full announced field even if
a driver has no practice or history. A prediction command expects just the
selected event/task's roster rows.

The grid columns are optional. `grid_position` is an independently known
grid slot (0 for pit lane), with a required `grid_known_at` and `grid_source` when
present. Leave all three blank if the grid/penalty was not known at race cutoff.
These columns are ignored for qualifying. Never populate them from later target
race results. Qualifying position is a separate feature automatically loaded
only for the race/podium task.

`team_history_id` is optional but recommended: use the explicit performance era
from the reviewed team mapping, with membership known by `known_at`. Without it,
feature generation uses `unverified:<year>:<team_id>`, preventing automatic
cross-season team priors. Driver identity stays independent of team membership.

For Phase 2 snapshots, `session_availability.csv` supplies reviewed completion
and availability timestamps, with an evidence reference. Do not populate these
from scheduled end times or invent historical publication times. A review binds
to the frozen source snapshot; strict publication mode still checks when the
exact source revision was retrieved. `event_context.csv` supplies reviewed stable
circuit/layout IDs; a changed layout needs a distinct ID when appropriate.
Without circuit evidence, circuit-prior features stay missing with zero counts.

Development inputs must exclude the manifest's `final_test` IDs. Source output,
manifest, feature and artifact paths are exclusive-create: choose a new path to
record a revision or repeat an experiment.
