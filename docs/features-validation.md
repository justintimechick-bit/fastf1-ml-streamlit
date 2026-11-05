# Feature validation — November 4, 2026

Implemented feature schema **3.0.0**, with separate qualifying and race datasets.
Podium models share the race cutoff and predictor matrix. The original
experiments, source snapshots and saved artifacts were preserved.

## Verification

```sh
.venv/bin/pytest -q
.venv/bin/ruff check src tests
.venv/bin/ruff format --check src tests
.venv/bin/python -m pip check
.venv/bin/python -m build --no-isolation
```

**78 tests passed**, including the existing ingestion/protocol tests. Lint,
formatting, dependency consistency, wheel and source-distribution builds passed.
The source archive includes feature modules, review CSV headers, the generated
dictionary and feature tests.

New checks cover future-result invariance; target Q/R exclusion from the
corresponding predictor; shifted 3/5-calendar-weekend windows; early prediction
cutoffs before a preceding race finishes; variable-field normalization;
teammate/team-era/circuit histories; pit/deletion/generated/disruption filters;
backward-only weather matching; Q1/Q3 separation; rookies and sparse appearances;
unseen categories and fold-only preprocessing; catalog review binding; missing
target labels; event-role disjointness; and deterministic artifact generation.

## Independent CLI rebuild

```sh
.venv/bin/f1ml demo --output data/phase3-final
.venv/bin/f1ml build-features --dataset data/phase3-final/dataset \
  --manifest data/phase3-final/manifest.json \
  --entrants data/phase3-final/entrants.csv --cutoffs data/phase3-final/cutoffs.csv \
  --output data/phase3-rebuild
```

Both feature directories have the identical receipt:

```text
ccc37825281ab0102fb44f15dc516e04211ad3896fec45bfbf78b71699c3a64d
```

The implementation fingerprint is
`bf5e59ce8e1a9ee32a44cc558aa072d1ec26e54a73c14a1645a6ea0cf812dcd0`.
The generated dictionary ID is
`4fe2154d9c170a21cf1f93d9148474ce982588ecd8bea18a77b5da74b20c59f4`.

| Synthetic dataset | Rows | Development weekends | Model columns | Rows missing practice pace |
| --- | ---: | ---: | ---: | ---: |
| Qualifying | 46 | 7 | 82 | 46 |
| Race | 46 | 7 | 97 | 46 |

This deliberately sparse example proves core retention: legacy practice summaries
establish participation but cannot prove comparable lap conditions, so pace
stays missing without removing any entrant. Separate richer offline fixtures
exercise actual condition-matched pace, disrupted laps and segmented Q1/Q2/Q3.
The synthetic final block (`2026-04` through `2026-07`) has no generated feature
rows or model metrics. Synthetic scores are software checks, not F1 performance
estimates.

## Real-data boundary

An attempted freeze of the existing immutable Phase 2 snapshot **without review
overlays** returned exit 2: zero 2026 events had verified completion times under
the experiment rules, so the requested four-weekend final block was not shrunk.
No manifest was written, no timestamps were fabricated, and no real-data models
or final-test metrics were produced. This does not mean the races did not occur;
it means the stored source evidence does not establish the needed cutoff times.

Use reviewed availability/completion evidence, cutoff rosters and circuit/layout
context as described in [features.md](features.md). Optional session data can
remain absent. Historical reconstructions still cannot establish the exact
publication-time state of later-revised data, and fuel loads/run plans remain
unknown even when pace conditions match.
