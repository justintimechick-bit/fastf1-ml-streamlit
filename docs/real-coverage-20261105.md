# Real-data coverage and exclusions · November 5, 2026

This report distinguishes raw acquisition from the reviewed modeling snapshot.
Eligibility was determined by source evidence, identity/roster review and cutoffs,
not model performance. All selected historical outcomes were consulted during
source QA, so none is described as an untouched holdout.

| Season | Catalog Q/R sessions | Catalog result rows | Reviewed model Q/R sessions |
| --- | ---: | ---: | ---: |
| 2024 | 48 | 958 | 0 |
| 2025 | 48 | 958 | 0 |
| 2026 | 36 | 787 | 38 |

Catalog ID: `90b38b17647bd0fe459242f2f85ba1eed5de53d440fa94f9f13fc415808a90f6`.
Model snapshot: `data/real-core-20261105-v4`, ID
`08bd7449688fda332cab42379d89cdeb295c3b200c45d2d9119e3aefc6befb2a`.
The model uses independent FIA final classifications for every selected session,
including the two missing Jolpica sessions (USA race, Mexico qualifying). Catalog
and FIA revisions are separate sources; their counts must not be added together.

## Detailed inventories

- [Catalog season coverage](reports/real-20261105/catalog-seasons.csv)
- [Catalog event coverage](reports/real-20261105/catalog-events.csv)
- [Catalog session and feature coverage](reports/real-20261105/catalog-sessions.csv)
- [Actual acquisition failures](reports/real-20261105/catalog-failures.csv)
- [Reviewed model session label counts](reports/real-20261105/model-sessions.csv)
- [Missing model feature groups by event](reports/real-20261105/model-feature-groups.csv)
- [Excluded future events](reports/real-20261105/model-event-exclusions.csv)

All 19 completed 2026 events have independently evidenced fields of 22 for both
qualifying and race: 418 qualifying roster rows and 418 race rows. Qualifying has
410 numeric ranks, three explicit special outcomes and five entrants absent from
the final document. Those five (three Australia, two Spain) retain unknown labels;
no last position or DNS was invented. Race has 339 numeric ranks and 79 explicit
special outcomes. All 418 race podium labels are observed, with three positives
per event. Qualifying does not carry a race-podium target.

Four future scheduled events (rounds 20–23) are excluded. Earlier-season results
are retained in the source catalog, but independent cutoff entry lists and
publication review for 2024/2025 have not been completed. They cannot yet enter a
fair pooled/weighted-history model comparison. The functional app exposes only
the reviewed 2026 cohort.

## Failure diagnosis and repairs

| Observation | Diagnosis | Treatment |
| --- | --- | --- |
| Official timing HTTP 401, documented mirror 404 | Source access restriction; does not mean the session never occurred | Cached results remain; practice/Q laps/weather unavailable, with no synthetic fallback |
| Jolpica HTTP 429 after a burst of requests | Upstream rate limit; existing catalog cooldown must be respected | Resumable acquisition retains successes; no retry storm or cooldown bypass |
| Tiny entry-list superscripts merged with car numbers | PDF parsing defect | Filter footnote-sized characters before parsing independently published entry lists |
| Barcelona short document facet selected another series | Event/category mapping defect | Reviewed explicit F1 Barcelona-Catalunya URL; verify 2026 document headers |
| Classification omits some independently listed entrants | Unobserved target, not evidence of an ordered tail | Keep field and unknown masks, exclude missing labels from metrics |
| Latest Monaco final issued after an appeal | Source revision published September 5 | Unavailable for earlier historical folds; preserve calendar windows and mask unavailable components |
| All-missing pace in both demo and real catalog | Synthetic fixture lacked observations; real timing access separately failed | No lap pace/weather columns enter the core model |

The FIA downloader caches raw evidence with URL, retrieval metadata and hashes,
spaces requests, verifies cache contents and supports offline replay. It fails
explicitly on an unavailable uncached document. Resuming never fabricates a row
or silently substitutes a different series. Raw PDF caches remain local.

The final reviewed parser was replayed offline into a new temporary directory:
all 11 saved table/provenance files were byte-identical. Original synthetic
archive verification found **0 changed files among 1,037 protected files**; its
locked source fingerprint is unchanged. The new experiment has separate IDs,
models, receipts and forecast locks, with no claim against the old ledger.

## Modeling exclusions and evaluation counts

Observed input groups are prior field-relative classification history, current
independent identity/team and, after publication, target qualifying for race and
podium. No practice, lap segment, weather or field-relative lap pace calculation
is represented as observed. Entirely missing and constant columns are excluded
using each training fold only; remaining preprocessing is fit on training only.

The saved Mexico replay has 18 eligible training events, three parameter-selection
events and two separate podium-calibration events. Target evaluation is **one
exploratory event**: qualifying 21/22 numeric labels, race 17/22 numeric labels,
podium 22/22 labels. Development and target results, including baseline losses,
are in [the actual-run report](real-model-lab.md#mexico-run-actually-executed).
A prospective final-test manifest and a genuine untouched evaluation have not
been created for this new historical lab.
