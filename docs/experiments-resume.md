# Phase 6 resume — November 4, 2026 (America/New_York)

Resumed from the saved Phase 6 state. Implementation, core models and paired
synthetic comparisons were already complete. All three failed live captures
were still within their 24-hour cooldown, so the bulk capture was not retried.

## Newly established access blocker

Performed two unauthenticated, read-only session-info checks for the already
selected **2024-01 GP Practice 1**. The session path came from public
`fastf1.get_session(2024, 1, 'Practice 1').api_path`, with the persistent schedule
cache enabled. These requests diagnosed endpoint access only; their responses
are not feature data and no final-test session was requested.

| Endpoint | Observed result |
| --- | --- |
| Official `livetiming.formula1.com` session-info archive | HTTP 401, empty response |
| FastF1's documented `livetiming-mirror.fastf1.dev` fallback, same path | HTTP 404 |

Exact URLs, timestamps, installed FastF1 version and response summaries are saved
in `data/phase6-resume-20261104/endpoint-check.json`. The fallback URL construction
was checked against the installed FastF1 3.8.1 source. No authentication tokens,
cookies or credentials were sent or recorded.

The upstream [maintainer's issue](https://github.com/theOehrly/Fast-F1/issues/978)
confirms that the archive now requires F1TV Access or a higher subscription.
The [3.9.0 release](https://github.com/theOehrly/Fast-F1/releases/tag/v3.9.0)
adds authentication support and also changes some result/cache behavior. This
explains the prior `DataNotLoadedError`; it is not evidence that every requested
session has no underlying data. Upgrading the package alone does not supply
authorization or a subscription.

## Preserved scope and results

Kept the original account-free scope and tested FastF1 3.8.x environment. No
dependency upgrade, sign-in, subscription purchase or new source was introduced.
No access-control workaround was attempted. A supported authenticated ingestion
environment or an already populated compatible local cache would be needed to
continue genuine timing/telemetry ingestion. Cache provenance and source binding
would still need verification before entering any experiment.

Verified receipt integrity for the core features, Phase 4 qualifying artifacts,
Phase 5 race/podium artifacts and Phase 6 supplements/comparisons. Also checked
the current core and optional implementation fingerprints separately. The frozen
manifest remains `3cc7b3bf005e808e360595275f3e24c35bbd8a26c962bafdef8e6ee766f968dd`,
with `2026-04` through `2026-07` still sealed. Nothing was rewritten under those
snapshots, and no optional group was promoted.

This resume changes documentation and adds diagnostic evidence only. The last
implementation verification remains 145 passing tests, lint/formatting checks
and successful builds, recorded in [Phase 6 validation](experiments-validation.md).
No model or feature code changed, so those tests were not unnecessarily rerun.
