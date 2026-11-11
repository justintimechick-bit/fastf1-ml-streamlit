# Streamlit Community Cloud deployment

Hosted URL: <https://fastf1-ml-pit-wall.streamlit.app/>.
Source: <https://github.com/justintimechick-bit/fastf1-ml-streamlit> (`main`).

Both entrypoints now open the **real-only F1 Model Lab**. Ordinary reruns read
verified saved tables. Explicit **Run historical test** and **Run forecast** actions
fit models; **Refresh FastF1 data** in the sidebar downloads new observations.
The bundle contains the 19-event 2026 base, both complete 24-race 2024/2025
historical datasets, and the sealed Mexico experiment with nine models. Older
seasons include explicitly labeled FastF1/Jolpica API fallback and timing proxies
where FIA documents could not be verified. Runtime refreshes update Predict Past
Races, Predict Future Races, and Data & Methodology together.

## Prepare a new release

```sh
.venv/bin/python -m f1ml_app.cloud --output deploy/streamlit-real-release
```

The output must be new. The packager verifies real snapshot/run receipts and
copies an explicit allowlist: application/source, pinned requirements, real tests,
documentation, the executed notebook, reviewed real facts and compatible saved
runs. It writes `DEPLOYMENT.json` with file hashes. It excludes synthetic
data/models/ledgers, raw FIA PDFs, FastF1 caches, `.git`, secrets and the private
class rubric PDF. Protected original artifacts remain unchanged locally.

Deploy the prepared directory through the existing repository on `main`, entrypoint
`streamlit_app.py`, Python **3.13**. `requirements.txt` includes pinned core, app and
real-PDF dependencies plus the local package. Community Cloud's writable runtime
can save refreshed snapshots under `data/live` and explicit new model runs;
those runtime files are temporary across reboots. Use local CLI runs for durable artifacts.
No API secrets are required for the saved core experiment.

Only the prepared F1 deployment repository was made public with user authorization.
The existing public GitHub connection is used; proposed private-repository OAuth
access was not granted. Other repositories were not changed.

## Validate a release

```sh
.venv/bin/pytest -q
.venv/bin/python -m f1ml_real backtest --event 2026-19
.venv/bin/streamlit run streamlit_app.py
```

Check Predict Past Races defaults to Mexico, all 18 earlier eligible events, three
model families and a baseline. Each qualifying/race forecast retains all 22
drivers. Metrics label partial numeric targets, development findings remain
separate, and status says exploratory historical replay. Inspect Three Models and
Data & Methodology. Missing real forecasts remain unavailable until an
explicit eligible test is run; no synthetic substitute is provided.

Press **Refresh FastF1 data** at the bottom of the sidebar and check the latest
weekend's schedule and source issues. Before qualifying, forecasts use the event
entry list or an explicitly labeled latest-known-field assumption. Each model's
predicted qualifying positions feed its race and podium forecasts. After qualifying,
race forecasts use the final published classification. Check all/12/6 training windows.
Finished weekends leave the upcoming selector by session time, even when source
results are delayed, and become available in Predict Past Races once collected. This release must be redeployed before the hosted app gains
these controls; local source changes alone do not update the hosted app.

The existing Streamlit configuration disables source watching. After publishing
changed imported modules, reboot the app from its management menu and verify the
new release. Check all three seasons, the 6/12/all-earlier training windows,
the shared sidebar refresh control, and updated methodology coverage/timestamps.
The current source tree no longer ships synthetic data;
older deployment commits retain their history without rewriting it.
