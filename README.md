# F1 Model Lab

A Streamlit app for testing three course-aligned machine learning families against
**real Formula 1 results**. Choose a completed Grand Prix, train on earlier 2026
weekends from the same season, and compare qualifying order, post-qualifying race order and podium
probabilities with simple baselines.

Hosted app: [F1 Model Lab](https://fastf1-ml-pit-wall.streamlit.app/).
[Deployment repository](https://github.com/justintimechick-bit/fastf1-ml-streamlit).

| Model family | Qualifying and race order | Podium classification |
| --- | --- | --- |
| Linear models | Ridge regression | Logistic regression |
| Random forest | Random forest regressor | Random forest classifier |
| Gradient boosting | Histogram gradient boosting regressor | Histogram gradient boosting classifier |

These are three families and **nine saved estimators**, covering regression,
regularization, classification, trees and ensembles from the course. Rank scores
are sorted into a complete forecast field; they are not predicted lap times.
Podium probabilities are independent and do not have to sum to three.

## Run the app and real tests

Use Python 3.13 for the pinned environment and saved models:

```sh
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
#if dependencies are installed, this boots the app
python -m streamlit run streamlit_app.py
```

The **Predict Past Races** page supports **2024, 2025 and 2026** and defaults to Mexico
2026. Select a season, a target event, and **Previous 6 races that year**,
**Previous 12 races that year**, or **All earlier races that year**, then press **Run historical
test**. Existing verified runs are reused. Fitting occurs only on that explicit
action, never during ordinary interface reruns. The
**Three Models** page explains each family and compares their saved results;
**Data & Methodology** shows coverage, provenance, cutoffs and
exclusions. **Predict Future Races** forecasts upcoming sessions after a data refresh.
No synthetic forecasts appear in this app.

Use the four tabs at the top of the page to switch pages. The navigation scrolls
horizontally on narrow screens. Each page opens with a
**Race setup** card (season, race and training history, plus **Run historical test**
on the main page, saved runs and the data update), and a sticky context ribbon
shows exactly which saved run is displayed. Results lead with a scoreboard of
each model's saved score against the simple baseline, then the podium spotlight,
then tabs for the full grid, every score and the audit trail. A **Qualifying order /
Race order & podium** switch above the results selects the prediction.

The sidebar holds app-wide preferences only: models, team or driver filters, and
**Detail level** (**Summary** or **Full detail**). All three models start selected;
choose a **Spotlight model** to set the grid order and podium cards. Multiple
selections add per-model ranks and podium probabilities, mean rank, and rank
spread. **Compare all three models** and **Reset all filters** restore selections
quickly, and clearing the model list keeps the spotlight model active.
Page switches preserve these controls. Saved run labels show season, round,
training scope and a short artifact hash.

Training, model selection and calibration use only eligible races before the
target weekend in the selected year. Short windows use all available earlier
races when fewer than 6/12 exist. A season opener cannot be trained because it
has no earlier races that year. Changing seasons resets the target and displayed
run; runs are stored separately by season, target, training window and snapshot.
The prepared 2024/2025 FIA datasets are under `data/historical/2024` and
`data/historical/2025`; `F1ML_HISTORICAL_DATA` overrides their parent directory.
Each contains all 24 races. FIA documents supply verified facts for 16 races in
2024 and 9 in 2025; the remaining 8 and 15 use the authorized, labeled
FastF1/Jolpica API fallback. The app shows those races and their source details.
API fallback results are real captured records, but historical publication times
and some entrant fields are reconstructed. FIA publication records act as timing
proxies; when absent, qualifying is placed just before the race and race labels
at race start plus 24 hours. These proxies cannot verify what a historical API
actually exposed or when a later revision appeared. They support exploratory
replays rather than a claim of prospective availability.

Press **Refresh FastF1 data** in **Refresh source data** on the main page to load new weekends,
Select an upcoming event (for example Las Vegas, `2026-21`) and press **Run forecast**.
Before the weekend starts, qualifying uses earlier form and the event entry list,
or an explicitly labeled assumption that the latest known same-season field continues.
Each model then feeds its own predicted qualifying positions into its race and podium
forecasts. These conditional forecasts do not propagate qualifying uncertainty or model
grid penalties. After final qualifying is published, race and podium forecasts use
the official qualifying positions until race start. All three tasks share the
**all earlier / last 12 / last 6** training window. Finished weekends leave the upcoming
selector by scheduled session time, even if final results are still being collected;
completed classifications enable them in **Historical test**. The refresh action is available on every
page and shows the season it refreshes. **Historical test**, **Refresh & predict**,
and **Data & methodology** share the active refreshed data; completed new events appear in the historical target
options once final qualifying and race classifications are available. Historical
tests train against that same data. **Data & methodology** updates coverage, source
details, completion counts and its last-refreshed timestamp immediately after a
refresh. Ordinary page reruns never download or train.

The refresh bypasses FastF1's cached calendar/results and reads current FIA indexes.
FastF1 supplies schedule/results captures; independent FIA entry lists and final
classification PDFs preserve the model lab's roster and numeric-rank checks.
Each refresh creates a verified snapshot under `data/live`, preserving the original
Mexico dataset and runs. Source failures are shown in the event's refresh details;
a failed refresh keeps the previous snapshot available. Live forecasts are saved
without target labels or target scores. Practice pace and weather remain excluded.

The same actions are available from the CLI:

```sh
python -m f1ml_real refresh
python -m f1ml_real forecast --event 2026-21
# Optional: --last-n 6 or --last-n 12
```

`F1ML_LIVE_DATA` sets the app's refresh storage directory (default `data/live`);
use CLI `--state` for the same override. The active season comes from the base
dataset, currently 2026. A new season needs a reviewed base dataset. On Streamlit
Cloud, runtime refreshes and new forecasts are temporary across restarts.

Reproduce or verify the saved Mexico experiment:

```sh
python -m f1ml_real backtest --dataset data/real-core-20261105-v4 \
  --event 2026-19 --runs artifacts/real-lab
# A separate historical experiment, without changing Mexico's sealed run:
python -m f1ml_real backtest --event 2026-18 --last-n 12
pytest -q
ruff check src/f1ml_real src/f1ml_app tests/real tools/export_real_notebook.py
```

`pytest` now selects **tests/real** only, using the sealed real FIA observations.
Missing real data fails explicitly. Older synthetic fixture tests remain archived
outside this default suite. Paths can be set with `F1ML_REAL_DATASET` and
`F1ML_REAL_RUNS` (the app also accepts `F1ML_LAB_DATASET`/`F1ML_LAB_RUNS`).

## What was actually run

The reviewed snapshot `data/real-core-20261105-v4` contains **19 completed 2026
weekends, 38 Q/R classifications and independent pre-cutoff entry lists**. Source
PDF URLs, hashes and official publication times are retained. There are 22 drivers
per field, including explicit unknown/nonclassified outcomes where a numeric rank
was not printed. These are real observations, with no synthetic replacements.

Mexico run: `artifacts/real-lab/replay-2026-19-all-bdc1c440786e`.
All 18 earlier 2026 events were eligible for the final fit. Hyperparameters used
three earlier development events (rounds 14–16); separate later events (17–18)
provided out-of-fold podium calibration. Models/configuration and complete-field
forecasts were sealed before target scoring.

| Mexico target metric (lower is better) | Baseline | Linear | Random forest | Boosting |
| --- | ---: | ---: | ---: | ---: |
| Qualifying MAE, 21 known ranks | 2.286 | 1.952 | 1.857 | 2.048 |
| Race MAE, 17 known ranks | **3.412** | 3.824 | 3.588 | 3.471 |
| Podium Brier score, 22 labels | .1275 | .0312 | .0589 | .0738 |

**One target event, exploratory retrospective replay.** Historical outcomes were
previously available and consulted; this is not an untouched final test. In the
three development events, the race baseline also beat all three race regressors;
only boosting beat the qualifying baseline, and only forest beat the podium
baseline. Calibration-fit diagnostics are reported separately and are not
independent validation. Rank metrics cover printed numeric positions only;
NC/DSQ drivers retain forecasts but are never assigned invented target ranks.

A second USA replay was run successfully through the Safari interface using all
17 earlier eligible events; its separate sealed artifacts are also included.

Saved runs include nine `.joblib` models, protocol/configuration, training rows,
feature audits, development and target predictions/metrics, a forecast lock and
content-hashed receipt. Changing configuration creates a separate run; altered
saved artifacts fail verification instead of being silently regenerated.

## Real-data ingestion and remaining limits

The resumable FastF1/Jolpica catalog contains **132 Q/R result sessions**: 48 in
2024 (958 rows), 48 in 2025 (958), and 36 in 2026 (787). Two completed 2026 catalog
sessions remain rate limited; the separate reviewed model snapshot uses official
FIA final documents for all 38 sessions, including those two. Cached source
captures and parsing are distinct from model training.

Practice laps, qualifying lap segments and weather remain unavailable through the
selected timing source: observed HTTP **401**, with the documented mirror **404**.
Missing pace affects the real pipeline too. Those groups are excluded from the
core experiment. Entirely missing and constant training columns are also excluded
before imputation; no median-filled empty column is presented as observed pace.

The functional lab supports **2024, 2025 and 2026**, with same-year training
windows, FIA entry lists/classifications, and labeled API fallback for older seasons.
Pooled and weighted-history real comparisons have **not** been run. A fresh future
holdout still needs a prospective manifest before model selection. Completed
historical tests in this app remain exploratory.

Rebuild reviewed facts into a **new** output directory (source downloads may need
network access; `--offline` requires the verified local cache):

```sh
python -m f1ml_real prepare-real --catalog data/real-20261105-catalog \
  --cache data/fia-2026-source-cache --output data/real-core-new \
  --as-of 2026-11-05T16:00:00Z --offline
```

Prepare an older season with `--year 2024` or `--year 2025`, its own cache directory
(`data/fia-2024-source-cache` or `data/fia-2025-source-cache`), and a new output
directory. For example:

```sh
python -m f1ml_real prepare-real --catalog data/real-20261105-catalog \
  --cache data/fia-2024-source-cache --output data/historical/2024 \
  --year 2024 --as-of 2026-11-09T21:00:00Z --allow-result-fallback
python -m f1ml_real backtest --dataset data/historical/2024 \
  --event 2024-19 --last-n 6
```

`--allow-result-fallback` applies only to 2024/2025 and uses verified local catalog
captures when FIA documents cannot be read. Omit it to require FIA evidence.
`--offline` requires cached source documents and performs no downloads. Each
preparation needs a new output directory; existing sealed datasets are preserved.

The public deployment includes parsed factual tables and provenance, not the raw
PDF cache. See the [coverage and exclusion report](docs/real-coverage-20261105.md),
[real model methodology and commands](docs/real-model-lab.md),
[project rubric alignment](docs/project-rubric-alignment.md), and the executed
[Mexico notebook](notebooks/real-mexico-2026.ipynb). The notebook contains actual
saved-result outputs; no report or class submission has been made.
The [validation record](docs/real-lab-validation.md) records the real tests,
archive integrity and hosted Safari smoke check.

## Deployment and preserved archive

```sh
python -m f1ml_app.cloud --output deploy/streamlit-real-release
```

The packager creates a new allowlisted real-only release, verifies its snapshot
and runs, and records file hashes. See [hosting instructions](docs/streamlit-cloud.md).
Only the prepared F1 deployment repository is public; no broader private GitHub
access was granted.

The earlier synthetic experiment, sealed predictions, consumed evaluation ledger
and locked `src/f1ml` implementation remain unchanged in the **local archive**.
They are removed from the active UI and current deployment data. The original
Phase 1–9 design/audit documents under `docs/` describe that earlier work and do
not supersede this real lab's measured status. The historical lab is implemented
separately in `src/f1ml_real`; `src/f1ml_app` contains its interface.
