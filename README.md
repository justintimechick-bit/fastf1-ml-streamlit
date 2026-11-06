# F1 Model Lab

A Streamlit app for testing three course-aligned machine learning families against
**real Formula 1 results**. Choose a completed Grand Prix, train on earlier 2026
weekends, and compare qualifying order, post-qualifying race order and podium
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
python -m streamlit run streamlit_app.py
```

The **Historical test** page defaults to Mexico 2026. Choose an event and all prior
eligible history, or the last 6/12 eligible events, then press **Run historical
test**. Existing verified runs are reused. Fitting occurs only on that explicit
action, never during ordinary interface reruns. Downloads are CLI-only. The
**Three models** page explains each family; **Data & methodology** shows coverage,
provenance, cutoffs and exclusions. No synthetic forecasts appear in this app.

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

The functional lab currently trains **2026 only**. Although 2024/2025 result rows
are cached, their independent cutoff rosters/publication review is unfinished.
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
