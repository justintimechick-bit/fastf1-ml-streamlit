# Streamlit Community Cloud deployment

Hosted URL: <https://fastf1-ml-pit-wall.streamlit.app/>.
Source: <https://github.com/justintimechick-bit/fastf1-ml-streamlit> (`main`).
Deployed through Safari on November 5, 2026 using Python **3.13.18**.

The root `streamlit_app.py` opens **Real-data experiment** by default. The existing
`src/f1ml_app/streamlit_app.py` remains the classroom entrypoint with its original
synthetic-demo home page. Both offer the same navigation and dedicated real-data
dashboard. Downloads, training and final scoring stay outside UI reruns.

The dashboard shows saved observations, missing data, eligibility, blockers and
separate development/final evaluation status. It does not suggest that a real
model has been trained: Phase 9 remains blocked before training.

## Prepare the repository

The working folder's `data/` is excluded from Git. Deploying the source alone
would produce missing-artifact errors. The packaging command creates an isolated
copy with verified synthetic artifacts, the real experiment, and the single
current source-catalog snapshot:

```sh
.venv/bin/python -m f1ml_app.cloud --output deploy/streamlit
```

The output path must be new. No protected artifact is changed or regenerated.
The deployment copy includes fixed model bytes and both consumed synthetic ledger
records; these are display evidence, and the app never claims a ledger. It also
includes `DEPLOYMENT.json` with file hashes and the preserved code fingerprint.
Future source edits need a new prepared copy; pushing from a stale copy does not
update the app.

The packager copies an explicit allowlist. It excludes FastF1 caches, unrelated
experiments, `.venv`, `.git`, local environment files and Streamlit secrets. Keep
the generated `.gitignore`: it allows the required `data/` snapshots while ignoring
credentials and runtime caches. Publish the prepared directory, not the whole
working folder. `requirements.txt` references the exact core and app locks plus
the local package so saved-model library versions remain pinned.

## Deploy on Community Cloud

1. Push the prepared directory to the deployment GitHub repository.
2. Sign in at [Streamlit Community Cloud](https://share.streamlit.io/).
3. Choose **Create app**, then **Yup, I have an app**.
4. Select repository `justintimechick-bit/fastf1-ml-streamlit`, branch `main`,
   and main file `streamlit_app.py`.
5. In **Advanced settings**, choose **Python 3.13**. No F1 account or secrets
   are required to display the prepared artifacts.
6. Deploy and inspect the build logs, then open the assigned `*.streamlit.app` URL.

The user explicitly approved making only this prepared deployment repository
public. Streamlit's existing public-repository connection is used; its proposed
private-repository OAuth access was not granted. Other repositories' visibility
and permissions were not changed. Visitor access is configured separately in Streamlit.

These fields and Python-version selection follow Streamlit's
[deployment documentation](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).
Dependencies are discovered from the root `requirements.txt` as described in the
[dependency documentation](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies).

## Validate after deployment

Open Real-data experiment and confirm 112 result sessions, 2,263 official result
rows and zero trained real models for the current saved experiment. Change the
season, filter missing results, inspect each task and download the report. The
evaluation table should show unrun records with blank metric values. Check a
synthetic weekend separately to confirm it remains clearly labeled synthetic.

Local checks:

```sh
LOKY_MAX_CPU_COUNT=4 .venv/bin/pytest -q tests/test_cloud_ui.py tests/test_app.py tests/test_real_experiment.py
.venv/bin/ruff check src tests streamlit_app.py
.venv/bin/ruff format --check src tests streamlit_app.py
```

The hosted build installed the pinned dependencies and started successfully under
Linux/Python 3.13.18. Safari loaded the real-data dashboard with 112 result sessions,
2,263 official result rows and zero trained real models. The app remains a source
review plus a separately labeled synthetic demo; deployment does not finish the
blocked real-data training experiment.
