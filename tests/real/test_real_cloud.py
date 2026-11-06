"""The shareable release contains real factual/model artifacts and runs read-only."""

import json
from pathlib import Path

import joblib
import requests
from sklearn.pipeline import Pipeline
from streamlit.testing.v1 import AppTest

from f1ml.storage import file_digest, verify_directory
from f1ml_app.cloud import DATASET, RUNS, prepare
from f1ml_real import lab

ROOT = Path(__file__).resolve().parents[2]


def test_real_cloud_release_keeps_provenance_and_excludes_demo_cache_and_secrets(
    tmp_path, monkeypatch
):
    output = tmp_path / "real-release"
    metadata = prepare(output, ROOT)
    assert metadata["data_kind"] == "real"
    assert metadata["source_snapshot_id"] == verify_directory(ROOT / DATASET)["id"]
    assert json.loads((output / "DEPLOYMENT.json").read_text()) == metadata
    assert metadata["saved_runs"]
    for run in metadata["saved_runs"]:
        assert run.startswith(str(RUNS))
        assert verify_directory(output / run) == verify_directory(ROOT / run)
    assert verify_directory(output / DATASET) == verify_directory(ROOT / DATASET)
    assert set(path.name for path in (output / "data").iterdir()) == {DATASET.name}
    assert not (output / "data/final-evaluation-ledger").exists()
    for relative, checksum in metadata["files"].items():
        path = Path(relative)
        assert path.suffix != ".pdf"
        assert path.name not in {".env", "secrets.toml"}
        assert "__pycache__" not in path.parts
        assert "fia-2026-source-cache" not in path.parts
        assert file_digest(output / relative) == checksum
    assert not list((output / "data").glob("phase*"))

    def forbidden(*args, **kwargs):
        raise AssertionError("The packaged landing page must only read real saved artifacts")

    monkeypatch.setenv("F1ML_PROJECT_ROOT", str(output))
    monkeypatch.setenv("F1ML_REAL_DATASET", str(output / DATASET))
    monkeypatch.setenv("F1ML_REAL_RUNS", str(output / RUNS))
    monkeypatch.setattr(Pipeline, "fit", forbidden)
    monkeypatch.setattr(joblib, "load", forbidden)
    monkeypatch.setattr(requests.Session, "request", forbidden)
    monkeypatch.setattr(lab, "cached_run", forbidden)
    at = AppTest.from_file(str(output / "streamlit_app.py"), default_timeout=30).run()
    assert not at.exception and not at.error
    assert at.metric[0].value == "18"
    assert at.button(key="lab_run").label == "Run historical test"
    assert at.selectbox(key="lab_target_event").value == "2026-19"
