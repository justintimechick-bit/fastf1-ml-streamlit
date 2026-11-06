"""Real FIA observations only; source data is read from a sealed local snapshot."""

import os
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def dataset_root():
    path = Path(os.environ.get("F1ML_REAL_DATASET", ROOT / "data/real-core-20261105-v4"))
    if not path.is_absolute():
        path = ROOT / path
    if not (path / "receipt.json").exists():
        pytest.fail(f"The real snapshot is required: {path}. See docs/real-model-lab.md.")
    return path


@pytest.fixture(scope="session")
def real_tables(dataset_root):
    return {
        name: pd.read_parquet(dataset_root / f"{name}.parquet")
        for name in ("events", "rosters", "labels", "documents", "exclusions")
    }


@pytest.fixture(scope="session")
def real_dataset(dataset_root):
    from f1ml_real.lab import RealDataset

    return RealDataset(dataset_root)


@pytest.fixture(scope="session")
def replay_path(dataset_root, tmp_path_factory):
    """One modest, real chronological integration run, shared by this test session."""
    from f1ml_real.lab import cached_run

    output = tmp_path_factory.mktemp("real-fia-replay")
    return cached_run(dataset_root, "2026-14", output, last_n=12)


@pytest.fixture(scope="session")
def replay(replay_path):
    from f1ml_real.lab import load_run

    return load_run(replay_path)
