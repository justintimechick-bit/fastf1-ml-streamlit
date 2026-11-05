import pandas as pd
import pytest

from f1ml.demo import SyntheticProvider, run_demo
from f1ml.domain import utc
from f1ml.evaluation.splits import gp_session, load_manifest
from f1ml.ingestion.pipeline import Dataset


@pytest.fixture(scope="session")
def workflow(tmp_path_factory):
    root = tmp_path_factory.mktemp("offline") / "run"
    run_demo(root)
    return root


@pytest.fixture
def dataset(workflow):
    return Dataset(workflow / "dataset")


@pytest.fixture
def manifest(workflow):
    return load_manifest(workflow / "manifest.json")


@pytest.fixture
def event_inputs(dataset):
    def create(event_id="2026-02", task="qualifying"):
        event = dataset.event(event_id)
        quali = gp_session(event, "Qualifying")
        cutoff = (
            utc(quali["scheduled_start"]) - pd.Timedelta(seconds=1)
            if task == "qualifying"
            else utc(quali["ended_at"]) + pd.Timedelta(hours=1)
        )
        entrants = pd.DataFrame(
            [
                {
                    "event_id": event_id,
                    "task": task,
                    "driver_id": driver,
                    "team_id": f"synthetic_team_{index // 2}",
                    "known_at": (cutoff - pd.Timedelta(days=1)).isoformat(),
                    "source": "reviewed_synthetic_entry_list",
                }
                for index, driver in enumerate(SyntheticProvider.drivers(event))
            ]
        )
        return event, cutoff, entrants

    return create
