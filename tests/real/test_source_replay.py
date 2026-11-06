"""Replay difficult downloaded FIA PDFs against the saved factual table."""

import json
from pathlib import Path

import pandas as pd
import pytest

from f1ml.storage import file_digest
from f1ml_real.fia import parse_classification, parse_roster, pdf_text


@pytest.mark.parametrize(
    "event_id,task",
    [
        ("2026-01", "qualifying"),
        ("2026-03", "race"),
        ("2026-19", "qualifying"),
        ("2026-19", "race"),
    ],
)
def test_retained_official_documents_reproduce_tricky_classifications(
    dataset_root, real_tables, event_id, task
):
    """Real vertical tables, omitted entrants, substitution and DSQ stay reproducible."""
    cache = Path(__file__).resolve().parents[2] / "data/fia-2026-source-cache"
    if not cache.is_dir():
        pytest.skip("Local-only raw FIA cache is excluded from deployment releases")
    documents = real_tables["documents"].query("event_id == @event_id and task == @task")
    paths = {}
    for item in documents.to_dict("records"):
        candidates = [p for p in cache.glob("*.pdf") if file_digest(p) == item["sha256"]]
        assert candidates, f"Retained PDF missing for {item['url']}"
        paths[item["purpose"]] = candidates[0]
    mapping = json.loads((dataset_root / "identity_mapping.json").read_text())
    roster = parse_roster(pdf_text(paths["roster"], roster=True), mapping)
    replay = parse_classification(
        pdf_text(paths["label"]), roster, "Qualifying" if task == "qualifying" else "Race"
    )
    expected = real_tables["labels"].query("event_id == @event_id and task == @task")
    columns = ["driver_id", "official_position", "outcome", "podium", "label_observed"]
    pd.testing.assert_frame_equal(
        replay[columns].sort_values("driver_id").reset_index(drop=True),
        expected[columns].sort_values("driver_id").reset_index(drop=True),
        check_dtype=False,
    )
