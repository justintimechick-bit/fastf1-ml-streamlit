"""Exercise published source facts and difficult classifications, never fake race rows."""

import json
from urllib.parse import urlparse

import pandas as pd
import pytest

from f1ml.storage import file_digest, verify_directory


def test_real_snapshot_integrity_and_source_evidence(dataset_root, real_tables):
    assert verify_directory(dataset_root)
    receipt = json.loads((dataset_root / "receipt.json").read_text())
    assert receipt["data_kind"] == "real"
    assert receipt["eligible_events"] == len(real_tables["events"]) == 19
    assert "exploratory" in receipt["mode"]
    assert "no untouched holdout" in receipt["exposure"]
    assert set(receipt["missing_groups"]) == {"practice_laps", "qualifying_laps", "weather"}
    for name, digest in receipt["files"].items():
        assert file_digest(dataset_root / name) == digest
    docs = real_tables["documents"]
    assert docs.sha256.str.fullmatch(r"[a-f0-9]{64}").all()
    assert docs.url.map(lambda u: urlparse(u).hostname == "www.fia.com").all()
    assert set(docs.purpose) == {"roster", "label"}
    assert len(docs) == len(real_tables["events"]) * 4


def test_every_driver_has_independent_precutoff_roster_and_unique_identity(real_tables):
    rosters, labels = real_tables["rosters"], real_tables["labels"]
    keys = ["event_id", "task", "driver_id"]
    assert not rosters.duplicated(keys).any()
    assert not labels.duplicated(keys).any()
    assert (
        rosters[keys]
        .sort_values(keys)
        .reset_index(drop=True)
        .equals(labels[keys].sort_values(keys).reset_index(drop=True))
    )
    assert not rosters[["driver_id", "team_id", "driver_number"]].isna().any().any()
    assert rosters.groupby(["event_id", "task"]).size().eq(22).all()
    assert (
        pd.to_datetime(rosters.known_at, utc=True) <= pd.to_datetime(rosters.cutoff, utc=True)
    ).all()
    assert rosters.source.str.contains("entry_list", case=False).all()
    assert labels.source.str.contains("classification", case=False).all()
    assert not rosters.source.eq(labels.source).any()


def test_cutoffs_separate_qualifying_and_postqualifying_race(real_tables):
    events = real_tables["events"]
    qcut = pd.to_datetime(events.qualifying_cutoff, utc=True)
    qavailable = pd.to_datetime(events.qualifying_available_at, utc=True)
    rcut = pd.to_datetime(events.race_cutoff, utc=True)
    ravailable = pd.to_datetime(events.race_available_at, utc=True)
    assert (qcut < qavailable).all()
    assert (qavailable < rcut).all()
    assert (rcut < ravailable).all()
    assert not real_tables["exclusions"].event_id.isin(events.event_id).any()
    # Future weekends remain exclusions even though their calendar entry exists.
    assert real_tables["exclusions"].reason.eq("race_not_started_as_of").all()


def test_official_special_statuses_never_receive_invented_tail_ranks(real_tables):
    labels = real_tables["labels"]
    special = labels.outcome.isin(["dsq", "non_classified", "dns", "withdrawn"])
    assert special.any()
    assert labels.loc[special, "official_position"].isna().all()
    assert labels.loc[special & labels.task.eq("race"), "podium"].eq(0).all()
    assert labels.loc[labels.task.eq("qualifying"), "podium"].isna().all()
    assert labels.loc[special, "label_observed"].all()
    unknown = labels.outcome.eq("unknown")
    assert unknown.sum() == 5
    assert labels.loc[unknown, ["official_position", "podium"]].isna().all().all()
    assert not labels.loc[unknown, "label_observed"].any()
    for _, group in labels.groupby(["event_id", "task"]):
        observed = group.official_position.dropna()
        assert not observed.duplicated().any()
        assert (observed >= 1).all() and (observed <= len(group)).all()
    assert labels[labels.task.eq("race")].groupby("event_id").podium.sum().eq(3).all()


def test_mexico_substitution_and_penalties_survive_normalization(real_tables):
    rosters = real_tables["rosters"].query("event_id == '2026-19'")
    labels = real_tables["labels"].query("event_id == '2026-19'")
    assert "ryo_hirakawa" in set(rosters.driver_id)
    assert "ocon" not in set(rosters.driver_id)
    assert rosters.loc[rosters.driver_id.eq("ryo_hirakawa"), "driver_number"].eq(50).all()
    lawson = labels.query("driver_id == 'lawson' and task == 'qualifying'").iloc[0]
    assert lawson.outcome == "dsq" and pd.isna(lawson.official_position)
    race = labels[labels.task.eq("race")]
    assert race.official_position.notna().sum() == 17
    assert set(race.loc[race.podium.eq(1), "driver_id"]) == {
        "max_verstappen",
        "hamilton",
        "antonelli",
    }


def test_loading_tampered_real_snapshot_fails_closed(dataset_root, tmp_path):
    """Corrupting a copied receipt is a software safeguard, not a new race observation."""
    import shutil

    from f1ml_real.lab import RealDataset

    copied = tmp_path / "changed"
    shutil.copytree(dataset_root, copied)
    with (copied / "labels.csv").open("a") as stream:
        stream.write("\n")
    with pytest.raises((ValueError, RuntimeError, AssertionError)):
        RealDataset(copied)
