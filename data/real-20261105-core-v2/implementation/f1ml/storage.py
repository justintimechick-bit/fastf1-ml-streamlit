"""Immutable local snapshots, canonical hashes and Parquet I/O."""

import hashlib
import json
from pathlib import Path

import pandas as pd


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation is deliberate: result revisions need a new snapshot.
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def read_json(path: Path):
    return json.loads(path.read_text())


def new_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=False)


def save_table(path: Path, frame: pd.DataFrame) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, engine="pyarrow")


def seal_directory(root: Path, metadata: dict) -> dict:
    files = {
        str(p.relative_to(root)): file_digest(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.name != "receipt.json"
    }
    payload = {**metadata, "files": files}
    receipt = {**payload, "id": digest(payload)}
    write_json(root / "receipt.json", receipt)
    return receipt


def verify_directory(root: Path) -> dict:
    receipt = read_json(root / "receipt.json")
    payload = {k: v for k, v in receipt.items() if k != "id"}
    if digest(payload) != receipt["id"]:
        raise ValueError("Snapshot receipt was modified")
    actual = {
        str(p.relative_to(root)): file_digest(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.name != "receipt.json"
    }
    if actual != receipt["files"]:
        raise ValueError("Snapshot contents changed; create and freeze a new revision")
    return receipt
