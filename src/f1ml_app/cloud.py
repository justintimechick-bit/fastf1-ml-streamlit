"""Prepare an isolated GitHub deployment copy, including only required saved artifacts.

No downloads, fitting, source changes, ledger claims or evaluation occur here.
The original experiment folders are copied byte-for-byte and verified before use.
"""

import argparse
import shutil
from pathlib import Path

from f1ml.finalization.locking import code_id
from f1ml.ingestion.catalog import current_snapshot
from f1ml.storage import digest, file_digest, read_json, verify_directory, write_json
from f1ml_app.settings import load_paths


def prepare(output: Path, root: Path):
    root = root.resolve()
    output = output.resolve()
    paths = load_paths(root=root)
    if (root / "configs/app.toml").exists():
        raise ValueError("Cloud packaging currently requires the reviewed default artifact paths")
    # Fail before creating a partial release if any required artifact is missing or altered.
    for path in (
        paths.dataset,
        paths.features,
        paths.qualifying,
        paths.race,
        paths.experiments,
        paths.lock,
        paths.final,
        paths.real_experiment,
    ):
        verify_directory(path)
    locked = read_json(paths.lock / "protocol.json")
    if code_id() != locked["code_id"]:
        raise ValueError("Synthetic source fingerprint differs from its preserved lock")
    snapshot, pointer_receipt = current_snapshot(paths.catalog)
    output.mkdir(parents=True, exist_ok=False)

    def copy(path):
        destination = output / path.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if path.is_dir():
            shutil.copytree(
                path, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
            )
        else:
            shutil.copy2(path, destination)

    for name in (
        "src/f1ml",
        "src/f1ml_app",
        "src/f1ml_real",
        "configs",
        "docs",
        "tests",
        ".streamlit",
        "README.md",
        "pyproject.toml",
        "MANIFEST.in",
        "examples",
        "streamlit_app.py",
        "requirements.txt",
        "requirements.lock.txt",
        "requirements-app.lock.txt",
    ):
        path = root / name
        if name == ".streamlit":
            # Do not copy a secrets.toml or local credentials into a deployment repository.
            copy(path / "config.toml")
        else:
            copy(path)
    for name in (
        "phase3-final",
        "phase4-final",
        "phase5-final",
        "phase6-verified-experiments",
        "phase7-inputs",
        "phase7-lock",
        "phase7-final",
        "final-evaluation-ledger",
        "real-20261105-core-v2",
        "phase9-preservation.json",
    ):
        copy(root / "data" / name)
    copy(paths.catalog / "CURRENT.json")
    copy(snapshot)
    if read_json(output / "data/ingestion-v2/CURRENT.json")["id"] != pointer_receipt["id"]:
        raise ValueError("Copied catalog changed during packaging")
    for path in (
        paths.dataset,
        paths.features,
        paths.qualifying,
        paths.race,
        paths.experiments,
        paths.lock,
        paths.final,
        paths.real_experiment,
    ):
        verify_directory(output / path.relative_to(root))
    (output / ".gitignore").write_text(
        ".venv/\n__pycache__/\n*.py[cod]\n.pytest_cache/\n.ruff_cache/\n"
        ".streamlit/secrets.toml\n.env\n.DS_Store\nbuild/\ndist/\n*.egg-info/\n"
    )
    files = {
        str(p.relative_to(output)): file_digest(p) for p in sorted(output.rglob("*")) if p.is_file()
    }
    metadata = {
        "kind": "streamlit_community_cloud_release",
        "entrypoint": "streamlit_app.py",
        "python": "3.13",
        "experiment_id": read_json(paths.real_experiment / "status.json")["experiment_id"],
        "real_experiment_receipt_id": verify_directory(paths.real_experiment)["id"],
        "synthetic_lock_code_id": locked["code_id"],
        "files": files,
        "runtime": "read-only saved artifacts; no training, downloads or scoring",
    }
    metadata["id"] = digest(metadata)
    write_json(output / "DEPLOYMENT.json", metadata)
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    release = prepare(args.output, args.root)
    print(f"Prepared {len(release['files'])} files at {args.output}; release {release['id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
