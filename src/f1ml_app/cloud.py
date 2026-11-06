"""Package the real-only Streamlit lab; never acquire data or fit models."""

import argparse
import shutil
from pathlib import Path

from f1ml.storage import digest, file_digest, verify_directory, write_json
from f1ml_real.lab import load_run

DATASET = Path("data/real-core-20261105-v4")
RUNS = Path("artifacts/real-lab")


def prepare(output: Path, root: Path):
    root, output = root.resolve(), output.resolve()
    receipt = verify_directory(root / DATASET)
    if receipt.get("data_kind") != "real":
        raise ValueError("Only reviewed real data may enter this release")
    runs = []
    for path in sorted((root / RUNS).glob("*/receipt.json")):
        run = load_run(path.parent)
        if run["summary"]["source_snapshot_id"] != receipt["id"]:
            continue
        runs.append(path.parent)
    if not runs:
        raise ValueError("Prepare a sealed real model replay before publishing")
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
        "tests/real",
        "tools/export_real_notebook.py",
        "notebooks/real-mexico-2026.ipynb",
        ".streamlit/config.toml",
        "README.md",
        "pyproject.toml",
        "MANIFEST.in",
        "streamlit_app.py",
        "requirements.txt",
        "requirements.lock.txt",
        "requirements-app.lock.txt",
        "requirements-real.lock.txt",
    ):
        copy(root / name)
    copy(root / DATASET)
    for path in runs:
        copy(path)
        verify_directory(output / path.relative_to(root))
    verify_directory(output / DATASET)
    (output / ".gitignore").write_text(
        ".venv/\n__pycache__/\n*.py[cod]\n.pytest_cache/\n.ruff_cache/\n"
        ".streamlit/secrets.toml\n.env\n.DS_Store\nbuild/\ndist/\n*.egg-info/\n"
    )
    files = {
        str(p.relative_to(output)): file_digest(p) for p in sorted(output.rglob("*")) if p.is_file()
    }
    metadata = {
        "kind": "streamlit_community_cloud_real_lab",
        "entrypoint": "streamlit_app.py",
        "python": "3.13",
        "data_kind": "real",
        "dataset": str(DATASET),
        "source_snapshot_id": receipt["id"],
        "saved_runs": [str(path.relative_to(root)) for path in runs],
        "files": files,
        "runtime": "Read saved facts/results on rerun; fit only after explicit Run historical test",
        "excluded": ["synthetic datasets/models/ledgers", "raw PDFs", "caches", "secrets"],
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
