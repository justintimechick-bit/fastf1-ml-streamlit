"""Run with ``python -m f1ml_real audit --config ... --output ...``."""

import argparse
import shutil
from pathlib import Path

from f1ml.storage import read_json, verify_directory
from f1ml_real.audit import audit


def fork_catalog(source, output):
    from f1ml.ingestion.catalog import current_snapshot, writer_lock
    from f1ml.storage import write_json

    # Wait for an acquisition writer before copying its resumable state. No lock is deleted.
    with writer_lock(source):
        _, receipt = current_snapshot(source)
        shutil.copytree(source, output)
    write_json(
        output / "fork.json",
        {
            "source": str(source),
            "snapshot_id": receipt["id"],
            "purpose": "isolated Phase 9 real-data acquisition",
        },
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Real-data experiment inventory and readiness")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("audit", help="Save an immutable source audit; never fit or score")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    ready = commands.add_parser("check-ready", help="Fail closed before any real-data training")
    ready.add_argument("--experiment", type=Path, required=True)
    fork = commands.add_parser(
        "fork-catalog", help="Copy a catalog for isolated resumable downloads"
    )
    fork.add_argument("--source", type=Path, required=True)
    fork.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "fork-catalog":
        fork_catalog(args.source, args.output)
        return 0
    if args.command == "audit":
        receipt = audit(args.config, args.output)
        print(f"{receipt['experiment_id']}: {receipt['state']} ({receipt['id']})")
        return 1 if receipt["state"] == "blocked_before_training" else 0
    verify_directory(args.experiment)
    status = read_json(args.experiment / "status.json")
    if not status["final_manifest_frozen"]:
        print("Training blocked: no reviewed modeling dataset or frozen final manifest.")
        for reason in status["blockers"]:
            print(reason)
        return 2
    print(
        "Readiness requires a separate reviewed training implementation; this command never fits."
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
