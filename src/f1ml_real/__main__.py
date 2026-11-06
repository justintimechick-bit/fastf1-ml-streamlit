"""Acquire reviewed real facts or explicitly run a historical model replay."""

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
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser(
        "prepare-real", help="Cache FIA evidence and seal a real snapshot"
    )
    capture.add_argument("--catalog", type=Path, required=True)
    capture.add_argument("--cache", type=Path, required=True)
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument("--as-of", required=True, help="UTC acquisition horizon")
    capture.add_argument("--offline", action="store_true", help="Require verified cached evidence")
    replay = commands.add_parser("backtest", help="Explicitly fit or reuse a sealed real replay")
    replay.add_argument("--dataset", type=Path, default=Path("data/real-core-20261105-v4"))
    replay.add_argument("--event", required=True, help="Reviewed event ID, e.g. 2026-19")
    replay.add_argument("--runs", type=Path, default=Path("artifacts/real-lab"))
    replay.add_argument(
        "--last-n", type=int, help="Limit fitted events; default all eligible history"
    )
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
    validate = commands.add_parser("validate-source", help="Replay normalization offline")
    validate.add_argument("--experiment", type=Path, required=True)
    validate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare-real":
        from f1ml_real.fia import prepare_dataset

        receipt = prepare_dataset(
            args.catalog, args.cache, args.output, as_of=args.as_of, offline=args.offline
        )
        print(f"Sealed {receipt['eligible_events']} real 2026 events: {receipt['id']}")
        return 0
    if args.command == "backtest":
        from f1ml_real.lab import cached_run, load_run

        path = cached_run(args.dataset, args.event, args.runs, last_n=args.last_n)
        run = load_run(path)
        print(f"Verified real historical replay: {path}")
        print(run["summary"]["evaluation_status"])
        print(run["metrics"].to_string(index=False))
        return 0
    if args.command == "fork-catalog":
        fork_catalog(args.source, args.output)
        return 0
    if args.command == "validate-source":
        from f1ml_real.validate import validate_sources

        result = validate_sources(args.experiment, args.output)
        print(
            f"{result['result_sessions']} sessions, {result['driver_rows']} rows, "
            f"{result['mismatched_sessions']} source replay mismatches"
        )
        return 1 if result["mismatched_sessions"] else 0
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
