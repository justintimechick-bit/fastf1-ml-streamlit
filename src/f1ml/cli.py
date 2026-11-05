"""Local commands for immutable experiments and resumable source acquisition."""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from f1ml.config import load_config
from f1ml.domain import MODES
from f1ml.evaluation.backtest import run_backtest
from f1ml.evaluation.splits import freeze, load_manifest
from f1ml.features.build import build_development
from f1ml.features.source import load_source
from f1ml.ingestion.pipeline import discover, ingest
from f1ml.storage import read_json

log = logging.getLogger(__name__)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Cutoff-aware F1 ML foundation (2024–2026)")
    root.add_argument("--config", type=Path, help="TOML config; defaults are safe local settings")
    root.add_argument("--verbose", action="store_true")
    commands = root.add_subparsers(dest="command", required=True)

    final_inputs = commands.add_parser(
        "synthetic-final-inputs", help="Independent fixture rosters; no target-result reads"
    )
    final_inputs.add_argument("--dataset", type=Path, required=True)
    final_inputs.add_argument("--manifest", type=Path, required=True)
    final_inputs.add_argument("--output", type=Path, required=True)
    final_lock = commands.add_parser(
        "final-lock", help="Audit development and freeze pre-block models/protocol"
    )
    final_lock.add_argument("--dataset", type=Path, required=True)
    final_lock.add_argument("--manifest", type=Path, required=True)
    final_lock.add_argument("--features", type=Path, required=True)
    final_lock.add_argument("--qualifying", type=Path, required=True)
    final_lock.add_argument("--race", type=Path, required=True)
    final_lock.add_argument("--experiments", type=Path, required=True)
    final_lock.add_argument("--entrants", type=Path, required=True)
    final_lock.add_argument("--cutoffs", type=Path, required=True)
    final_lock.add_argument("--output", type=Path, required=True)
    final_test = commands.add_parser(
        "final-evaluate", help="One recorded holdout evaluation after model lock"
    )
    final_test.add_argument("--lock", type=Path, required=True)
    final_test.add_argument("--dataset", type=Path, required=True)
    final_test.add_argument("--output", type=Path, required=True)
    for command in (final_lock, final_test):
        command.add_argument("--ledger", type=Path, default=Path("data/final-evaluation-ledger"))
        command.add_argument(
            "--exploratory", action="store_true", help="Explicit test reuse; needs a future holdout"
        )
    for command in (final_inputs, final_lock, final_test):
        command.add_argument("--availability", type=Path)
        command.add_argument("--event-context", type=Path)

    telemetry = commands.add_parser(
        "telemetry-download", help="Capture a bounded explicit telemetry pilot"
    )
    telemetry.add_argument("--dataset", type=Path, required=True)
    telemetry.add_argument("--manifest", type=Path, required=True)
    telemetry.add_argument("--requests", type=Path, required=True)
    telemetry.add_argument("--output", type=Path, required=True)
    telemetry.add_argument("--retry-failed", action="store_true")
    optional = commands.add_parser(
        "build-experiments", help="Build cutoff-bound optional supplements"
    )
    optional.add_argument("--dataset", type=Path, required=True)
    optional.add_argument("--manifest", type=Path, required=True)
    optional.add_argument("--features", type=Path, required=True)
    optional.add_argument("--telemetry", type=Path, required=True)
    optional.add_argument("--output", type=Path, required=True)
    experiments = commands.add_parser(
        "experiment-compare", help="Paired development feature-group ablations"
    )
    experiments.add_argument("--manifest", type=Path, required=True)
    experiments.add_argument("--features", type=Path, required=True)
    experiments.add_argument("--optional", type=Path, required=True)
    experiments.add_argument("--output", type=Path, required=True)

    for command, help_text in (
        ("download", "Discover and download normalized Phase 2 data into a resumable catalog"),
        ("resume", "Resume pending jobs; reuse completed jobs and observe retry cooldowns"),
        ("refresh-event", "Refresh one event into a new cache and source revision"),
    ):
        acquisition = commands.add_parser(command, help=help_text)
        acquisition.add_argument("--output", type=Path, required=True, help="Catalog directory")
        acquisition.add_argument(
            "--as-of",
            required=command == "download",
            help="Timezone-bearing discovery/download horizon",
        )
        acquisition.add_argument("--event", action="append", required=command == "refresh-event")
        acquisition.add_argument(
            "--retry-failed",
            action="store_true",
            help="Explicitly bypass failure cooldown and reopen timing circuit",
        )

    coverage = commands.add_parser("coverage", help="Inspect a verified normalized snapshot")
    coverage.add_argument("--catalog", type=Path, required=True)
    coverage.add_argument(
        "--level", choices=("seasons", "events", "sessions", "failures"), default="seasons"
    )
    coverage.add_argument("--event", help="Filter event/session/failure coverage")
    coverage.add_argument("--format", choices=("table", "json", "csv"), default="table")

    reparse = commands.add_parser("reparse", help="Reapply mappings/normalization offline")
    reparse.add_argument("--output", type=Path, required=True, help="Catalog directory")

    discovery = commands.add_parser("discover", help="Discover schedules; exclude testing")
    discovery.add_argument("--output", type=Path, required=True)

    ingestion = commands.add_parser(
        "ingest", help="Cache downloads and save a new Parquet revision"
    )
    ingestion.add_argument("--schedule", type=Path, required=True)
    ingestion.add_argument(
        "--as-of", required=True, help="Explicit timezone, e.g. 2026-11-04T22:00:00Z"
    )
    ingestion.add_argument("--event", action="append", help="Optional event IDs from discovery")
    ingestion.add_argument("--refresh", action="store_true", help="Renew cached source responses")
    ingestion.add_argument("--output", type=Path, required=True)

    freezing = commands.add_parser("freeze", help="Freeze eligible weekends and reserve final test")
    freezing.add_argument("--dataset", type=Path, required=True)
    freezing.add_argument("--output", type=Path, required=True)

    build = commands.add_parser(
        "build-features", help="Build only frozen development inputs/labels"
    )
    build.add_argument("--dataset", type=Path, required=True)
    build.add_argument("--manifest", type=Path, required=True)
    build.add_argument("--entrants", type=Path, required=True, help="Cutoff-aware roster CSV")
    build.add_argument("--cutoffs", type=Path, required=True, help="One row/event/task CSV")
    build.add_argument("--output", type=Path, required=True)

    backtest = commands.add_parser(
        "backtest", help="Fit baselines in expanding 2026 validation folds"
    )
    backtest.add_argument("--manifest", type=Path, required=True)
    backtest.add_argument("--features", type=Path, required=True)
    backtest.add_argument("--output", type=Path, required=True)

    comparison = commands.add_parser(
        "qualifying-compare", help="Nested chronological qualifying models and fixed rules"
    )
    comparison.add_argument("--manifest", type=Path, required=True)
    comparison.add_argument("--features", type=Path, required=True)
    comparison.add_argument("--output", type=Path, required=True)

    race_comparison = commands.add_parser(
        "race-compare", help="Chronological race classification, podium and calibration comparison"
    )
    race_comparison.add_argument("--manifest", type=Path, required=True)
    race_comparison.add_argument("--features", type=Path, required=True)
    race_comparison.add_argument("--output", type=Path, required=True)

    prediction = commands.add_parser(
        "predict", help="Predict an unseen event; final test is sealed"
    )
    prediction.add_argument("--model", type=Path, required=True, help="Trusted local model.joblib")
    prediction.add_argument("--dataset", type=Path, required=True)
    prediction.add_argument("--manifest", type=Path, required=True)
    prediction.add_argument("--event", required=True)
    prediction.add_argument("--entrants", type=Path, required=True)
    prediction.add_argument("--cutoff", required=True)
    prediction.add_argument("--mode", choices=MODES, default="historical_reconstruction")
    prediction.add_argument("--output", type=Path, required=True)

    quali_prediction = commands.add_parser(
        "predict-qualifying", help="Predict a full field using a trusted qualifying model bundle"
    )
    quali_prediction.add_argument(
        "--model", type=Path, required=True, help="Model bundle directory"
    )
    quali_prediction.add_argument("--dataset", type=Path, required=True)
    quali_prediction.add_argument("--manifest", type=Path, required=True)
    quali_prediction.add_argument("--event", required=True)
    quali_prediction.add_argument("--entrants", type=Path, required=True)
    quali_prediction.add_argument("--cutoff", required=True)
    quali_prediction.add_argument("--mode", choices=MODES, default="historical_reconstruction")
    quali_prediction.add_argument("--output", type=Path, required=True)

    race_prediction = commands.add_parser(
        "predict-race", help="After-qualifying classification and independent podium probabilities"
    )
    race_prediction.add_argument("--race-model", type=Path, required=True)
    race_prediction.add_argument("--podium-model", type=Path, required=True)
    race_prediction.add_argument("--dataset", type=Path, required=True)
    race_prediction.add_argument("--manifest", type=Path, required=True)
    race_prediction.add_argument("--event", required=True)
    race_prediction.add_argument("--entrants", type=Path, required=True)
    race_prediction.add_argument("--cutoff", required=True)
    race_prediction.add_argument("--mode", choices=MODES, default="historical_reconstruction")
    race_prediction.add_argument("--output", type=Path, required=True)

    for command in (
        freezing,
        build,
        prediction,
        quali_prediction,
        race_prediction,
        telemetry,
        optional,
    ):
        command.add_argument(
            "--availability",
            type=Path,
            help="Reviewed session completion/availability CSV for a Phase 2 snapshot",
        )
        command.add_argument(
            "--event-context", type=Path, help="Reviewed circuit/layout identity CSV"
        )

    dictionary = commands.add_parser(
        "feature-dictionary", help="Write the versioned feature dictionary"
    )
    dictionary.add_argument("--output", type=Path, required=True)

    demo = commands.add_parser("demo", help="Run deterministic offline SYNTHETIC workflow")
    demo.add_argument("--output", type=Path, required=True)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        config = load_config(args.config)
        logging.basicConfig(
            level="DEBUG" if args.verbose else config.log_level,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
        if args.command == "feature-dictionary":
            from f1ml.features.dictionary import write_dictionary
            from f1ml.storage import new_directory

            new_directory(args.output)
            print(
                json.dumps(
                    {"dictionary_id": write_dictionary(args.output), "output": str(args.output)},
                    indent=2,
                )
            )
            return 0
        if args.command == "telemetry-download":
            from f1ml.experiments.acquisition import TelemetryCatalog, TelemetryProvider

            result = TelemetryCatalog(args.output).run(
                load_source(args.dataset, args.availability, args.event_context),
                load_manifest(args.manifest),
                pd.read_csv(args.requests, dtype=str),
                TelemetryProvider(Path(config.cache_dir) / "schedules", config.experiments),
                config.experiments,
                args.retry_failed,
            )
            print(json.dumps(result, indent=2))
            return 0 if all(r["status"] == "complete" for r in result) else 1
        if args.command == "reparse":
            from f1ml.ingestion.catalog import Catalog

            result = Catalog(args.output, config, None).reparse()
            print(
                json.dumps(
                    {"id": result["id"], "snapshot": result["snapshot"], "run": result["run"]},
                    indent=2,
                )
            )
            return 1 if result["partial_jobs"] else 0
        if args.command in ("download", "resume", "refresh-event"):
            from f1ml.ingestion.catalog import Catalog
            from f1ml.ingestion.provider import FastF1Provider

            if args.command == "refresh-event" and len(args.event) != 1:
                raise ValueError("refresh-event requires exactly one --event")
            provider = FastF1Provider(
                Path(config.cache_dir), max_api_pages=config.ingestion.max_api_pages
            )
            result = Catalog(args.output, config, provider).run(
                as_of=args.as_of,
                selected=args.event,
                create=args.command == "download",
                retry_failed=args.retry_failed,
                refresh=args.event[0] if args.command == "refresh-event" else None,
            )
            print(
                json.dumps(
                    {
                        "id": result["id"],
                        "snapshot": result["snapshot"],
                        "run": result["run"],
                        "partial_jobs": result["partial_jobs"],
                        "failure_rows": result["failure_rows"],
                        "timing_circuit": result["timing_circuit"],
                        "results_circuit": result.get("results_circuit"),
                    },
                    indent=2,
                )
            )
            return (
                1
                if result["discovery_failures"]
                or result["partial_jobs"]
                or result["run"]["deferred_jobs"]
                else 0
            )
        if args.command == "coverage":
            from f1ml.ingestion.coverage import inspect_coverage

            frame = inspect_coverage(args.catalog, args.level)
            if args.event:
                if "event_id" not in frame:
                    raise ValueError("Event filtering requires events/sessions/failures coverage")
                frame = frame.loc[frame.event_id.eq(args.event)]
            if args.format == "json":
                print(frame.to_json(orient="records", indent=2))
            elif args.format == "csv":
                print(frame.to_csv(index=False), end="")
            else:
                columns = [
                    c
                    for c in frame
                    if c
                    in (
                        "year",
                        "event_id",
                        "name",
                        "session_id",
                        "kind",
                        "event_format",
                        "events",
                        "driver_count",
                        "results_rows",
                        "laps_rows",
                        "weather_rows",
                        "completion_state",
                        "practice_expected",
                        "practice_with_laps",
                        "result_sessions",
                        "sessions_with_weather",
                        "sessions_missing_targets",
                        "missing_rank_targets",
                        "unusual_classifications",
                        "failures",
                        "blocked_sessions",
                        "future_sessions",
                        "type",
                        "message",
                    )
                ]
                print(frame[columns].to_string(index=False))
            return 0
        if args.command in ("discover", "ingest"):
            from f1ml.ingestion.provider import FastF1Provider

            provider = FastF1Provider(Path(config.cache_dir), getattr(args, "refresh", False))
            if args.command == "discover":
                result = discover(provider, config, args.output)
                log.info("Discovered %s GP events", len(result["events"]))
                return 0
            result = ingest(provider, read_json(args.schedule), args.output, args.as_of, args.event)
            log.info("Saved source revision %s", result["id"])
            # Partial ingestion is preserved for diagnosis but visibly fails the command.
            return 1 if result["failures"] else 0
        if args.command == "synthetic-final-inputs":
            from f1ml.finalization.locking import synthetic_inputs

            source = load_source(args.dataset, args.availability, args.event_context)
            result = synthetic_inputs(source, load_manifest(args.manifest, source), args.output)
        elif args.command == "final-lock":
            from f1ml.finalization.locking import lock_models

            source = load_source(args.dataset, args.availability, args.event_context)
            result = lock_models(
                source,
                load_manifest(args.manifest, source),
                args.features,
                args.qualifying,
                args.race,
                args.experiments,
                args.entrants,
                args.cutoffs,
                args.output,
                args.ledger,
                args.exploratory,
            )
        elif args.command == "final-evaluate":
            from f1ml.finalization.evaluate import evaluate_once

            result = evaluate_once(
                args.lock,
                load_source(args.dataset, args.availability, args.event_context),
                args.output,
                args.ledger,
                args.exploratory,
            )
        elif args.command == "demo":
            from f1ml.demo import run_demo

            result = run_demo(args.output)
        elif args.command == "freeze":
            result = freeze(
                load_source(args.dataset, args.availability, args.event_context),
                config,
                args.output,
            )
        elif args.command == "build-features":
            dataset = load_source(args.dataset, args.availability, args.event_context)
            manifest = load_manifest(args.manifest, dataset)
            result = build_development(
                dataset,
                manifest,
                pd.read_csv(args.entrants, dtype=str),
                pd.read_csv(args.cutoffs, dtype=str),
                args.output,
            )
        elif args.command == "build-experiments":
            from f1ml.experiments.build import build_supplements

            result = build_supplements(
                load_source(args.dataset, args.availability, args.event_context),
                load_manifest(args.manifest),
                args.features,
                args.telemetry,
                args.output,
                config.experiments,
            )
        elif args.command == "experiment-compare":
            from f1ml.experiments.evaluate import run_experiments

            result = run_experiments(
                load_manifest(args.manifest), args.features, args.optional, args.output, config
            )
        elif args.command == "backtest":
            result = run_backtest(load_manifest(args.manifest), args.features, args.output)
        elif args.command == "race-compare":
            from f1ml.evaluation.race import run_comparison

            result = run_comparison(
                load_manifest(args.manifest),
                args.features,
                args.output,
                config.race,
                config.random_seed,
            )
            print(
                json.dumps(
                    {"id": result["id"], "output": str(args.output), "status": result["status"]},
                    indent=2,
                )
            )
            return 0 if all(result["selected_models"].values()) else 1
        elif args.command == "predict-race":
            from f1ml.models.race import load_model
            from f1ml.prediction.race import predict_event

            race_model = load_model(args.race_model, "race")
            podium_model = load_model(args.podium_model, "podium")
            result = predict_event(
                race_model,
                podium_model,
                load_source(args.dataset, args.availability, args.event_context),
                load_manifest(args.manifest),
                args.event,
                args.cutoff,
                pd.read_csv(args.entrants, dtype=str),
                args.mode,
                args.output,
                model_ids={
                    "race": read_json(args.race_model / "receipt.json")["id"],
                    "podium": read_json(args.podium_model / "receipt.json")["id"],
                },
            )
        elif args.command == "qualifying-compare":
            from f1ml.evaluation.qualifying import run_comparison

            result = run_comparison(
                load_manifest(args.manifest),
                args.features,
                args.output,
                config.qualifying,
                config.random_seed,
            )
            print(
                json.dumps(
                    {"id": result["id"], "output": str(args.output), "status": result["status"]},
                    indent=2,
                )
            )
            return 0 if result["common_evaluation_events"] else 1
        elif args.command == "predict-qualifying":
            from f1ml.models.qualifying import load_model
            from f1ml.prediction.qualifying import predict_event

            dataset = load_source(args.dataset, args.availability, args.event_context)
            result = predict_event(
                load_model(args.model),
                dataset,
                load_manifest(args.manifest),
                args.event,
                args.cutoff,
                pd.read_csv(args.entrants, dtype=str),
                args.mode,
                args.output,
            )
        elif args.command == "predict":
            from f1ml.prediction.service import load_model, predict_event

            dataset = load_source(args.dataset, args.availability, args.event_context)
            manifest = load_manifest(args.manifest)
            result = predict_event(
                load_model(args.model),
                dataset,
                manifest,
                args.event,
                args.cutoff,
                pd.read_csv(args.entrants, dtype=str),
                args.mode,
                args.output,
            )
        print(json.dumps({"id": result["id"], "output": str(args.output)}, indent=2))
        return 0
    except (ValueError, KeyError, FileNotFoundError, FileExistsError, StopIteration) as exc:
        log.error("%s", exc, exc_info=args.verbose)
        return 2
