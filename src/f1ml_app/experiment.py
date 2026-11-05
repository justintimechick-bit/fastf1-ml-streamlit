"""Read-only access to the frozen experiment: dataset, manifest, features, models, results.

Every artifact is integrity-checked with the pipeline's own verifiers (``load_source``,
``load_manifest``, ``verify_directory``). Labels are held in separate attributes from
feature inputs and are only ever joined to predictions for display, after prediction.
"""

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import pandas as pd

from f1ml.domain import PRACTICES, utc
from f1ml.evaluation.splits import gp_session, load_manifest
from f1ml.features.source import available, load_source
from f1ml.storage import read_json, verify_directory
from f1ml_app.settings import AppPaths

ROLE_LABELS = {
    "final_test": "Final test · frozen",
    "development_validation": "Development · validation fold",
    "development_training": "Development · training only",
    "excluded": "Excluded at freeze",
    "outside_manifest": "Outside manifest",
}
STRATEGY_LABELS = {
    "2026_only": "2026-only",
    "equal": "Pooled 2024–26 (equal)",
    "weighted": "Weighted history (2026-favored)",
    "fixed_rule": "Fixed rule",
}
FAMILY_LABELS = {
    "ridge": "Ridge regression",
    "random_forest": "Random forest",
    "gradient_boosting": "Histogram gradient boosting",
    "logistic": "Logistic regression",
    "baseline_recent": "Recent-form rule",
    "baseline_practice": "Clean-practice rule",
    "baseline_qualifying": "Qualifying-order rule",
}


class ArtifactUnavailable(RuntimeError):
    """A saved artifact needed by one view is missing or failed integrity checks."""


@dataclass(frozen=True)
class ModelCard:
    """Lineage summary of one saved model bundle (metadata only; no unpickling)."""

    task: str
    family: str
    strategy: str
    role: str
    path: Path
    model_id: str
    validation_event: str | None
    metadata: dict = field(repr=False, compare=False, hash=False)

    @property
    def train_events(self) -> list[str]:
        return list(self.metadata["train_events"])

    @property
    def dependency_events(self) -> list[str]:
        return list(self.metadata["dependency_events"])

    @property
    def training_end(self) -> str:
        return self.metadata["max_weekend_end"]

    @property
    def label(self) -> str:
        return f"{FAMILY_LABELS.get(self.family, self.family)} · {STRATEGY_LABELS[self.strategy]}"

    def lineage_problems(self, manifest, event_id, cutoff, weekend_start) -> list[str]:
        """Mirror of the guards in ``f1ml.prediction.*.predict_checked``, for explanation.

        ``predict_checked`` remains the authority at prediction time; this lets the interface
        say *why* a saved model is not eligible without attempting a prediction.
        """
        meta, problems = self.metadata, []
        if meta["manifest_id"] != manifest["id"]:
            problems.append("belongs to a different frozen manifest")
        if event_id in meta["dependency_events"]:
            problems.append("this weekend was used for its training, tuning or selection")
        if cutoff is not None:
            for name, text in (
                ("max_feature_cutoff", "training features"),
                ("max_label_end", "training labels"),
                ("max_weekend_end", "training weekends"),
            ):
                if utc(meta[name]) >= utc(cutoff):
                    problems.append(
                        f"{text} reach {utc(meta[name]):%Y-%m-%d %H:%M} UTC, not strictly "
                        f"before the {utc(cutoff):%Y-%m-%d %H:%M} UTC cutoff"
                    )
        if weekend_start is not None and utc(meta["max_weekend_end"]) >= utc(weekend_start):
            problems.append("depends on a weekend that had not finished before this one began")
        return problems


def _first(frame: pd.DataFrame, **match) -> pd.Series | None:
    mask = pd.Series(True, index=frame.index)
    for key, value in match.items():
        mask &= frame[key].eq(value)
    rows = frame.loc[mask]
    return rows.iloc[0] if len(rows) else None


class Experiment:
    """The frozen experiment named by ``AppPaths`` (Phase 3 dataset → Phase 7 results)."""

    def __init__(self, paths: AppPaths):
        self.paths = paths
        for required in (paths.dataset, paths.manifest, paths.features):
            if not required.exists():
                raise ArtifactUnavailable(f"Missing prepared artifact: {required}")
        self.source = load_source(paths.dataset)
        self.manifest = load_manifest(paths.manifest, self.source)
        self.features_receipt = verify_directory(paths.features)
        if self.features_receipt["manifest_id"] != self.manifest["id"]:
            raise ArtifactUnavailable("Saved features belong to a different manifest")
        root = paths.features
        # Prediction inputs (cutoff-bound features) and outcomes (labels) stay separate.
        self.qualifying_features = pd.read_parquet(root / "qualifying_features.parquet")
        self.race_features = pd.read_parquet(root / "race_features.parquet")
        self._qualifying_labels = pd.read_parquet(root / "qualifying_labels.parquet")
        self._race_labels = pd.read_parquet(root / "race_labels.parquet")
        self.feature_coverage = pd.read_parquet(root / "coverage.parquet")
        cutoffs = [pd.read_csv(paths.development_cutoffs, dtype=str).assign(origin="development")]
        if (paths.lock / "cutoffs.csv").exists():
            cutoffs.append(
                pd.read_csv(paths.lock / "cutoffs.csv", dtype=str).assign(origin="final_lock")
            )
        self.cutoffs = pd.concat(cutoffs, ignore_index=True)
        if self.cutoffs.duplicated(["event_id", "task"]).any():
            raise ArtifactUnavailable("Duplicate event/task cutoffs across saved inputs")

    # ----- identity, roles and data kind -------------------------------------------------

    @property
    def manifest_id(self) -> str:
        return self.manifest["id"]

    @cached_property
    def is_synthetic(self) -> bool:
        return any(
            "synthetic" in str(spec.get("availability_basis", ""))
            or "SYNTHETIC" in str(event.get("name", ""))
            for event in self.source.events
            for spec in event["sessions"]
        )

    @property
    def data_kind(self) -> str:
        return "synthetic_validation" if self.is_synthetic else "real"

    def role(self, event_id: str) -> str:
        manifest = self.manifest
        if event_id in manifest["final_test"]:
            return "final_test"
        if event_id in {fold["validate"] for fold in manifest["folds"]}:
            return "development_validation"
        if event_id in manifest["development"]:
            return "development_training"
        if event_id in {e["event_id"] for e in manifest["excluded"]}:
            return "excluded"
        return "outside_manifest"

    def exclusion_reason(self, event_id: str) -> str | None:
        for item in self.manifest["excluded"]:
            if item["event_id"] == event_id:
                return item["reason"]
        return None

    @cached_property
    def events(self) -> pd.DataFrame:
        rows = []
        for event in self.source.events:
            sessions = {s["kind"]: s for s in event["sessions"]}
            rows.append(
                {
                    "event_id": event["event_id"],
                    "year": int(event["year"]),
                    "round": int(event["round"]),
                    "name": event.get("name") or event["event_id"],
                    "location": event.get("location"),
                    "circuit_id": event.get("circuit_id"),
                    "role": self.role(event["event_id"]),
                    "weekend_start": self.weekend_start(event["event_id"]),
                    "qualifying_completed": bool(sessions.get("Qualifying", {}).get("completed")),
                    "race_completed": bool(sessions.get("Race", {}).get("completed")),
                }
            )
        return pd.DataFrame(rows).sort_values(["year", "round"]).reset_index(drop=True)

    def event(self, event_id: str) -> dict:
        return self.source.event(event_id)

    def weekend_start(self, event_id: str) -> str:
        # Same definition as the CLI prediction commands: earliest scheduled session start.
        starts = [utc(s["scheduled_start"]) for s in self.event(event_id)["sessions"]]
        return min(starts).isoformat()

    def cutoff(self, event_id: str, task: str) -> tuple[str, str] | None:
        row = _first(self.cutoffs, event_id=event_id, task=task)
        return None if row is None else (row["cutoff"], row["mode"])

    def gp_session(self, event_id: str, kind: str) -> dict | None:
        try:
            return gp_session(self.event(event_id), kind)
        except ValueError:
            return None

    def session_rows(self, event_id: str) -> pd.DataFrame:
        """Weekend timeline with availability at each prediction cutoff (pipeline rule)."""
        q_cut, r_cut = self.cutoff(event_id, "qualifying"), self.cutoff(event_id, "race")
        rows = []
        for spec in sorted(self.event(event_id)["sessions"], key=lambda s: s["scheduled_start"]):
            kind = spec["kind"]
            in_scope = kind in PRACTICES or kind in ("Qualifying", "Race")
            row = {
                "session": kind,
                "scheduled_start": utc(spec["scheduled_start"]),
                "completed": bool(spec.get("completed")),
                "official_results": bool(spec.get("official_results")),
                "saved_table": bool(spec.get("table")),
                "in_model_scope": in_scope,
            }
            for label, cut, permitted in (
                ("usable_at_qualifying_cutoff", q_cut, kind in PRACTICES),
                ("usable_at_race_cutoff", r_cut, kind in PRACTICES or kind == "Qualifying"),
            ):
                # Same rule as build_event: practice needs timing availability before cutoff.
                row[label] = bool(
                    in_scope
                    and permitted
                    and cut
                    and available(spec, utc(cut[0]), cut[1], timing=kind in PRACTICES)
                )
            if not in_scope:
                row["status"] = "Sprint format session · outside model scope"
            elif row["completed"] and row["saved_table"]:
                row["status"] = "Completed · saved"
            elif row["completed"]:
                row["status"] = "Completed · no saved data"
            else:
                row["status"] = "Not completed"
            rows.append(row)
        return pd.DataFrame(rows)

    # ----- saved feature inputs and label-side outcomes --------------------------------

    def saved_inputs(self, event_id: str, task: str) -> pd.DataFrame:
        frame = self.qualifying_features if task == "qualifying" else self.race_features
        return frame.loc[frame.event_id.eq(event_id)].reset_index(drop=True)

    def development_labels(self, event_id: str, task: str) -> pd.DataFrame:
        """Official outcomes for display only; never passed to a predictor."""
        frame = self._qualifying_labels if task == "qualifying" else self._race_labels
        return frame.loc[frame.event_id.eq(event_id)].reset_index(drop=True)

    # ----- saved model bundles ----------------------------------------------------------

    def _card(self, root: Path, role: str) -> ModelCard:
        meta = read_json(root / "model.json")
        return ModelCard(
            task=meta["task"],
            family=meta["family"],
            strategy=meta["strategy"],
            role=role,
            path=root,
            model_id=read_json(root / "receipt.json")["id"],
            validation_event=meta.get("validation_event"),
            metadata=meta,
        )

    @cached_property
    def model_cards(self) -> list[ModelCard]:
        cards = []
        folds = self.paths.qualifying / "fold_models"
        cards += [
            self._card(p.parent, "outer_fold") for p in sorted(folds.glob("*/*/*/model.json"))
        ]
        folds = self.paths.race / "fold_models"
        cards += [
            self._card(p.parent, "outer_fold") for p in sorted(folds.glob("*/*/*/*/model.json"))
        ]
        for task in ("qualifying", "race", "podium"):
            root = self.paths.lock / "models" / task
            if (root / "model.json").exists():
                cards.append(self._card(root, "locked_refit"))
        return cards

    def fold_cards(self, task: str, event_id: str) -> list[ModelCard]:
        return [
            c
            for c in self.model_cards
            if c.task == task and c.role == "outer_fold" and c.validation_event == event_id
        ]

    def locked_card(self, task: str) -> ModelCard | None:
        return next(
            (c for c in self.model_cards if c.task == task and c.role == "locked_refit"), None
        )

    def eligible_cards(self, task: str, event_id: str) -> list[ModelCard]:
        cut = self.cutoff(event_id, "qualifying" if task == "qualifying" else "race")
        start = self.weekend_start(event_id)
        return [
            c
            for c in self.model_cards
            if c.task == task
            and not c.lineage_problems(self.manifest, event_id, cut[0] if cut else None, start)
        ]

    def selected_config(self, task: str) -> tuple[str, str] | None:
        """Family/strategy chosen in development (Phase 4/5 summaries)."""
        if task == "qualifying":
            chosen = self.qualifying_comparison["summary"].get("selected_model")
        else:
            chosen = self.race_comparison["summary"].get("selected_models", {}).get(task)
        return (chosen["family"], chosen["strategy"]) if chosen else None

    # ----- development comparisons (Phase 4/5/6) -----------------------------------------

    @cached_property
    def qualifying_comparison(self) -> dict:
        root = self.paths.qualifying
        if not (root / "receipt.json").exists():
            raise ArtifactUnavailable(f"Missing Phase 4 comparison: {root}")
        return {
            "receipt": verify_directory(root),
            "summary": read_json(root / "summary.json"),
            "eligibility": pd.DataFrame(read_json(root / "eligibility.json")).assign(
                task="qualifying"
            ),
            "event_metrics": pd.read_parquet(root / "event_metrics.parquet"),
            "predictions": pd.read_parquet(root / "predictions.parquet"),
            "search": read_json(root / "search_config.json"),
            "tuning": read_json(root / "tuning.json"),
        }

    @cached_property
    def race_comparison(self) -> dict:
        root = self.paths.race
        if not (root / "receipt.json").exists():
            raise ArtifactUnavailable(f"Missing Phase 5 comparison: {root}")
        return {
            "receipt": verify_directory(root),
            "summary": read_json(root / "summary.json"),
            "eligibility": pd.DataFrame(read_json(root / "eligibility.json")),
            "event_metrics": pd.read_parquet(root / "event_metrics.parquet"),
            "predictions": pd.read_parquet(root / "predictions.parquet"),
            "calibration": pd.read_parquet(root / "calibration.parquet"),
            "target_coverage": pd.read_parquet(root / "target_coverage.parquet"),
            "search": read_json(root / "search_config.json"),
        }

    @cached_property
    def feature_experiments(self) -> dict:
        root = self.paths.experiments
        if not (root / "receipt.json").exists():
            raise ArtifactUnavailable(f"Missing Phase 6 experiments: {root}")
        return {
            "receipt": verify_directory(root),
            "retention": pd.DataFrame(read_json(root / "retention.json")),
            "summary": pd.read_parquet(root / "summary.parquet"),
            "missingness": pd.read_parquet(root / "missingness.parquet"),
        }

    # ----- frozen final evaluation (Phase 7) --------------------------------------------

    @cached_property
    def final(self) -> dict:
        root, lock = self.paths.final, self.paths.lock
        if not (root / "receipt.json").exists() or not (lock / "receipt.json").exists():
            raise ArtifactUnavailable(
                "The locked final evaluation has not been run; no final-test results exist"
            )
        receipt = verify_directory(root)
        lock_receipt = verify_directory(lock)
        if receipt["lock_id"] != lock_receipt["id"]:
            raise ArtifactUnavailable("Final evaluation belongs to a different model lock")
        if receipt["manifest_id"] != self.manifest_id:
            raise ArtifactUnavailable("Final evaluation belongs to a different manifest")
        seals = read_json(root / "prediction_seals.json")
        seal_receipts = {task: verify_directory(root / f"{task}_prediction_seal") for task in seals}
        for task, sealed in seal_receipts.items():
            if sealed["id"] != seals[task]:
                raise ArtifactUnavailable(f"{task} prediction seal does not match its record")
        from f1ml.finalization.locking import code_id

        lock_protocol = read_json(lock / "protocol.json")
        claim = self.paths.ledger / self.manifest_id / "claim.json"
        result = self.paths.ledger / self.manifest_id / "result.json"
        return {
            "receipt": receipt,
            "lock_receipt": lock_receipt,
            "protocol": read_json(root / "protocol.json"),
            "lock_protocol": lock_protocol,
            # The interface lives outside src/f1ml, so it does not alter this fingerprint.
            "code_matches_lock": lock_protocol["code_id"] == code_id(),
            "summary": read_json(root / "summary.json"),
            "seals": seal_receipts,
            "qualifying_seal": pd.read_parquet(
                root / "qualifying_prediction_seal/predictions.parquet"
            ),
            "qualifying_inputs": pd.read_parquet(
                root / "qualifying_prediction_seal/features.parquet"
            ),
            "race_seal": pd.read_parquet(root / "race_prediction_seal/predictions.parquet"),
            "race_inputs": pd.read_parquet(root / "race_prediction_seal/features.parquet"),
            "race_and_podium": pd.read_parquet(root / "race_and_podium_forecasts.parquet"),
            "top_three": read_json(root / "top_three_comparison.json"),
            "event_metrics": pd.read_parquet(root / "event_metrics.parquet"),
            "aggregate_metrics": pd.read_parquet(root / "aggregate_metrics.parquet"),
            "paired": pd.read_parquet(root / "paired_baseline_differences.parquet"),
            "coverage": pd.read_parquet(root / "coverage.parquet"),
            "dataset_sessions": pd.read_parquet(root / "dataset_sessions.parquet"),
            "dataset_coverage": pd.read_parquet(root / "dataset_coverage.parquet"),
            "subgroups": pd.read_parquet(root / "subgroup_errors.parquet"),
            "conditions": pd.read_parquet(root / "conditions_after_prediction.parquet"),
            "qualifying_vs_actual": pd.read_parquet(
                root / "qualifying_predicted_vs_actual.parquet"
            ),
            "race_vs_actual": pd.read_parquet(root / "race_predicted_vs_actual.parquet"),
            "podium_vs_actual": pd.read_parquet(root / "podium_predicted_vs_actual.parquet"),
            "feature_audit": read_json(root / "feature_audit.json"),
            "ledger_claim": read_json(claim) if claim.exists() else None,
            "ledger_result": read_json(result) if result.exists() else None,
            "figures": {
                name: root / f"calibration-{name}.png"
                for name in ("development", "final")
                if (root / f"calibration-{name}.png").exists()
            },
        }

    def final_available(self) -> tuple[bool, str | None]:
        try:
            _ = self.final
        except (ArtifactUnavailable, ValueError, FileNotFoundError) as exc:
            return False, str(exc)
        return True, None
