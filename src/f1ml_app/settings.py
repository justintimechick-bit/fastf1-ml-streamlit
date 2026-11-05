"""Artifact locations for the interface. Defaults match the Phase 3–7 commands in the README."""

import os
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

DEFAULTS = {
    "dataset": "data/phase3-final/dataset",
    "manifest": "data/phase3-final/manifest.json",
    "features": "data/phase3-final/features",
    "development_cutoffs": "data/phase3-final/cutoffs.csv",
    "qualifying": "data/phase4-final",
    "race": "data/phase5-final",
    "experiments": "data/phase6-verified-experiments",
    "lock": "data/phase7-lock",
    "final": "data/phase7-final",
    "ledger": "data/final-evaluation-ledger",
    "catalog": "data/ingestion-v2",
    "real_experiment": "data/real-20261105-core-v2",
}


@dataclass(frozen=True)
class AppPaths:
    root: Path
    dataset: Path
    manifest: Path
    features: Path
    development_cutoffs: Path
    qualifying: Path
    race: Path
    experiments: Path
    lock: Path
    final: Path
    ledger: Path
    catalog: Path
    real_experiment: Path

    def describe(self) -> dict[str, str]:
        return {
            f.name: str(getattr(self, f.name).relative_to(self.root))
            if getattr(self, f.name).is_relative_to(self.root)
            else str(getattr(self, f.name))
            for f in fields(self)
            if f.name != "root"
        }


def project_root() -> Path:
    """Explicit override, then the working directory, then this source checkout."""
    if os.environ.get("F1ML_PROJECT_ROOT"):
        return Path(os.environ["F1ML_PROJECT_ROOT"]).resolve()
    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "configs").is_dir():
            return candidate
    return Path(__file__).resolve().parents[2]


def load_paths(config: Path | None = None, root: Path | None = None) -> AppPaths:
    """Read optional ``[paths]`` overrides from ``configs/app.toml`` (or ``F1ML_APP_CONFIG``)."""
    root = (root or project_root()).resolve()
    values = dict(DEFAULTS)
    config = config or (
        Path(os.environ["F1ML_APP_CONFIG"])
        if os.environ.get("F1ML_APP_CONFIG")
        else root / "configs" / "app.toml"
    )
    if config.exists():
        with config.open("rb") as stream:
            overrides = tomllib.load(stream).get("paths", {})
        unknown = set(overrides) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown app path keys: {sorted(unknown)}")
        values.update(overrides)
    resolved = {
        key: (Path(value) if Path(value).is_absolute() else root / value)
        for key, value in values.items()
    }
    return AppPaths(root=root, **resolved)
