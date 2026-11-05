"""Small, validated configuration with no machine-specific paths."""

import tomllib
from dataclasses import asdict, dataclass, field
from math import isfinite
from pathlib import Path


@dataclass(frozen=True)
class IngestionConfig:
    max_attempts: int = 3
    retry_base_seconds: float = 1.0
    retry_max_seconds: float = 8.0
    retry_cooldown_hours: float = 24.0
    timing_circuit_event_threshold: int = 3
    timing_circuit_cooldown_hours: float = 24.0
    rate_limit_cooldown_hours: float = 1.0
    max_api_pages: int = 20
    team_mapping_path: str = "configs/team_mapping.json"

    def __post_init__(self):
        for key in ("max_attempts", "timing_circuit_event_threshold", "max_api_pages"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"ingestion.{key} must be a positive integer")
        for key in (
            "retry_base_seconds",
            "retry_max_seconds",
            "retry_cooldown_hours",
            "timing_circuit_cooldown_hours",
            "rate_limit_cooldown_hours",
        ):
            if (
                not isinstance(getattr(self, key), (int, float))
                or not isfinite(getattr(self, key))
                or getattr(self, key) < 0
            ):
                raise ValueError(f"ingestion.{key} must be nonnegative")
        if self.retry_max_seconds > 30 or self.max_attempts > 5:
            raise ValueError(
                "Ingestion retry bounds are at most 5 attempts and 30 seconds per wait"
            )


@dataclass(frozen=True)
class FeatureConfig:
    pace_time_bin_seconds: int = 300
    weather_max_age_seconds: int = 120
    track_temperature_bin_c: float = 5.0
    min_comparable_drivers: int = 2

    def __post_init__(self):
        for name in ("pace_time_bin_seconds", "weather_max_age_seconds", "min_comparable_drivers"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"features.{name} must be a positive integer")
        if self.min_comparable_drivers < 2:
            raise ValueError("Comparable pace needs at least two drivers")
        if not isfinite(self.track_temperature_bin_c) or self.track_temperature_bin_c <= 0:
            raise ValueError("features.track_temperature_bin_c must be positive and finite")


@dataclass(frozen=True)
class QualifyingConfig:
    """Small, predeclared search; outer validation labels never select parameters."""

    min_training_events: int = 2
    min_2026_training_events: int = 2
    max_inner_folds: int = 3
    ridge_alphas: tuple[float, ...] = (1.0, 10.0, 100.0)
    forest_depths: tuple[int, ...] = (4, 8)
    forest_trees: int = 80
    boosting_rates: tuple[float, ...] = (0.05, 0.1)
    boosting_iterations: int = 100
    recency_half_lives: tuple[float, ...] = (6.0, 12.0)
    season_2026_weights: tuple[float, ...] = (2.0, 4.0)

    def __post_init__(self):
        for name in (
            "min_training_events",
            "min_2026_training_events",
            "max_inner_folds",
            "forest_trees",
            "boosting_iterations",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"qualifying.{name} must be a positive integer")
        for name in (
            "ridge_alphas",
            "forest_depths",
            "boosting_rates",
            "recency_half_lives",
            "season_2026_weights",
        ):
            values = tuple(getattr(self, name))
            if not values or any(not isfinite(x) or x <= 0 for x in values):
                raise ValueError(f"qualifying.{name} must contain positive finite values")
            object.__setattr__(self, name, values)
        if any(type(x) is not int for x in self.forest_depths):
            raise ValueError("qualifying.forest_depths must be integers")
        if min(self.season_2026_weights) <= 1:
            raise ValueError("2026 season weights must exceed the 2025 weight of 1")


@dataclass(frozen=True)
class RaceConfig(QualifyingConfig):
    """Race/podium searches share the historical strategies, with temporal postprocessing."""

    logistic_cs: tuple[float, ...] = (0.1, 1.0, 10.0)
    postprocess_folds: int = 2
    min_calibration_events: int = 2
    min_threshold_events: int = 2
    thresholds: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    calibration_bins: int = 5

    def __post_init__(self):
        super().__post_init__()
        for name in (
            "postprocess_folds",
            "min_calibration_events",
            "min_threshold_events",
            "calibration_bins",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"race.{name} must be a positive integer")
        for name in ("logistic_cs", "thresholds"):
            values = tuple(getattr(self, name))
            if not values or any(not isfinite(x) or x <= 0 for x in values):
                raise ValueError(f"race.{name} must contain positive finite values")
            object.__setattr__(self, name, values)
        if any(x >= 1 for x in self.thresholds):
            raise ValueError("race.thresholds must be strictly between 0 and 1")
        if self.min_calibration_events > self.postprocess_folds:
            raise ValueError("Calibration minimum exceeds reserved postprocess folds")
        if self.min_threshold_events > self.postprocess_folds:
            raise ValueError("Threshold minimum exceeds reserved postprocess folds")


@dataclass(frozen=True)
class ExperimentConfig:
    max_sessions: int = 4
    max_laps_per_driver: int = 3
    max_attempts: int = 2
    failure_cooldown_hours: float = 24.0
    max_sample_gap_seconds: float = 1.0
    min_channel_coverage: float = 0.8
    full_throttle_percent: float = 99.0
    families: tuple[str, ...] = ("linear", "random_forest", "gradient_boosting")
    strategies: tuple[str, ...] = ("equal",)
    min_paired_events: int = 3
    min_observed_fraction: float = 0.25
    min_position_improvement: float = 0.05
    min_log_loss_improvement: float = 0.005
    max_runtime_ratio: float = 3.0

    def __post_init__(self):
        for key in ("max_sessions", "max_laps_per_driver", "max_attempts", "min_paired_events"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"experiments.{key} must be a positive integer")
        if self.max_attempts > 3 or self.max_sessions > 12:
            raise ValueError("Selective capture permits at most 12 sessions and 3 attempts")
        for key in (
            "failure_cooldown_hours",
            "max_sample_gap_seconds",
            "min_channel_coverage",
            "full_throttle_percent",
            "min_observed_fraction",
            "min_position_improvement",
            "min_log_loss_improvement",
            "max_runtime_ratio",
        ):
            value = getattr(self, key)
            if not isfinite(value) or value < 0:
                raise ValueError(f"experiments.{key} must be finite and nonnegative")
        if not 0 < self.min_channel_coverage <= 1 or not 0 <= self.min_observed_fraction <= 1:
            raise ValueError("Experiment coverage fractions must lie in [0, 1]")
        if not 0 < self.full_throttle_percent <= 100 or self.max_sample_gap_seconds <= 0:
            raise ValueError("Invalid telemetry threshold/gap")
        for key, allowed in (
            ("families", {"linear", "random_forest", "gradient_boosting"}),
            ("strategies", {"2026_only", "equal", "weighted"}),
        ):
            value = tuple(getattr(self, key))
            if not value or len(set(value)) != len(value) or not set(value) <= allowed:
                raise ValueError(f"Unknown or duplicate experiments.{key}")
            object.__setattr__(self, key, value)


@dataclass(frozen=True)
class Config:
    years: tuple[int, ...] = (2024, 2025, 2026)
    evaluation_year: int = 2026
    final_test_weekends: int = 4
    min_training_weekends: int = 3
    min_validation_weekends: int = 1
    history_window: int = 5
    cache_dir: str = "data/cache/fastf1"
    log_level: str = "INFO"
    random_seed: int = 42
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    qualifying: QualifyingConfig = field(default_factory=QualifyingConfig)
    race: RaceConfig = field(default_factory=RaceConfig)
    experiments: ExperimentConfig = field(default_factory=ExperimentConfig)

    def __post_init__(self):
        if tuple(self.years) != (2024, 2025, 2026) or self.evaluation_year != 2026:
            raise ValueError("This phase supports exactly 2024–2026; evaluation is on 2026.")
        for key in (
            "final_test_weekends",
            "min_training_weekends",
            "min_validation_weekends",
            "history_window",
        ):
            value = getattr(self, key)
            if type(value) is not int or value < 1:
                raise ValueError(f"{key} must be a positive integer")

    def to_dict(self):
        return asdict(self)


def load_config(path: str | Path | None = None) -> Config:
    if path is None:
        return Config()
    with Path(path).open("rb") as stream:
        values = tomllib.load(stream)
    if "years" in values:
        values["years"] = tuple(values["years"])
    if "ingestion" in values:
        values["ingestion"] = IngestionConfig(**values["ingestion"])
    if "features" in values:
        values["features"] = FeatureConfig(**values["features"])
    if "qualifying" in values:
        values["qualifying"] = QualifyingConfig(**values["qualifying"])
    if "race" in values:
        values["race"] = RaceConfig(**values["race"])
    if "experiments" in values:
        values["experiments"] = ExperimentConfig(**values["experiments"])
    return Config(**values)
