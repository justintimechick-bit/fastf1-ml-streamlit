"""Source batches deliberately separate network loading from normalization."""

from dataclasses import dataclass, field

import pandas as pd

TRANSFORM_VERSION = "2.1.0"
SCHEMA_VERSION = 2
SESSION_CODES = {
    "Practice 1": "FP1",
    "Practice 2": "FP2",
    "Practice 3": "FP3",
    "Sprint Qualifying": "SQ",
    "Sprint": "S",
    "Qualifying": "Q",
    "Race": "R",
}


def session_id(event_id: str, kind: str) -> str:
    return f"{event_id}/{SESSION_CODES[kind]}"


@dataclass
class SourceBatch:
    frames: dict[str, pd.DataFrame] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


class SourceUnavailable(RuntimeError):
    """The public interface returned no usable component data; not a transient retry."""
