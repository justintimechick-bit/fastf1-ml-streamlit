"""Session identities and timezone rules shared across all layers."""

from enum import StrEnum

import pandas as pd


class SessionKind(StrEnum):
    PRACTICE_1 = "Practice 1"
    PRACTICE_2 = "Practice 2"
    PRACTICE_3 = "Practice 3"
    SPRINT_QUALIFYING = "Sprint Qualifying"
    SPRINT = "Sprint"
    QUALIFYING = "Qualifying"
    RACE = "Race"


PRACTICES = {SessionKind.PRACTICE_1, SessionKind.PRACTICE_2, SessionKind.PRACTICE_3}
GP_SESSIONS = PRACTICES | {SessionKind.QUALIFYING, SessionKind.RACE}
TASKS = ("qualifying", "race", "podium")
MODES = ("historical_reconstruction", "publication_snapshot")


def utc(value) -> pd.Timestamp:
    """Reject ambiguous naive timestamps rather than interpreting local time."""
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError(f"An explicit timezone is required: {value!r}")
    return stamp.tz_convert("UTC")


def event_key(year: int, round_number: int) -> str:
    if year not in (2024, 2025, 2026) or round_number < 1:
        raise ValueError("Only 2024–2026 Grand Prix events (round > 0) are supported")
    return f"{year}-{round_number:02d}"


def session_kind(name: str) -> SessionKind:
    # FastF1 also recognizes the former name; never confuse it with GP qualifying.
    return SessionKind("Sprint Qualifying" if name == "Sprint Shootout" else name)
