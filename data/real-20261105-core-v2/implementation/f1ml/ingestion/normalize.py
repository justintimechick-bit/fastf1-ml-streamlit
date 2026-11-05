"""Normalize official result rows without manufacturing a finishing order."""

import json

import numpy as np
import pandas as pd

RESULT_COLUMNS = [
    "driver_id",
    "driver_number",
    "team_id",
    "official_position",
    "classification_code",
    "outcome",
    "podium",
]
PRACTICE_COLUMNS = ["driver_id", "best_lap_seconds", "clean_laps"]


def positive_position(series: pd.Series) -> pd.Series:
    numbers = pd.to_numeric(series, errors="coerce")
    return numbers.where((numbers > 0) & (numbers % 1 == 0)).astype("Float64")


def validate_ids(frame: pd.DataFrame) -> None:
    if frame.driver_id.isna().any() or frame.driver_id.astype(str).str.strip().eq("").any():
        raise ValueError("Missing stable driver identity; supply a reviewed identity mapping")
    if frame.driver_id.duplicated().any():
        raise ValueError("Duplicate driver identity in a session")


def normalize_results(
    raw: pd.DataFrame, kind: str, *, allow_ambiguous_positions=False
) -> pd.DataFrame:
    """Accept Jolpica result rows, not FastF1's timing-derived result fallback.

    position is the provider's official order, including any ordered DNS/DSQ tail.
    positionText/status are retained separately; missing positions stay missing.
    """
    if kind not in ("Qualifying", "Race"):
        raise ValueError("Only Grand Prix qualifying/race labels are supported")
    if raw.empty:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    raw = raw.reset_index(drop=True)
    result = pd.DataFrame(index=raw.index)
    result["driver_id"] = raw["driverId"].astype("string")
    result["driver_number"] = raw.get("number", pd.Series(index=raw.index, dtype="str"))
    result["team_id"] = raw.get("constructorId", pd.Series(index=raw.index, dtype="str"))
    position = positive_position(raw["position"])
    codes = raw.get("positionText", pd.Series("", index=raw.index)).fillna("").astype(str)
    status = raw.get("status", pd.Series("", index=raw.index)).fillna("").astype(str)
    result["official_position"] = position
    result["classification_code"] = codes
    upper = codes.str.upper().str.strip()
    dsq = upper.isin(["D", "DSQ"]) | status.str.contains("disqual", case=False)
    excluded = upper.eq("E") | status.str.contains("excluded", case=False)
    withdrawn = upper.eq("W") | status.str.contains("withdr", case=False)
    dns = upper.eq("DNS") | status.str.contains(r"did not start|\bdns\b", case=False)
    dnq = upper.eq("F") | status.str.contains("did not qualify|failed to qualify", case=False)
    nc = upper.isin(["N", "R", "DNF"]) | status.str.contains("not classified", case=False)
    finished = status.str.match(r"(?i)^(finished|lapped|\+\s*\d+\s*laps?)$")
    result["is_classified"] = pd.Series(pd.NA, index=raw.index, dtype="boolean")
    classified = positive_position(codes).notna() | finished
    if kind == "Qualifying":
        classified |= position.notna()
    result.loc[classified, "is_classified"] = True
    result.loc[nc | dsq | excluded | withdrawn | dns | dnq, "is_classified"] = False
    result["is_retired"] = pd.Series(pd.NA, index=raw.index, dtype="boolean")
    if kind == "Race":
        known_status = status.str.strip().ne("")
        result.loc[known_status | upper.eq("R"), "is_retired"] = False
        retirement = (
            known_status
            & ~finished
            & ~(dsq | excluded | withdrawn | dns | dnq)
            & ~status.str.contains("not classified", case=False)
        ) | upper.eq("R")
        result.loc[retirement, "is_retired"] = True
    result["outcome"] = "unknown"
    result.loc[position.notna(), "outcome"] = "ordered"
    result.loc[result.is_retired.eq(True) & result.is_classified.eq(True), "outcome"] = (
        "retired_classified"
    )
    result.loc[result.is_retired.eq(True) & result.is_classified.isna(), "outcome"] = (
        "retired_classification_unknown"
    )
    result.loc[nc, "outcome"] = "non_classified"
    result.loc[upper.eq("R"), "outcome"] = "retired_non_classified"
    result.loc[dnq, "outcome"] = "dnq"
    result.loc[withdrawn, "outcome"] = "withdrawn"
    result.loc[dns, "outcome"] = "dns"
    result.loc[excluded, "outcome"] = "excluded"
    result.loc[dsq, "outcome"] = "dsq"
    result["podium"] = pd.Series(pd.NA, index=raw.index, dtype="Float64")
    if kind == "Race":
        result.loc[position.notna(), "podium"] = position.between(1, 3).astype(float)
        result.loc[
            result.outcome.isin(
                [
                    "dns",
                    "dsq",
                    "excluded",
                    "withdrawn",
                    "dnq",
                    "non_classified",
                    "retired_non_classified",
                ]
            ),
            "podium",
        ] = 0.0
    result["position_missing"] = position.isna()
    result["status_missing"] = status.str.strip().eq("")
    result["classification_missing"] = codes.str.strip().eq("")
    validate_ids(result)
    known = result.official_position.dropna()
    result["position_ambiguous"] = position.notna() & position.duplicated(keep=False)
    if known.duplicated().any() and not allow_ambiguous_positions:
        raise ValueError("Duplicate official positions require source review")
    # Preserve every source column, including literal position/status, grid and Q times.
    # They are label-side ingestion fields; the model allowlist remains unchanged.
    preserved = raw.map(lambda value: None if pd.isna(value) else str(value)).add_prefix("source_")
    preserved["source_record_json"] = raw.apply(
        lambda row: json.dumps(
            {k: None if pd.isna(v) else str(v) for k, v in row.items()}, sort_keys=True
        ),
        axis=1,
    )
    return pd.concat([result, preserved], axis=1).reset_index(drop=True)


def summarize_practice(session) -> pd.DataFrame:
    """Practice pace only: accurate, nondeleted, timed, non-pit laps.

    No weather, pit counts, race laps or result positions are exposed here.
    Unknown deletion/accuracy flags are excluded conservatively.
    """
    laps = pd.DataFrame(session.laps).copy()
    if laps.empty:
        return pd.DataFrame(columns=PRACTICE_COLUMNS)
    mask = (
        laps.IsAccurate.eq(True)
        & laps.Deleted.eq(False)
        & laps.PitInTime.isna()
        & laps.PitOutTime.isna()
        & laps.LapTime.notna()
    )
    clean = laps.loc[mask].copy()
    clean["best_lap_seconds"] = clean.LapTime.dt.total_seconds()
    clean = clean.loc[np.isfinite(clean.best_lap_seconds) & (clean.best_lap_seconds > 0)]
    if clean.empty:
        return pd.DataFrame(columns=PRACTICE_COLUMNS)
    identities = pd.DataFrame(session.results)[["DriverNumber", "DriverId"]]
    identities = identities.rename(
        columns={"DriverNumber": "driver_number", "DriverId": "driver_id"}
    )
    validate_ids(identities)
    if identities.driver_number.duplicated().any():
        raise ValueError("Duplicate driver numbers in practice")
    summary = clean.groupby("DriverNumber").best_lap_seconds.agg(["min", "count"]).reset_index()
    summary = summary.rename(
        columns={"DriverNumber": "driver_number", "min": "best_lap_seconds", "count": "clean_laps"}
    )
    summary = summary.merge(identities, on="driver_number", how="left", validate="one_to_one")
    validate_ids(summary)
    return summary[PRACTICE_COLUMNS]
