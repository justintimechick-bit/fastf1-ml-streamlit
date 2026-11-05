"""Shift by complete calendar weekends, then aggregate driver/team observations."""

import numpy as np
import pandas as pd

from f1ml.domain import utc
from f1ml.evaluation.splits import gp_session
from f1ml.features.dictionary import WINDOWS
from f1ml.features.pace import boolean
from f1ml.features.source import audited_spec, available


def team_history(year, team_id, explicit=None):
    if explicit is not None and pd.notna(explicit) and str(explicit).strip():
        return str(explicit)
    return f"unverified:{year}:{team_id}"


def normalized_results(raw):
    """Keep source positions separate from comparable, complete-field rank features."""
    result = raw.copy()
    if result.empty:
        return result
    if result.driver_id.isna().any() or result.driver_id.duplicated().any():
        raise ValueError("Result rows need unique stable driver IDs")
    n = len(result)
    rank = pd.to_numeric(result.official_position, errors="coerce")
    ambiguous = rank.notna() & rank.duplicated(keep=False)
    if "position_ambiguous" in result:
        ambiguous |= result.position_ambiguous.fillna(True).astype(bool)
    valid = rank.between(1, n) & rank.mod(1).eq(0) & ~ambiguous
    if (
        "official_field_complete" in result
        and not result.official_field_complete.fillna(False).all()
    ):
        valid[:] = False
    result["rank_norm"] = ((rank - 1) / (n - 1)).where(valid) if n > 1 else np.nan
    result["field_n"] = n
    result["retirement"] = (
        boolean(result.is_retired)
        if "is_retired" in result
        else pd.Series(pd.NA, index=result.index)
    )
    return result


def shifted_events(source, manifest, target, cutoff):
    """Strict event shift; late sessions/overlapping weekends cannot sneak into history."""
    start = min(utc(s["scheduled_start"]) for s in target["sessions"])
    output = []
    for event in source.events:
        if (
            event["event_id"] == target["event_id"]
            or event["event_id"] not in manifest["development"]
        ):
            continue
        race = gp_session(event, "Race")
        if (
            race.get("completed")
            and race.get("ended_at")
            and utc(race["ended_at"]) < start
            and utc(race["ended_at"]) < cutoff
            and utc(race["scheduled_start"]) < start
        ):
            output.append(event)
    return sorted(
        output, key=lambda e: (utc(gp_session(e, "Race")["scheduled_start"]), e["event_id"])
    )


def historical_observations(source, manifest, target, cutoff, mode):
    weekends = shifted_events(source, manifest, target, cutoff)
    frames, audit = [], []
    for event in weekends:
        for kind in ("Qualifying", "Race"):
            spec = gp_session(event, kind)
            if not spec.get("official_results") or not available(spec, cutoff, mode):
                continue
            raw = source.component(spec, "results")
            audit.append(audited_spec(event, spec))
            frame = normalized_results(raw)
            if frame.empty:
                continue
            frame["event_id"] = event["event_id"]
            frame["season"] = event["year"]
            frame["kind"] = kind
            frame["circuit_id"] = event.get("circuit_id")
            frame["history_id"] = [
                team_history(
                    event["year"], row.get("team_id", "unknown"), row.get("team_history_id")
                )
                for row in frame.to_dict("records")
            ]
            frames.append(frame)
    return weekends, pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(), audit


def statistics(weekends, observations, driver, target, history_id, circuit):
    result = {}

    def put(name, values):
        values = pd.to_numeric(pd.Series(values, dtype="object"), errors="coerce").dropna()
        result[name] = float(values.mean()) if len(values) else np.nan
        result[name + "_missing"] = int(not len(values))
        return len(values)

    for kind, prefix in (("Qualifying", "qualifying"), ("Race", "race")):
        frame = (
            observations.loc[observations.kind.eq(kind)]
            if not observations.empty
            else pd.DataFrame()
        )
        mine = frame.loc[frame.driver_id.eq(driver)] if not frame.empty else frame
        previous = weekends[-1]["event_id"] if weekends else None
        prior = mine.loc[mine.event_id.eq(previous)].rank_norm.tolist() if not mine.empty else []
        put(f"previous_{prefix}_rank", prior)
        for window in WINDOWS:
            ids = [e["event_id"] for e in weekends[-window:]]
            recent = mine.loc[mine.event_id.isin(ids)] if not mine.empty else mine
            count = put(
                f"{prefix}_rank_mean_{window}",
                recent.rank_norm.tolist() if not recent.empty else [],
            )
            result[f"{prefix}_rank_count_{window}"] = count
            deltas = []
            if not recent.empty:
                for row in recent.loc[recent.season.eq(target["year"])].itertuples():
                    peers = frame.loc[
                        frame.event_id.eq(row.event_id)
                        & frame.history_id.eq(row.history_id)
                        & ~frame.driver_id.eq(driver)
                    ].rank_norm.dropna()
                    if pd.notna(row.rank_norm) and len(peers):
                        deltas.append(row.rank_norm - peers.mean())
            result[f"teammate_{prefix}_delta_count_{window}"] = put(
                f"teammate_{prefix}_delta_mean_{window}", deltas
            )
            team = (
                frame.loc[
                    frame.event_id.isin(ids)
                    & frame.season.eq(target["year"])
                    & frame.history_id.eq(history_id)
                ]
                if not frame.empty
                else frame
            )
            strengths = (
                team.groupby("event_id").rank_norm.mean().dropna().tolist()
                if not team.empty
                else []
            )
            result[f"team_{prefix}_strength_count_{window}"] = put(
                f"team_{prefix}_strength_mean_{window}", strengths
            )
        prior_ids = [e["event_id"] for e in weekends if e["year"] < target["year"]][-5:]
        old = (
            frame.loc[frame.event_id.isin(prior_ids) & frame.history_id.eq(history_id)]
            if not frame.empty
            else frame
        )
        strengths = (
            old.groupby("event_id").rank_norm.mean().dropna().tolist() if not old.empty else []
        )
        result[f"team_prior_{prefix}_count"] = put(f"team_prior_{prefix}_strength", strengths)
        same = (
            mine.loc[mine.circuit_id.eq(circuit)].rank_norm.dropna().tolist()
            if not mine.empty and circuit != "unknown"
            else []
        )
        result[f"circuit_prior_{prefix}_count"] = put(f"circuit_prior_{prefix}_rank", same)
    for metric, column in (("podium", "podium"), ("retirement", "retirement")):
        race = (
            observations.loc[observations.kind.eq("Race") & observations.driver_id.eq(driver)]
            if not observations.empty
            else observations
        )
        for window in WINDOWS:
            ids = [e["event_id"] for e in weekends[-window:]]
            selected = race.loc[race.event_id.isin(ids)] if not race.empty else race
            values = selected[column].tolist() if not selected.empty and column in selected else []
            result[f"{metric}_count_{window}"] = put(f"{metric}_rate_{window}", values)
    result.update(
        history_count=result["race_rank_count_5"],
        history_quali_mean=result["qualifying_rank_mean_5"],
        history_race_mean=result["race_rank_mean_5"],
        history_podium_rate=result["podium_rate_5"],
    )
    return result
