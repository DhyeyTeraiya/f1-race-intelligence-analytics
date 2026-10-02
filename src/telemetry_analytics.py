"""Telemetry-derived tire stint pace and degradation analytics."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "season",
    "round",
    "driver_code",
    "stint",
    "compound",
    "tire_age_laps",
    "lap_time_sec",
}


def _prepare_telemetry(telemetry: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize the bundled telemetry frame."""
    missing = REQUIRED_COLUMNS.difference(telemetry.columns)
    if missing:
        raise ValueError(f"telemetry is missing required columns: {sorted(missing)}")
    df = telemetry.copy()
    for column in ("season", "round", "stint", "tire_age_laps", "lap_time_sec"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["compound"] = df["compound"].astype(str).str.upper()
    df["driver_code"] = df["driver_code"].astype(str).str.upper()
    return df.dropna(subset=["season", "round", "stint", "tire_age_laps", "lap_time_sec"])


def build_stint_pace_summary(telemetry: pd.DataFrame, min_laps: int = 5) -> pd.DataFrame:
    """Summarize pace and modeled degradation for each driver tire stint.

    ``degradation_sec_per_lap`` is the least-squares slope of lap time against
    tire age within a stint. It is a descriptive telemetry signal, not a claim
    of causal tire-only degradation because fuel load and traffic are not fully
    observed in the bundled derived dataset.
    """
    if min_laps < 2:
        raise ValueError("min_laps must be at least 2")
    df = _prepare_telemetry(telemetry)
    rows: list[dict] = []
    group_columns = ["season", "round", "driver_code", "stint", "compound"]
    for keys, group in df.groupby(group_columns, dropna=False):
        group = group.sort_values("tire_age_laps")
        if len(group) < min_laps:
            continue
        x = group["tire_age_laps"].to_numpy(dtype=float)
        y = group["lap_time_sec"].to_numpy(dtype=float)
        slope = float(np.polyfit(x, y, 1)[0]) if np.unique(x).size > 1 else 0.0
        baseline_count = min(3, len(group))
        baseline = float(group["lap_time_sec"].head(baseline_count).median())
        keys_dict = dict(zip(group_columns, keys))
        rows.append(
            {
                **keys_dict,
                "stint_laps": int(len(group)),
                "start_tire_age": int(group["tire_age_laps"].min()),
                "end_tire_age": int(group["tire_age_laps"].max()),
                "baseline_lap_time_sec": round(baseline, 3),
                "best_lap_time_sec": round(float(y.min()), 3),
                "median_lap_time_sec": round(float(np.median(y)), 3),
                "degradation_sec_per_lap": round(slope, 5),
                "pace_loss_per_10_laps_sec": round(slope * 10, 3),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["season", "round", "driver_code", "stint"], ignore_index=True
    )


def build_degradation_curve(
    telemetry: pd.DataFrame,
    driver_code: str | None = None,
    compounds: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Return median lap pace by tire age for an interactive degradation chart."""
    df = _prepare_telemetry(telemetry)
    if driver_code and driver_code != "All drivers":
        df = df[df["driver_code"] == driver_code.upper()]
    if compounds:
        wanted = {str(value).upper() for value in compounds}
        df = df[df["compound"].isin(wanted)]
    if df.empty:
        return pd.DataFrame(columns=["compound", "tire_age_laps", "median_lap_time_sec", "q25_lap_time_sec", "q75_lap_time_sec", "lap_count"])
    curve = (
        df.groupby(["compound", "tire_age_laps"], as_index=False)["lap_time_sec"]
        .agg(
            median_lap_time_sec="median",
            q25_lap_time_sec=lambda values: values.quantile(0.25),
            q75_lap_time_sec=lambda values: values.quantile(0.75),
            lap_count="count",
        )
        .sort_values(["compound", "tire_age_laps"])
    )
    return curve.reset_index(drop=True)
