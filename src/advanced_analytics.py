"""Advanced scenario analytics for the F1 Race Intelligence dashboard."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from strategy_simulator import StrategySimulator


def build_prediction_sensitivity(
    base_input: pd.DataFrame,
    model_artifact: dict,
    grid_positions: Iterable[int] = range(1, 23),
    pit_stops: Iterable[int] = range(1, 4),
) -> pd.DataFrame:
    """Evaluate podium probability across grid and pit-stop what-if scenarios.

    The selected driver's other inputs remain fixed. This is a scenario analysis,
    not a calibrated confidence interval, and is intentionally labeled that way
    in the dashboard.
    """
    if base_input.empty:
        raise ValueError("base_input must contain one modeling row")
    if not model_artifact or "pipeline" not in model_artifact or "features" not in model_artifact:
        raise ValueError("model_artifact must contain pipeline and features")

    row = base_input.iloc[0].copy()
    features = list(model_artifact["features"])
    pipeline = model_artifact["pipeline"]
    difficulty = float(row.get("overtake_difficulty", 0))
    scenarios: list[dict] = []

    for grid in grid_positions:
        grid = int(grid)
        for stops in pit_stops:
            stops = int(stops)
            scenario = row.copy()
            scenario["grid_position"] = grid
            scenario["pit_stops_count"] = stops
            scenario["grid_circuit_difficulty_interaction"] = grid * difficulty
            scenario["is_front_row"] = int(grid <= 2)
            scenario["is_top_5_grid"] = int(grid <= 5)
            scenario["is_top_10_grid"] = int(grid <= 10)
            probability = float(pipeline.predict_proba(pd.DataFrame([scenario])[features])[0][1] * 100)
            scenarios.append(
                {
                    "grid_position": grid,
                    "pit_stops": stops,
                    "podium_probability": round(probability, 3),
                }
            )

    return pd.DataFrame(scenarios)


def build_strategy_robustness_grid(
    simulator: StrategySimulator | None = None,
    gap_values: Iterable[float] = np.arange(0.5, 4.1, 0.5),
    tire_ages: Iterable[int] = range(10, 41, 5),
    chaser_pit_lap: int = 22,
    leader_pit_lap: int = 24,
    chaser_compound: str = "HARD",
) -> pd.DataFrame:
    """Build a net-margin grid for undercut decision robustness.

    Each row is one scenario. Positive net margin means the modeled undercut
    gains time relative to the starting gap; negative values favor extending.
    """
    simulator = simulator or StrategySimulator()
    rows: list[dict] = []
    for gap in gap_values:
        for tire_age in tire_ages:
            result = simulator.simulate_undercut(
                gap_before_pit_sec=float(gap),
                chaser_pit_lap=int(chaser_pit_lap),
                leader_pit_lap=int(leader_pit_lap),
                chaser_compound=chaser_compound,
                laps_on_leader_tire=int(tire_age),
            )
            rows.append(
                {
                    "gap_before_pit_sec": float(gap),
                    "leader_tire_age_laps": int(tire_age),
                    "net_margin_sec": float(result.get("net_margin_sec", np.nan)),
                    "recommendation": result.get("recommendation", "UNAVAILABLE"),
                    "success": bool(result.get("success", False)),
                }
            )
    return pd.DataFrame(rows)
