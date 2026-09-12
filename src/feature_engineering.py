"""
Feature Engineering Pipeline for Real Formula 1 Data (2021-2026)
Constructs predictive feature matrices from real FIA Grand Prix results,
grid positions, constructor strength, and driver form.
"""

import os
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)


def load_raw_data():
    races_path = os.path.join(RAW_DIR, "f1_races.csv")
    telemetry_path = os.path.join(RAW_DIR, "f1_lap_telemetry.csv")
    df_races = pd.read_csv(races_path)
    df_telemetry = pd.read_csv(telemetry_path) if os.path.exists(telemetry_path) else pd.DataFrame()
    return df_races, df_telemetry


def build_engineered_features(df_races: pd.DataFrame) -> pd.DataFrame:
    df = df_races.copy()

    # Sort chronologically
    df.sort_values(by=["season", "round", "grid_position"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 1. Driver Rolling Form (Points and Avg Finish in previous 4 races)
    df["driver_rolling_points"] = (
        df.groupby(["driver_code"])["points"]
        .transform(lambda x: x.shift(1).rolling(window=4, min_periods=1).mean())
        .fillna(0.0)
    )

    df["driver_rolling_finish"] = (
        df.groupby(["driver_code"])["finish_position"]
        .transform(lambda x: x.shift(1).rolling(window=4, min_periods=1).mean())
        .fillna(12.0)
    )

    # 2. Constructor Rolling Dominance (Constructor points over last 4 races)
    df["team_rolling_pts"] = (
        df.groupby(["constructor"])["points"]
        .transform(lambda x: x.shift(1).rolling(window=4, min_periods=1).sum())
        .fillna(0.0)
    )

    # 3. Grid Conversion & Interaction Terms
    df["grid_circuit_difficulty_interaction"] = df["grid_position"] * df["overtake_difficulty"]
    df["is_front_row"] = (df["grid_position"] <= 2).astype(int)
    df["is_top_5_grid"] = (df["grid_position"] <= 5).astype(int)
    df["is_top_10_grid"] = (df["grid_position"] <= 10).astype(int)

    # 4. Positions Gained/Lost
    df["positions_change"] = df["grid_position"] - df["finish_position"]

    # 5. Constructor Tier Categorization based on historical average points
    team_pts = df.groupby("constructor")["points"].mean()
    tier_map = {}
    for team, avg_pt in team_pts.items():
        if avg_pt >= 15.0:
            tier_map[team] = 1
        elif avg_pt >= 7.0:
            tier_map[team] = 2
        elif avg_pt >= 2.5:
            tier_map[team] = 3
        else:
            tier_map[team] = 4
    df["team_tier"] = df["constructor"].map(tier_map).fillna(3)

    # Save to processed directory
    processed_path = os.path.join(PROCESSED_DIR, "f1_features.csv")
    df.to_csv(processed_path, index=False)
    print(f"[SUCCESS] Real engineered features saved to {processed_path} ({len(df)} rows, {len(df.columns)} columns)")
    return df


def get_modeling_data(df: pd.DataFrame = None):
    if df is None:
        processed_path = os.path.join(PROCESSED_DIR, "f1_features.csv")
        if os.path.exists(processed_path):
            df = pd.read_csv(processed_path)
        else:
            df_races, _ = load_raw_data()
            df = build_engineered_features(df_races)

    feature_cols = [
        "grid_position",
        "overtake_difficulty",
        "is_street_circuit",
        "team_tier",
        "driver_rolling_points",
        "driver_rolling_finish",
        "team_rolling_pts",
        "grid_circuit_difficulty_interaction",
        "is_front_row",
        "is_top_5_grid",
        "is_top_10_grid",
        "pit_stops_count"
    ]

    target_podium = "podium_finish"
    target_winner = "race_winner"

    X = df[feature_cols].copy()
    y_podium = df[target_podium].copy()
    y_winner = df[target_winner].copy()

    # Chronological Train-Test Split: Train on 2021-2024, Test on real 2025 & 2026 seasons!
    train_mask = df["season"] < 2025
    test_mask = df["season"] >= 2025

    X_train, X_test = X[train_mask], X[test_mask]
    y_train_podium, y_test_podium = y_podium[train_mask], y_podium[test_mask]
    y_train_winner, y_test_winner = y_winner[train_mask], y_winner[test_mask]

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train_podium": y_train_podium,
        "y_test_podium": y_test_podium,
        "y_train_winner": y_train_winner,
        "y_test_winner": y_test_winner,
        "feature_cols": feature_cols,
        "df_full": df
    }


if __name__ == "__main__":
    races, _ = load_raw_data()
    build_engineered_features(races)
