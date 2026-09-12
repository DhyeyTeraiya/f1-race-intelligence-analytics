"""
Feature Engineering Pipeline for F1 Race Intelligence & Modeling
Transforms raw race entries, qualifying deltas, and circuit metrics
into ML-ready feature matrices for classification and regression tasks.
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
    df_telemetry = pd.read_csv(telemetry_path)
    return df_races, df_telemetry


def build_engineered_features(df_races: pd.DataFrame) -> pd.DataFrame:
    df = df_races.copy()

    # Sort sequentially by season, round, and grid position
    df.sort_values(by=["season", "round", "grid_position"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 1. Driver Rolling Form (Points and Avg Finish in prior races)
    df["driver_rolling_points"] = (
        df.groupby(["season", "driver_code"])["points"]
        .transform(lambda x: x.shift(1).rolling(window=4, min_periods=1).mean())
        .fillna(0.0)
    )

    df["driver_rolling_finish"] = (
        df.groupby(["season", "driver_code"])["finish_position"]
        .transform(lambda x: x.shift(1).rolling(window=4, min_periods=1).mean())
        .fillna(10.0)
    )

    # 2. Team Rolling Dominance (Constructor points percentage)
    df["team_rolling_pts"] = (
        df.groupby(["season", "constructor"])["points"]
        .transform(lambda x: x.shift(1).rolling(window=4, min_periods=1).sum())
        .fillna(0.0)
    )

    # 3. Grid Conversion Potential: Interaction between grid position and overtake difficulty
    # Higher difficulty makes grid position more determinant of finish
    df["grid_circuit_difficulty_interaction"] = df["grid_position"] * df["overtake_difficulty"]

    # 4. Qualifying Delta Normalized
    df["quali_delta_clipped"] = df["qualifying_delta"].clip(0.0, 3.0)

    # 5. Categorical Mappings
    downforce_map = {"Low": 1, "Medium": 2, "High": 3}
    df["downforce_numeric"] = df["downforce_level"].map(downforce_map).fillna(2)

    weather_map = {"Dry": 0, "Mixed": 1, "Wet": 2}
    df["weather_numeric"] = df["weather"].map(weather_map).fillna(0)

    # 6. Positions Gained / Lost (Target & Diagnostic metric)
    df["positions_change"] = df["grid_position"] - df["finish_position"]

    # 7. Front-row start flag
    df["is_front_row"] = (df["grid_position"] <= 2).astype(int)
    df["is_top_5_grid"] = (df["grid_position"] <= 5).astype(int)

    # Save to processed directory
    processed_path = os.path.join(PROCESSED_DIR, "f1_features.csv")
    df.to_csv(processed_path, index=False)
    print(f"Engineered features saved to {processed_path} ({len(df)} rows, {len(df.columns)} columns)")
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
        "qualifying_delta",
        "team_tier",
        "driver_skill",
        "overtake_difficulty",
        "downforce_numeric",
        "weather_numeric",
        "track_temp_c",
        "driver_rolling_points",
        "driver_rolling_finish",
        "team_rolling_pts",
        "grid_circuit_difficulty_interaction",
        "is_front_row",
        "is_top_5_grid"
    ]

    target_podium = "podium_finish"
    target_winner = "race_winner"

    X = df[feature_cols].copy()
    y_podium = df[target_podium].copy()
    y_winner = df[target_winner].copy()

    # Chronological Train-Test Split (e.g. 2021-2023 Train, 2024 Test)
    train_mask = df["season"] < 2024
    test_mask = df["season"] == 2024

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
